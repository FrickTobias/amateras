"""
Finds big cells and calculates shortest path in between them
"""

import logging
import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
from collections import defaultdict
import math
import inspect
from amateras import utils
from typing import List, Tuple
from argparse import ArgumentTypeError
from python_tsp.heuristics import solve_tsp_local_search
from python_tsp.distances import euclidean_distance_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s %(levelname)s %(asctime)s:\t%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

ARG_MIN_VAL = 0
ARG_MAX_VAL = 1


def add_arguments(parser):
    parser.add_argument("input", help="input image")

    # TODO: Make optional and maybe just write coords
    parser.add_argument("--qc-outdir", help="Output directory")
    parser.add_argument("--n-cells", type=int, default=20, help="Number of cells find.")
    parser.add_argument("--details", action="store_true", help="Writes extra files")
    parser.add_argument("--size-min", type=int, default=30,
                        help="Min size for cells. Default: %(default)s")
    parser.add_argument("--size-max", type=int, default=500,
                        help="Max size for cells. Default: %(default)s")

    # TODO: Add that some of these are between 0 and 1
    parser.add_argument("--convexity-min", type=range_limited_float_type, default=0.875,
                        help="Min convexity for cells. Default: %(default)s")
    parser.add_argument("--inertia-min", type=range_limited_float_type, default=0.6,
                        metavar="",
                        help="Min inertia for cells. Default: %(default)s")

    # TODO: Add arguments for final filtering
    # TODO: Add argument for logfile writing
    return parser


def main(args):
    big_cells = find_big_cells(
        args.input, args.n_cells, args.qc_outdir, args.details, args.size_min,
        args.size_max,
        args.convexity_min, args.inertia_min
    )
    big_cells_ordered, dist = find_short_path(big_cells, args.qc_outdir)
    print(big_cells_ordered)


def find_big_cells(input, n_cells: int, qc_outdir=None, details: bool = False,
                   size_min: int = 30, size_max: int = 500,
                   convexity_min: float = 0.875, inertia_min: float = 0.6):
    logger.info("Opening file")
    input_img = cv2.imread(input)

    if qc_outdir:
        logger.info("Setting up")
        mkdir(qc_outdir)
        function_name = inspect.stack()[0][3]
        if details:
            mkdir(f"{qc_outdir}/rejections")
            mkdir(f"{qc_outdir}/center-contours")

    # Add to img
    out = input_img.copy()
    out_big_cells = input_img.copy()

    # Remove huge objects
    logger.info("Masking dust")
    out = mask_dust(out, thresh_min=60, min_size=4000, details=details,
                    qc_outdir=qc_outdir)

    # Detect cells
    logger.info("Detecting cells")
    cell_contours = cell_detector_2(out, details=details, qc_outdir=qc_outdir)

    # Write to terminal
    detected_cells = len(cell_contours)
    logger.info(f"Detected: {detected_cells}")

    # Size filters
    logger.info("Filtering cells")
    cell_contours_sfilt = utils.filter_by_area(cell_contours, size_min=size_min,
                                         size_max=size_max)
    sfilt_cells = len(cell_contours_sfilt)
    logger.info(f"{size_min} < Size < {size_max}: {sfilt_cells}")

    # Inertia filters
    cell_contours_sfilt_ifilt = utils.filter_by_inertia(
        cell_contours_sfilt, threshold=inertia_min, keep_NA=True
    )
    sfilt_ifilt_cells = len(cell_contours_sfilt_ifilt)
    logger.info(f"Inertia > {inertia_min}: {sfilt_ifilt_cells}")

    # convexity filters
    cell_contours_final = utils.filter_for_convexity(
        cell_contours_sfilt_ifilt, convexity_threshold=convexity_min, keep_NA=True
    )
    cfilt_cells = len(cell_contours_final)
    logger.info(f"Convexity > {convexity_min}: {cfilt_cells} (green)")

    # TODO: Filter on circularity (maybe)
    # TODO: Filter on n_cells in mask

    areas = [cv2.contourArea(a) for a in cell_contours_final]

    # Save areas to histogram
    if qc_outdir:
        logger.info("Making histogram")
        save_histogram(areas, f"{qc_outdir}/histogram.png", bins=range(0, 500, 10),
                       title=f"AMATERAS cell areas, {size_min}<size<{size_max}, "
                             f"inrt>{inertia_min}, cxt>{convexity_min}",
                       x_axis="area", x_unit="px",
                       y_axis="count", y_unit="")

    # Set up dataframe for finding proximal cells
    logger.info("Making dataframes for biggest cell detections")
    all_cell_centroids = [utils.cnt_centroid(c) for c in cell_contours_sfilt]
    cf = CentroidFinder(all_cell_centroids)

    # TODO: Tidy this mess up
    # Create dataframe and get the xy positions for the biggest cells in list
    di = OrderedDict()
    di["area"] = areas
    di["contour"] = cell_contours_final
    centroids = [utils.cnt_centroid(c) for c in cell_contours_final]
    di["centroid"] = centroids
    di["cX"] = [c[0] for c in centroids]
    di["cY"] = [c[1] for c in centroids]

    df = pd.DataFrame(di)
    dfsort = df.sort_values(by="area", ascending=False)

    dfsort_a = dfsort.area.tolist()
    dfsort_c = dfsort.centroid.tolist()
    dfsort_cont = dfsort.contour.tolist()

    # Loop through centroids and find biggest cells
    logger.info(f"Locating the {n_cells} biggest cells")
    good_cells = 0
    big_cells = list()
    for i, (centroid, area, cnt) in enumerate(zip(dfsort_c, dfsort_a, dfsort_cont)):
        bbox = cv2.boundingRect(cnt)
        x, y, w, h = bbox
        cell_raw = input_img[y:y + h, x:x + w]

        # Find cell center contours
        center_contours = cell_center_detector(cell_raw, cnt, cnt_start=(x, y))

        # Filter one last time
        filter_pass = final_qc_filtering(
            center_contours, candidate_no=i, cell_img=cell_raw, details=details,
            qc_outdir=qc_outdir
        )

        if qc_outdir and details:
            out_cnt = cell_raw.copy()
            out_cnt = cv2.drawContours(out_cnt, center_contours, -1, (50, 255, 50), -1)
            cv2.imwrite(f"{qc_outdir}/center-contours/candidate-{i}-raw.tif", cell_raw)
            cv2.imwrite(f"{qc_outdir}/center-contours/candidate-{i}-cnt.tif", out_cnt)

        if filter_pass:
            # Find and mark proximal cells in output
            cX, cY = centroid
            sX1, sY1 = cX - 40, cY - 40
            sX2, sY2 = cX + 40, cY + 40
            proximals = cf.find_proximals(centroid, search_window=40)
            for proximal_c in proximals:
                out = add_distances_to_img(out, p1=centroid, p2=proximal_c)

            # Mark big cell in output
            big_cells.append(centroid)
            out = cv2.putText(out, f"big cell: {good_cells}", (sX1, sY1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 255), 1)
            out = cv2.rectangle(out, (sX1, sY1), (sX2, sY2), (50, 50, 255), 5)
            out_big_cells = cv2.putText(
                out_big_cells, f"big cell: {good_cells}", (sX1, sY1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 255), 1
            )
            out_big_cells = cv2.rectangle(out_big_cells, (sX1, sY1), (sX2, sY2),
                                          (50, 50, 255), 5)

            # Check if done
            good_cells += 1
            if good_cells >= n_cells:
                logger.info(f"All {n_cells} found")
                logger.info(f"Total no evaluated cells: {i}")
                break

    # Adding contours to output
    logger.info("Adding contours to image")
    out = cv2.drawContours(out, cell_contours, -1, (200, 200, 200), 1)
    out = cv2.drawContours(out, cell_contours_sfilt, -1, (75, 75, 175), 1)
    out = cv2.drawContours(out, cell_contours_sfilt_ifilt, -1, (75, 175, 175), 1)
    out = add_contours_to_img(out, contours=cell_contours_final, add_area=True,
                              add_centroid=True, color=(50, 255, 50))

    # Write output
    if qc_outdir:
        logger.info("Writing final file")
        cv2.imwrite(f"{qc_outdir}/detections-and-{n_cells}-biggest-cells.tif", out)
        cv2.imwrite(f"{qc_outdir}/{n_cells}-biggest-cells.tif", out_big_cells)

        logger.info("Finding picking path (only for example image)")
        ordered_big_cells, _ = find_short_path(big_cells)
        out_picking_path = out.copy()
        for point1, point2 in zip(ordered_big_cells, ordered_big_cells[1:]):
            out_picking_path = cv2.line(
                out_picking_path, point1, point2, [125, 255, 50], 2
            )
        tX, tY = ordered_big_cells[0]
        tX -= 20
        tY -= 60
        out_picking_path = cv2.putText(
            out_picking_path, "start", (tX, tY), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (125, 255, 50), 2
        )

        cv2.imwrite(f"{qc_outdir}/{function_name}.picking-path-example.tif",
                    out_picking_path)

        if details:
            logger.info("Writing QC outputs")
            cv2.imwrite(f"{qc_outdir}/{function_name}.input.tif", input_img)

    return big_cells


def add_distances_to_img(img, p1, p2):
    middle = middlepoint(p1, p2)
    text_pos = (middle[0] + 5, middle[1])
    dist = round(math.dist(p1, p2))

    img = cv2.putText(img, str(dist), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                      (255, 125, 50), 1)
    img = cv2.line(img, p1, p2, (255, 125, 50), 1)

    return img


def final_qc_filtering(center_contours, candidate_no: int, inertia_thresh: float = 0.6,
                       convexity_thresh: float = 0.8, circularity: float = 0.7,
                       cell_img=None, details: bool = False, qc_outdir=None):
    # Inits
    filter_fails = defaultdict(bool)
    # function_name = inspect.stack()[0][3]
    n_cells = len(center_contours)

    # Filter special case where only one cell and center is only 1 pixel
    if len(center_contours) == 1 and cv2.contourArea(center_contours[0]) < 2:
        filter_fails["special-case"] = True

    # Filter for contours for number of cells
    if n_cells < 1:
        filter_fails["no-cells"] = True
    elif n_cells > 1:
        filter_fails["several-cells"] = True
    else:
        # Since we know there is only one cell, we don't need it in a list
        center_contour = center_contours[0]

        # Filter for inertia (=elongation, 1 means height = width, where height = length
        # along principal axis)
        inertia = utils.cnt_inertia(center_contour)
        if inertia is not None and inertia < inertia_thresh:
            filter_fails[f"inertia-over-{inertia_thresh}"] = True

        # Filter for convexity (How many indents something has, where 1 means it does
        # not have any)
        convexity = utils.cnt_convexity(center_contour)
        if convexity is not None and convexity < convexity_thresh:
            filter_fails[f"convexity-under-{convexity_thresh}"] = True

        # Filter for circularity (A measurement of how circle-like the perimeter is of
        # the object, where a line < triangle < square < pentagon < ... < circle = 1)
        circularity = utils.cnt_circularity(center_contour)
        if circularity is not None and circularity < circularity:
            filter_fails[f"circularity-under-{circularity}"] = True

    # Write rejected to output
    if details:
        reasons = [reason for reason, filter_fail in filter_fails.items() if
                   filter_fail is True]
        cv2.imwrite(
            f"{qc_outdir}/rejections/candidate-{candidate_no}.{'.'.join(reasons)}.tif",
            cell_img
        )

    # Check if there are any fails
    if any(filter_fails.values()):
        filter_pass = False
    else:
        filter_pass = True

    return filter_pass


def mask_dust(img, thresh_min: int = 60, thresh_max: int = 255, min_size: int = 4000,
              blurring_kernel: int = 11, erosion_kernel_number: int = 5,
              erosion_iterations: int = 3, details: bool = False, qc_outdir=None):
    # Convert to gray for analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blurring
    blurred_median = cv2.medianBlur(gray, blurring_kernel)

    # Creating morphological erosion mask
    lower = thresh_min
    upper = 255
    inrange = cv2.inRange(blurred_median, lower, upper)
    erosion_kernel = np.ones((erosion_kernel_number, erosion_kernel_number), np.uint8)
    erosion = cv2.erode(inrange, erosion_kernel, iterations=erosion_iterations)

    # remove small objects in threshed
    contours, _ = cv2.findContours((255 - erosion), cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    bigcnts = []
    for cnt in contours:
        if min_size < cv2.contourArea(cnt):
            bigcnts.append(cnt)

    # Prep mask
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask = cv2.drawContours(mask, bigcnts, -1, (255, 255, 255), -1)
    mask_inv = 255 - mask

    # Mask image
    out = img.copy()
    masked = cv2.bitwise_and(out, mask_inv)
    if qc_outdir and details:
        function_name = inspect.stack()[0][3]
        cv2.imwrite(f"{qc_outdir}/{function_name}.1-input.tif", img)
        cv2.imwrite(f"{qc_outdir}/{function_name}.2-blurred_median.tif", blurred_median)
        cv2.imwrite(
            f"{qc_outdir}/{function_name}.3-blurred-inrange-{lower}-{upper}.tif",
            inrange
        )
        cv2.imwrite(f"{qc_outdir}/{function_name}.4-erosion.tif", erosion)
        cv2.imwrite(f"{qc_outdir}/{function_name}.5-huge-hairs-mask.tif", mask_inv)
    return masked


def cell_detector_2(img, blur_kernel: Tuple[int, int] = (3, 3), black_thresh: int = 70,
                    white_thresh: int = 125, details: bool = False, qc_outdir=None):
    # Find black spots
    blurred = cv2.blur(img, blur_kernel)
    blurred_gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _, black_thresh = cv2.threshold(blurred_gray, black_thresh, 255, cv2.THRESH_BINARY)
    black_thresh_inv = 255 - black_thresh

    # find white spots
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, white_thresh = cv2.threshold(img_gray, white_thresh, 255, cv2.THRESH_BINARY)

    # combine white + black
    combined = cv2.bitwise_or(black_thresh_inv, white_thresh)

    # Get contours of masked imagee
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if qc_outdir and details:
        # show results
        function_name = inspect.stack()[0][3]
        cv2.imwrite(f"{qc_outdir}/{function_name}.1-input.tif", img)
        cv2.imwrite(f"{qc_outdir}/{function_name}.2-blurred.tif", blurred)
        cv2.imwrite(f"{qc_outdir}/{function_name}.3-black-thresh.tif", black_thresh_inv)
        cv2.imwrite(f"{qc_outdir}/{function_name}.4-white-thresh.tif", white_thresh)
        cv2.imwrite(f"{qc_outdir}/{function_name}.5-combined-thresh.tif", combined)

        img_show = img.copy()
        cv2.drawContours(img_show, contours, -1, (50, 255, 50), 1)
        cv2.imwrite(f"{qc_outdir}/{function_name}.6-contours.tif", img_show)

    return contours


def cell_center_detector(roi, contour, cnt_start: Tuple[int, int] = (0, 0)):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Make mask from contour (otherwise cnt detection will find areas outside cell)
    contour = contour - cnt_start
    contour = [contour]
    mask = np.zeros(shape=gray.shape, dtype=np.uint8)
    mask = cv2.drawContours(mask, contour, -1, 255, -1)

    # Run dynamic thresholding on image to find cell centers (white spots)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 115, 1)

    # Apply mask, removing detections outside of cell
    thresh_masked = cv2.bitwise_and(mask, thresh)

    # Find contours of binary image (which should be one per cell if it works)
    contours, _ = cv2.findContours(thresh_masked, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    return contours


def cnt_convexity(cnt):
    hull = cv2.convexHull(cnt)

    cnt_area = cv2.contourArea(cnt)
    hull_area = cv2.contourArea(hull)

    if hull_area == 0:
        return None

    convexity = cnt_area / hull_area
    return convexity


def cnt_inertia(cnt):
    # Cannot approximate as ellipse if less than 5 points
    if len(cnt) < 5:
        return None

    ellipse = cv2.fitEllipse(cnt)
    center, shape, angle = ellipse
    width, height = shape

    inertia = width / height
    return inertia


def cnt_circularity(cnt):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0:
        return None

    circularity = 4 * np.pi * area / (perimeter ** 2)
    return circularity


def cnt_centroid(cnt):
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    c = (cX, cY)
    return c


def cnt_principal_axis(contours):
    principals = list()
    for cnt in contours:
        if len(cnt) < 5:
            continue
        e = cv2.fitEllipse(cnt)
        c = cnt_centroid(cnt)
        pa = ellipse_principal_axis(c, e)
        principals.append(pa)

    return principals


def ellipse_principal_axis(centroid, ellipse):
    e = ellipse
    cx, cy = centroid
    x1 = int(np.round(cx + e[1][1] / 2 * np.cos((e[2] + 90) * np.pi / 180.0)))
    y1 = int(np.round(cy + e[1][1] / 2 * np.sin((e[2] + 90) * np.pi / 180.0)))
    x2 = int(np.round(cx + e[1][1] / 2 * np.cos((e[2] - 90) * np.pi / 180.0)))
    y2 = int(np.round(cy + e[1][1] / 2 * np.sin((e[2] - 90) * np.pi / 180.0)))
    pa = ((x1, y1), (x2, y2))
    return pa


def middlepoint(p1, p2):
    middlepoint = []
    for i in range(len(p1)):
        line = [p1[i], p2[i]]
        middle = int((max(line) - min(line)) / 2) + min(line)
        middlepoint.append(middle)
    return middlepoint

#def filter_by_inertia(contours, threshold: float = 0.01, keep_NA: bool = False):
#    filtered = list()
#    for cnt in contours:
#        inertia = utils.cnt_inertia(cnt)
#
#        # Very samll contours can sometimtes not be approximated as ellipses
#        if keep_NA and inertia is None:
#            filtered.append(cnt)
#            continue
#        elif inertia >= threshold:
#            filtered.append(cnt)
#    return filtered
#
#
#def filter_for_convexity(contours, convexity_threshold: float = 0.9,
#                         keep_NA: bool = False):
#    filtered = list()
#    for cnt in contours:
#        convexity = utils.cnt_convexity(cnt)
#        if keep_NA and convexity is None:
#            filtered.append(cnt)
#            continue
#        elif convexity >= convexity_threshold:
#            filtered.append(cnt)
#
#    return filtered


def add_contours_to_img(img, contours, add_centroid: bool = False,
                        add_area: bool = False,
                        color: Tuple[int, int, int] = (255, 255, 255)):
    # Draw contours
    out = img.copy()
    out = cv2.drawContours(out, contours, -1, color, 1)

    # Add area text/centroid
    if add_area or add_centroid:
        for cnt in contours:

            # calculate area
            a = cv2.contourArea(cnt)

            # calculate centroid
            c = utils.cnt_centroid(cnt)

            # Set color according to area
            h = 150 + a
            color_text = (50, h, h)
            color_centroid = (h, 50, h)

            if add_area:
                out = cv2.putText(out, str(round(a)), (c[0], c[1] - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.2, color_text, 0)

            if add_centroid:
                out = cv2.circle(out, c, 1, color_centroid, -1)

    return out


def save_histogram(data: List[int], file_name: str, bins: range, title=None,
                   x_axis=None, x_unit=None, y_axis=None, y_unit=None):
    ar = np.array(data)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(ar, bins=bins)
    plt.xlabel(f"{x_axis} [{x_unit}]")
    plt.ylabel(f"{y_axis} [{y_unit}]")
    fig.suptitle(title)
    plt.savefig(file_name)
    plt.close()


def mkdir(path, verbose=False):
    if not os.path.isdir(path):
        if verbose:
            logger.info(f"creating {path}")
        os.mkdir(path)
    else:
        if verbose:
            logger.info(f"recreating {path}")
        shutil.rmtree(path)
        os.mkdir(path)


def find_short_path(coords, qc_outdir=None):
    logger.info("Finding short path in between points")
    distance_matrix = tsp_dist_matrix(coords, tsp_is_open=True)
    logging.getLogger("python_tsp.heuristics.local_search").setLevel(logging.WARNING)
    permutation, distance = solve_tsp_local_search(distance_matrix)
    ordered_coords = [coords[p] for p in permutation]

    # Write plot with points and path
    if qc_outdir:
        ordered_data = np.array(ordered_coords)
        plt.scatter(ordered_data[:, 0], ordered_data[:, 1], color="r")
        plt.plot(ordered_data[:, 0], ordered_data[:, 1])
        plt.scatter(ordered_data[0, 0], ordered_data[0, 1], color="b")
        plt.text(ordered_data[0, 0], ordered_data[0, 1], "start")
        plt.gca().invert_yaxis()
        plt.savefig(f"{qc_outdir}/actual-picking-path.png")

    return ordered_coords, distance


def tsp_dist_matrix(coords, tsp_is_open: bool = False):
    distance_matrix = euclidean_distance_matrix(coords)
    if tsp_is_open:
        distance_matrix[:, 0] = 0
    return distance_matrix


def range_limited_float_type(arg):
    """ Type function for argparse - a float within some predefined bounds """
    try:
        f = float(arg)
    except ValueError:
        raise ArgumentTypeError("Must be a floating point number")
    if f < ARG_MIN_VAL or f >= ARG_MAX_VAL:
        raise ArgumentTypeError(
            f"Argument must be within [{str(ARG_MIN_VAL)}, {str(ARG_MAX_VAL)})"
        )
    return f


class CentroidFinder():

    def __init__(self, centroids):
        # TODO: Rewrite this so we don't have to make a dict in between
        tmp_dict = OrderedDict()
        tmp_dict["centroid"] = centroids
        # TODO: Rewrite so it gets cX/cY from centroid
        tmp_dict["cX"] = [c[0] for c in centroids]
        tmp_dict["cY"] = [c[1] for c in centroids]

        self.centroid_df = pd.DataFrame(tmp_dict)

    def find_proximals(self, centroid: Tuple[int, int], search_window: int = 40):
        cX, cY = centroid
        sX1, sY1 = cX - search_window, cY - search_window
        sX2, sY2 = cX + search_window, cY + search_window

        centroid_df_xfilt = self.centroid_df[self.centroid_df["cX"].between(sX1, sX2)]
        centroid_df_xyfilt = centroid_df_xfilt[
            centroid_df_xfilt["cY"].between(sY1, sY2)]

        proximals = centroid_df_xyfilt.centroid.tolist()

        return proximals
