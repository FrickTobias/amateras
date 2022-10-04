"""
Finds big cells and calculates shortest path in between them
"""

import logging
import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple
from amateras import utils
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
    parser.add_argument("--n-cells", type=int, default=20, help="Number of cells find.")
    parser.add_argument("--qc-outdir", help="Output directory")

    # TODO: Make optional and maybe just write coords
    parser.add_argument("--details", action="store_true", help="Writes extra files")
    parser.add_argument("--size-min", type=int, default=30,
                        help="Min size for cells. Default: %(default)s")
    parser.add_argument("--size-max", type=int, default=500,
                        help="Max size for cells. Default: %(default)s")

    # TODO: Add that some of these are between 0 and 1
    parser.add_argument("--convexity-min", type=utils.range_limited_float_type,
                        default=0.875,
                        help="Min convexity for cells. Default: %(default)s")
    parser.add_argument("--inertia-min", type=utils.range_limited_float_type,
                        default=0.6,
                        metavar="",
                        help="Min inertia for cells. Default: %(default)s")

    # TODO: Add arguments for final filtering
    # TODO: Add argument for logfile writing

    return parser


def main(args):
    fluorescent_cells = find_fluorescencent_spots(
        args.input, args.n_cells, args.qc_outdir
    )
    fluorescent_cells_ordered, dist = find_short_path(fluorescent_cells, args.qc_outdir)
    utils.print_to_out(fluorescent_cells_ordered, header=True)


def find_fluorescencent_spots(input, n_cells, qc_outdir=None):
    logger.info("Opening file")
    img = cv2.imread(input)

    img_mod = img.copy()

    # Blurring
    BLUR_KERNEL = 3
    gray = cv2.cvtColor(img_mod, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, BLUR_KERNEL)

    MIN_PX_VAL = 30
    white_thresh = find_white_spots(blur, MIN_PX_VAL)

    contours, _ = cv2.findContours(white_thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    logger.info(f"Detected spots: {len(contours)}")

    logger.info("Filtering cells")

    size_min = 5
    size_max = 500
    convexity_min = 0.8
    inertia_min = 0.6
    circularity_min = 0.7

    # size filt
    contours_afilt = utils.filter_by_area(contours, size_min=size_min,
                                          size_max=size_max)
    logger.info(f"{size_min} < Size < {size_max}: {len(contours_afilt)}")
    # inertia filt
    contours_afilt_ifilt = utils.filter_by_inertia(
        contours_afilt, threshold=inertia_min, keep_NA=True
    )
    logger.info(f"Inertia > {inertia_min}: {len(contours_afilt_ifilt)}")
    # convexity filt
    contours_afilt_ifilt_cfilt = utils.filter_for_convexity(
        contours_afilt_ifilt, convexity_threshold=convexity_min, keep_NA=True
    )
    logger.info(f"Convexity > {convexity_min}: {len(contours_afilt_ifilt_cfilt)}")
    # circularity filt
    contours_afilt_ifilt_cfilt_cfilt = utils.filter_for_circularity(
        contours_afilt_ifilt_cfilt, circularity_threshold=circularity_min, keep_NA=True
    )
    logger.info(
        f"Circularity > {circularity_min}: {len(contours_afilt_ifilt_cfilt_cfilt)}")

    contours_final = contours_afilt_ifilt_cfilt_cfilt

    logger.info("Drawing contours")
    cv2.drawContours(img_mod, contours, -1, (125, 125, 125), 1)
    cv2.drawContours(img_mod, contours_afilt, -1, (50, 50, 125), 1)
    cv2.drawContours(img_mod, contours_afilt_ifilt, -1, (125, 50, 50), 1)
    cv2.drawContours(img_mod, contours_afilt_ifilt_cfilt, -1, (125, 50, 125), 1)
    cv2.drawContours(img_mod, contours_afilt_ifilt_cfilt_cfilt, -1, (50, 125, 50), 1)

    logger.info("Calculating intensities and magnitudes")
    areas = [cv2.contourArea(c) for c in contours_final]
    intensities = []
    magnitudes = []
    for cnt, a in tqdm(zip(contours_final, areas), total=len(contours_final)):
        bbox = cv2.boundingRect(cnt)
        x, y, w, h = bbox
        cell_raw = img[y:y + h, x:x + w]

        img_masked = cnt_mask_img(cell_raw, cnt, cnt_start=(x, y))

        magnitude = np.sum(img_masked)
        intesity = magnitude / a

        magnitudes.append(magnitude)
        intensities.append(intesity)

    logger.info("Finding highest intesity signals")

    intensities_array = np.array(intensities, dtype=float)
    contours_final_array = np.array(contours_final, dtype=object)

    sort_index = (-intensities_array).argsort()[:n_cells]
    contours_final_sorted = contours_final_array[sort_index]
    intensities_sorted = intensities_array[sort_index]

    centroids = []
    for i, (cnt, intensity) in enumerate(
            zip(contours_final_sorted, intensities_sorted)):
        bbox = cv2.boundingRect(cnt)
        x, y, w, h = bbox
        cv2.rectangle(img_mod, (x, y), (x + w, y + h), (50, 50, 255), 1)
        cv2.putText(img_mod, f"{i}: {round(intensity, 2)}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 255), 1)
        centroids.append(utils.cnt_centroid(cnt))

        # Marking high intensity cells
        cv2.putText(
            img_mod, f"high int: {i}", (x + w // 2, y + h // 2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (50, 50, 255), 1
        )
        img_mod = cv2.rectangle(
            img_mod, (x + w // 2 - 40, y + h // 2 - 40),
            (x + w // 2 + 40, y + h // 2 + 40), (50, 50, 255), 5
        )

    logger.info("Writing output")
    if qc_outdir:
        mkdir(qc_outdir)
        cv2.imwrite(f"{qc_outdir}/input.tif", img)
        cv2.imwrite(f"{qc_outdir}/blur.tif", blur)
        cv2.imwrite(f"{qc_outdir}/thresh.tif", white_thresh)
        cv2.imwrite(f"{qc_outdir}/cnts.tif", img_mod)
        xmax = int(np.average(intensities) * 2)
        plt.hist(intensities, range(xmax))
        plt.title("intensities")
        plt.ylabel("count []")
        plt.xlabel("intensity [lx/px]")
        plt.savefig(f"{qc_outdir}/intesities-historam.png")

        logger.info("Finding picking path (only for example image)")
        ordered_centroids, _ = find_short_path(centroids)
        img_path = img_mod.copy()
        for point1, point2 in zip(ordered_centroids, ordered_centroids[1:]):
            img_path = cv2.line(
                img_path, point1, point2, [125, 255, 50], 2
            )
        tX, tY = ordered_centroids[0]
        tX -= 20
        tY -= 60
        img_path = cv2.putText(
            img_path, "start", (tX, tY), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (125, 255, 50), 2
        )

        cv2.imwrite(f"{qc_outdir}/picking-path-example.tif", img_path)

    return centroids


def cnt_mask_img(img, cnt, cnt_start: Tuple[int, int] = (0, 0)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Make mask from contour (otherwise cnt detection will find areas outside cell)
    cnt = cnt - cnt_start
    cnt = [cnt]

    # Make mask from contour (otherwise cnt detection will find areas outside cell)
    mask = np.zeros(shape=gray.shape, dtype=np.uint8)
    mask = cv2.drawContours(mask, cnt, -1, 255, -1)

    # Apply mask, removing detections outside of cell
    img_masked = cv2.bitwise_and(mask, gray)
    return img_masked


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


def middlepoint(p1, p2):
    middlepoint = []
    for i in range(len(p1)):
        line = [p1[i], p2[i]]
        middle = int((max(line) - min(line)) / 2) + min(line)
        middlepoint.append(middle)
    return middlepoint


def find_white_spots(img_gray, white_thresh):
    _, white_thresh = cv2.threshold(img_gray, white_thresh, 255, cv2.THRESH_BINARY)

    return white_thresh


def find_short_path(coords):
    logger.info("Finding short path in between points")
    distance_matrix = tsp_dist_matrix(coords, tsp_is_open=True)
    logging.getLogger("python_tsp.heuristics.local_search").setLevel(logging.WARNING)
    permutation, distance = solve_tsp_local_search(distance_matrix)
    ordered_coords = [coords[p] for p in permutation]

    return ordered_coords, distance


def tsp_dist_matrix(coords, tsp_is_open: bool = False):
    distance_matrix = euclidean_distance_matrix(coords)
    if tsp_is_open:
        distance_matrix[:, 0] = 0
    return distance_matrix
