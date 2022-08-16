import cv2
import numpy as np

# TODO: Do this for image
def color_histogram(grayImage, thresholdImage):
    # PLOT HISTOGRAM OF THRESHOLDED AND GRAYSCALE IMAGES
    plt.figure(figsize=(14, 12))
    plt.subplot(2, 2, 1), plt.imshow(grayImage, 'gray'), plt.title('Grayscale Image')
    plt.subplot(2, 2, 2), plt.hist(grayImage.ravel(), 256), plt.title(
        'Color Histogram of Grayscale Image')
    plt.subplot(2, 2, 3), plt.imshow(thresholdImage, 'gray'), plt.title(
        'Binary (Thresholded)  Image')
    plt.subplot(2, 2, 4), plt.hist(thresholdImage.ravel(), 256), plt.title(
        'Color Histogram of Binary (Thresholded) Image')
    plt.savefig('fig1.png')
    plt.show()

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


def filter_by_area(contours, size_min: int = 10, size_max: int = 500):
    filtered = []
    for cnt in contours:
        if size_min < cv2.contourArea(cnt) < size_max:
            filtered.append(cnt)
    return filtered


def filter_by_inertia(contours, threshold: float = 0.01, keep_NA: bool = False):
    filtered = list()
    for cnt in contours:
        inertia = cnt_inertia(cnt)
        if inertia is None:
            if keep_NA:
                filtered.append(cnt)
        elif inertia >= threshold:
            filtered.append(cnt)
    return filtered


def filter_for_convexity(contours, convexity_threshold: float = 0.9,
                         keep_NA: bool = False):
    filtered = list()
    for cnt in contours:
        convexity = cnt_convexity(cnt)
        if convexity is None:
            if keep_NA:
                filtered.append(cnt)
        elif convexity >= convexity_threshold:
            filtered.append(cnt)

    return filtered


def filter_for_circularity(contours, circularity_threshold: float = 0.7,
                           keep_NA: bool = False):
    filtered = list()
    for cnt in contours:
        circularity = cnt_circularity(cnt)
        if circularity is None:
            if keep_NA:
                filtered.append(cnt)
        elif circularity >= circularity_threshold:
            filtered.append(cnt)

    return filtered
