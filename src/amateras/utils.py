import cv2
from matplotlib import pyplot as plt
import numpy as np
import logging
import shutil
import os
from pathlib import Path
from typing import Tuple
from typing import List
from argparse import ArgumentTypeError

logger = logging.getLogger(__name__)

ARG_MIN_VAL = 0
ARG_MAX_VAL = 1


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


def mkdir(path: Path, verbose: bool = False, overwrite: bool = False):
    if not os.path.isdir(path):
        if verbose:
            logger.info(f"creating {path}")
        os.mkdir(path)
    else:
        if overwrite:
            if verbose:
                logger.info(f"recreating {path}")
            shutil.rmtree(path)
            os.mkdir(path)
        else:
            if verbose:
                logger.info(f"{path} already exist")


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


def print_to_out(sorted_coordinates: List[Tuple[int, int]], header: bool = False, arealist: List[int] = []):
    if header:
        print("No,X,Y,area") if arealist else print("No,X,Y")
    if arealist:
        for i, ((x, y), area) in enumerate(zip(sorted_coordinates, arealist)):
            print(f"{i},{x},{y},{area}")
    else:
        for i, (x, y) in enumerate(sorted_coordinates):
            print(f"{i},{x},{y}")
