"""Contains image preprocessing functionality required for Wavelet-based image
indexing and search.

The preprocessing stage of WBIIS application results in a collection of images 
that have been cropped, resized and converted into another colour space (namely, 
the component colour space).

"""


import concurrent.futures
import functools
import sys
import time

import cv2


def lazy_preprocess(file_paths):
    """Returns a generator for reading and processing images."""
    for file_path in file_paths:
        img = process_image(file_path)
        yield img


def multi_preprocess(file_paths):
    """Returns a processed set of images."""
    with concurrent.futures.ProcessPoolExecutor() as executor:
        imgs = list(executor.map(process_image, file_paths))
        return imgs


def process_image(file_path, cropped=True, scaled=True, mapped=False):
    """Loads and performs the relevant preprocessing on an image.

    NOTE:   Change the relevant kwargs to reflect the level of preprocessing
            that your data has already undergone.
    """
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if not cropped:
        img = crop_centre_square(img)
    if not scaled:
        img = rescale(img)
    if not mapped:
        img = map_color_space(img)
    return img


def crop_centre_square(img):
    """Crops the largest centre square of an image"""
    y, x, _ = img.shape
    crop_length = min(y, x)
    startx = x // 2 - (crop_length // 2)
    starty = y // 2 - (crop_length // 2)
    cropped = img[starty:starty + crop_length, startx:startx + crop_length,:]
    return cropped


def rescale(src, dim=(128, 128)):
    """Resizes the image to the specified dimensions"""
    dst = cv2.resize(src, dim, interpolation=cv2.INTER_LINEAR)
    return dst


def map_color_space(src, max_val=255):
    """Converts an RGB image to an image in component colour space."""
    b, g, r = cv2.split(src)
    c_1 = (b + g + r)/3
    c_2 = (r + (max_val - b))/2
    c_3 = (r + 2 * (max_val - g) + b)/4
    dst = cv2.merge((c_1, c_2, c_3))
    return dst


def save(file_name, img):
    """Writes out an image to a file with the specified file name"""
    cv2.imwrite(file_name, img)    
