import concurrent.futures
import os
import random

import cv2
import numpy as np
from matplotlib import pyplot as plt


def preprocess(db_path, num_imgs=1000): 
    with concurrent.futures.ProcessPoolExecutor() as executor:
        file_names = random.sample(os.listdir(db_path), num_imgs)
        file_paths = list(map(lambda f: db_path + f, file_names))
        imgs = list(executor.map(process_image, file_paths)) 
        return file_names, imgs


def process_image(file_path, cropped=True, scaled=True, mapped=False):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if not cropped:
        img = crop_centre_square(img)
    if not scaled:
        img = rescale(img)
    if not mapped:
        img = map_color_space(img)
    return img
    

def crop_centre_square(img):
    y, x, _ = img.shape
    crop_length = min(y, x)
    startx = x // 2 - (crop_length // 2)
    starty = y // 2 - (crop_length // 2)
    cropped = img[starty:starty + crop_length, startx:startx + crop_length,:]
    return cropped


def rescale(src, dim=(128, 128)):
    dst = cv2.resize(src, dim, interpolation=cv2.INTER_LINEAR)
    return dst


def map_color_space(src, max_val=255):
    b, g, r = cv2.split(src)
    c_1 = (b + g + r)/3
    c_2 = (r + (max_val - b))/2
    c_3 = (r + 2 * (max_val - g) + b)/4
    dst = cv2.merge((c_1, c_2, c_3))
    return dst

