import cv2
import numpy as np
import pywt
from matplotlib import pyplot as plt

import visualise

def fast_query():
    pass


def slow_query(query_img, db_imgs, num_matches=10, level=3):
    stds, wavelet_vectors = construct_channel_vector(query_img, level)
    filtered = []
    for index, db_img in enumerate(db_imgs):
        db_stds, db_wavelet_vectors = construct_channel_vector(db_img, level)
        if std_accept(stds, db_stds, 50):
            dist = sum(list(map(distance, wavelet_vectors, db_wavelet_vectors)))
            filtered.append((index, dist))
    final = sorted(filtered, key = lambda x: x[1])[:num_matches]
    return final


def init_kd_tree(db_imgs):
    pass


def init_vectors(db_imgs):
    pass


def construct_channel_vector(img, level):
   channels = cv2.split(img)
   feature_vector = [construct_feature_vector(c, level) for c in channels]
   stds = [tup[0] for tup in feature_vector]
   wavelet_vectors = [tup[1] for tup in feature_vector]
   return stds, wavelet_vectors


def construct_feature_vector(img, level):
    coeffs = pywt.wavedec2(img, 'db8', 'sym', level)
    cA, (cH, cV, cD) = coeffs[0], coeffs[1]
    std = np.std(cA)
    wavelet_vector = [cA, cH, cV, cD]
    return std, wavelet_vector 


def std_accept(query_stds, db_stds, percent):
    beta = 1 - (percent/100)
    s1, s2, s3 = tuple(query_stds)
    t1, t2, t3 = tuple(db_stds)
    condition = lambda t, s: True if t > s * beta and t < s / beta else False
    return condition(t1, s1) or (condition(t2, s2) and condition(t3, s3))


def distance(wavelet_vector_1, wavelet_vector_2):
    euclid_dist = lambda a, b: np.linalg.norm(a-b)
    dist = sum(list(map(euclid_dist, wavelet_vector_1, wavelet_vector_2)))
    return dist
