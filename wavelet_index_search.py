import cv2
import numpy as np
import pywt
import scipy.spatial
from matplotlib import pyplot as plt

import visualise

def fast_query(feature_vector, kd_tree, num_matches=12):
    _, wavelet_vectors = feature_vector
    vector = np.asarray(wavelet_vectors).flatten()
    dists, indices = kd_tree.query(vector, num_matches)
    return zip(indices, dists)


def slow_query(feature_vector, db_feature_vectors, num_matches=10, std_thresh=50):
    closest = []
    stds, wavelet_vectors = feature_vector
    for index, db_feature_vector in enumerate(db_feature_vectors):
        db_stds, db_wavelet_vectors = db_feature_vector 
        if std_accept(stds, db_stds, std_thresh):
            dist = sum(list(map(distance, wavelet_vectors, db_wavelet_vectors)))
            closest.append((index, dist))
    return sorted(closest, key = lambda x: x[1])[:num_matches]


def init_kd_tree(feature_vectors):
    data = []
    for _, wavelet_vectors in feature_vectors:
        data.append(np.asarray(wavelet_vectors).flatten()) 
    return scipy.spatial.KDTree(data)


def init_feature_vectors(db_imgs, level):
    feature_vectors = []
    for db_img in db_imgs:
        feature_vector = construct_feature_vector(db_img, level)
        feature_vectors.append(feature_vector)
    return feature_vectors


def construct_feature_vector(img, level):
    channels = cv2.split(img)
    channel_vectors = [construct_channel_vector(c, level) for c in channels]
    stds = [tup[0] for tup in channel_vectors]
    wavelet_vectors = [tup[1] for tup in channel_vectors]
    return stds, wavelet_vectors


def construct_channel_vector(img, level):
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
