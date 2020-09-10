"""WBIIS module contains functionality for constructing and querying feature vectors.

It is important to note that fast query ommits standard deviation thresholding. Slow
query is similar to the query process described in the paper 'Content-based image
indexing and searching using Daubechies wavelets' (Wang, Wiederhold, Firschein and
Xin Wei, 1998)

Typical usage example:

query_feature_vector = construct_feature_vector(img, level=3)
db_feature_vectors = init_feature_vectors(db_imgs, level=3)
kd_tree = init_kd_tree(feature_vectors)
result = fast_query(query_feature_vector, kd_tree)
"""

import functools
import sys
import time

import cv2
import numpy as np
import pywt
import scipy.spatial


def timer(func):
    """Print the runtime of the decorated function."""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        sys.stdout.flush()
        return value
    return wrapper_timer 


@timer
def fast_query(feature_vector, kd_tree, num_matches=10):
    """Queries a kd-tree for the nearest neighbours of a feature vector"""
    _, vector = feature_vector
    dists, indices = kd_tree.query(vector, num_matches)
    return zip(indices, dists)


@timer
def slow_query(feature_vector, db_feature_vectors, num_matches=10, std_thresh=50):
    """Queries a feature vector through standard deviation thresholding and wavelet
    vector nearest neighbours.
    """
    closest = []
    stds, vector = feature_vector
    for index, db_feature_vector in enumerate(db_feature_vectors):
        db_stds, db_vector = db_feature_vector
        if std_accept(stds, db_stds, std_thresh):
            dist = distance(vector, db_vector)
            closest.append((index, dist))
    return sorted(closest, key = lambda x: x[1])[:num_matches]


def init_kd_tree(feature_vectors):
    """Initialises the kd-tree for a set of feature vectors"""
    data = [vector for _, vector in feature_vectors]
    return scipy.spatial.KDTree(data)


def init_feature_vectors(db_imgs, level):
    """Generates the feature vectors of a collection of database images"""
    feature_vectors = []
    for db_img in db_imgs:
        feature_vector = construct_feature_vector(db_img, level)
        feature_vectors.append(feature_vector)
    return feature_vectors


def construct_feature_vector(img, level):
    """Combines channel vectors and standard deviation values of an image"""
    channels = cv2.split(img)
    channel_vectors = [construct_channel_vector(c, level) for c in channels]
    stds = [tup[0] for tup in channel_vectors]
    wavelet_vectors = [tup[1] for tup in channel_vectors]
    return stds, np.asarray(wavelet_vectors).flatten()


def construct_channel_vector(img, level):
    """Generates the standard deviation and wavelet vector for an image channel"""
    coeffs = pywt.wavedec2(img, 'db8', 'sym', level)
    cA, (cH, cV, cD) = coeffs[0], coeffs[1]
    std = np.std(cA)
    wavelet_vector = [cA, cH, cV, cD]
    return std, wavelet_vector


def std_accept(query_stds, db_stds, percent):
    """Defines standard deviation acceptance criteria for database a image"""
    beta = 1 - (percent/100)
    s1, s2, s3 = tuple(query_stds)
    t1, t2, t3 = tuple(db_stds)
    condition = lambda t, s: s * beta < t < s / beta
    return condition(t1, s1) or (condition(t2, s2) and condition(t3, s3))


def distance(vector_1, vector_2):
    """Calculates the sum of the distances between two vectors"""
    euclid_dists = np.linalg.norm(vector_1 - vector_2)
    dist = np.sum(euclid_dists)
    return dist
