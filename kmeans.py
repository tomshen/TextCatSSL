#!/usr/bin/env python2.7
import csv
import math
import random

import numpy as np
import numpy.random as nprandom
from numpy.linalg import norm

import util

def load_data(data_set):
    num_docs = util.get_num_docs(data_set)
    num_feats = util.get_num_features(data_set)
    data = np.zeros((num_docs, num_feats))
    with util.open_data_file(data_set) as data_file:
        data_reader = csv.reader(data_file, delimiter=' ')
        for row in data_reader:
            doc = int(row[0]) - 1
            word = int(row[1]) - 1
            count = int(row[2])
            data[doc][word] = count
    return data

def generate_feature_matrix(data):
    # TODO: use tfidf instead of counts
    return data

def tfidf(ftd, ndt, nd):
    tf = math.log(ftd + 1)
    idf = math.log(float(nd) / ndt)
    return tf * idf

def cosine_distance(u, v):
    dist = np.dot(u, v) / (norm(u) * norm(v))
    return dist

def initialize_means(data, k):
    '''
    Uses k-means++ to initialize seeds for clustering
    '''
    num_docs = len(data)
    num_feats = len(data[0])
    seeds = np.empty((k, num_feats))

    # Choose one center uniformly at random from among the data points.
    doc = random.randint(0, num_docs-1)
    seeds[0] = data[doc]
    dist_from_seed = np.zeros(num_docs)
    for s in xrange(1, k):
        for i in xrange(1, num_docs):
            # For each data point x, compute D(x), the distance between x and the
            # nearest center that has already been chosen.
            dist_from_seed[i] = min([cosine_distance(data[i], seeds[j])
                                     for j in xrange(0, s)])

        # Choose one new data point at random as a new center, using a weighted
        # probability distribution where a point x is chosen with probability
        # proportional to D(x)^2.
        dist_from_seed = dist_from_seed / sum(dist_from_seed)
        doc = nprandom.choice(np.arange(num_docs), p=dist_from_seed)
        seeds[s] = data[doc]

    return seeds

