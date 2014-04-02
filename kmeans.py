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

def generate_feature_matrix(data, tfidf_threshold=0.2):
    feature_matrix = np.copy(data)
    word_doc_counts = np.array([np.count_nonzero(data.T[w])
                                for w in xrange(data.shape[1])])

    for doc in xrange(data.shape[0]):
        for word in xrange(data.shape[1]):
            if data[doc][word] == 0 or word_doc_counts[word] == 0:
                continue
            count = data[doc][word]
            tfidf = calculate_tfidf(count, word_doc_counts[word], data.shape[0])
            feature_matrix[doc][word] = count if tfidf > tfidf_threshold else 0

    return feature_matrix

def calculate_tfidf(ftd, ndt, nd):
    tf = math.log(ftd + 1)
    idf = math.log(float(nd) / ndt)
    return tf * idf

def cosine_distance(u, v):
    dist = 1.0 - np.dot(u, v) / (norm(u) * norm(v))
    if dist < 0:
        return 0.0
    return dist

def initialize_means(data, k):
    '''
    Uses k-means++ to initialize seeds for clustering
    '''
    num_docs = data.shape[0]
    num_feats = data.shape[1]
    seeds = np.empty((k, num_feats))

    # Choose one center uniformly at random from among the data points.
    doc = random.randint(0, num_docs-1)
    seeds[0] = data[doc]
    dist_from_seed = np.zeros(num_docs)
    for s in xrange(1, k):
        for i in xrange(1, num_docs):
            # For each data point x, compute D(x), the distance between x and
            # the nearest center that has already been chosen.
            # TODO: use matrix multiplication for more efficient calculation
            dist_from_seed[i] = min([cosine_distance(data[i], seeds[j])
                                     for j in xrange(0, s)])

        # Choose one new data point at random as a new center, using a weighted
        # probability distribution where a point x is chosen with probability
        # proportional to D(x)^2.
        dist_from_seed = dist_from_seed / sum(dist_from_seed)
        doc = nprandom.choice(np.arange(num_docs), p=dist_from_seed)
        seeds[s] = data[doc]

    return seeds

def cluster_data(data, k):
    def different_clusters(X, Y):
        if X is None or Y is None:
            return True
        for i in xrange(k):
            if cosine_distance(X[i], Y[i]) > 0.01:
                return True
        return False

    num_docs = data.shape[0]
    num_feats = data.shape[1]
    data = generate_feature_matrix(data)

    cluster_centers = initialize_means(data, k)
    cluster_membership = np.zeros(num_docs)
    new_cluster_centers = None
    iteration = 0
    while different_clusters(cluster_centers, new_cluster_centers):
        if new_cluster_centers is not None:
            cluster_centers = new_cluster_centers
        cluster_doc_count = np.zeros(k)
        new_cluster_centers = np.zeros((k, num_feats))
        for d in xrange(num_docs):
            doc = data[d]
            dists = [cosine_distance(doc, cluster_centers[c]) for c in xrange(k)]
            closest, min_dist = np.argmin(dists), min(dists)
            weight = 1.0 - min_dist
            new_cluster_centers[closest] += data[d] * weight
            cluster_doc_count[closest] += weight
        for i in xrange(k):
            new_cluster_centers[i] /= cluster_doc_count[i]
        iteration += 1
        print iteration

    cluster_centers = new_cluster_centers
    cluster_membership = np.array([np.argmin([cosine_distance(data[d],
            cluster_centers[c]) for c in xrange(k)]) for d in xrange(num_docs)])

    return cluster_membership

if __name__ == '__main__':
    import sys
    print cluster_data(load_data('20NG'), int(sys.argv[1]))
