#!/usr/bin/env python
import os
import csv
import math
import multiprocessing as mp
import string
import sys
from datetime import datetime
from collections import Counter

import numpy as np

from kmeans import load_data, cluster_data
from lsh import MultiLSHasher
from util import open_data_file, open_graph_file, get_counts
from analyze import find_ambiguous_words, get_pred_labels

def get_doc_features(data_set):
    doc_features = {}
    with open_data_file(data_set) as data:
        datareader = csv.reader(data, delimiter=' ')
        for row in datareader:
            doc = int(row[0])
            word = int(row[1])
            count = int(row[2])
            if doc not in doc_features:
                doc_features[doc] = []
            doc_features[doc].append((word, count))
    return doc_features

def __cluster_data(params):
    i, data_set, k = params
    return (string.ascii_lowercase[i], cluster_data(load_data(data_set), k))

def generate_kmeans_graph(data_set, num_clusterings=1, k=8, verbose=False):
    if verbose: print 'Generating k-means graph'
    process_pool = mp.Pool(processes=num_clusterings)
    labels = dict(process_pool.map(__cluster_data, [
        (i, data_set, k) for i in xrange(num_clusterings)]))
    process_pool.close()
    process_pool.join()
    if verbose: print 'Documents clustered'

    data_counts = get_counts(data_set)
    num_docs = data_counts[0]
    num_features = data_counts[1]

    doc_features = {}
    word_counts = Counter()
    with open_data_file(data_set) as data:
        datareader = csv.reader(data, delimiter=' ')
        for row in datareader:
            doc = int(row[0])
            word = int(row[1])
            count = int(row[2])
            word_counts[word] += 1
            if doc not in doc_features:
                doc_features[doc] = []
            doc_features[doc].append((word, count))
    if verbose: print 'Loaded doc features'
    for doc, features in doc_features.items():
        feature_tfidf = []
        for w, c in features:
            tfidf = math.log(c+1) * math.log(num_docs/float(word_counts[w]))
            feature_tfidf.append((w,tfidf))
        doc_features[doc] = feature_tfidf

    doc_features = {}
    words_doc_count = Counter()
    with open_data_file(data_set) as data:
        datareader = csv.reader(data, delimiter=' ')
        for row in datareader:
            doc = int(row[0])
            count = int(row[2])
            for hl, label in labels.items():
                word = str(row[1]) + hl + str(label[doc-1])
                words_doc_count[word] += 1
                if doc not in doc_features:
                    doc_features[doc] = []
                doc_features[doc].append((word, count))
    if verbose: print 'Generated labeled doc features'

    filename = '%s-km-k%dn%d' % (data_set, k, num_clusterings)
    with open_graph_file(filename) as graph:
        datawriter = csv.writer(graph, delimiter='\t')
        for doc, feature_counts in doc_features.items():
            for feature, count in feature_counts:
                tfidf = math.log(count+1) * math.log(num_docs/float(
                  words_doc_count[feature]))
                datawriter.writerow([doc, feature, tfidf])

            for feat in xrange(1, num_features+1):
                w = 'w%d' % feat
                for hl in labels:
                    for label in xrange(k):
                        lw = str(feat) + hl + str(k)
                        datawriter.writerow([w, lw, '1.0'])

    if verbose: print 'Wrote graph file %s' % filename

def generate_lsh_graph(data_set, num_hashes=3, num_bits=5, verbose=False):
    hashers = MultiLSHasher(num_hashes, num_bits)
    if verbose: print 'Hashers initialized'

    data_counts = get_counts(data_set)
    num_docs = data_counts[0]
    num_features = data_counts[1]

    doc_features = {}
    word_counts = Counter()
    with open_data_file(data_set) as data:
        datareader = csv.reader(data, delimiter=' ')
        for row in datareader:
            doc = int(row[0])
            word = int(row[1])
            count = int(row[2])
            word_counts[word] += 1
            if doc not in doc_features:
                doc_features[doc] = []
            doc_features[doc].append((word, count))
    if verbose: print 'Loaded doc features'
    for doc, features in doc_features.items():
        feature_tfidf = []
        for w, c in features:
            tfidf = math.log(c+1) * math.log(num_docs/float(word_counts[w]))
            feature_tfidf.append((w,tfidf))
        doc_features[doc] = feature_tfidf

    hashers.compute_stream(doc_features)
    signatures = hashers.compute_signatures()
    if verbose: print 'Computed signatures'

    doc_features = {}
    words_doc_count = Counter()
    with open_data_file(data_set) as data:
        datareader = csv.reader(data, delimiter=' ')
        for row in datareader:
            doc = int(row[0])
            count = int(row[2])
            for hl, s in signatures.items():
                word = str(row[1]) + hl + s[doc]
                words_doc_count[word] += 1
                if doc not in doc_features:
                    doc_features[doc] = []
                doc_features[doc].append((word, count))
    if verbose: print 'Generated hashed doc features'

    filename = '%s-lsh-h%db%d' % (data_set, num_hashes, num_bits)
    with open_graph_file(filename) as graph:
        datawriter = csv.writer(graph, delimiter='\t')
        for doc, feature_counts in doc_features.items():
            for feature, count in feature_counts:
                tfidf = math.log(count+1) * math.log(num_docs/float(
                  words_doc_count[feature]))
                datawriter.writerow([doc, feature, tfidf])
    if verbose: print 'Wrote graph file %s' % filename

def get_new_doc_features(data_set, output_file):
    doc_features = get_doc_features(data_set)
    ambiguous_words = find_ambiguous_words(output_file)
    doc_labels = get_pred_labels(output_file)
    for doc, features in doc_features.items():
        label = doc_labels[doc][0]
        new_features = []
        for word, count in features:
            if word in ambiguous_words:
                word_labels = ambiguous_words[word]
                if str(label) in word_labels:
                    new_features.append((str(word) + 'w' + str(label), count))
                else:
                    new_features.append((str(word) + 'w', count))
            else:
                new_features.append((str(word) + 'w', count))
        doc_features[doc] = new_features
    return doc_features

def generate_baseline_graph(data_set, verbose=False):
    data_counts = get_counts(data_set)
    num_docs = data_counts[0]
    num_features = data_counts[1]
    test_data = []

    words_doc_count = Counter()
    for doc, features in get_doc_features(data_set).items():
        for word, count in features:
            words_doc_count[word] += 1
            test_data.append([doc, word, count])
    if verbose: print 'Loaded doc features'

    filename = data_set + '-baseline'
    with open_graph_file(filename) as graph:
        datawriter = csv.writer(graph, delimiter='\t')
        for d,w,c in test_data:
            tfidf = math.log(c+1) * math.log(num_docs/float(words_doc_count[w]))
            datawriter.writerow([str(d), str(w) + 'w', tfidf])
        if verbose: print 'Wrote graph file %s' % filename

def generate_labeled_baseline_graph(output_file, verbose=False):
    data_set = output_file.split('-')[0]
    data_counts = get_counts(data_set)
    num_docs = data_counts[0]
    num_features = data_counts[1]
    test_data = []

    words_doc_count = Counter()
    for doc, features in get_new_doc_features(data_set, output_file).items():
        for word, count in features:
            words_doc_count[word] += 1
            test_data.append([doc, word, count])
    if verbose: print 'Loaded doc features'

    filename = data_set + '-baseline'
    with open_graph_file(filename) as graph:
        datawriter = csv.writer(graph, delimiter='\t')
        for d,w,c in test_data:
            tfidf = math.log(c+1) * math.log(num_docs/float(words_doc_count[w]))
            datawriter.writerow([str(d), str(w), tfidf])
        if verbose: print 'Wrote graph file %s' % filename



# TODO remove dependency on num_docs / num_features
def generate_knn_graph(data_set, k, verbose=False):
    data_counts = get_counts(data_set)
    num_docs = data_counts[0]
    num_features = data_counts[1]

    assert k < num_docs

    feature_matrix = np.matrix(np.zeros((num_docs, num_features)))
    words_doc_count = np.zeros(num_features)
    docs = set()
    with open_data_file(data_set) as data:
        datareader = csv.reader(data, delimiter=' ')
        for row in datareader:
            doc = int(row[0]) - 1
            word = int(row[1]) - 1
            count = int(row[2])
            words_doc_count[word] += 1
            docs.add(doc)
            feature_matrix.itemset((doc,word), count)
    if verbose: print 'Loaded test data'

    if verbose: print 'Generating feature matrix'
    for doc in xrange(num_docs):
        if doc in docs:
            for word in xrange(num_features):
                if words_doc_count[word] != 0:
                    count = feature_matrix.item((doc,word))
                    tfidf = math.log(count+1) * math.log(num_docs/float(words_doc_count[word]))
                    feature_matrix.itemset((doc,word), tfidf)
        if doc % 10 == 9:
            if verbose: print 'Processed %d out of %d documents' % (doc+1, num_docs)
    if verbose: print 'Generated feature matrix'

    normalizing_matrix = np.matrix(np.zeros((num_docs, num_docs)))
    for i in xrange(num_docs):
        f = feature_matrix[i]
        fft = math.sqrt(f * f.transpose())
        if fft < 1e-9:
            normalizing_matrix.itemset((i,i), 0.0)
        else:
            normalizing_matrix.itemset((i,i), 1.0 / fft)
    if verbose: print 'Generated normalizing matrix'

    if verbose: print 'Generating folded graph'
    edges = []
    N = normalizing_matrix
    F = feature_matrix
    for doc in xrange(num_docs):
        Nv = np.matrix(np.zeros((num_docs,1)))
        Nv.itemset(doc, N.item((doc, doc)))
        FtNv = F[doc].transpose() * N.item((doc,doc))
        doc_weights = np.array(N * (F * FtNv)).transpose()
        nearest_neighbors = np.argsort(doc_weights)
        for neighbor in nearest_neighbors[0][-k:]:
            if doc_weights.item(neighbor) < 1e-9:
                continue
            edges.append(((doc+1, int(neighbor)+1), doc_weights.item(neighbor)))
        if doc % 10 == 9:
            if verbose: print 'Processed %d out of %d documents' % (doc+1, num_docs)
    if verbose: print 'Generated folded graph'

    filename = '%s-knn-k%d' % (data_set, k)
    with open_graph_file(filename) as graph:
        datawriter = csv.writer(graph, delimiter='\t')
        for edge, weight in edges:
            datawriter.writerow([edge[0], edge[1], weight])
    if verbose: print 'Wrote graph file %s' % filename

def generate_knn_graphs(data_set, ks=[5,10,20,30,50,100], verbose=False):
    '''
    since we get a list of *all* the neighbors ordered by "nearness",
    it makes more sense to iterate through the different k's within
    the function rather than outside it
    '''
    max_k = max(ks)
    feature_matrix = np.matrix(np.zeros((num_docs, num_features)))
    words_doc_count = np.zeros(num_features)
    with open_data_file(data_set) as data:
        datareader = csv.reader(data, delimiter=' ')
        for row in datareader:
            doc = int(row[0]) - 1
            word = int(row[1]) - 1
            count = int(row[2])
            words_doc_count[word] += 1
            feature_matrix.itemset((doc,word), count)
    if verbose: print('[%s]: Loaded test data.' % str(datetime.now().time()))

    if verbose: print('[%s]: Generating feature matrix' % str(datetime.now().time()))
    for doc in xrange(num_docs):
        for word in xrange(num_features):
            if words_doc_count[word] != 0:
                count = feature_matrix.item((doc,word))
                tfidf = math.log(count+1) * math.log(num_docs/float(words_doc_count[word]))
                feature_matrix.itemset((doc,word), tfidf)
        if doc % 10 == 9:
            if verbose: print('[%s]: Processed %d out of %d documents' % (str(datetime.now().time()),
                (doc+1), num_docs))
    if verbose: print('[%s]: Generated feature matrix' % str(datetime.now().time()))

    normalizing_matrix = np.matrix(np.zeros((num_docs, num_docs)))
    for i in xrange(num_docs):
        f = feature_matrix[i]
        normalizing_matrix.itemset((i,i), 1.0 / math.sqrt(f * f.transpose()))
    if verbose: print('[%s]: Generated normalizing matrix' % str(datetime.now().time()))

    if verbose: print('[%s]: Generating folded graph' % str(datetime.now().time()))
    doc_neighbors = {}
    N = normalizing_matrix
    F = feature_matrix
    for doc in xrange(num_docs):
        Nv = np.matrix(np.zeros((num_docs,1)))
        Nv.itemset(doc, N.item((doc, doc)))
        FtNv = F[doc].transpose() * N.item((doc,doc))
        doc_weights = np.array(N * (F * FtNv)).transpose()
        neighbors = np.argsort(doc_weights)[0]
        doc_neighbors[doc] = [(neighbor, doc_weights.item(neighbor)) for neighbor in neighbors[-max_k:]]
        if doc % 10 == 9:
            if verbose: print('[%s]: Processed %d out of %d documents' % (
                str(datetime.now().time()), (doc+1), num_docs))
    if verbose: print('[%s]: Generated folded graph' % str(datetime.now().time()))

    for k in ks:
        filename = '%s-knn-k%d' % (data_set, k)
        with open_graph_file(filename) as graph:
            datawriter = csv.writer(graph, delimiter='\t')
            for doc in xrange(num_docs):
                for neighbor,weight in doc_neighbors[doc][-k:]:
                    if weight >= 1e-9:
                        datawriter.writerow([str(doc+1), str(neighbor+1), weight])
            if verbose: print 'Wrote graph file %s' % filename
