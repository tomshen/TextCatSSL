#!/usr/bin/python
import os
import csv
import string
import math
import random
from datetime import datetime

import numpy as np

from lsh import LSHasher

'''
We will test by generating the edges for the graph connecting documents to
feature+hashes, labelling the train data (seed), propagating labels, and then
comparing the labels assigned to train data with the gold labels.
'''

DATA_DIR = os.path.join('data', '20_newsgroups')
TEST_DATA = 'test.data'
NUM_DOCS = 7505
NUM_FEATURES = 61188


def lsh_equivalent_k(num_hashes, num_bits):
    '''
    Given num_hashes and num_bits for LSH, what k for k-NN does that
    correspond to
    '''
    num_buckets = num_hashes * (1 << num_bits)
    return NUM_DOCS / num_buckets

def generate_lsh_graph(num_hashes=3, num_bits=5, verbose=False):
    # first hash labelled 'a', second labelled 'b', etc
    hashers = dict([
        (string.ascii_lowercase[i],
            LSHasher(num_bits, 10000)) for i in xrange(num_hashes)])
    if verbose: print 'Hashers initialized'
    doc_features = {}
    words_doc_count = [0 for i in xrange(NUM_FEATURES+1)]
    with open(os.path.join(DATA_DIR, TEST_DATA), 'rb') as data:
        datareader = csv.reader(data, delimiter=' ')
        for row in datareader:
            doc = int(row[0])
            word = int(row[1])
            count = int(row[2])
            words_doc_count[word] += 1
            if doc not in doc_features:
                doc_features[doc] = []
            doc_features[doc].append((word, count))
    if verbose: print 'Loaded doc features'
    signatures = {}
    for hl, h in hashers.items():
        h.compute_stream(doc_features)
        signatures[hl] = h.compute_signatures()
        if verbose: print 'Computed signatures for hasher %s' % hl
    if verbose: print 'Computed signatures'
    with open(os.path.join(DATA_DIR, TEST_DATA), 'rb') as data:
        filename = 'test-lsh-h%db%d.data' % (num_hashes, num_bits)
        with open(os.path.join(DATA_DIR, filename), 'wb') as hashed:
            datareader = csv.reader(data, delimiter=' ')
            datawriter = csv.writer(hashed, delimiter='\t')
            for row in datareader:
                doc = int(row[0])
                word = int(row[1])
                count = int(row[2])
                for hl, s in signatures.items():
                    hashed_word = str(word) + hl + s[doc]
                    tfidf = math.log(count+1) * math.log(NUM_DOCS/float(words_doc_count[word]))
                    datawriter.writerow([doc, hashed_word, tfidf])

def generate_baseline_graph():
    test_data = []
    words_doc_count = [0 for i in xrange(NUM_FEATURES+1)]
    with open(os.path.join(DATA_DIR, TEST_DATA), 'rb') as data:
        datareader = csv.reader(data, delimiter=' ')
        for row in datareader:
            doc = int(row[0])
            word = int(row[1])
            count = int(row[2])
            words_doc_count[word] += 1
            test_data.append([doc, word, count])
    with open(os.path.join(DATA_DIR, 'test-baseline.data'), 'wb') as unhashed:
        datawriter = csv.writer(unhashed, delimiter='\t')
        for d,w,c in test_data:
            tfidf = math.log(c+1) * math.log(NUM_DOCS/float(words_doc_count[w]))
            datawriter.writerow([str(d), 'w' + str(w), tfidf])

def generate_knn_graph(k=10, verbose=False):
    assert k < NUM_DOCS
    feature_matrix = np.matrix(np.zeros((DATA_SIZE, NUM_FEATURES)))
    words_doc_count = np.zeros(NUM_FEATURES)
    with open(os.path.join(DATA_DIR, TEST_DATA), 'rb') as data:
        datareader = csv.reader(data, delimiter=' ')
        for row in datareader:
            doc = int(row[0]) - 1
            word = int(row[1]) - 1
            count = int(row[2])
            words_doc_count[word] += 1
            feature_matrix[doc][word] = count
    if verbose: print('[%s]: Loaded test data.' % str(datetime.now().time()))

    if verbose: print('[%s]: Generating feature matrix' % str(datetime.now().time()))
    for doc in xrange(NUM_DOCS):
        for word in xrange(NUM_FEATURES):
            if words_doc_count[word] != 0:
                count = feature_matrix.item((doc,word))
                tfidf = math.log(count+1) * math.log(NUM_DOCS/float(words_doc_count[word]))
                feature_matrix.itemset((doc,word), tfidf)
        if doc % 10 == 9:
            if verbose: print('[%s]: Processed %d out of %d documents' % (str(datetime.now().time()),
                (doc+1), NUM_DOCS))
    if verbose: print('[%s]: Generated feature matrix' % str(datetime.now().time()))

    normalizing_matrix = np.matrix(np.zeros((NUM_DOCS, NUM_DOCS)))
    for i in xrange(NUM_DOCS):
        f = feature_matrix[i]
        normalizing_matrix.itemset((i,i), 1.0 / math.sqrt(f * f.transpose()))
    if verbose: print('[%s]: Generated normalizing matrix' % str(datetime.now().time()))

    if verbose: print('[%s]: Generating folded graph' % str(datetime.now().time()))
    edges = []
    N = normalizing_matrix
    F = feature_matrix
    for doc in xrange(NUM_DOCS):
        Nv = np.matrix(np.zeros((NUM_DOCS,1)))
        Nv.itemset(doc, N.item((doc, doc)))
        FtNv = F[doc].transpose() * N.item((doc,doc))
        doc_weights = array(N * (F * FtNv)).transpose()
        nearest_neighbors = [i for i in np.argsort(doc_weights)[-k:]]
        for neighbor in nearest_neighbors:
            # so that we don't have duplicate edges
            edges.add(((min(doc+1, neighbor+1), max(doc+1, neighbor+1)),
                doc_weights[neighbor]))
        if doc % 10 == 9:
            if verbose: print('[%s]: Processed %d out of %d documents' % (
                str(datetime.now().time()), (doc+1), NUM_DOCS))
    if verbose: print('[%s]: Generated folded graph' % str(datetime.now().time()))

    with open(os.path.join(DATA_DIR, 'test-knn-k%d.data' % k), 'wb') as unhashed:
        datawriter = csv.writer(unhashed, delimiter='\t')
        for edge, weight in edges:
            datawriter.writerow([edge[0], edge[1], weight])

def make_seeds(perc_seeds=0.1):
    labels = {}
    with open(os.path.join(DATA_DIR, 'test.label'), 'r') as f:
        doc_idx = 1
        for line in f:
            label = int(line.strip())
            if label not in labels:
                labels[label] = []
            labels[label].append(doc_idx)
            doc_idx += 1
    with open(os.path.join(DATA_DIR, 'seeds.data'), 'w') as f:
        for label, docs in labels.items():
            for doc in random.sample(docs, int(len(docs) * perc_seeds)): # take perc_seeds of labels
                f.write(str(doc) + '\t' + str(label) + '\t1.0\n')
    with open(os.path.join(DATA_DIR, 'gold.data'), 'w') as f:
        for label, docs in labels.items():
            for doc in docs:
                f.write(str(doc) + '\t' + str(label) + '\t1.0\n')
