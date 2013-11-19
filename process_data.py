#!/usr/bin/python
import os
import csv
import string
import math
import random
from datetime import datetime

import numpy as np

from lsh import LSHasher

"""
We will test by generating the edges for the graph connecting documents to
feature+hashes, labelling the train data (seed), propagating labels, and then
comparing the labels assigned to train data with the gold labels.
"""

DATA_DIR = 'data/20_newsgroups'
DATA_SIZE = 7505
NUM_HASHES = 3
NUM_FEATURES = 61188

def process_hashed_dataset():
    # first hash labelled 'a', second labelled 'b', etc
    hashers = dict([(string.ascii_lowercase[i], LSHasher(int(math.log(
        DATA_SIZE / 100, 2)), 10000)) for i in xrange(NUM_HASHES)])
    print 'Hashers initialized'
    doc_features = {}
    words_doc_count = [0 for i in xrange(NUM_FEATURES+1)]
    with open(os.path.join(DATA_DIR, 'test.data'), 'rb') as data:
        datareader = csv.reader(data, delimiter=' ')
        for row in datareader:
            doc = int(row[0])
            word = int(row[1])
            count = int(row[2])
            words_doc_count[word] += 1
            if doc not in doc_features:
                doc_features[doc] = []
            doc_features[doc].append((word, count))
    print 'Loaded doc features'
    signatures = {}
    for hl, h in hashers.items():
        h.compute_stream(doc_features)
        signatures[hl] = h.compute_signatures()
        print 'Computed signatures for hasher %s' % hl
    print 'Computed signatures'
    with open(os.path.join(DATA_DIR, 'test.data'), 'rb') as data:
        with open(os.path.join(DATA_DIR, 'hashed_test.data'), 'wb') as hashed:
            datareader = csv.reader(data, delimiter=' ')
            datawriter = csv.writer(hashed, delimiter='\t')
            for row in datareader:
                doc = int(row[0])
                word = int(row[1])
                count = int(row[2])
                for hl, s in signatures.items():
                    hashed_word = str(word) + hl + s[doc]
                    tfidf = math.log(count+1) * math.log(DATA_SIZE/float(words_doc_count[word]))
                    datawriter.writerow([doc, hashed_word, tfidf])

def process_baseline_dataset():
    test_data = []
    words_doc_count = [0 for i in xrange(NUM_FEATURES+1)]
    with open(os.path.join(DATA_DIR, 'test.data'), 'rb') as data:
        datareader = csv.reader(data, delimiter=' ')
        for row in datareader:
            doc = int(row[0])
            word = int(row[1])
            count = int(row[2])
            words_doc_count[word] += 1
            test_data.append([doc, word, count])
    with open(os.path.join(DATA_DIR, 'unhashed_test.data'), 'wb') as unhashed:
        datawriter = csv.writer(unhashed, delimiter='\t')
        for d,w,c in test_data:
            tfidf = math.log(c+1) * math.log(DATA_SIZE/float(words_doc_count[w]))
            datawriter.writerow([str(d), 'w' + str(w), tfidf])

def process_knn_dataset(k=30):
    assert k < DATA_SIZE
    feature_matrix = np.zeros((DATA_SIZE, NUM_FEATURES))
    words_doc_count = np.zeros(NUM_FEATURES)
    with open(os.path.join(DATA_DIR, 'test.data'), 'rb') as data:
        datareader = csv.reader(data, delimiter=' ')
        for row in datareader:
            doc = int(row[0]) - 1
            word = int(row[1]) - 1
            count = int(row[2])
            words_doc_count[word] += 1
            feature_matrix[doc][word] = count
    print('[%s]: Loaded test data.' % str(datetime.now().time()))

    print('[%s]: Generating feature matrix' % str(datetime.now().time()))
    for doc in xrange(DATA_SIZE):
        for word in xrange(NUM_FEATURES):
            if words_doc_count[word] != 0:
                count = feature_matrix[doc][word]
                tfidf = math.log(count+1) * math.log(DATA_SIZE/float(words_doc_count[word]))
                feature_matrix[doc][word] = tfidf
        if doc % 10 == 9:
            print('[%s]: Processed %d out of %d documents' % (str(datetime.now().time()),
                (doc+1), DATA_SIZE))
    print('[%s]: Generated feature matrix' % str(datetime.now().time()))

    normalizing_matrix = np.zeros((DATA_SIZE, DATA_SIZE))
    for i in xrange(DATA_SIZE):
        normalizing_matrix[i][i] = 1.0 / math.sqrt(feature_matrix[i] * feature_matrix[i].transpose())

    print('[%s]: Generated normalizing matrix' % str(datetime.now().time()))

    print('[%s]: Generating folded graph' % str(datetime.now().time()))
    edges = set([])
    for doc in xrange(DATA_SIZE):
        v = np.zeros(DATA_SIZE)
        v[doc] = 1
        N = normalizing_matrix
        F = feature_matrix
        doc_weights = N * (F * (F.transpose() * (N * v)))
        nearest_neighbors = np.argsort(doc_weights)[-k:]
        for neighbor in nearest_neighbors:
            # so that we don't have duplicate edges
            edges.add(((min(doc+1, neighbor+1), max(doc+1, neighbor+1)), doc_weights[neighbor]))
        if doc % 10 == 9:
            print('[%s]: Processed %d out of %d documents' % (str(datetime.now().time()),
                (doc+1), DATA_SIZE))
    print('[%s]: Generated folded graph' % str(datetime.now().time()))

    with open(os.path.join(DATA_DIR, 'knn_test.data'), 'wb') as unhashed:
        datawriter = csv.writer(unhashed, delimiter='\t')
        for edge, weight in edges:
            datawriter.writerow([edge[0], edge[1], weight])

def make_seeds(perc_seeds):
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

if __name__ == '__main__':
    process_knn_dataset()
    '''
    print 'Processing dataset with LSH'
    process_hashed_dataset()
    print 'Processing dataset without LSH'
    process_baseline_dataset()
    print 'Choosing seed labels'
    make_seeds(0.5)
    '''
