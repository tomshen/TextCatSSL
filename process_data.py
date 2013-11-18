#!/usr/bin/python
import os
import math
import csv
import string
import math
import random
from datetime import datetime

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
    test_data = {}
    words_doc_count = [0 for i in xrange(NUM_FEATURES+1)]
    with open(os.path.join(DATA_DIR, 'test.data'), 'rb') as data:
        datareader = csv.reader(data, delimiter=' ')
        for row in datareader:
            doc = int(row[0])
            word = int(row[1])
            count = int(row[2])
            words_doc_count[word] += 1
            if doc not in test_data:
                test_data[doc] = []
            test_data[doc].append((word, count))
    print('[%s]: Loaded test data.' % str(datetime.now().time()))
    for doc, words in test_data.items():
        features = [0 for i in xrange(NUM_FEATURES)]
        for w,c in test_data[doc]:
            tfidf = math.log(c+1) * math.log(DATA_SIZE/float(words_doc_count[w]))
            features[w-1] = tfidf
        test_data[doc] = features
    print('[%s]: Generated feature vectors' % str(datetime.now().time()))

    def cosine_similarity(u,v):
        dot_product = float(sum([u[i] * v[i] for i in xrange(NUM_FEATURES)]))
        norm_u = math.sqrt(sum([x**2 for x in u]))
        norm_v = math.sqrt(sum([x**2 for x in v]))
        return dot_product / (norm_u * norm_v)

    doc_similarity = dict([(doc,[]) for doc in test_data.keys()])
    edges = set([])
    graph_gen_counter = 0
    for docA, featA in test_data.items():
        for docB, featB in test_data.items():
            cs = cosine_similarity(featA, featB)
            doc_similarity[docA].append((cs, docB))
            doc_similarity[docB].append((cs, docA))
        doc_similarity[docA].sort(reverse=True)
        for i in xrange(k):
            similarity = doc_similarity[docA][i]
            docB = similarity[0]
            weight = similarity[1]
            edge = (docA, docB) if docA < docB else (docB, docA)
            edges.add((edge, weight))
        graph_gen_counter += 1
        if graph_gen_counter % 10 == 0:
            print('[%s]: Processed %d of %d documents' % str(datetime.now().time()),
                graph_gen_counter, DATA_SIZE)
        test_data[docA] = None
    print('[%s]: Generated k-NN graph' % str(datetime.now().time()))

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
