import os
import math
import csv
import string
import random

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

def process_dataset():
    # first hash labelled 'a', second labelled 'b', etc
    hashers = dict([(string.ascii_lowercase[i], LSHasher(int(math.log(
        DATA_SIZE / 100, 2)), 1000)) for i in xrange(NUM_HASHES)])
    print 'Hahsers initialized'
    doc_features = {}
    with open(os.path.join(DATA_DIR, 'test.data'), 'rb') as data:
        datareader = csv.reader(data, delimiter=' ')
        for row in datareader:
            doc = int(row[0])
            word = int(row[1]) - 1
            count = int(row[2])
            if doc not in doc_features:
                doc_features[doc] = [0 for i in xrange(NUM_FEATURES)]
            doc_features[doc][word] = count
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
                word = int(row[1]) - 1
                count = int(row[2])
                for hl, s in signatures.items():
                    hashed_word = str(word) + hl + s[doc]
                    weight = '1.0'
                    datawriter.writerow([doc, hashed_word, weight])

def make_seeds():
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
            for doc in random.sample(docs, len(docs) / 2): # take half of labels
                f.write(str(doc) + '\t' + str(label) + '\t1.0\n')
    with open(os.path.join(DATA_DIR, 'gold.data'), 'w') as f:
        for label, docs in labels.items():
            for doc in docs:
                f.write(str(doc) + '\t' + str(label) + '\t1.0\n')

# process_dataset()
make_seeds()