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

def process_hashed_dataset():
    # first hash labelled 'a', second labelled 'b', etc
    hashers = dict([(string.ascii_lowercase[i], LSHasher(int(math.log(
        DATA_SIZE / 100, 2)), 10000)) for i in xrange(NUM_HASHES)])
    print 'Hashers initialized'
    doc_features = {}
    with open(os.path.join(DATA_DIR, 'test.data'), 'rb') as data:
        datareader = csv.reader(data, delimiter=' ')
        for row in datareader:
            doc = int(row[0])
            word = int(row[1]) - 1
            count = int(row[2])
            if doc not in doc_features:
                doc_features[doc] = [] # [0 for i in xrange(NUM_FEATURES)]
            doc_features[doc].append(word) #[word] = count
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

def process_baseline_dataset():
    test_data = []
    max_count = 1 # if we want to weight the edges by word count
    with open(os.path.join(DATA_DIR, 'test.data'), 'rb') as data:
        datareader = csv.reader(data, delimiter=' ')
        for row in datareader:
            doc = int(row[0])
            word = int(row[1]) - 1
            count = int(row[2])
            if count > max_count:
                max_count = float(count)
            test_data.append([doc, word, count])

    with open(os.path.join(DATA_DIR, 'unhashed_test.data'), 'wb') as unhashed:
        datawriter = csv.writer(unhashed, delimiter='\t')
        for d,w,c in test_data:
            datawriter.writerow([str(d), 'w' + str(w), '1.0'])

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
    print 'Processing dataset with LSH'
    process_hashed_dataset()
    print 'Processing dataset without LSH'
    process_baseline_dataset()
    print 'Choosing seed labels'
    make_seeds(0.5)