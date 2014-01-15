#!/usr/bin/python
import os
import csv
import string
import math
import random
from datetime import datetime
import json

import numpy as np

from lsh import LSHasher, MultiLSHasher

DATA_DIR = os.path.join('data', '20_newsgroups')
TEST_DATA = 'test.data'
NUM_DOCS = 7505
NUM_FEATURES = 61188

def generate_lsh_graph(num_hashes=3, num_bits=5, verbose=False):
    # first hash labelled 'a', second labelled 'b', etc
    hashers = MultiLSHasher(num_hashes, num_bits)
    if verbose: print 'Hashers initialized'
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
    if verbose: print 'Loaded doc features'
    hashers.compute_stream(doc_features)
    signatures = hashers.compute_signatures()
    if verbose: print 'Computed signatures'
    hw_doc = {}
    doc_hw = {}
    with open(os.path.join(DATA_DIR, 'test.data'), 'rb') as data:
        filename = 'test-lsh-h%db%d.data' % (num_hashes, num_bits)
        with open(os.path.join(DATA_DIR, filename), 'wb') as hashed:
            datareader = csv.reader(data, delimiter=' ')
            datawriter = csv.writer(hashed, delimiter='\t')
            for row in datareader:
                doc = int(row[0])
                word = int(row[1])
                count = int(row[2])
                if doc not in doc_hw:
                    doc_hw[doc] = []
                for hl, s in signatures.items():
                    hashed_word = str(word) + hl + s[doc]
                    tfidf = math.log(count+1) * math.log(NUM_DOCS/float(words_doc_count[word]))
                    doc_hw[doc].append([hashed_word, tfidf])
                    if hashed_word not in hw_doc:
                        hw_doc[hashed_word] = []
                    hw_doc[hashed_word].append([doc, tfidf])
                    datawriter.writerow([doc, hashed_word, tfidf])
    with open('debug-lsh.json', 'wb') as f:
        json.dump(hw_doc, f, indent=4, separators=(',', ': '))
    with open('debug2-lsh.json', 'wb') as f:
        json.dump(doc_hw, f, indent=4, separators=(',', ': '))
generate_lsh_graph(10,2,True)

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
    hw_doc = {}
    doc_hw = {}
    with open(os.path.join(DATA_DIR, 'test-baseline.data'), 'wb') as unhashed:
        datawriter = csv.writer(unhashed, delimiter='\t')
        for d,w,c in test_data:
            if d not in doc_hw:
                doc_hw[d] = []
            if w not in hw_doc:
                hw_doc[w] = []
            tfidf = math.log(c+1) * math.log(NUM_DOCS/float(words_doc_count[w]))
            doc_hw[d].append([w, tfidf])
            hw_doc[w].append([d, tfidf])
            # datawriter.writerow([str(d), 'w' + str(w), tfidf])
    with open('debug.json', 'wb') as f:
        json.dump(hw_doc, f, indent=4, separators=(',', ': '))
    with open('debug2.json', 'wb') as f:
        json.dump(doc_hw, f, indent=4, separators=(',', ': '))
# generate_baseline_graph()
'''
with open('debug-lsh.json', 'r') as f1:
    with open('debug2-lsh.json', 'r') as f2:
        hwd = json.load(f1)
        dhw = json.load(f2)
        print('Average neighborhood size: ' + str(float(sum([len(docs) for docs in hwd.values()])) / len(hwd)))
        num_neighbors = []
        for d, hws in dhw.items():
            neighbors = set([int(d)])
            # hws = sorted(hws, key=lambda hw: hw[1])
            for hwtf in hws:
                hw = hwtf[0]
                tf = hwtf[1]
                for doc,tf in hwd[str(hw)]:
                    neighbors.add(doc)
            print('%d [%d / %s]: %s' % (doc, len(neighbors), str([len(hwd[str(hw[0])]) for hw in hws]), str(neighbors)))
            num_neighbors.append(len(neighbors))
        print('Average neighborhood size: ' + str(float(sum(num_neighbors)) / len(num_neighbors)))
'''
