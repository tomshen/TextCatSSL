#!/usr/bin/env python
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import config
from lsh import MultiLSHasher
import util

data_set = '20NG'

def gen_lsh(num_hashes, num_bits, verbose=True):
    # first hash labelled 'a', second labelled 'b', etc
    hashers = MultiLSHasher(num_hashes, num_bits)
    print '%d hashes, %d bits' % (num_hashes, num_bits)
    if verbose: print 'Hashers initialized'
    doc_features = {}
    words_doc_count = [0 for i in xrange(util.get_num_features(data_set)+1)]
    with util.open_data_file(data_set) as data:
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
    hd = {}
    for hl, s in signatures.items():
        for doc, sig in s.items():
            h = hl + sig
            if h not in hd:
                hd[h] = 0
            hd[h] += 1
    return hd

def plot_graphs_cdf(h,b):
    data = gen_lsh(h,b)
    X = list(data.values())
    plt.hist(X, bins=max(X), cumulative=True)
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'img/lsh-h%db%d-cdf' % (h,b)))
    plt.clf()

def plot_graphs(h,b):
    data = gen_lsh(h,b)
    X = list(data.values())
    plt.hist(X, bins=max([1,sum(X)/10]))
    print 'Average: %f' % (sum(X) / len(X))
    print data
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'img/lsh-h%db%d' % (h,b)))
    plt.clf()

if __name__ == '__main__':
    for h in [1,3,5]:
        for b in [1,2,3,4,5,8,10]:
            plot_graphs(h,b)
            plot_graphs_cdf(h,b)
