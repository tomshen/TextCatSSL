#!/usr/bin/env python
import json
import time
import sys

import analyze
import mk

data_sets = ['R8', 'webkb']

def run_lsh(data_set):
    error_rates = {}
    for h in [1,3,5]:
        error_rates[h] = {}
        for b in [1,3,5]:
            t0 = time.clock()
            mk.make_lsh_graph(data_set, h, b)
            filename = '%s-lsh-h%db%d' % (data_set, h, b)
            mk.make_config(filename)
            mk.run_junto(filename)
            error_rates[h][b] = (analyze.compare_to_true_labels(filename), time.clock() - t0)
    return error_rates

def run_baseline(data_set):
    t0 = time.clock()
    mk.make_baseline_graph(data_set)
    filename = data_set + '-baseline'
    mk.make_config(filename)
    mk.run_junto(filename)
    return analyze.compare_to_true_labels(filename), time.clock() - t0


def run_ib(data_set):
    error_rates = {}
    for pc in [50,75,90,95,99]:
        error_rates[pc] = {}
        for i in [2,3,4,5]:
            t0 = time.clock()
            mk.make_iterative_baseline_graph(data_set, i, pc)
            filename = '%s-pc%di%d' % (data_set, pc, i)
            mk.make_config(filename)
            mk.run_junto(filename)
            error_rates[pc][i] = (analyze.compare_to_true_labels(filename), time.clock() - t0)
    return error_rates

def run_knn(data_set):
    error_rates = {}
    for k in [10,30,50]:
        t0 = time.clock()
        filename = '%s-knn-k%d' % (data_set, k)
        mk.make_config(filename)
        mk.run_junto(filename)
        error_rates[k] = (analyze.compare_to_true_labels(filename), time.clock() - t0)
    return error_rates

if __name__ == '__main__':
    error_rates = {
        'baseline': [],
        'lsh': [],
        'knn': [],
        'ib': []
    }
    data_set = sys.argv[1]
    for i in xrange(1):
        for ps in [0.01, 0.05, 0.1]:
            mk.make_seeds(data_set, perc_seeds=ps)
            error_rates['baseline'].append(run_baseline(data_set))
            error_rates['lsh'].append(run_lsh(data_set))
            error_rates['knn'].append(run_knn(data_set))
            error_rates['ib'].append(run_ib(data_set))
    with open('error-rates-%s.txt' % data_set, 'w') as f:
        json.dump(error_rates, f)

