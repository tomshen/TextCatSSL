#!/usr/bin/env python
import json
import time
import sys

import analyze
import mk

data_sets = ['20NG', 'R8', 'webkb']

def run_lsh(data_set):
    mk.make_lsh_graph(data_set, 6, 6)
    filename = '%s-lsh-h5b5' % data_set
    mk.make_config(filename)
    mk.run_junto(filename)
    return analyze.compare_to_true_labels(filename)

def run_ib(data_set):
    i = 5
    pc = 90
    mk.make_iterative_baseline_graph(data_set, i, pc)
    filename = '%s-pc%di%d' % (data_set, pc, i)
    mk.make_config(filename)
    mk.run_junto(filename)
    return analyze.compare_to_true_labels(filename)

def run_baseline(data_set):
    mk.make_baseline_graph(data_set)
    filename = data_set + '-baseline'
    mk.make_config(filename)
    mk.run_junto(filename)
    return analyze.compare_to_true_labels(filename)

def run_knn(data_set):
    filename = '%s-knn-k50' % data_set
    mk.make_config(filename)
    mk.run_junto(filename)
    return analyze.compare_to_true_labels(filename)

if __name__ == '__main__':
    for i in xrange(5):
        data_error = {}
        for data_set in data_sets:
            error_rates = {}
            error_rates['b'] = run_baseline(data_set)
            error_rates['ib'] = run_ib(data_set)
            error_rates['lsh'] = run_lsh(data_set)
            error_rates['knn'] = run_knn(data_set)
            data_error[data_set] = error_rates
        print data_error
        with open('error-rates%d.txt' % i, 'w') as f:
            json.dump(data_error, f)

