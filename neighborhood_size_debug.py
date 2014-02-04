#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from debug import gen_lsh

def plot_graphs_cdf(h,b):
    data = gen_lsh(h,b)
    X = list(data.values())
    plt.hist(X, bins=max(X), cumulative=True)
    plt.savefig('img/lsh-h%db%d-cdf' % (h,b))
    plt.clf()

def plot_graphs(h,b):
    data = gen_lsh(h,b)
    X = list(data.values())
    plt.hist(X, bins=max([1,sum(X)/10]))
    print 'Average: %f' % (sum(X) / len(X))
    print data
    plt.savefig('img/lsh-h%db%d' % (h,b))
    plt.clf()

if __name__ == '__main__':
    plot_graphs(1,1)
    '''
    for h in [1,3,5]:
        for b in [1,2,3,4,5,8,10]:
            plot_graphs(h,b)
            plot_graphs_cdf(h,b)'''
