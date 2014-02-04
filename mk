#!/usr/bin/env python
import os
import subprocess
import sys

import analyze
import config
import graph
import random
import util

def make_lsh_graph(data_set, num_hashes, num_bits):
    graph.generate_lsh_graph(data_set, num_hashes, num_bits, verbose=True)

def make_knn_graph(data_set, num_neighbors):
    graph.generate_knn_graph(data_set, num_neighbors, verbose=True)

def make_baseline_graph(data_set):
    graph.generate_baseline_graph(data_set, verbose=True)

def make_config(data_set):
    util.make_config(data_set)

def make_seeds(data_set, perc_seeds=0.1):
    labels = {}
    num_docs = util.get_num_docs(data_set)
    with util.open_label_file(data_set) as f:
        doc_idx = 1
        for line in f:
            label = int(line.strip())
            if label not in labels:
                labels[label] = []
            labels[label].append(doc_idx)
            doc_idx += 1
    with util.open_seeds_file(data_set, 'wb') as f:
        for label, docs in labels.items():
            for doc in random.sample(docs, int(len(docs) * perc_seeds)): # take perc_seeds of labels
                f.write(str(doc) + '\t' + str(label) + '\t%d\n' % num_docs)
    with util.open_gold_file(data_set, 'wb') as f:
        for label, docs in labels.items():
            for doc in docs:
                f.write(str(doc) + '\t' + str(label) + '\t%d\n' % num_docs)

def run_junto(config_file):
    junto_env = os.environ.copy()
    junto_env['JUNTO_DIR'] = os.path.join(os.getcwd(), 'lib/junto')
    junto_env['PATH'] = junto_env['JUNTO_DIR'] + ':' + junto_env['PATH']
    subprocess.Popen(['./lib/junto/bin/junto', 'config',
        os.path.join(config.CONFIG_DIR, config_file)], env=junto_env)

def analyze_output(graph_file):
    analyze.compare_to_true_labels(graph_file)

def clean():
    dirs = [config.CONFIG_DIR, config.GRAPH_DIR, config.OUTPUT_DIR]
    for d in dirs:
        print 'Deleting all files in %s...' % d,
        for f in os.listdir(d):
            try:
                os.remove(os.path.join(d, f))
            except:
                d2 = os.path.join(d, f)
                if os.path.isdir(d2):
                    for f in os.listdir(d2):
                        os.remove(os.path.join(d2, f))
        print 'Done.'

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == 'graph':
            graph_type = sys.argv[2]
            data_set = sys.argv[3]
            if graph_type == 'lsh':
                h = int(sys.argv[4])
                b = int(sys.argv[5])
                make_lsh_graph(data_set, h, b)
            elif graph_type == 'knn':
                k = int(sys.argv[4])
                make_knn_graph(data_set, k)
            elif graph_type == 'base':
                make_baseline_graph(data_set)
        elif sys.argv[1] == 'config':
            make_config(sys.argv[2])
        elif sys.argv[1] == 'seeds':
            make_seeds(sys.argv[2])
        elif sys.argv[1] == 'analyze':
            analyze_output(sys.argv[2])
        elif sys.argv[1] == 'run':
            run_junto(sys.argv[2])
        elif sys.argv[1] == 'clean':
            clean()


if __name__ == '__main__':
    main()