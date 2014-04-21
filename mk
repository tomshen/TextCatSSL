#!/usr/bin/env python
import os
import subprocess
import sys

import analyze
import config
import graph
import random
import util

def make_kmeans_graph(data_set, num_clusterings, k):
    graph.generate_kmeans_graph(data_set, num_clusterings, k, verbose=True)

def make_lsh_graph(data_set, num_hashes, num_bits):
    graph.generate_lsh_graph(data_set, num_hashes, num_bits, verbose=True)

def make_knn_graph(data_set, num_neighbors):
    graph.generate_knn_graph(data_set, num_neighbors, verbose=True)

def make_baseline_graph(data_set):
    graph.generate_baseline_graph(data_set, verbose=True)

def make_iterative_baseline_graph(data_set, iterations, percentile):
    output_file = data_set + '-pc%di%d' % (percentile, iterations)
    print 'Iteration 1'
    graph.generate_baseline_graph(data_set, filename=output_file, verbose=True)
    if iterations == 1:
        return
    make_config(output_file)
    run_junto(output_file)
    for i in xrange(2, iterations):
        run_junto(output_file)
        print 'Iteration %d' % i
        graph.generate_labeled_baseline_graph(output_file, verbose=True,
                                              percentile=percentile)
    run_junto(output_file)
    print 'Iteration %d' % iterations
    graph.generate_labeled_baseline_graph(output_file, verbose=True,
                                          percentile=percentile)


def make_config(data_set):
    util.make_config(data_set)

def make_seeds(data_set, perc_seeds=0.1):
    labels = {}
    num_docs = util.get_num_docs(data_set)
    labels = util.get_label_docs(data_set)
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
    subprocess.call(['./lib/junto/bin/junto', 'config',
        os.path.join(config.CONFIG_DIR, config_file)], env=junto_env)

def analyze_output(graph_file):
    analyze.save_pred_labels(graph_file)
    analyze.compare_to_true_labels(graph_file)
    def analysis_string(a):
        return '\n'.join([str(k) + ': ' + str(v) for k,v in a.items()]) + '\nAverage: %f' % (float(sum(a.values())) / len(a))
    print 'Precision:\n' + analysis_string(analyze.get_precision(graph_file))
    print 'Recall:\n' + analysis_string(analyze.get_recall(graph_file))
    print 'F1:\n' + analysis_string(analyze.get_f1_scores(graph_file))

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

def clean_data(data_set):
    print 'Deleting all files in %s with suffix %s...' % (config.DATA_DIR,
        data_set)
    for f in os.listdir(config.DATA_DIR):
        if '_' in f and f.split('.')[0].split('_')[1] == data_set:
            os.remove(os.path.join(config.DATA_DIR, f))
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
            elif graph_type.lower() == 'knn':
                k = int(sys.argv[4])
                make_knn_graph(data_set, k)
            elif graph_type == 'base' or graph_type == 'baseline':
                make_baseline_graph(data_set)
            elif graph_type == 'ib':
                try:
                    i = int(sys.argv[4])
                except:
                    i = 2
                try:
                    pc = int(sys.argv[5])
                except:
                    pc = 95
                make_iterative_baseline_graph(data_set, i, pc)
            elif graph_type == 'kmeans':
                k = int(sys.argv[4])
                try:
                    nc = int(sys.argv[5])
                except:
                    nc = 1
                make_kmeans_graph(data_set, k, nc)
        elif sys.argv[1] == 'config':
            make_config(sys.argv[2])
        elif sys.argv[1] == 'seeds':
            perc_seeds = 0.1
            if len(sys.argv) > 3:
                perc_seeds = float(sys.argv[3])
            make_seeds(sys.argv[2], perc_seeds)
        elif sys.argv[1] == 'analyze':
            analyze_output(sys.argv[2])
        elif sys.argv[1] == 'plot':
            analyze.hash_class_image(sys.argv[2], save=False)
        elif sys.argv[1] == 'run':
            if sys.argv[2] not in os.listdir(config.CONFIG_DIR):
                make_config(sys.argv[2])
            run_junto(sys.argv[2])
        elif sys.argv[1] == 'clean':
            clean()
            if len(sys.argv) > 2:
                clean_data(sys.argv[2])
        elif sys.argv[1] == 'small':
            labels = [1,2]
            if len(sys.argv) > 5 and sys.argv[4] == 'random':
                labels = random.sample(range(1,21), int(sys.argv[5]))
            analyze.make_small_data_set(sys.argv[2], int(sys.argv[3]), labels)


if __name__ == '__main__':
    main()
