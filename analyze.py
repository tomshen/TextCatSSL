#!/usr/bin/env python

import csv
import os
import random
import sys
from collections import Counter

import util

def get_pred_labels(graph_file):
    pred_labels = {}
    with util.open_output_file(graph_file) as f:
        datareader = csv.reader(f, delimiter='\t')
        for row in datareader:
            try:
                doc = int(row[0])
                label_info = row[3].split(' ')
                try:
                    label = int(label_info[0])
                    weight = float(label_info[1])
                except:
                    label = int(label_info[2])
                    weight = float(label_info[3])

                pred_labels[doc] = (label, weight)
            except:
                # a word, not a doc (contains hash), or if seed label
                continue
    return pred_labels

def get_seeds(data_set):
    with util.open_seeds_file(data_set) as f:
        datareader = csv.reader(f, delimiter='\t')
        return set([int(row[0]) for row in datareader])

def calculate_f1_score(precision, recall):
    return 2.0 * (precision * recall) / (precision + recall)

def compare_to_true_labels(graph_file):
    pred_labels = get_pred_labels(graph_file)
    data_set = graph_file.split('-')[0]
    seeds = get_seeds(data_set)
    true_labels = {}
    num_pred = 0
    num_incorrect = 0
    with util.open_label_file(data_set) as f:
        curr_doc = 1
        for label in f:
            if curr_doc not in seeds and curr_doc in pred_labels:
                num_pred += 1
                if int(label) != pred_labels[curr_doc][0]:
                    num_incorrect += 1
            curr_doc += 1
    error_rate = float(num_incorrect) / num_pred
    print '%s - error_rate: %.3f' % (graph_file, error_rate)

def label_feature_probs(data_set):
    label_counters = util.get_label_features(data_set)
    label_sums = { i: float(sum(label_counters[i].values()))
        for i in label_counters }
    with util.open_output_file(data_set + '-feat-prob', 'wb') as out:
        datawriter = csv.writer(out)
        for label, feature_counter in label_counters.items():
            for feature, count in feature_counter.items():
                datawriter.writerow([
                    label,
                    feature,
                    count / label_sums[label]
                ])

def make_small_data_set(data_set, num_docs, labels):
    small_set = 's' + ''.join(map(str, labels)) + '-' + data_set

    doc_features = util.get_doc_features(data_set)
    label_docs = util.get_label_docs(data_set)
    samp_size = num_docs / len(labels)
    small_doc_labels = {}
    with util.open_data_file(small_set, 'wb') as data:
        datawriter = csv.writer(data, delimiter=' ')
        for label in labels:
            docs = random.sample(label_docs[label], samp_size)
            for doc in docs:
                for feature, count in doc_features[doc].items():
                    datawriter.writerow([
                        doc,
                        feature,
                        count
                    ])
    util.duplicate_label_file(data_set, small_set)
    util.duplicate_count_file(data_set, small_set)
    print('Smaller dataset with labels [%s] and %d docs created from %s.' %
        (','.join(map(str, labels)), num_docs, data_set))

def main():
    # make_small_data_set('20NG', 10, [1,2])
    label_feature_probs(sys.argv[1])

if __name__ == '__main__':
    main()