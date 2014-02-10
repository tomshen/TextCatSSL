#!/usr/bin/env python

import csv
import os
import random
import string
import sys
from collections import Counter

import matplotlib.pyplot as plt

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

def save_pred_labels(graph_file):
    pred_labels = get_pred_labels(graph_file)
    with util.open_output_file(graph_file + '.pred', 'wb') as f:
        for doc, label_weights in sorted(pred_labels.items()):
            f.write('%d %d %f\n' % (doc, label_weights[0], label_weights[1]))
    print 'Saved predictions for %s.pred' % graph_file

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

def get_label_feature_probs(data_set):
    label_fps = {}
    with util.open_output_file(data_set + '-feat-prob') as data:
        datareader = csv.reader(data)
        for datum in datareader:
            label = int(datum[0])
            feature = int(datum[1])
            prob = float(datum[2])

            if label not in label_fps:
                label_fps[label] = []
            label_fps[label].append((feature, prob))
    return label_fps

def make_small_data_set(data_set, num_docs, labels):
    small_set = 's' + ''.join(map(str, labels)) + '_' + data_set

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

def get_doc_hashes(graph_file):
    assert 'lsh' in graph_file
    doc_hashes = {}
    with util.open_graph_file(graph_file, 'rb') as graph:
        datareader = csv.reader(graph, delimiter='\t')
        for row in datareader:
            doc = int(row[0])
            for i in xrange(len(row[1])):
                if row[1][i] in string.ascii_lowercase:
                    hl = row[1][i]
                    h = row[1][i+1:]
                    if doc not in doc_hashes:
                        doc_hashes[doc] = {}
                    if hl in doc_hashes[doc]:
                        assert h == doc_hashes[doc][hl]
                    else:
                        doc_hashes[doc][hl] = h
                    break
    return doc_hashes

def get_hash_docs(graph_file):
    assert 'lsh' in graph_file
    hash_docs = {}
    with util.open_graph_file(graph_file, 'rb') as graph:
        datareader = csv.reader(graph, delimiter='\t')
        for row in datareader:
            doc = int(row[0])
            for i in xrange(len(row[1])):
                if row[1][i] in string.ascii_lowercase:
                    hl = row[1][i]
                    h = row[1][i+1:]
                    if hl not in hash_docs:
                        hash_docs[hl] = {}
                    if not h in hash_docs[hl]:
                        hash_docs[hl][h] = set()
                    hash_docs[hl][h].add(doc)
                    break
    return hash_docs

def analyze_hash_labels(graph_file, hl='a'):
    pred_labels = get_pred_labels(graph_file)
    doc_hashes = { k: v[hl] for k,v in get_doc_hashes(graph_file).items() }
    hash_docs = get_hash_docs(graph_file)[hl]
    label_hashes = {}
    for doc, label_weights in pred_labels.items():
        label = label_weights[0]
        if label not in label_hashes:
            label_hashes[label] = set()
        label_hashes[label].add(doc_hashes[doc])
    label_hashes = { k: sorted(list(v)) for k,v in label_hashes.items() }
    hash_labels = {}
    for h, docs in hash_docs.items():
        for doc in docs:
            if doc in pred_labels:
                if h not in hash_labels:
                    hash_labels[h] = set()
                label = pred_labels[doc][0]
                hash_labels[h].add(label)
            #else:
                #hash_labels[h].add('UNKNOWN')
    hash_labels = { k: sorted(list(v)) for k,v in hash_labels.items() }
    util.print_as_json(label_hashes)
    util.print_as_json(hash_labels)

def plot_label_feature_probs(data_set1, data_set2, label=None):
    if label is None:
        lfps = zip(*sorted(get_label_feature_probs(data_set1).popitem()[1]))
        lfps2 = zip(*sorted(get_label_feature_probs(data_set2).popitem()[1]))
    else:
        lfps = zip(*sorted(get_label_feature_probs(data_set1)[label]))
        lfps2 = zip(*sorted(get_label_feature_probs(data_set2)[label]))
    plt.plot(lfps[0], lfps[1], 'r')
    plt.plot(lfps2[0], lfps2[1], 'b')

    plt.show()

def main():
    label_feature_probs(sys.argv[1])
    label_feature_probs(sys.argv[2])
    if len(sys.argv) > 3:
        plot_label_feature_probs(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        plot_label_feature_probs(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
    main()