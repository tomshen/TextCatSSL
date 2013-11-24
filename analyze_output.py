#!/usr/bin/python
import os
import csv
import sys

DATA_DIR = os.path.join('data', '20_newsgroups')
OUTPUT_DIR = 'out'

def get_pred_labels(junto_output_filename):
    pred_labels = {}
    with open(junto_output_filename, 'rb') as f:
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

def get_seeds():
    with open(os.path.join(DATA_DIR, 'seeds.data'), 'rb') as f:
        datareader = csv.reader(f, delimiter='\t')
        return set([int(row[0]) for row in datareader])

def calculate_f1_score(precision, recall):
    return 2.0 * (precision * recall) / (precision + recall)

def compare_to_true_labels(junto_output_filename):
    pred_labels = get_pred_labels(junto_output_filename)
    seeds = get_seeds()
    true_labels = {}
    num_pred = 0
    num_incorrect = 0
    with open(os.path.join(DATA_DIR, 'test.label'), 'r') as f:
        curr_doc = 1
        for label in f:
            if curr_doc not in seeds and curr_doc in pred_labels:
                num_pred += 1
                if int(label) != pred_labels[curr_doc][0]:
                    num_incorrect += 1
            curr_doc += 1
    error_rate = float(num_incorrect) / num_pred
    info = '%s - error_rate: %.3f' % (junto_output_filename, error_rate)
    print(info)
    return info

if __name__ == '__main__':
    compare_to_true_labels(sys.argv[1])
