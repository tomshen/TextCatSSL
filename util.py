import csv
import json
import os
from os.path import join
import re
import shutil
import subprocess

from config import *

def open_map_file(data_set, flags='rb'):
    return open(join(DATA_DIR, data_set + '.map'), flags)

def open_vocab_file(data_set, flags='rb'):
    return open(join(DATA_DIR, data_set + '.vocab'), flags)

def open_data_file(data_set, flags='rb'):
    return open(join(DATA_DIR, data_set + '.data'), flags)

def open_label_file(data_set, flags='rb'):
    return open(join(DATA_DIR, data_set + '.label'), flags)

def open_seeds_file(data_set, flags='rwb'):
    return open(join(DATA_DIR, data_set + '.seeds'), flags)

def open_gold_file(data_set, flags='rwb'):
    return open(join(DATA_DIR, data_set + '.gold'), flags)

def open_graph_file(graph_file, flags='wb'):
    return open(join(GRAPH_DIR, graph_file), flags)

def open_output_file(graph_file, flags='rb'):
    return open(join(OUTPUT_DIR, graph_file), flags)

def duplicate_label_file(old_data_set, new_data_set):
    return shutil.copyfile(join(DATA_DIR, old_data_set + '.label'), join(DATA_DIR, new_data_set + '.label'))

def duplicate_count_file(old_data_set, new_data_set):
    return shutil.copyfile(join(DATA_DIR, old_data_set + '.count'), join(DATA_DIR, new_data_set + '.count'))

def save_plot(plt, filename):
    plt.savefig(join(OUTPUT_DIR, join('img', filename)))

# returns [num_docs, num_features]
def get_counts(data_set):
    with open(join(DATA_DIR, data_set + '.count')) as f:
        return [int(line.strip()) for line in f.readlines()]

def get_num_docs(data_set):
    return get_counts(data_set)[0]

def get_num_features(data_set):
    return get_counts(data_set)[1]

def get_num_labels(data_set):
    return get_counts(data_set)[2]

def get_doc_labels(data_set):
    doc_labels = {}
    with open_label_file(data_set) as labels:
        docs = set(get_doc_features(data_set).keys())
        i = 1
        for line in labels:
            if i in docs:
                try:
                    doc_labels[i] = int(line.strip())
                except:
                    doc_labels[i] = float(line.strip())
            i += 1
    return doc_labels

def get_class_names(data_set):
    with open_map_file(data_set) as f:
        return {i+1: w.split()[0] for i, w in enumerate(f)}

def get_actual_words(data_set):
    with open_vocab_file(data_set) as f:
        return {i+1: w.strip() for i, w in enumerate(f)}

__vocab_cache = {}
def get_word_index(word, data_set='20NG'):
  if data_set not in __vocab_cache:
      __vocab_cache[data_set] = get_actual_words(data_set)
  return next((i for i, w in __vocab_cache[data_set].items()
              if word == w), None)

def get_label_docs(data_set):
    label_docs = {}
    with open_label_file(data_set) as labels:
        docs = set(get_doc_features(data_set).keys())
        i = 1
        for line in labels:
            if i in docs:
                try:
                    label = int(line.strip())
                except:
                    label = float(line.strip())
                if label not in label_docs:
                    label_docs[label] = set()
                label_docs[label].add(i)
            i += 1
    return label_docs

def get_doc_features(data_set):
    doc_features = {}
    with open_data_file(data_set) as data:
        datareader = csv.reader(data, delimiter=' ')
        for datum in datareader:
            doc = int(datum[0])
            feature = int(datum[1])
            try:
                weight = int(datum[2])
            except:
                weight = float(datum[2])
            if doc not in doc_features:
                doc_features[doc] = {}
            doc_features[doc][feature] = weight
    return doc_features

def get_label_features(data_set):
    label_features = {}
    doc_labels = get_doc_labels(data_set)
    with open_data_file(data_set) as data:
        datareader = csv.reader(data, delimiter=' ')
        for datum in datareader:
            label = doc_labels[int(datum[0])]
            feature = int(datum[1])
            count = int(datum[2])
            if label not in label_features:
                label_features[label] = {}
            label_features[label][feature] = count
    return label_features

def make_config(graph_file):
    data_set = graph_file.split('-')[0]
    junto_config = {
        'data_format': 'edge_factored',
        'iters': '10',
        'verbose': 'false',
        'prune_threshold': '0',
        'algo': 'mad',
        'mu1': '1',
        'mu2': '1e-2',
        'mu3': '1e-2',
        'beta': '2',
        'seed_file': join(DATA_DIR, data_set + '.seeds'),
        'test_file': join(DATA_DIR, data_set + '.gold'),
        'graph_file': join(GRAPH_DIR, graph_file),
        'output_file': join(OUTPUT_DIR, graph_file)
    }
    config_string = '\n'.join([k + ' = ' + v for k,v in junto_config.items()])
    with open(join(CONFIG_DIR, graph_file), 'w') as f:
        f.write(config_string)

def is_word(s):
    try:
        i = int(s) # relies on words containing non-digits
        return False
    except:
        return True

def parse_output_file(output_file):
    def parse_label_scores(s):
        scores = s.split()
        return {scores[i]: float(scores[i+1]) for i in xrange(0, len(scores), 2)}

    with open_output_file(output_file) as f:
        data = csv.reader(f, delimiter='\t')
        words = {}
        docs = {}
        for row in data:
            name = row[0]
            # gold_label, gold_score = [int(x) for x in row[1].split(' ')]
            # seed_label, seed_score = [int(x) for x in row[2].split(' ')]
            label_scores = parse_label_scores(row[3])
            if is_word(name):
                words[int(re.split('\D', name)[0])] = label_scores
            else:
                docs[int(name)] = label_scores

        return docs, words

def print_as_json(obj, sort=True):
    print(json.dumps(obj, sort_keys=sort, indent=2, separators=(',', ': ')))

def run_junto(config_file):
    junto_env = os.environ.copy()
    junto_env['JUNTO_DIR'] = os.path.join(os.getcwd(), 'lib/junto')
    junto_env['PATH'] = junto_env['JUNTO_DIR'] + ':' + junto_env['PATH']
    junto_env['JAVA_MEM_FLAG'] = '-Xmx32g'
    subprocess.call(['./lib/junto/bin/junto', 'config',
        os.path.join(CONFIG_DIR, config_file)], env=junto_env)
