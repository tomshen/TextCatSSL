import csv
from os.path import join

from config import *

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
        i = 1
        for line in labels:
            doc_labels[i] = int(line.strip())
            i += 1
    return doc_labels

def get_doc_features(data_set):
    doc_features = {}
    with open_data_file(data_set) as data:
        datareader = csv.reader(data, delimiter=' ')
        for datum in datareader:
            doc = int(datum[0])
            feature = int(datum[1])
            count = int(datum[2])
            if doc not in doc_features:
                doc_features[doc] = {}
            doc_features[doc][feature] = count

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
