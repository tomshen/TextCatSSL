#!/usr/bin/python
import os
import subprocess

from graph_gen import *
from analyze_output import *

OUTPUT_DIR = 'out'
CONFIG_DIR = 'config'
JUNTO = 'lib/junto/bin/junto'

num_hashes = [1,3,5]
num_bits = [1,2,5,8,10,11,12,13,15]
perc_seeds = [0.01,0.02,0.05,0.10]
num_nearest_neighbors = [5,10,20,30,50,100]

def generate_graphs():
    generate_baseline_graph()
    print('Generated baseline graph')

    for h in num_hashes:
        for b in num_bits:
            generate_lsh_graph(h,b)
            print('Generated LSH graph with %d hashes of %d bits' % (h,b))
    print('Generated LSH graphs')

    generate_knn_graphs(num_nearest_neighbors)
    print('Generated k-NN graphs')

default_config = {
    'data_format': 'edge_factored',
    'iters': '10',
    'verbose': 'false',
    'prune_threshold': '0',
    'algo': 'mad',
    'mu1': '1',
    'mu2': '1e-2',
    'mu3': '1e-2',
    'beta': '2',
    'seed_file': os.path.join(DATA_DIR, 'seeds.data'),
    'test_file': os.path.join(DATA_DIR, 'gold.data')
}
default_config_string = '\n'.join([k + ' = ' + v for k,v in default_config.items()])
def make_junto_config(test_data_filename):
    test_data = test_data_filename.split('.')[0]
    graph_file = os.path.join(DATA_DIR, test_data_filename)
    output_file = os.path.join(OUTPUT_DIR, test_data + '.out')
    with open(os.path.join(CONFIG_DIR, test_data + '.config'), 'w') as f:
        f.write(default_config_string)
        f.write('\ngraph_file = %s\n' % graph_file)
        f.write('output_file = %s\n' % output_file)

def prepare_for_propagation():
    make_seeds(0.1)
    print('Made seed data')

    make_junto_config('test-baseline.data')

    for h in num_hashes:
        for b in num_bits:
            make_junto_config('test-lsh-h%db%d.data' % (h,b))

    for k in num_nearest_neighbors:
        make_junto_config('test-knn-k%d.data' % k)

    print('Made junto config files')

def propagate_labels():
    junto_env = os.environ.copy()
    junto_env['JUNTO_DIR'] = os.path.join(os.getcwd(), 'lib/junto')
    junto_env['PATH'] = junto_env['JUNTO_DIR'] + ':' + junto_env['PATH']
    def run_junto(config_filename):
        subprocess.Popen([JUNTO, 'config', os.path.join(CONFIG_DIR,
            config_filename)], env=junto_env)

    run_junto('test-baseline.config')
    print('Propagated labels on baseline graph')

    for h in num_hashes:
        for b in num_bits:
            run_junto('test-lsh-h%db%d.config' % (h,b))
            print('Propagated labels on LSH graph with %d hashes of %d bits' % (h,b))
    print('Propagated labels on LSH graphs')

    for k in num_nearest_neighbors:
        run_junto('test-knn-k%d.config' % k)
        print('Propagated labels on %d-NN graph' % k)
    print('Propagated labels on kNN graphs')

def analyze_outputs():
    with open('results.txt', 'w') as f:
        output_files = os.listdir()
        for output_file in output_files:
            f.write(compare_to_true_labels(output_file) + '\n')

if __name__ == '__main__':
    generate_graphs()
    prepare_for_propagation()
    propagate_labels()
    analyze_outputs()