#!/usr/bin/env python
import os.path as path
import subprocess

from config import PROPPR_DIR, PROPPR_PROGRAM_DIR
import util


def generate_facts(data_set):
    with open(path.join(PROPPR_PROGRAM_DIR, data_set + '.facts'), 'w') as f:
        for i in xrange(1, util.get_num_labels(data_set) + 1):
            f.write('isLabel(l%d)\n' % i)


def generate_data(data_set):
    labels = [str(i) for i in xrange(1, util.get_num_labels(data_set) + 1)]
    with util.open_seeds_file(data_set, 'r') as seeds:
        with open(path.join(PROPPR_PROGRAM_DIR, data_set + '.data'), 'w') as f:
            for line in seeds:
                doc, label, weight = line.split('\t')
                f.write('predict(d%s,Y)\t' % doc + '\t'.join(
                    '%spredict(d%s,l%s)' % (
                        '+' if label == l else '-',
                        doc,
                        l) for l in labels) + '\n')


def generate_graph(data_set):
    with util.open_data_file(data_set) as data:
        with open(path.join(PROPPR_PROGRAM_DIR, data_set + '.graph'), 'w') as f:
            for line in data:
                doc, feature, weight = line.split()
                f.write('\t'.join(['hasWord', 'd' + doc, 'w' + feature]) + '\n')


def run_proppr(data_set):
    commands = {
        'compile': 'cd {0} && sh scripts/compile.sh {1}'.format(
            PROPPR_DIR, PROPPR_PROGRAM_DIR),
        'train': '''
            java -cp {0}/bin:{0}/lib/*:{0}/conf/ edu.cmu.ml.praprolog.Trainer \
            --programFiles {1}/textcat.crules:{1}/{2}.cfacts:{1}/{2}.graph \
            --train {1}/{2}.data --output {1}/{2}.cooked --params {1}/{2}.wts \
            --prover dpr:0.0001:0.01:boost --threads 8
        '''.format(PROPPR_DIR, PROPPR_PROGRAM_DIR, data_set)
    }
    generate_facts(data_set)
    subprocess.call(commands['compile'], shell=True)
    subprocess.call(commands['train'], shell=True)

