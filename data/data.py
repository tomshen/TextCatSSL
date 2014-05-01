#!/usr/bin/env python
from collections import Counter
from os.path import join
import subprocess
import sys

setup = {
    '20NG': [
        'wget http://qwone.com/~jason/20Newsgroups/20news-bydate-matlab.tgz',
        'tar xf 20news-bydate-matlab.tgz',
        'cp 20news-bydate/matlab/test.map 20NG.map',
        'wget -O 20NG.vocab http://qwone.com/~jason/20Newsgroups/vocabulary.txt'
    ],
    'webkb': [
        'wget http://web.ist.utl.pt/~acardoso/datasets/webkb-train-stemmed.txt',
        'wget http://web.ist.utl.pt/~acardoso/datasets/webkb-test-stemmed.txt'
    ],
    'R8': [
        'wget http://web.ist.utl.pt/~acardoso/datasets/r8-train-all-terms.txt',
        'wget http://web.ist.utl.pt/~acardoso/datasets/r8-test-all-terms.txt'
    ]
}
cleanup = {
    '20NG': [
        'rm -f 20news-bydate-matlab.tgz',
        'rm -rf 20news-bydate'
    ],
    'webkb': [
        'rm -f webkb-train-stemmed.txt',
        'rm -f webkb-test-stemmed.txt'
    ],
    'R8': [
        'rm -f r8-train-all-terms.txt',
        'rm -f r8-test-all-terms.txt'
    ]
}
def process_20NG():
    base = '20news-bydate/matlab'
    def o(fn):
        with open(join(base, fn)) as f:
            return [l for l in f.read().split('\n') if l]
    tsd = [[int(x) for x in l.split()] for l in o('test.data')]
    tsl = {(i+1):int(l) for i, l in enumerate(o('test.label'))}
    trd = [[int(x) for x in l.split()] for l in o('train.data')]
    for l in trd:
        l[0] += len(tsl)
    trl = {(i+len(tsl)+1):int(l) for i, l in enumerate(o('train.label'))}
    tsd += trd
    tsl.update(trl)
    with open('20NG.data', 'w') as f:
        for d, w, c in tsd:
            f.write('%d %d %d\n' % (d, w, c))
    with open('20NG.label', 'w') as f:
        for d, l in tsl.items():
            f.write('%d\n' % l)
    with open('20NG.count', 'w') as f:
        f.write('\n'.join([str(len(tsl)), '61188', '20']))

def process_data(data_set, train_file, test_file):
    def o(fn):
        with open(fn) as f:
            return [(l.split('\t')[0], l.split('\t')[1].split())
                for l in f.read().split('\n') if l]
    tr = o(train_file)
    ts = o(test_file)
    docs = tr + ts
    dct = {w:(i+1) for i, w in enumerate(set(
        [word for _, words in docs for word in words]))}
    mp = {l:(i+1) for i, l in enumerate(set(
        [label for label,_ in docs]))}
    docs = [(mp[l], list(Counter([dct[w] for w in ws]).items()))
        for l, ws in docs]
    with open(data_set + '.data', 'w') as df:
        with open(data_set + '.label', 'w') as dl:
            doc = 1
            for l, words in docs:
                dl.write('%d\n' % l)
                for w, c in words:
                    df.write('%d %d %d\n' % (doc, w, c))
                doc += 1

    with open(data_set + '.vocab', 'w') as f:
        f.write('\n'.join(sorted(dct.keys(), key=dct.get)))

    with open(data_set + '.map', 'w') as f:
        for i, l in sorted((i,l) for l, i in mp.items()):
            f.write('%s %d\n' % (l, i))

    with open(data_set + '.count', 'w') as f:
        f.write('\n'.join([str(len(docs)), str(len(dct)), str(len(mp))]))

if __name__ == '__main__':
    if len(sys.argv) == 0:
        pass
    elif sys.argv[1] == '20NG':
        for cmd in setup['20NG']:
            subprocess.call(cmd.split())
        process_20NG()
        for cmd in cleanup['20NG']:
            subprocess.call(cmd.split())
    elif sys.argv[1] == 'webkb':
        for cmd in setup['webkb']:
            subprocess.call(cmd.split())
        process_data('webkb', 'webkb-train-stemmed.txt',
                'webkb-test-stemmed.txt')
        for cmd in cleanup['webkb']:
            subprocess.call(cmd.split())
    elif sys.argv[1] == 'R8':
        for cmd in setup['R8']:
            subprocess.call(cmd.split())
        process_data('R8', 'r8-train-all-terms.txt',
                'r8-test-all-terms.txt')
        for cmd in cleanup['R8']:
            subprocess.call(cmd.split())
