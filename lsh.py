import multiprocessing
import random
import string
import sys

def _LSHasher_compute_stream(h):
    h[0][1].compute_stream(h[1])
    return h[0]

def _LSHasher_compute_signatures(h):
    return h[0], h[1].compute_signatures()

class MultiLSHasher:
    def __init__(self, num_hashes, num_bits):
        self.num_hashes = num_hashes
        self.hashers = [(string.ascii_lowercase[i], LSHasher(num_bits))
            for i in xrange(num_hashes)]

    def compute_stream(self, doc_features):
        process_pool = multiprocessing.Pool(processes=self.num_hashes)
        self.hashers = process_pool.map(_LSHasher_compute_stream, [
            (self.hashers[i], doc_features)
            for i in xrange(len(self.hashers))])
        process_pool.close()
        process_pool.join()

    def compute_signatures(self):
        process_pool = multiprocessing.Pool(processes=self.num_hashes)
        sigs = dict(process_pool.map(_LSHasher_compute_signatures, self.hashers))
        process_pool.close()
        process_pool.join()
        return sigs

class LSHasher:
    """
    Implements streaming locality-sensitive hashing as defined by Van Durme
    and Lall (2010)[http://personal.denison.edu/~lalla/online-lsh.pdf].
    """
    def __init__(self, num_bits, pool_size=10000, seed=None):
        self.pool_size = pool_size
        self.num_bits = num_bits
        self.dot_products = {}
        if seed:
            self.seed = hash(seed)
            random.seed(self.seed)
        else:
            self.seed = random.randint(1, sys.maxint)
        self.pool = [random.gauss(0,1) for i in xrange(pool_size)]

    def profile_hash(self, iterations=10**6):
        import time
        start = time.clock()
        [self.hash_feature(1000, i) for i in xrange(iterations)]
        return (time.clock() - start) / float(iterations)

    def hash_feature(self, j, f):
        p = 16777619
        h = 1315423911
        h = (h ^ self.seed) * p

        key = str(j) + 'f' + str(f)
        for c in key:
            h = (h ^ ord(c)) * p
        h += h << 13
        h ^= h >> 7
        h += h << 3
        h ^= h >> 17
        h += h << 5
        return h % self.pool_size

    def compute_stream(self, doc_features):
        """
        Compute entire stream of documents and features at once.
        """
        for doc, features in doc_features.items():
            self.compute_doc(doc, features)

    def compute_doc(self, doc, features):
        """Compute for one doc and associated features"""
        if doc not in self.dot_products:
            self.dot_products[doc] = [0 for i in xrange(self.num_bits)]
        for j in xrange(self.num_bits):
            self.dot_products[doc][j] = sum(
                (self.pool[self.hash_feature(j+1, f+1)]*w for f,w in features))

    def compute_signatures(self):
        """Generate bit string signatures for each document"""
        signatures = {}

        # calculating threshold takes 0.6 extra seconds per fp_array
        fs = [f for fp_array in self.dot_products.values() for f in fp_array]
        threshold = sorted(fs)[len(fs)/2]
        for doc, fp_array in self.dot_products.items():
            signatures[doc] = ''.join(['0' if f <= threshold else '1' for f in fp_array])
        return signatures