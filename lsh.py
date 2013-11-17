import random
import sys
import hashlib

class LSHasher(object):
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
        for doc, fp_array in self.dot_products.items():
            signatures[doc] = ''.join(['0' if f <= 0 else '1' for f in fp_array])
        return signatures

def test_LSHasher():
    test_map = {
        'foo': [1,2,3,4],
        'bar': [18,17,16,15],
        'qux': [1,2,3,7]
    }
    hasherA = LSHasher(3)
    hasherB = LSHasher(5)

    """
    Note that hasherA and hasherB will produce different hashes because each
    has a different seed and a different set of random values drawn from
    Normal(0,1).
    """

    hasherA.compute_stream(test_map)
    hasherB.compute_stream(test_map)

    print 'Hasher A:', hasherA.compute_signatures()
    print 'Hasher B:', hasherB.compute_signatures()

if __name__ == '__main__':
    test_LSHasher()