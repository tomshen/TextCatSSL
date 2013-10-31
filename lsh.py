import random
import sys

class LSHasher(object):
    """
    Implements streaming locality-sensitive hashing as defined by Van Durme
    and Lall (2010)[http://personal.denison.edu/~lalla/online-lsh.pdf].
    """
    def __init__(self, num_bits, pool_size=100, seed=None):
        self.pool_size = pool_size
        self.num_bits = num_bits
        self.seed = seed or random.randint(1,sys.maxint)
        self.dot_products = {}

        random.seed(seed)
        self.pool = [random.gauss(0,1) for i in xrange(pool_size)]


    def compute_stream(self, doc_features):
        """
        Compute entire stream of documents and features at once. Note that
        this function will work even if doc_features is a generator.
        """
        for doc, features in doc_features.items():
            self.compute_doc(doc, features)

    def compute_doc(self, doc, features):
        """Compute for one doc and associated features"""
        if doc not in self.dot_products:
            self.dot_products[doc] = [0 for i in xrange(self.num_bits)]
        for feature in features:
            for j in xrange(len(self.dot_products[doc])):
                # the following can and should be replaced by a hash function
                # that properly maps (seed, feature) -> 0...(pool_size-1)
                pool_idx = hash(self.seed * j * feature) % self.pool_size
                self.dot_products[doc][j] += self.pool[pool_idx]

    def compute_signatures(self):
        """Generate bit string signatures for each document"""
        signatures = {}
        for doc, fp_array in self.dot_products.items():
            signatures[doc] = [0 if f <= 0 else 1 for f in fp_array]
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