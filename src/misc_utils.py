import numpy as np
from copy import deepcopy

class PermutationHandler():
    def __init__(self, n, k=None, random_seed=0): #n choose k
        if k is None: k = n
        self.n = n
        self.k = k
        self.n_perms = np.math.factorial(n)//np.math.factorial(n-k)
        self.random_seed = random_seed
        self.rng = np.random.RandomState(seed=random_seed)

    def generate_random_permutation(self):
        n, k = self.n, self.k
        item = []
        a = list(range(n))
        for i in range(k):
            entry_idx = self.rng.randint(n-i)
            item.append(a[entry_idx])
            del a[entry_idx]
        return tuple(item)

    def get_generator(self, count=None):
        if count is None:
            for idx in range(self.n_perms):
                yield self.index_to_permutation(idx)
        else:
            for idx in range(count):
                yield self.generate_random_permutation()

    def index_to_permutation(self, idx):
        assert idx < self.n_perms, "Index must be less than {}".format(self.n_perms)
        n_perms = self.n_perms
        n, k = self.n, self.k
        a = list(range(n))
        item = []
        for i in range(k):
            stride = n_perms // (n-i)
            entry_index = int(idx // stride)
            idx = idx - entry_index * stride
            item.append(a[entry_index])
            del a[entry_index]
            n_perms = n_perms//(n-i)
        return tuple(item)

    def permutation_to_index(self, item):
        n_perms = self.n_perms
        n, k = self.n, self.k
        a = list(range(n))
        idx = 0
        for i in range(len(item)):
            idx = idx + n_perms//(n-i) * a.index(item[i])
            a.remove(item[i])
            n_perms = n_perms//(n-i)
        return int(idx)

    def mutate(self, item, pos=None):
        item = list(item)
        n, k = self.n, self.k
        if pos is None: pos = self.rng.randint(k)
        perturbation = self.rng.randint(n-1) + 1
        existing_val = item[pos]
        new_val = (existing_val + perturbation) % n
        try:
            new_val_idx = item.index(new_val)
        except:
            new_val_idx = None
        if new_val_idx is not None:
            item[new_val_idx] = existing_val
        item[pos] = new_val
        return tuple(item)
   
    def crossover(self, item1, item2):
        item1, item2 = list(item1), list(item2)
        n, k = self.n, self.k
        pos = 1 + self.rng.randint(k-1)
        citem1 = item1[:pos] + [entry for entry in item2 if entry not in item1[:pos]][-k+pos:]
        citem2 = item2[:pos] + [entry for entry in item1 if entry not in item2[:pos]][-k+pos:]
        citem3 = [entry for entry in item2 if entry not in item1[pos:]][:pos] + item1[pos:]
        citem4 = [entry for entry in item1 if entry not in item2[pos:]][:pos] + item2[pos:]
        return tuple(citem1), tuple(citem2), tuple(citem3), tuple(citem4)
