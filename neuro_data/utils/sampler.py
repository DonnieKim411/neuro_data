from collections import Counter

import torch
import numpy as np
from torch.utils.data.sampler import Sampler
from itertools import chain


class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class RepeatSubsetRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement for a repeated number of times.

    Arguments:
        indices (sequence): a sequence of indices
        repeat (int)
    """

    def __init__(self, indices, repeat=1):
        self.indices = indices
        self.repeat = repeat

    def __iter__(self):
        return chain.from_iterable(
            (self.indices[i] for i in torch.randperm(len(self.indices)))
            for _ in range(self.repeat))

    def __len__(self):
        return len(self.indices) * self.repeat


class RepeatSubsetSequentialSampler(Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement for a repeated number of times.

    Arguments:
        indices (sequence): a sequence of indices
        repeat (int)
    """

    def __init__(self, indices, repeat=1):
        self.indices = indices
        self.repeat = repeat

    def __iter__(self):
        return chain.from_iterable(
            (self.indices[i] for i in range(len(self.indices)))
            for _ in range(self.repeat))

    def __len__(self):
        return len(self.indices) * self.repeat


class SubSubsetRandomSequentialSampler(Sampler):
    """Samples a subset from a dataset randomly. 
    Then iterates through those indices sequentially.

    Arguments:
        indices (list): a list of indices
        subsubset_size (int): size of subset of subset
    """

    def __init__(self, indices, subsubset_size):
        self.indices = indices
        self.subsubset_size = subsubset_size

    def __iter__(self):
        subsubset_indices = np.random.choice(self.indices, size=self.subsubset_size,
                                             replace=False)
        return (subsubset_indices[i] for i in range(len(subsubset_indices)))

    def __len__(self):
        return self.subsubset_size


class BalancedSubsetSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement, balanced by occurence of types.

    Arguments:
        indices (list): a list of indices
    """

    def configure_sampler(self, indices, types, mode='shortest'):
        self.indices = indices
        c = Counter(types[indices])
        if mode == 'longest':
            self.num_samples = max(c.values())
            self.replacement = True
        elif mode == 'shortest':
            self.num_samples = min(c.values())
            self.replacement = False

        for e, n in c.items():
            c[e] = 1 / n
        self.types = types
        self.weights = torch.DoubleTensor([c[types[i]] for i in indices])

    def __init__(self, indices, types, mode='shortest'):
        self.configure_sampler(indices, types, mode)

    def __iter__(self):
        selection = torch.multinomial(
            self.weights, self.num_samples, self.replacement)
        # print(Counter(self.types[self.indices[i]] for i in selection), len(selection)))
        return (self.indices[i] for i in selection)

    def __len__(self):
        return self.num_samples
