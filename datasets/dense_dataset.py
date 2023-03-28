import numpy as np
from scipy.sparse import *
import torch

class DensePairwise:
    '''
    Input pairwise similarity
    '''

    def __init__(self, mtx, k=100):
        self.mtx = mtx
        self.N = self.mtx.shape[0]
        self.K = k
        if self.K > self.N - 1:
            self.K = self.N - 1
        self.knn = []
        np.fill_diagonal(self.mtx, 0)
        for i in range(self.N):
            self.knn.append( (-self.mtx[i, ]).argsort()[:self.K] )

    def sample_triplet_neighbor(self, rng):
        i1 = rng.integers(low=0, high=self.N)
        i2 = rng.choice(self.knn[i1])
        i3 = rng.choice(self.knn[i2])
        while i3 == i1:
            i3 = rng.choice(self.knn[i2])
        s12 = self.mtx[i1, i2]
        s13 = self.mtx[i1, i3]
        s23 = self.mtx[i2, i3]
        tri_i = np.array([i1, i2, i3], dtype=int)
        tri_w = np.array([s12, s13, s23])
        return torch.from_numpy(tri_i), torch.from_numpy(tri_w)

    def sample_triplet_random(self, rng):
        i1, i2, i3 = rng.choice(np.arange(self.N), size=3, replace=False, shuffle=True)
        s12 = self.mtx[i1, i2]
        s13 = self.mtx[i1, i3]
        s23 = self.mtx[i2, i3]
        tri_i = np.array([i1, i2, i3], dtype=int)
        tri_w = np.array([s12, s13, s23])
        return torch.from_numpy(tri_i), torch.from_numpy(tri_w)
