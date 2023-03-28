import numpy as np
from scipy.sparse import *
import torch
import torch.utils.data

class HCDataset(torch.utils.data.Dataset):
    """Hierarchical clustering dataset"""

    def __init__(self, obj, n_sample, rng, sample_local=.5, sample_decay=-1,\
                 adj_ratio_per=100, debug = 0):
        """Hierarchical Clustering dataset with a triple sampler"""
        self.obj = obj
        self.n_triples = 0
        self.total_triples = n_sample
        self.rng = rng
        self.p_local = sample_local
        self.kappa = sample_decay
        self.adj_ratio = adj_ratio_per
        if self.kappa < 1 and self.p_local > 0:
            self.kappa = np.exp(-np.log(self.p_local)/(self.total_triples//self.adj_ratio))
        self.debug = debug

    def __len__(self):
        return self.total_triples

    def __getitem__(self, idx):
        if self.p_local <= 0:
            return self.obj.sample_triplet_random(self.rng)
        else:
            if idx % self.adj_ratio == 0:
                if self.debug:
                    print(torch.utils.data.get_worker_info().id, idx, self.p_local, '\n')
                self.p_local *= self.kappa
                self.p_local = np.clip(self.p_local, 0, 1)
            coin = self.rng.random()
            if coin > self.p_local:
                return self.obj.sample_triplet_random(self.rng)
            else:
                return self.obj.sample_triplet_neighbor(self.rng)


# class HCDataset2(torch.utils.data.IterableDataset):
#     """Hierarchical clustering dataset"""

#     def __init__(self, obj, n_sample, rng,\
#                  sample_local=.5, sample_decay=-1, adj_ratio_per=100):
#         """Hierarchical Clustering dataset with a triple sampler"""
#         self.obj = obj
#         self.n_triples = 0
#         self.total_triples = n_sample
#         self.rng = rng
#         self.p_local = sample_local
#         self.kappa = sample_decay
#         self.adj_ratio = adj_ratio_per
#         if self.kappa < 1 and self.p_local > 0:
#             self.kappa = np.exp(-np.log(self.p_local)/(self.total_triples//self.adj_ratio))

#     def __iter__(self):
#         return self.samples_triple()

#     def samples_triple(self):
#         while self.n_triples < self.total_triples:
#             self.n_triples += 1
#             if self.p_local <= 0:
#                 yield self.obj.sample_triplet_random(self.rng)
#             else:
#                 if self.n_triples % self.adj_ratio == 0:
#                     self.p_local *= self.kappa
#                     self.p_local = np.clip(self.p_local, 0, 1)
#                 coin = self.rng.random()
#                 if coin > self.p_local:
#                     yield self.obj.sample_triplet_random(self.rng)
#                 else:
#                     yield self.obj.sample_triplet_neighbor(self.rng)
