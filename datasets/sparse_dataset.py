import numpy as np
import networkx as nx
import sklearn.metrics
from scipy.sparse import *
from pynndescent import NNDescent
from networkx.algorithms.community import k_clique_communities
import torch
import torch.utils.data

class SparsePairwise:
    '''
    Construct sufficient information to approximate pairwise distances
    Input: sparse matrix with feature x sample
    '''

    def __init__(self, mtx, n_thread=1,\
                 clique_size=-1, max_community_size=1000):
        self.mtx = csr_array(mtx)
        self.mtx.eliminate_zeros()
        self.M, self.N = mtx.shape
        self.nnz_rowsum = (self.mtx != 0).sum(axis = 1)
        self.n_thread = n_thread
        self.clique_size = clique_size
        self.max_community_size = max_community_size

    def knn_init(self, metric="manhattan", k=100, k_init=30, verbose=True):
        nnd_index = NNDescent(self.mtx.T, metric=metric, n_jobs=self.n_thread,\
                              n_neighbors=k_init, low_memory=True,\
                              compressed=True, parallel_batch_queries=True,\
                              verbose=verbose)
        knn_indx, knn_dist = nnd_index.query(self.mtx.T, k=k+1)
        self.knn_precomputed(knn_indx, knn_dist)

    def knn_precomputed(self, knn_indx, knn_dist):
        assert knn_indx.shape[0] == self.N, "Invalid dimension of knn_indx"
        assert knn_indx.shape == knn_dist.shape, "Inconsistent input"
        self.K = knn_indx.shape[1] - 1

        knn_dist = np.clip(knn_dist, 1, np.ma.masked_invalid(knn_dist).max())
        self.knn_indx = [[knn_indx[i, j] for j in range(self.K+1) if knn_indx[i, j] != i] for i in range(self.N)]
        self.knn = [ {knn_indx[i, j]: knn_dist[i, j] for j in range(self.K+1)} for i in range(self.N) ]
        self.knn_max_dist = knn_dist.max(axis = 1)
        self.long_range_approx(knn_indx)

    def long_range_approx(self, knn_indx):
        k = self.clique_size
        if k > knn_indx.shape[1] - 1 or k < 0:
            k = knn_dist.shape[1] - 1
        # KNN graph
        G = nx.Graph()
        G.add_nodes_from(range(self.N))
        G.add_edges_from([(i, knn_indx[i, j]) for i in range(self.N) for j in range(k)])
        G.remove_edges_from(nx.selfloop_edges(G))
        # Connected components or closely linked communities
        cc = [c for c in nx.connected_components(G) if len(c) > 1]
        min_clique_size = k-1
        communities = []
        for i, v in enumerate(cc):
            if len(v) < self.max_community_size:
                communities.append(list(v))
                continue
            cliq = [c for c in nx.find_cliques(G.subgraph(v)) if len(c) >= min_clique_size]
            if len(cliq) <= 1:
                communities.append(list(v))
                continue
            kc = list(k_clique_communities(G.subgraph(v), min_clique_size, cliq ) )
            communities += [ list(x) for x in kc ]
            if i % 20 == 0:
                print(len(v), len(cliq), len(kc), sum([len(x) for x in kc]) )
        # Approximate community center (~ancestral sequence)
        community_centers = csc_array((self.mtx.shape[0],  0))
        self.mtx = self.mtx.tocsc()
        for i, v in enumerate(communities):
            u = (self.mtx[:, v] != 0).sum(axis = 1)
            private = (u == self.nnz_rowsum) & (u < len(v))
            u = self.mtx[:, v].mean(axis = 1)
            u[private] = 0
            community_centers = hstack([community_centers, csr_array(u).T ])
        community_centers.eliminate_zeros()
        # Map leaf nodes to centers
        center_index = NNDescent(community_centers.T, metric="manhattan",\
                        n_neighbors=np.min([30, len(communities)-1]),\
                        low_memory=True, n_jobs=self.n_thread,\
                        compressed=True, parallel_batch_queries=True,\
                        verbose=False)
        center_i, center_d = center_index.query(self.mtx.T, k=1)
        center_i = center_i.reshape(-1)
        center_d = center_d.reshape(-1)
        self.node_center_map = {i:[center_i[i], center_d[i]] for i in range(self.N) }
        n_center = community_centers.shape[1]
        # Add outlier nodes as centers
        dcut = center_d[center_d.argsort()[int(len(center_d)*.995 ) ]]
        ind_isolated = np.arange(self.N)[center_d > dcut]
        community_centers = hstack([community_centers, self.mtx[:, ind_isolated]])
        self.node_center_map.update({ x:[n_center+i, 0] for i,x in enumerate(ind_isolated) })
        # Pairwise distance between communities
        self.center_pairwise = sklearn.metrics.pairwise_distances(community_centers.T, metric = "manhattan")
        self.n_center = self.center_pairwise.shape[0]

    def sample_triplet_neighbor(self, rng):
        i1 = rng.integers(low=0, high=self.N)
        i2 = rng.choice(self.knn_indx[i1])
        i3 = rng.choice(self.knn_indx[i2])
        while i1 == i3:
            i3 = rng.choice(self.knn_indx[i2])
        s12 = self.knn[i1][i2]
        s23 = self.knn[i2][i3]
        if i3 in self.knn[i1]:
            s13 = self.knn[i1][i3]
        elif i1 in self.knn[i3]:
            s13 = self.knn[i3][i1]
        else:
            c1, d1 = self.node_center_map[i1]
            c3, d3 = self.node_center_map[i3]
            if c1 != c3:
                s13 = d1 + d3 + self.center_pairwise[c1, c3]
            else:
                s13 = np.max([s12, s23, self.knn_max_dist[i1]+1, self.knn_max_dist[i3]+1])
        tri_i = np.array([i1, i2, i3], dtype=int)
        tri_w = 1./np.array([s12, s13, s23])
        return torch.from_numpy(tri_i), torch.from_numpy(tri_w)

    def sample_triplet_random(self, rng):
        i1, i3 = rng.integers(low=0, high=self.N, size=2)
        i2 = rng.choice(self.knn_indx[i1])
        while i1 == i3 or i2 == i3:
            i1, i3 = rng.integers(low=0, high=self.N, size=2)
            i2 = rng.choice(self.knn_indx[i1])
        s12 = self.knn[i1][i2]
        c1, d1 = self.node_center_map[i1]
        c2, d2 = self.node_center_map[i2]
        c3, d3 = self.node_center_map[i3]
        if c2 != c3:
            s23 = d2 + d3 + self.center_pairwise[c2, c3]
        elif i3 in self.knn[i2]:
            s23 = self.knn[i2][i3]
        elif i2 in self.knn[i3]:
            s23 = self.knn[i3][i2]
        else:
            s23 = np.max([self.knn_max_dist[i2]+1, self.knn_max_dist[i3]+1])
        if c1 != c3:
            s13 = d1 + d3 + self.center_pairwise[c1, c3]
        elif i3 in self.knn[i1]:
            s13 = self.knn[i1][i3]
        elif i1 in self.knn[i3]:
            s13 = self.knn[i3][i1]
        else:
            s13 = np.max([self.knn_max_dist[i1]+1, self.knn_max_dist[i3]+1])
        tri_i = np.array([i1, i2, i3], dtype=int)
        tri_w = 1./np.array([s12, s13, s23])
        return torch.from_numpy(tri_i), torch.from_numpy(tri_w)


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
        if self.kappa < 1:
            self.kappa = np.exp(-np.log(self.p_local)/(self.total_triples//self.adj_ratio))
        self.debug = debug

    def __len__(self):
        return self.total_triples

    def __getitem__(self, idx):
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


class HCDataset2(torch.utils.data.IterableDataset):
    """Hierarchical clustering dataset"""

    def __init__(self, obj, n_sample, rng,\
                 sample_local=.5, sample_decay=-1, adj_ratio_per=100):
        """Hierarchical Clustering dataset with a triple sampler"""
        self.obj = obj
        self.n_triples = 0
        self.total_triples = n_sample
        self.rng = rng
        self.p_local = sample_local
        self.kappa = sample_decay
        self.adj_ratio = adj_ratio_per
        if self.kappa < 1:
            self.kappa = np.exp(-np.log(self.p_local)/(self.total_triples//self.adj_ratio))

    def __iter__(self):
        return self.samples_triple()

    def samples_triple(self):
        while self.n_triples < self.total_triples:
            self.n_triples += 1
            if self.n_triples % self.adj_ratio == 0:
                self.p_local *= self.kappa
                self.p_local = np.clip(self.p_local, 0, 1)
            coin = self.rng.random()
            if coin > self.p_local:
                yield self.obj.sample_triplet_random(self.rng)
            else:
                yield self.obj.sample_triplet_neighbor(self.rng)
