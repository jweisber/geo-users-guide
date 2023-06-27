from scipy.stats import betabinom
from sklearn.metrics.pairwise import manhattan_distances
import numpy as np

class Community:

    def __init__(self, a, w):
        self.probs   = np.ones((a, w)) / w
        self.n_last  = np.zeros((a, 1))
        self.k_last  = np.zeros((a, 1))
        self.n_total = np.zeros((a, 1))
        self.k_total = np.zeros((a, 1))

    def update(self, n, k, c):
        n, k = n.reshape(n.size, 1), k.reshape(k.size, 1)
        self.n_last, self.k_last = n, k
        self.n_total += n
        self.k_total += k

        w = self.probs.shape[1]
        p = np.arange(w) / (w - 1)
        p = p.reshape(1, p.size)
        likelihood = (p ** k) * ((1 - p) ** (n - k))

        #posteriors = self.probs * likelihood / (self.probs * likelihood).sum(axis=1)[:, np.newaxis]
        posteriors = self.probs * likelihood

        freq = self.k_total / self.n_total
        tot_likelihood = (p ** freq) * ((1 - p) ** (1 - freq))
        posteriors += c * (tot_likelihood == tot_likelihood.max(axis=1)[:, np.newaxis])

        #self.probs = posteriors / (1.0 + c)
        self.probs = posteriors / posteriors.sum(axis=1)[:, np.newaxis]

    def pool_geo(self, d):
        new_probs = np.zeros(self.probs.shape)
        nbhds = manhattan_distances(self.probs) <= d

        for a, row in enumerate(nbhds):
            probs = self.probs[row].prod(axis=0)
            probs = probs ** (1 / row.sum())
            if probs.sum() > 0:
                new_probs[a] = probs / probs.sum()
            else:
                new_probs[a] = self.probs[a]
        
        self.probs = new_probs

    def mask_weights(self, d):
        mask = (manhattan_distances(self.probs) <= d).astype(int)
        self.weights = mask / mask.sum(axis=1)[:, np.newaxis]                     # normalize and update weights