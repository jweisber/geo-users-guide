import numpy as np
from mod.community import Community


def score_community(b, a, w, cmn):
    V = np.zeros((a, w))          # start with 0s
    b_col = int(10 * b)           # identify truth columns
    V[:, b_col] = 1               # set those to 1
    score = ((V - cmn.probs) ** 2).sum(axis=1).mean()
    return score

def run_simulation(b, a, w, n, k, c, d):
    cmn = Community(a, w)
    score = score_community(b, a, w, cmn)

    for i in range(n.shape[1]):
        cmn.update(n[:, i], k[:, i], c)
        cmn.pool_geo(d)
        score += score_community(b, a, w, cmn)
    
    return score

def run_grid(a, w, n, t, biases, boosts, distances):
    results = []
    
    for b in biases:
        n = n * np.ones((a, t), dtype=int)
        np.random.seed()
        k = np.random.binomial(n, b)
        
        for c in boosts:
            for d in distances:
                score = run_simulation(b, a, w, n, k, c, d)
                results += [[b, c, d, score]]
    
    return results