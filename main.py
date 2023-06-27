import numpy as np
import pandas as pd
from itertools import chain
from multiprocessing import Pool
from mod import run_grid

if __name__ == '__main__':
    a, w, n, t      = 50, 11, 1, 500    # a agents, w worlds (biases), n flips per time-step, t time-steps
    biases          = np.linspace(0, .5, 6)
    distances       = np.linspace(0, 2, 21)
    boosts          = np.linspace(0, 1, 11)
    args = 50 * ((a, w, n, t, biases, boosts, distances),)    # run 50 simulations for each combination of parameters

    pool = Pool()
    results = pool.starmap(run_grid, args)
    pool.close()
    pool.join()

    results = list(chain(*results))
    df = pd.DataFrame(results)
    df.columns = ['bias', 'boost', 'dist', 'score']
    df.to_csv("data/50-agents.csv", index=False)
