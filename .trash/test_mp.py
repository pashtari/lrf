import numpy as np
from matplotlib.path import Path
from joblib import Parallel, delayed
import time
import sys

## Check if one line segment contains another. 

def check_paths(path):
    for other_path in a:
        res = 'no cross'
        chck = Path(other_path)
        if chck.contains_path(path):
            res = 'cross'
            break
    return res

if __name__ == '__main__':
    ## Create pairs of points for line segments
    a = list(zip(np.random.rand(5000, 2), np.random.rand(5000, 2)))
    b = list(zip(np.random.rand(300, 2), np.random.rand(300, 2)))

    now = time.time()
    if len(sys.argv) >= 2:
        res = Parallel(n_jobs=int(sys.argv[1]))(delayed(check_paths)(Path(points)) for points in b)
    else:
        res = [check_paths(Path(points)) for points in b]
    print("Finished in", time.time() - now, "sec")
