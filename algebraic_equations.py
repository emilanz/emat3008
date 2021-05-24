import numpy as np

def alg_cubic(x, c):
    x = x[0]
    return np.array([x*x*x - x + c])