import math
import numpy as np
from synthetic_problem import  Synthetic_problem
import sys

problem = None

def main(seed, params):

    global problem

    if problem is None:
        problem = Synthetic_problem(seed)

    problem.plot()
    return problem.f_noisy(params)

if __name__ == '__main__':
    seed = int(sys.argv[1])
    x = float(sys.argv[2])
    y = float(sys.argv[3])
    main(seed, {'X': x, 'Y': y}) 
