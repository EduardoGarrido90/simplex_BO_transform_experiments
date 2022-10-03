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
    result = problem.f_noisy(params)
    return str(result['o1'])

def wrapper(seed, x, y):
    return main(seed, {'X': x, 'Y': y}) 

if __name__ == '__main__':
    seed = int(sys.argv[1])
    x = float(sys.argv[2])
    y = float(sys.argv[3])
    wrapper(seed, x, y)
