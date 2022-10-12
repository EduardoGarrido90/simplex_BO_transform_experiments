import math
import numpy as np
from synthetic_problem import  Synthetic_problem
import sys

problem = None

def main(seed, params):

    global problem

    if problem is None:
        problem = Synthetic_problem(seed)

    #problem.plot()
    result = problem.f(params)
    return str(result['o1'])

def plot_objective_function(seed, l_bound, h_bound):
    global problem

    if problem is None:
        problem = Synthetic_problem(seed)

    problem.plot(l_bound, h_bound)

def wrapper(seed, x, y):
    return main(seed, {'X': x, 'Y': y}) 

def initiate(seed):
    global problem

    if problem is None:
        problem = Synthetic_problem(seed)

    problem.sleep_until_call()

def get_optimum(seed):
    global problem

    if problem is None:
        problem = Synthetic_problem(seed)

    problem.get_optimum(seed)

def get_worst(seed):
    global problem

    if problem is None:
        problem = Synthetic_problem(seed)

    problem.get_worst(seed)

if __name__ == '__main__':
    seed = int(sys.argv[1])
    if len(sys.argv)==2:
        initiate(seed)
    elif sys.argv[2]==1:
        get_optimum(seed)
    else:
        get_worst(seed)
#    x = float(sys.argv[2])
#    y = float(sys.argv[3])
#    wrapper(seed, x, y)
