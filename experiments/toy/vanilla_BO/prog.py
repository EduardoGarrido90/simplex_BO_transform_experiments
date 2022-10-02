import math
import numpy as np
from synthetic_problem import  Synthetic_problem

problem = None
NUM_EXP = 1

def main(job_id, params):

    global problem

    if problem is None:
        problem = Synthetic_problem(NUM_EXP)

    import pdb; pdb.set_trace();
    problem.plot()
    return problem.f_noisy(params)

if __name__ == '__main__':
    main(1,{'X':0.1, 'Y':0.2}) 
