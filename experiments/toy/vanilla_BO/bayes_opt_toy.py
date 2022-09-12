import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.special import softmax


#TODO 1: Propose new objective function with highest value in the simplex then less.
#TODO 2: Propose to penalize the exit of the obj fun as another baseline.
#TODO 3: Use the transformation in the GP (kernels).
#TODO 4: Use Daniel transformation.
#TODO 5: Experiments.

GLOBAL_MAXIMUM = 1.8

def ci(y, n_exps): #Confidence interval.
    return 1.96 * y.std(axis=1) / np.sqrt(n_exps)

def obj_fun_2(X_train): #Objective function. Needs to be only valid for the diagonal, or at least, more valid there.
    return torch.tensor([np.sin(x[0])/(np.cos(x[1]) + np.sinh(x[2])) + torch.rand(1)/10.0 for x in X_train])

def obj_fun(x): #Length of x: 5. Range [0,1]^5. To be maximized.
    if torch.any(x > 1.0):
        raise Exception("Hypercube violated")

    y = torch.sum(x)
    distance_wrt_simplex = torch.abs(torch.tensor(1.0)-y)
    penalization = distance_wrt_simplex**x.shape[0] 
    return y - penalization

def wrapped_obj_fun(X_train):
    transformed_inputs = (X_train - 0.5) / 0.05
    X_train = torch.exp(transformed_inputs)/torch.sum(torch.exp(transformed_inputs)) #Sums to 1: ps assert(torch.sum(X_train)==1.0)
    return obj_fun(X_train)

def plot_results(n_iters, results):
    X_plot = np.linspace(1, n_iters, n_iters)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.errorbar(
        X_plot, np.log10(GLOBAL_MAXIMUM - results[0].mean(axis=1)), yerr=ci(results[0], results.shape[2]), label="Vanilla BO", linewidth=1.5, capsize=3, alpha=0.6
    )
    ax.errorbar(
        X_plot, np.log10(GLOBAL_MAXIMUM - results[1].mean(axis=1)), yerr=ci(results[1], results.shape[2]), label="Random Search", linewidth=1.5, capsize=3, alpha=0.6,
    )
    ax.errorbar(
        X_plot, np.log10(GLOBAL_MAXIMUM - results[2].mean(axis=1)), yerr=ci(results[2], results.shape[2]), label="Wrapped objective function", linewidth=1.5, capsize=3, alpha=0.6,
    )
    #ax.set(xlabel='number of observations (beyond initial points)', ylabel='Log10 Regret')
    ax.set(xlabel='Number of observations', ylabel='Log10 Regret')
    ax.legend(loc="lower left")
    plt.title('Bayesian optimization results of the different methods')
    plt.show()

def get_initial_results(initial_design_size):
    n_dims = 3
    X = torch.rand(initial_design_size, n_dims)
    Y = torch.tensor([obj_fun(x) for x in X]).reshape(X.shape[0], 1)
    return X, Y

def perform_BO_iteration(X, Y, wrapped=False):
    gp = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)

    UCB = UpperConfidenceBound(gp, beta=0.1)


    bounds = torch.stack([torch.zeros(X.shape[1]), torch.ones(X.shape[1])])
    new_X, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
    )

    if wrapped:
        new_y = wrapped_obj_fun(new_X)
    else:
        new_y = obj_fun(new_X)

    X = torch.cat((X, new_X),0)
    Y = torch.cat((Y, new_y.reshape(1,1)),0)
    return X, Y

def perform_random_iteration(X, Y):
    new_X = torch.rand(1, X.shape[1])
    new_y = obj_fun(new_X)
    X = torch.cat((X, new_X),0)
    Y = torch.cat((Y, new_y.reshape(1,1)),0)
    return X, Y

def perform_wrapper_rounding_experiment(seed : int, initial_design_size: int, budget: int) -> torch.Tensor:
    random.seed(seed)
    torch.random.manual_seed(seed)
    X, Y = get_initial_results(initial_design_size)

    for i in range(budget):
        X, Y = perform_BO_iteration(X, Y, wrapped=True)

    return Y

def perform_BO_experiment(seed : int, initial_design_size: int, budget: int) -> torch.Tensor:
    random.seed(seed)
    torch.random.manual_seed(seed)
    X, Y = get_initial_results(initial_design_size)

    for i in range(budget):
        X, Y = perform_BO_iteration(X, Y)

    return Y

def perform_random_experiment(seed, initial_design_size, budget) -> torch.Tensor:
    random.seed(seed)
    torch.random.manual_seed(seed)
    X, Y = get_initial_results(initial_design_size)

    for i in range(budget):
        X, Y = perform_random_iteration(X, Y)

    return Y

if __name__ == '__main__' :
    total_exps = 5
    initial_design_size = 5
    budget = 10
    n_methods = 3
    total_its = initial_design_size + budget
    results = torch.ones((n_methods, total_its, total_exps))
    for exp in range(total_exps):
        results[0, :, exp] = perform_BO_experiment(exp, initial_design_size, budget).reshape((total_its))
        results[1, :, exp] = perform_random_experiment(exp, initial_design_size, budget).reshape((total_its))
        results[2, :, exp] = perform_wrapper_rounding_experiment(exp, initial_design_size, budget).reshape((total_its))
        print(exp)
    plot_results(initial_design_size+budget, results)
