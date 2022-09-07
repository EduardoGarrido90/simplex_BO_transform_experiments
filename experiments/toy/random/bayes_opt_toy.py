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

#Naive BO.
#TODO TASK 1: Create the loop. DONE! 
#TODO TASK 2: Plot the results. DONE! (Extend it to several experiments)
#TODO TASK 3: Modularize the code. DONE! 
#TODO TASK 4: Improve the objective function.
#TODO TASK 5: Extend it to several experiments with different seed.


def ci(y, n_exps): #Confidence interval.
    return 1.96 * y.std(axis=1) / np.sqrt(n_exps)

def obj_fun(X_train): #Objective function.
    return torch.tensor([np.sin(x[0])/(np.cos(x[1]) + np.sinh(x[2])) for x in X_train])

def plot_results(n_iters, results):
    X_plot = np.linspace(1, n_iters, n_iters)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.errorbar(
        X_plot, results.mean(axis=1), yerr=ci(results, results.shape[1]), label="Vanilla BO", linewidth=1.5, capsize=3, alpha=0.6
    )
    #ax.errorbar(
    #    iters, y_ei.mean(axis=0), yerr=ci(y_ei), label="NEI", linewidth=1.5, capsize=3, alpha=0.6,
    #)
    #ax.errorbar(
    #    iters, y_ei_warp.mean(axis=0), yerr=ci(y_ei_warp), label="NEI + Input Warping",
    #    linewidth=1.5, capsize=3, alpha=0.6,
    #)
    #ax.set(xlabel='number of observations (beyond initial points)', ylabel='Log10 Regret')
    ax.set(xlabel='number of observations', ylabel='Objective function')
    ax.legend(loc="lower right")
    plt.title('Bayesian optimization results of the different methods')
    plt.show()

    '''
    plt.plot(X_plot, results)
    plt.title('Bayesian optimization results')
    plt.xlabel('Number of iterations')
    plt.ylabel('Objective function')
    plt.show()
    '''

def perform_BO_iteration(X, Y):
    gp = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)


    UCB = UpperConfidenceBound(gp, beta=0.1)


    bounds = torch.stack([torch.zeros(X.shape[1]), torch.ones(X.shape[1])])
    new_X, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
    )

    new_y = obj_fun(new_X)
    X = torch.cat((X, new_X),0)
    Y = torch.cat((Y, new_y.reshape(1,1)),0)
    return X, Y

def perform_BO_experiment(seed, initial_design_size, budget):
    random.seed(seed)
    torch.random.manual_seed(seed)
    n_dims = 3
    X = torch.rand(initial_design_size, n_dims)
    Y = (obj_fun(X) + torch.rand(initial_design_size)/10.0).reshape((initial_design_size), 1) #Adding a little bit of noise.
    counter = 0

    while counter < budget:
        X, Y = perform_BO_iteration(X, Y)
        counter = counter + 1

    return Y

if __name__ == '__main__' :
    total_exps = 3
    initial_design_size = 20
    budget = 35
    total_its = initial_design_size + budget
    results = torch.ones((total_its, total_exps))
    counter = 0
    while counter < total_exps:
        results[:, counter] = perform_BO_experiment(counter, initial_design_size, budget).reshape((total_its))
        counter = counter + 1
        print(counter)
    plot_results(initial_design_size+budget, results)
