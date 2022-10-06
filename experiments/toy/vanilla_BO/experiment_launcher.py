import torch
import sobol
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.models.transforms.input import Simplex, Normalize, ChainedInputTransform
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib.pyplot as plt
import numpy as np
import random
import execnet
from scipy.special import softmax

#TAREAS DEL SIMPLEX:
#DONE 0.a: Hacer el random 2D.
#DONE 0.b: Hacer el random 3D.
#1. Visualización del simplex.
#2. Extraccion del peor punto del simplex.
#3. Para penalizacion:
#3.1 Hallar si el punto pertenece al simplex, sino, coger el peor punto y penalizar linealmente por distancia de la proyeccion de forma suave.
#3.2 Si pertenece, entonces hacer la transformación inversa y ya esta.
#4 Hacer una visualizacion exhaustiva de la funcion objetivo en 2D y de su representacion en el simplex para enseñar a Daniel que [0,1]^2 -> No es exhaustivo el simplex. Esto se arregla con -0.5/0.05.
#5 Arreglar con -0.5/0.05 que sería una escala de [0,1]^2 a R^2, luego de R^2 a S^3 tienes el simplex de ese punto. 
#6 Comparar todos los baselines. 
GLOBAL_MAXIMUM = 1000

def call_python_version(Version, Module, Function, ArgumentList):
    gw      = execnet.makegateway("popen//python=python%s" % Version)
    channel = gw.remote_exec("""
        from %s import %s as the_function
        channel.send(the_function(*channel.receive()))
    """ % (Module, Function))
    channel.send(ArgumentList)
    return channel.receive()

def ci(y, n_exps): #Confidence interval.
    return 1.96 * y.std(axis=1) / np.sqrt(n_exps)

def objective_function(x, seed, to_simplex=False, penalize=False):
    if(to_simplex):
        x = inverse_biyective_transformation(simplex_transformation(x))
    print("Evaluating objective function X=", str(x))
    y = torch.tensor(float(call_python_version("2.7", "prog", "wrapper", [seed, float(x[0]), float(x[1])])))
    if(penalize):
        y = penalization_approach(x, y)
    print("Objective function evaluated. Y=", str(y))
    return y

def simplex_transformation(x):
    transformed_inputs = (x - 0.5) / 0.05
    return torch.exp(transformed_inputs)/torch.sum(torch.exp(transformed_inputs)) #Sums to 1: ps assert(torch.sum(X_train)==1.0)

def penalization_approach(x, y):
    sum_values = torch.sum(x)
    penalization = torch.abs(torch.tensor(1.0)-sum_values)*10
    return y+penalization

def plot_results_log10_regret(n_iters, results):
    X_plot = np.linspace(1, n_iters, n_iters)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.errorbar(
        X_plot, np.log10(GLOBAL_MAXIMUM - results[0].mean(axis=1)), yerr=0.1*ci(results[0], results.shape[2]), label="Biyective Transformation", linewidth=1.5, capsize=3, alpha=0.6
    )
    ax.errorbar(
        X_plot, np.log10(GLOBAL_MAXIMUM - results[1].mean(axis=1)), yerr=0.1*ci(results[1], results.shape[2]), label="Simplex Transformation", linewidth=1.5, capsize=3, alpha=0.6,
    )
    ax.errorbar(
        X_plot, np.log10(GLOBAL_MAXIMUM - results[2].mean(axis=1)), yerr=0.1*ci(results[2], results.shape[2]), label="Penalization Approach", linewidth=1.5, capsize=3, alpha=0.6,
    )
    ax.errorbar(
        X_plot, np.log10(GLOBAL_MAXIMUM - results[3].mean(axis=1)), yerr=0.1*ci(results[3], results.shape[2]), label="Random Search BT", linewidth=1.5, capsize=3, alpha=0.6,
    )
    ax.errorbar(
        X_plot, np.log10(GLOBAL_MAXIMUM - results[4].mean(axis=1)), yerr=0.1*ci(results[4], results.shape[2]), label="Random Search ST", linewidth=1.5, capsize=3, alpha=0.6,
    )
    #ax.set(xlabel='number of observations (beyond initial points)', ylabel='Log10 Regret')
    ax.set(xlabel='Number of observations', ylabel='Log10 Regret')
    ax.legend(loc="lower left")
    plt.title('Bayesian optimization results of the different methods')
    plt.show()

def get_best_results_list(y):
    best = y[0]
    counter = 0
    for y_i in y:
        if(y_i<best):
            best = y_i
        y[counter] = best
        counter = counter + 1
    return y

def plot_results_log10_regret_acum(n_iters, results):
    X_plot = np.linspace(1, n_iters, n_iters)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    y_0 = np.log10(GLOBAL_MAXIMUM - results[0].mean(axis=1))
    y_1 = np.log10(GLOBAL_MAXIMUM - results[1].mean(axis=1))
    y_2 = np.log10(GLOBAL_MAXIMUM - results[2].mean(axis=1))
    y_3 = np.log10(GLOBAL_MAXIMUM - results[3].mean(axis=1))
    y_4 = np.log10(GLOBAL_MAXIMUM - results[4].mean(axis=1))

    ax.errorbar(
        X_plot, get_best_results_list(y_0), yerr=0.1*ci(results[0], results.shape[2]), label="Biyective Transformation", linewidth=1.5, capsize=3, alpha=0.6
    )
    ax.errorbar(
        X_plot, get_best_results_list(y_1), yerr=0.1*ci(results[1], results.shape[2]), label="Simplex Transformation", linewidth=1.5, capsize=3, alpha=0.6,
    )
    ax.errorbar(
        X_plot, get_best_results_list(y_2), yerr=0.1*ci(results[2], results.shape[2]), label="Penalization Approach", linewidth=1.5, capsize=3, alpha=0.6,
    )
    ax.errorbar(
        X_plot, get_best_results_list(y_3), yerr=0.1*ci(results[3], results.shape[2]), label="Random Search BT", linewidth=1.5, capsize=3, alpha=0.6,
    )
    ax.errorbar(
        X_plot, get_best_results_list(y_4), yerr=0.1*ci(results[4], results.shape[2]), label="Random Search ST", linewidth=1.5, capsize=3, alpha=0.6,
    )
    #ax.set(xlabel='number of observations (beyond initial points)', ylabel='Log10 Regret')
    ax.set(xlabel='Number of observations', ylabel='Best observed Log10 Regret')
    ax.legend(loc="lower left")
    plt.title('Bayesian optimization results of the different methods')
    plt.show()

def get_initial_results(initial_design_size, seed, dims, simplex_transformation=False, penalizing_approach=False):
    X = torch.rand(initial_design_size, dims)
    Y = torch.tensor([objective_function(x, seed, to_simplex=simplex_transformation, penalize=penalizing_approach) for x in X]).reshape(X.shape[0], 1)
    return X, Y

def meshgrid_to_2d_grid(X, Y):
    final_piece = torch.vstack((X[0,0].repeat(len(X[0])),Y[0])).T
    for i in range(len(X[0])-1):
        final_piece = torch.cat((final_piece,torch.vstack((X[i+1,0].repeat(len(X[0])),Y[0])).T))
    return final_piece

def plot_acq_fun_model_posterior(acq_fun, obs_input, model, iteration, method_name):
    grid_x = torch.linspace(0.0, 1.0, 100)
    grid_y = torch.linspace(0.0, 1.0, 100)
    X, Y = torch.meshgrid(grid_x, grid_y)
    grid = meshgrid_to_2d_grid(X, Y)
    acq_fun_grid = acq_fun.forward(grid.reshape((grid.shape[0],1,grid.shape[1]))).detach()
    posterior_grid = model.posterior(grid).mean[:,0].detach()
    #function_grid = torch.sum(grid**2.0, axis=1)
    function_grid = torch.tensor([objective_function(x, name_obj_fun, bounds) for x in grid])
    
    fig,ax=plt.subplots(1,1)
    grid_dim = len(grid_x)
    cp = ax.contourf(grid[:,0].reshape(grid_dim, grid_dim), grid[:,1].reshape(grid_dim, grid_dim), function_grid.reshape(grid_dim, grid_dim))
    fig.colorbar(cp) # Add a colorbar to a plot
    ax.set_title('Objective function. Iteration ' + str(iteration) + '. ' + method_name)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.scatter(obs_input[:,0], obs_input[:,1], color="black", marker="X")
    plt.scatter(obs_input[len(obs_input)-1,0], obs_input[len(obs_input)-1,1], color="red", marker="X")
    plt.savefig('./images/obj_fun_' + str(iteration) + '_' + method_name + '.png')
    #plt.show()
    plt.clf()
    plt.close()

    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(grid[:,0].reshape(grid_dim, grid_dim), grid[:,1].reshape(grid_dim, grid_dim), posterior_grid.reshape(grid_dim, grid_dim))
    ax.set_title('Mean model posterior. Iteration ' + str(iteration) + '. ' + method_name)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.scatter(obs_input[:,0], obs_input[:,1], color="black", marker="X")
    plt.scatter(obs_input[len(obs_input)-1,0], obs_input[len(obs_input)-1,1], color="red", marker="X")
    plt.savefig('./images/mean_model_posterior_' + str(iteration) + '_' + method_name + '.png')
    #plt.show()
    plt.clf()
    plt.close()
    
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(grid[:,0].reshape(grid_dim, grid_dim), grid[:,1].reshape(grid_dim, grid_dim), acq_fun_grid.reshape(grid_dim, grid_dim))
    fig.colorbar(cp) # Add a colorbar to a plot
    ax.set_title('Acquisition function. Iteration ' + str(iteration) + '. ' + method_name)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.scatter(obs_input[:,0], obs_input[:,1], color="black", marker="X")
    plt.scatter(obs_input[len(obs_input)-1,0], obs_input[len(obs_input)-1,1], color="red", marker="X")
    plt.savefig('./images/acq_fun_' + str(iteration) + '_' + method_name + '.png')
    #plt.show()
    plt.clf()
    plt.close()

def perform_BO_iteration(X, Y, seed, method_name, apply_simplex=False, apply_penalization=False, plot_acq_model=False):

    gp = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    
    try:
        fit_gpytorch_model(mll)
    except: #Numerical issues.
        import gpytorch
        gpytorch.settings.cholesky_jitter._global_float_value = 1e-02 #Ill-conditioned matrix, adding more jitter to diagonal for cholesky.
        fit_gpytorch_model(mll)
        gpytorch.settings.cholesky_jitter._global_float_value = 1e-06 #Restoring.
    
    UCB = UpperConfidenceBound(gp, beta=0.3, maximize=False)
    bounds_cube = torch.stack([torch.zeros(X.shape[1]), torch.ones(X.shape[1])])
    if plot_acq_model:
        plot_acq_fun_model_posterior(UCB, X, gp, bounds, name_obj_fun, seed+1, method_name)
    new_X, acq_value = optimize_acqf(
            UCB, bounds=bounds_cube, q=1, num_restarts=5, raw_samples=20,
    )
    if apply_simplex:
        new_y = objective_function(new_X[0], seed, to_simplex=True)
    elif apply_penalization:
        new_y = objective_function(new_X[0], seed, penalize=True)
    else:
        new_y = objective_function(new_X[0], seed)
    X = torch.cat((X, new_X),0)
    Y = torch.cat((Y, new_y.reshape(1,1)),0)
    return X, Y

def perform_random_iteration(X, Y, seed, method_name, apply_simplex=False):
    new_X = torch.rand(1, X.shape[1])
    if apply_simplex:
        new_y = objective_function(new_X[0], seed, to_simplex=True)
    else:
        new_y = objective_function(new_X[0], seed)
    X = torch.cat((X, new_X),0)
    Y = torch.cat((Y, new_y.reshape(1,1)),0)
    return X, Y

def perform_simplex_transformation_experiment(seed : int, initial_design_size: int, budget: int, name_obj_fun : str, bounds) -> torch.Tensor:
    random.seed(seed)
    torch.random.manual_seed(seed)
    X, Y = get_initial_results(initial_design_size, name_obj_fun, bounds)

    for i in range(budget):
        X, Y = perform_BO_iteration(X, Y, name_obj_fun, bounds, i, "Simplex transformation", apply_simplex=True)

    return Y

def perform_BO_experiment(seed : int, initial_design_size: int, budget: int, name_obj_fun : str, bounds) -> torch.Tensor:
    random.seed(seed)
    torch.random.manual_seed(seed)
    X, Y = get_initial_results(initial_design_size, name_obj_fun, bounds, seed)
    for i in range(budget):
        X, Y = perform_BO_iteration(X, Y, name_obj_fun, bounds, i, "Vanilla BO")

    return Y

def perform_random_experiment(seed, initial_design_size, budget, name_obj_fun, bounds) -> torch.Tensor:
    random.seed(seed)
    torch.random.manual_seed(seed)
    X, Y = get_initial_results(initial_design_size, name_obj_fun, bounds)

    for i in range(budget):
        X, Y = perform_random_iteration(X, Y, name_obj_fun, bounds)

    return Y

def biyective_transformation(x):
    simplex_representation = torch.zeros(x.shape[0]+1)
    p = torch.exp(x) / (1.0 + torch.sum(np.exp(x)))
    p_extra = 1.0 - torch.sum(p)
    simplex_representation[0:simplex_representation.shape[0]-1] = p
    simplex_representation[simplex_representation.shape[0]-1] = p_extra
    return simplex_representation

def inverse_biyective_transformation(x_simplex):
    return torch.log(x_simplex/(1-torch.sum(x_simplex)+x_simplex[x_simplex.shape[0]-1]))[0:x_simplex.shape[0]-1]

def perform_penalizing_approach_experiment(seed, initial_design_size, budget, dims_simplex) -> torch.Tensor:
    print('Initiating simplex transformation experiment')
    random.seed(seed)
    torch.random.manual_seed(seed)
    X, Y = get_initial_results(initial_design_size, seed, dims_simplex, penalizing_approach=True)
    for i in range(budget):
        X, Y = perform_BO_iteration(X, Y, seed, "Penalizing approach", apply_penalization=True)
        print("Iteration: " + str(i+1))
    print('Ending simplex transformation experiment')
    return Y

def perform_simplex_transformation_experiment(seed, initial_design_size, budget, dims_simplex) -> torch.Tensor:
    print('Initiating simplex transformation experiment')
    random.seed(seed)
    torch.random.manual_seed(seed)
    X, Y = get_initial_results(initial_design_size, seed, dims_simplex, simplex_transformation=True)
    for i in range(budget):
        X, Y = perform_BO_iteration(X, Y, seed, "Simplex transformation", apply_simplex=True)
        print("Iteration: " + str(i+1))
    print('Ending simplex transformation experiment')
    return Y

def perform_biyective_transformation_experiment(seed, initial_design_size, budget, dims_simplex) -> torch.Tensor:
    print('Initiating biyective transformation experiment')
    random.seed(seed)
    torch.random.manual_seed(seed)
    X, Y = get_initial_results(initial_design_size, seed, dims_simplex-1)
    for i in range(budget):
        X, Y = perform_BO_iteration(X, Y, seed, "Biyective transformation")
        print("Iteration: " + str(i+1))
    print('Ending biyective transformation experiment')
    return Y

def perform_RS_BT_experiment(seed, initial_design_size, budget, dims_simplex) -> torch.Tensor:
    print('Initiating Random search BT experiment')
    random.seed(seed)
    torch.random.manual_seed(seed)
    X, Y = get_initial_results(initial_design_size, seed, dims_simplex-1)
    for i in range(budget):
        X, Y = perform_random_iteration(X, Y, seed, "RS-BT")
        print("Iteration: " + str(i+1))
    print('Ending Random search BT experiment')
    return Y

def perform_RS_ST_experiment(seed, initial_design_size, budget, dims_simplex) -> torch.Tensor:
    print('Initiating Random search ST experiment')
    random.seed(seed)
    torch.random.manual_seed(seed)
    X, Y = get_initial_results(initial_design_size, seed, dims_simplex, simplex_transformation=True)
    for i in range(budget):
        X, Y = perform_random_iteration(X, Y, seed, "RS-ST", apply_simplex=True)
        print("Iteration: " + str(i+1))
    print('Ending Random search ST experiment')
    return Y

if __name__ == '__main__' :
    #Tests.
    #normalize_points(torch.tensor([3,-1]), torch.tensor([[-4.5,-4.5],[4.5,4.5]]))
    #branin(torch.tensor([9.42478, 2.475]))
    x_simplex = biyective_transformation(torch.tensor([0.3,0.9]))
    x = inverse_biyective_transformation(x_simplex)
    dims_simplex = 3
    total_exps = 10
    initial_design_size = 5
    budget = 10
    n_methods = 5
    total_its = initial_design_size + budget
    results = torch.ones((n_methods, total_its, total_exps))
    for exp in range(total_exps):
        results[0, :, exp] = perform_biyective_transformation_experiment(exp, initial_design_size, budget, dims_simplex).reshape((total_its))
        results[1, :, exp] = perform_simplex_transformation_experiment(exp, initial_design_size, budget, dims_simplex).reshape((total_its))
        results[2, :, exp] = perform_penalizing_approach_experiment(exp, initial_design_size, budget, dims_simplex).reshape((total_its))
        results[3, :, exp] = perform_RS_BT_experiment(exp, initial_design_size, budget, dims_simplex).reshape((total_its))
        results[4, :, exp] = perform_RS_ST_experiment(exp, initial_design_size, budget, dims_simplex).reshape((total_its))
        print(exp)
    #plot_results_log10_regret_acum(initial_design_size+budget, results)
    plot_results(initial_design_size+budget, results)
