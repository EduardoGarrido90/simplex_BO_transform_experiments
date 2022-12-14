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
import math
import matplotlib.tri as tri
import os 
import time
from os.path import exists

#TAREAS DEL SIMPLEX:
#DONE 0.a: Hacer el random 2D.
#DONE 0.b: Hacer el random 3D.
#1. Visualización del simplex. Generar los puntos de la transf. biyectiva para ver que no cubre con [0,1]² y arreglar visualizacion.)
#DONE 2. Extraccion del mejor punto del simplex.
#DONE 2b. Extraccion del peor punto del simplex.
#3. Para penalizacion:
#3.1 Hallar si el punto pertenece al simplex, sino, coger el peor punto y penalizar linealmente por distancia de la proyeccion de forma suave.
#3.2 Si pertenece, entonces hacer la transformación inversa y ya esta.
#4 Hacer una visualizacion exhaustiva de la funcion objetivo en 2D y de su representacion en el simplex para enseñar a Daniel que [0,1]^2 -> No es exhaustivo el simplex. Esto se arregla con -0.5/0.05.
#5 Arreglar con -0.5/0.05 que sería una escala de [0,1]^2 a R^2, luego de R^2 a S^3 tienes el simplex de ese punto. 
#6 Comparar todos los baselines.
GLOBAL_MAXIMUM = 1000
NO_ACTION = 1
QUERY_SYN_PROBLEM = 2
FINISHED = 3

def xy2bc(xy, tol=1.e-4):
    '''Converts 2D Cartesian coordinates to barycentric.'''
    coords = np.array([tri_area(xy, p) for p in pairs]) / AREA
    return np.clip(coords, tol, 1.0 - tol)

def plot_simplex(seed, iteration, obs_input=None, **kwargs):
    corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
    AREA = 0.5 * 1 * 0.75**0.5
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=2)
    pvals = [objective_function(xy, seed) for xy in zip(trimesh.x, trimesh.y)]
    nlevels = 200
    fig1, _ = plt.subplots()
    tcf = plt.tricontourf(trimesh, pvals, nlevels, cmap='jet', origin='lower', **kwargs)
    fig1.colorbar(tcf)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    if obs_input != None:
        observations = []
        for observation in obs_input:
            observations.append(biyective_transformation(observation))
        observations = np.array([x.numpy() for x in observations])
        observations = observations.dot(corners)
        plt.plot(observations[:, 0], observations[:, 1], 'Xk')
        plt.plot(observations[len(observations)-1,0], observations[len(observations)-1,1], 'Xr')
    plt.title('Objective function transformed into simplex')
    plt.show()
    plt.savefig('./images/simplex_objective_function_' + str(iteration) + '.png')
    plt.axis('off')

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

def plot_objective_function(seed, iteration, l_bound=0, h_bound=1, obs_input=None):
    if obs_input != None:
        generate_observations_file(obs_input)
    call_python_version("2.7", "prog", "plot_objective_function", [seed, iteration, l_bound, h_bound])
    plot_simplex(seed, iteration, obs_input)

def generate_observations_file(obs_input):
    f = open("outputs/obs_input.txt","w")
    np.savetxt('outputs/obs_input.txt', obs_input.numpy())

def delete_files():
    os.remove("outputs/action.txt")
    os.remove("outputs/result_ts.txt")
    os.remove("outputs/action_core.txt")
    os.remove("outputs/params_is.txt")

def action_call():
    if exists("outputs/action_core.txt"):
       f = open("outputs/action_core.txt", "r") 
       return f.read()
    else:
        return NO_ACTION

def get_result_obj_fun():
    f = open("outputs/result_ts.txt")
    return float(f.read())

def objective_function(x, seed, to_simplex=False, penalize=False, expand_point=False):
    if(expand_point):
        x = (x - 0.5) / 0.05
    if(to_simplex):
        x = inverse_biyective_transformation(simplex_transformation(x))
    print("Evaluating objective function X=", str(x))
    #y = torch.tensor(float(call_python_version("2.7", "prog", "wrapper", [seed, float(x[0]), float(x[1])])))
    f = open("outputs/params_is.txt", "w")
    f.write(str(float(x[0])) + " " + str(float(x[1])))
    f.close()
    f = open("outputs/action.txt", "w")
    f.write(str(QUERY_SYN_PROBLEM))
    f.close()
    action=NO_ACTION
    print("Querying GP")
    while(action==NO_ACTION):
        time.sleep(0.1)
        action = action_call()
        if(action != NO_ACTION):
            y = get_result_obj_fun()
            delete_files()
    print("GP queried")
    if(penalize):
        y = penalization_approach(x, y)
    print("Objective function evaluated. Y=", str(y))
    return torch.tensor(y)

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

def plot_results_log10_regret_acum(n_iters, results, optimums):
    X_plot = np.linspace(1, n_iters, n_iters)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    optimums = np.repeat(optimums, results.shape[1]).reshape((optimums.shape[0], results.shape[1])).T
    y_0 = np.log10(np.abs(optimums - results[0])).mean(axis=1)
    y_1 = np.log10(np.abs(optimums - results[1])).mean(axis=1)
    y_2 = np.log10(np.abs(optimums - results[2])).mean(axis=1)
    y_3 = np.log10(np.abs(optimums - results[3])).mean(axis=1)
    y_4 = np.log10(np.abs(optimums - results[4])).mean(axis=1)
    y_5 = np.log10(np.abs(optimums - results[5])).mean(axis=1)

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
    ax.errorbar(
        X_plot, get_best_results_list(y_5), yerr=0.1*ci(results[4], results.shape[2]), label="Biyective Transformation extended", linewidth=1.5, capsize=3, alpha=0.6,
    )
    #ax.set(xlabel='number of observations (beyond initial points)', ylabel='Log10 Regret')
    ax.set(xlabel='Number of observations', ylabel='Best observed Log10 Regret')
    ax.legend(loc="lower left")
    plt.title('Bayesian optimization results of the different methods')
    plt.show()

def get_initial_results(initial_design_size, seed, dims, simplex_transformation=False, penalizing_approach=False, expand=False):
    X = torch.rand(initial_design_size, dims)
    Y = torch.tensor([objective_function(x, seed, to_simplex=simplex_transformation, penalize=penalizing_approach, expand_point=expand) for x in X]).reshape(X.shape[0], 1)
    return X, Y

def meshgrid_to_2d_grid(X, Y):
    final_piece = torch.vstack((X[0,0].repeat(len(X[0])),Y[0])).T
    for i in range(len(X[0])-1):
        final_piece = torch.cat((final_piece,torch.vstack((X[i+1,0].repeat(len(X[0])),Y[0])).T))
    return final_piece

def plot_acq_fun_model_posterior(acq_fun, obs_input, model, iteration, method_name, seed):
    grid_x = torch.linspace(0.0, 1.0, 100)
    grid_y = torch.linspace(0.0, 1.0, 100)
    grid_dim = len(grid_x)
    X, Y = torch.meshgrid(grid_x, grid_y)
    grid = meshgrid_to_2d_grid(X, Y)
    acq_fun_grid = acq_fun.forward(grid.reshape((grid.shape[0],1,grid.shape[1]))).detach()
    posterior_grid = model.posterior(grid).mean[:,0].detach()
    plot_objective_function(seed, iteration, obs_input=obs_input)
    #function_grid = torch.sum(grid**2.0, axis=1)
    #function_grid = torch.tensor([objective_function(x, name_obj_fun, bounds) for x in grid])
    
    '''
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
    '''

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

def perform_BO_iteration(X, Y, seed, method_name, iteration, apply_simplex=False, apply_penalization=False, plot_acq_model=True, expand=False):

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
        plot_acq_fun_model_posterior(UCB, X, gp, iteration, method_name, seed)
    new_X, acq_value = optimize_acqf(UCB, bounds=bounds_cube, q=1, num_restarts=5, raw_samples=20,)
    new_y = objective_function(new_X[0], seed, to_simplex=apply_simplex, penalize=apply_penalization, expand_point=expand)
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
        X, Y = perform_BO_iteration(X, Y, seed, "Penalizing approach", i, apply_penalization=True)
        print("Iteration: " + str(i+1))
    print('Ending simplex transformation experiment')
    return Y

def perform_simplex_transformation_experiment(seed, initial_design_size, budget, dims_simplex) -> torch.Tensor:
    print('Initiating simplex transformation experiment')
    random.seed(seed)
    torch.random.manual_seed(seed)
    X, Y = get_initial_results(initial_design_size, seed, dims_simplex, simplex_transformation=True)
    for i in range(budget):
        X, Y = perform_BO_iteration(X, Y, seed, "Simplex transformation", i, apply_simplex=True)
        print("Iteration: " + str(i+1))
    print('Ending simplex transformation experiment')
    return Y

def perform_biyective_transformation_experiment(seed, initial_design_size, budget, dims_simplex) -> torch.Tensor:
    print('Initiating biyective transformation experiment')
    random.seed(seed)
    torch.random.manual_seed(seed)
    X, Y = get_initial_results(initial_design_size, seed, dims_simplex-1)
    for i in range(budget):
        X, Y = perform_BO_iteration(X, Y, seed, "Biyective transformation", i)
        print("Iteration: " + str(i+1))
    print('Ending biyective transformation experiment')
    return Y

def perform_biyective_transformation_experiment_expanded(seed, initial_design_size, budget, dims_simplex) -> torch.Tensor:
    print('Initiating biyective transformation experiment')
    random.seed(seed)
    torch.random.manual_seed(seed)
    X, Y = get_initial_results(initial_design_size, seed, dims_simplex-1, expand=True)
    for i in range(budget):
        X, Y = perform_BO_iteration(X, Y, seed, "Biyective transformation expanded", i, expand=True)
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

def generate_synthetic_problem(seed):
    #call_python_version("2.7", "prog", "initiate", [seed])
    os.system("python2 prog.py " + str(seed) + " &")

def generate_optimum(seed):
    #call_python_version("2.7", "prog", "initiate", [seed])
    if not exists("optimums/best_result_" + str(seed) + ".txt"):
        os.system("python2 prog.py " + str(seed) + " 1")
    f = open("optimums/best_result_" + str(seed) + ".txt", "r")
    optimum = float(f.read())
    f.close()
    return optimum

def generate_worst(seed):
    #call_python_version("2.7", "prog", "initiate", [seed])
    if not exists("optimums/worst_result_" + str(seed) + ".txt"):
        os.system("python2 prog.py " + str(seed) + " 2")
    f = open("optimums/worst_result_" + str(seed) + ".txt", "r")
    optimum = float(f.read())
    f.close()
    return optimum

if __name__ == '__main__' :
    #Tests.
    #normalize_points(torch.tensor([3,-1]), torch.tensor([[-4.5,-4.5],[4.5,4.5]]))
    #branin(torch.tensor([9.42478, 2.475]))
    #x_simplex = biyective_transformation(torch.tensor([0.3,0.9]))
    #x = inverse_biyective_transformation(x_simplex)
    dims_simplex = 3
    total_exps = 5
    initial_design_size = 5
    budget = 20
    n_methods = 6
    total_its = initial_design_size + budget
    results = torch.ones((n_methods, total_its, total_exps))
    optimums = torch.ones((total_exps))
    worsts = torch.ones((total_exps))
    for exp in range(total_exps):
        optimums[exp] = generate_optimum(exp)
        worsts[exp] = generate_worst(exp)
        generate_synthetic_problem(exp)
        plot_objective_function(exp, 0)
        results[0, :, exp] = perform_biyective_transformation_experiment(exp, initial_design_size, budget, dims_simplex).reshape((total_its))
        results[1, :, exp] = perform_biyective_transformation_experiment_expanded(exp, initial_design_size, budget, dims_simplex).reshape((total_its))
        results[2, :, exp] = perform_simplex_transformation_experiment(exp, initial_design_size, budget, dims_simplex).reshape((total_its))
        results[3, :, exp] = perform_penalizing_approach_experiment(exp, initial_design_size, budget, dims_simplex).reshape((total_its))
        results[4, :, exp] = perform_RS_BT_experiment(exp, initial_design_size, budget, dims_simplex).reshape((total_its))
        results[5, :, exp] = perform_RS_ST_experiment(exp, initial_design_size, budget, dims_simplex).reshape((total_its))
        print(exp)
    f = open("outputs/action.txt", "a") #kills the other process
    f.write(str(FINISHED))
    f.close()
    plot_results_log10_regret_acum(initial_design_size+budget, results, optimums)
    #plot_results(initial_design_size+budget, results)
