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

#1. Design Daniel transformation.
#2. Averiguar por qué el simplex lo hace al revés y porque no está funcionando la optimización, es MUY raro.
# OK 2.1. Asegurarse de que se está imprimiendo bien. Contrastar que el grid y valor por valor te da lo mismo para it = 10 en adelante.
#2.2. PARECE QUE SI Averiguar si el modelo está aplicando la transformación de entrada cuando hace la media. Probar con y sin transformación.
# HECHO, PERO PARECE INMUNE 2.3. Exagerar valores para disperar 0.5 0.05 a 0.1 y 0.01 a ver que pasa. 
#2.4. Ver si la idea de la transformacion funciona, ver un punto que colisione con otro del simplex y comprobar que ambos puntos valen lo mismo de acorde al posterior. 
#3. Debuggeo grande a los valores. 

#No es que se penalizen soluciones que no valen. Es que la función por dentro hace el método del simplex con carteras incorrectas. Luego si ya le
#metes esa información al modelo, entonces va a hacerlo mejor.  Aunque es raro, tecnicamente no hay pendiente, es o 0, o valor. Hacer una funcion del 
#estilo ya que debe ser cartera valida o pdf valida. Hacer el planteamiento de la transformacion por si el programa te lo traspasa automatico, en ese caso si vale. 
#Igual esto se puede resolver con un prior en vez de con una transformación?

GLOBAL_MAXIMUM = 1000

def call_python_version(Version, Module, Function, ArgumentList):
    gw      = execnet.makegateway("popen//python=python%s" % Version)
    channel = gw.remote_exec("""
        from %s import %s as the_function
        channel.send(the_function(*channel.receive()))
    """ % (Module, Function))
    channel.send(ArgumentList)
    return channel.receive()

def normalize_points(X, bounds):
    Y= X.clone()
    bounds = bounds.T
    min_bounds = bounds[0].repeat(Y.shape[0]).reshape((Y.shape[0], Y.shape[1]))
    max_bounds = bounds[1].repeat(Y.shape[0]).reshape((Y.shape[0], Y.shape[1]))
    return (Y-min_bounds)/(max_bounds-min_bounds) #(x_i-x_min)/(x_max-x_min)

def unnormalize_points(X, bounds):
    Y = X.clone()
    for dim in range(X.shape[1]):
        Y[:,dim] = Y[:,dim]*(bounds[dim,1]-bounds[dim,0])+bounds[dim,0]
    return Y

def ci(y, n_exps): #Confidence interval.
    return 1.96 * y.std(axis=1) / np.sqrt(n_exps)

def obj_fun_2(X_train): #Objective function. Needs to be only valid for the diagonal, or at least, more valid there.
    return torch.tensor([np.sin(x[0])/(np.cos(x[1]) + np.sinh(x[2])) + torch.rand(1)/10.0 for x in X_train])

def simplex_penalization(x, bounds):
    max_bounds = torch.abs(bounds[0,0]-bounds[0,1]) #The maximum range. 
    x = x * max_bounds
    sum_values = torch.sum(x)
    distance_wrt_simplex = torch.abs(max_bounds-sum_values)
    penalization = 10000 * distance_wrt_simplex
    return penalization

def branin_function(x, bounds): #To minimize, tested OK.
    x = unnormalize_points(x.reshape(1, x.shape[0]), bounds)[0]
    a=1.0
    b=5.1/(4.0*np.pi**2)
    c=5.0/np.pi
    r=6.0
    s=10.0
    t=1.0/(8.0*np.pi)
    f=a*(x[1]-b*x[0]**2+c*x[0]-r)**2+s*(1-t)*torch.cos(x[0])+s
    return f

def penalized_branin(x, bounds):
    if len(x.shape) == 2:
        x = x.reshape(x.shape[1])
    return branin_function(x, bounds) + simplex_penalization(x, bounds)

def sphere_obj_function(x, bounds): #Length of x: 5. Range [0,1]^5. To be maximized.
    if torch.any(x > 1.0):
        raise Exception("Hypercube violated")
    if len(x.shape) == 2:
        x = x.reshape(x.shape[1])
    x = unnormalize_points(x.reshape(1, x.shape[0]), bounds)[0]
    sum_values = torch.sum(x)
    y = torch.sum(x**2.0) #Sphere function.
    #y = (100.0*torch.sin(x[0]) + 100.0*torch.cos(x[1])) / (1.0 + 30.0*torch.sin(x[2]))
    distance_wrt_simplex = torch.abs(torch.tensor(1.0)-sum_values)
    penalization = 100 * distance_wrt_simplex
    return y - penalization

def sphere_obj_function_old(x, bounds): #Length of x: 5. Range [0,1]^5. To be maximized.
    if torch.any(x > 1.0):
        raise Exception("Hypercube violated")
    if len(x.shape) == 2:
        x = x.reshape(x.shape[1])
    x = unnormalize_points(x.reshape(1, x.shape[0]), bounds)[0]
    sum_values = torch.sum(x)
    y = torch.sum(x**2.0) #Sphere function.
    #y = (100.0*torch.sin(x[0]) + 100.0*torch.cos(x[1])) / (1.0 + 30.0*torch.sin(x[2]))
    distance_wrt_simplex = torch.abs(torch.tensor(1.0)-sum_values)
    penalization = 20.0 * distance_wrt_simplex**x.shape[0] 
    return y - penalization

def wrapper_gp_sample(x, seed):
    import pdb; pdb.set_trace();
    return torch.tensor(float(call_python_version("2.7", "prog", "wrapper", [seed, float(x[0]), float(x[1])])))

def obj_fun(x, name, bounds, seed):
    if name == 'sphere':
        y = sphere_obj_function(x, bounds)
    elif name == 'branin':
        y = penalized_branin(x, bounds)
    else:
        y = wrapper_gp_sample(x, seed)
    return y

def wrapped_obj_fun(X_train, name_obj_fun, bounds):
    transformed_inputs = (X_train - 0.5) / 0.05
    X_train = torch.exp(transformed_inputs)/torch.sum(torch.exp(transformed_inputs)) #Sums to 1: ps assert(torch.sum(X_train)==1.0)
    return obj_fun(X_train, name_obj_fun, bounds)

def penalize_obj_fun(X_train, name_obj_fun, bounds):
    y = obj_fun(X_train, name_obj_fun, bounds)
    sum_values = torch.sum(X_train)
    penalization = torch.abs(torch.tensor(1.0)-sum_values)
    return y-penalization

def plot_results_log10_regret(n_iters, results):
    X_plot = np.linspace(1, n_iters, n_iters)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.errorbar(
        X_plot, np.log10(GLOBAL_MAXIMUM - results[0].mean(axis=1)), yerr=0.1*ci(results[0], results.shape[2]), label="Vanilla BO", linewidth=1.5, capsize=3, alpha=0.6
    )
    ax.errorbar(
        X_plot, np.log10(GLOBAL_MAXIMUM - results[1].mean(axis=1)), yerr=0.1*ci(results[1], results.shape[2]), label="Random Search", linewidth=1.5, capsize=3, alpha=0.6,
    )
    ax.errorbar(
        X_plot, np.log10(GLOBAL_MAXIMUM - results[2].mean(axis=1)), yerr=0.1*ci(results[2], results.shape[2]), label="Wrapped objective function", linewidth=1.5, capsize=3, alpha=0.6,
    )
    ax.errorbar(
        X_plot, np.log10(GLOBAL_MAXIMUM - results[3].mean(axis=1)), yerr=0.1*ci(results[3], results.shape[2]), label="Penalized objective function", linewidth=1.5, capsize=3, alpha=0.6,
    )
    ax.errorbar(
        X_plot, np.log10(GLOBAL_MAXIMUM - results[4].mean(axis=1)), yerr=0.1*ci(results[4], results.shape[2]), label="Simplex transformation", linewidth=1.5, capsize=3, alpha=0.6,
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
        X_plot, get_best_results_list(y_0), yerr=0.1*ci(results[0], results.shape[2]), label="Vanilla BO", linewidth=1.5, capsize=3, alpha=0.6
    )
    ax.errorbar(
        X_plot, get_best_results_list(y_1), yerr=0.1*ci(results[1], results.shape[2]), label="Random Search", linewidth=1.5, capsize=3, alpha=0.6,
    )
    ax.errorbar(
        X_plot, get_best_results_list(y_2), yerr=0.1*ci(results[2], results.shape[2]), label="Wrapped objective function", linewidth=1.5, capsize=3, alpha=0.6,
    )
    ax.errorbar(
        X_plot, get_best_results_list(y_3), yerr=0.1*ci(results[3], results.shape[2]), label="Penalized objective function", linewidth=1.5, capsize=3, alpha=0.6,
    )
    ax.errorbar(
        X_plot, get_best_results_list(y_4), yerr=0.1*ci(results[4], results.shape[2]), label="Simplex transformation", linewidth=1.5, capsize=3, alpha=0.6,
    )
    #ax.set(xlabel='number of observations (beyond initial points)', ylabel='Log10 Regret')
    ax.set(xlabel='Number of observations', ylabel='Best observed Log10 Regret')
    ax.legend(loc="lower left")
    plt.title('Bayesian optimization results of the different methods')
    plt.show()

def plot_results(n_iters, results):
    X_plot = np.linspace(1, n_iters, n_iters)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.errorbar(
        X_plot, results[0].mean(axis=1), yerr=0.1 * ci(results[0], results.shape[2]), label="Vanilla BO", linewidth=1.5, capsize=3, alpha=0.6
    )
    ax.errorbar(
        X_plot, results[1].mean(axis=1), yerr=0.1 * ci(results[1], results.shape[2]), label="Random Search", linewidth=1.5, capsize=3, alpha=0.6,
    )
    ax.errorbar(
        X_plot, results[2].mean(axis=1), yerr=0.1 * ci(results[2], results.shape[2]), label="Wrapped objective function", linewidth=1.5, capsize=3, alpha=0.6,
    )
    ax.errorbar(
        X_plot, results[3].mean(axis=1), yerr=0.1 * ci(results[3], results.shape[2]), label="Penalized objective function", linewidth=1.5, capsize=3, alpha=0.6,
    )
    ax.errorbar(
        X_plot, results[4].mean(axis=1), yerr=0.1 * ci(results[4], results.shape[2]), label="Simplex transformation", linewidth=1.5, capsize=3, alpha=0.6,
    )
    #ax.set(xlabel='number of observations (beyond initial points)', ylabel='Log10 Regret')
    ax.set(xlabel='Number of observations', ylabel='Objective function')
    ax.legend(loc="lower left")
    plt.title('Bayesian optimization results of the different methods')
    plt.show()

def get_initial_results(initial_design_size, name_obj_fun, bounds, seed):
    X = []
    n_dims = bounds.shape[1]
    X = torch.rand(initial_design_size, n_dims)
    import pdb; pdb.set_trace();
    Y = torch.tensor([obj_fun(x, name_obj_fun, bounds, seed) for x in X]).reshape(X.shape[0], 1)
    return X, Y

def meshgrid_to_2d_grid(X, Y):
    final_piece = torch.vstack((X[0,0].repeat(len(X[0])),Y[0])).T
    for i in range(len(X[0])-1):
        final_piece = torch.cat((final_piece,torch.vstack((X[i+1,0].repeat(len(X[0])),Y[0])).T))
    return final_piece

def plot_acq_fun_model_posterior(acq_fun, obs_input, model, bounds, fun_name, iteration, method_name):
    grid_x = torch.linspace(0.0, 1.0, 100)
    grid_y = torch.linspace(0.0, 1.0, 100)
    X, Y = torch.meshgrid(grid_x, grid_y)
    grid = meshgrid_to_2d_grid(X, Y)
    acq_fun_grid = acq_fun.forward(grid.reshape((grid.shape[0],1,grid.shape[1]))).detach()
    posterior_grid = model.posterior(grid).mean[:,0].detach()
    #function_grid = torch.sum(grid**2.0, axis=1)
    function_grid = torch.tensor([obj_fun(x, name_obj_fun, bounds) for x in grid])
    
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

def perform_BO_iteration(X, Y, name_obj_fun, bounds, seed, method_name, normalize=False, wrapped=False, penalize=False, apply_simplex=False, plot_acq_model=True):

    if not apply_simplex:
        gp = SingleTaskGP(X, Y)
    else:
        #normalize = Normalize(d=X.shape[1], bounds=bounds)
        simplex = Simplex(indices=list(range(X.shape[-1]))) #Print acq. fun.
        #tf = ChainedInputTransform(tf1=normalize, tf2=simplex)
        #gp = SingleTaskGP(X, Y, input_transform=tf)
        gp = SingleTaskGP(X, Y, input_transform=simplex)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    
    try:
        fit_gpytorch_model(mll)
    except: #Numerical issues.
        import gpytorch
        gpytorch.settings.cholesky_jitter._global_float_value = 1e-02 #Ill-conditioned matrix, adding more jitter to diagonal for cholesky.
        fit_gpytorch_model(mll)
        gpytorch.settings.cholesky_jitter._global_float_value = 1e-06 #Restoring.
    
    UCB = UpperConfidenceBound(gp, beta=0.1, maximize=False)
    bounds_cube = torch.stack([torch.zeros(X.shape[1]), torch.ones(X.shape[1])])
    if plot_acq_model:
        plot_acq_fun_model_posterior(UCB, X, gp, bounds, name_obj_fun, seed+1, method_name)
    new_X, acq_value = optimize_acqf(
            UCB, bounds=bounds_cube, q=1, num_restarts=5, raw_samples=20,
    )
    if wrapped:
        new_y = wrapped_obj_fun(new_X, name_obj_fun, bounds)
    elif penalize:
        new_y = penalize_obj_fun(new_X, name_obj_fun, bounds)
    else:
        new_y = obj_fun(new_X, name_obj_fun, bounds)
    X = torch.cat((X, new_X),0)
    Y = torch.cat((Y, new_y.reshape(1,1)),0)
    return X, Y

def perform_random_iteration(X, Y, name_obj_fun, bounds):
    new_X = torch.rand(1, X.shape[1])
    new_y = obj_fun(new_X, name_obj_fun, bounds)
    X = torch.cat((X, new_X),0)
    Y = torch.cat((Y, new_y.reshape(1,1)),0)
    return X, Y

def perform_wrapper_rounding_experiment(seed : int, initial_design_size: int, budget: int, name_obj_fun : str, bounds) -> torch.Tensor:
    random.seed(seed)
    torch.random.manual_seed(seed)
    X, Y = get_initial_results(initial_design_size, name_obj_fun, bounds)

    for i in range(budget):
        X, Y = perform_BO_iteration(X, Y, name_obj_fun, bounds, i, "Wrapper rounding", wrapped=True)

    return Y

def perform_wrapper_penalizing_experiment(seed : int, initial_design_size: int, budget: int, name_obj_fun : str, bounds) -> torch.Tensor:
    random.seed(seed)
    torch.random.manual_seed(seed)
    X, Y = get_initial_results(initial_design_size, name_obj_fun, bounds)

    for i in range(budget):
        X, Y = perform_BO_iteration(X, Y, name_obj_fun, bounds, i, "Wrapper penalizing", penalize=True)

    return Y


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

if __name__ == '__main__' :
    #Tests.
    #normalize_points(torch.tensor([3,-1]), torch.tensor([[-4.5,-4.5],[4.5,4.5]]))
    #branin(torch.tensor([9.42478, 2.475]))
    import pdb; pdb.set_trace();
    x_simplex = biyective_transformation(torch.tensor([0.3,0.9]))
    x = inverse_biyective_transformation(x_simplex)
    total_exps = 1
    initial_design_size = 5
    budget = 10
    n_methods = 5
    name_obj_fun = 'GP-synthetic'
    bounds = torch.stack([torch.tensor([0,1]), torch.tensor([0,1])])
    total_its = initial_design_size + budget
    results = torch.ones((n_methods, total_its, total_exps))
    for exp in range(total_exps):
        results[0, :, exp] = perform_BO_experiment(exp, initial_design_size, budget, name_obj_fun, bounds).reshape((total_its))
        results[1, :, exp] = perform_random_experiment(exp, initial_design_size, budget, name_obj_fun, bounds).reshape((total_its))
        results[2, :, exp] = perform_wrapper_rounding_experiment(exp, initial_design_size, budget, name_obj_fun, bounds).reshape((total_its))
        results[3, :, exp] = perform_wrapper_penalizing_experiment(exp, initial_design_size, budget, name_obj_fun, bounds).reshape((total_its))
        results[4, :, exp] = perform_simplex_transformation_experiment(exp, initial_design_size, budget, name_obj_fun, bounds).reshape((total_its))
        print(exp)
    #plot_results_log10_regret_acum(initial_design_size+budget, results)
    plot_results(initial_design_size+budget, results)
