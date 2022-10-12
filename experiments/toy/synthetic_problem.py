import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import time
from spearmint.models.gp import GP
from spearmint.acquisition_functions.predictive_entropy_search import sample_gp_with_random_features
from spearmint.utils.parsing          import parse_config_file
from spearmint.tasks.input_space      import InputSpace
from spearmint.tasks.input_space      import paramify_no_types
from spearmint.utils.parsing          import parse_tasks_from_jobs
import numpy as np

NUM_RANDOM_FEATURES = 1000

NO_ACTION = 1
QUERY_SYN_PROBLEM = 2
FINISHED = 3
import numpy as np
import ghalton
from os.path import exists

class Synthetic_problem:

    def __init__(self, num_experiment):
        
        state = np.random.get_state()

        np.random.seed(num_experiment)

        options = parse_config_file('.', 'config.json')
        input_space = InputSpace(options["variables"])
        tasks = parse_tasks_from_jobs(None, options["experiment-name"], options, input_space)

        for key in tasks:
            tasks[ key ].options['likelihood'] = "NOISELESS"

        sequence_size = 1000
        sequencer = ghalton.Halton(input_space.num_dims)
        X = np.array(sequencer.get(sequence_size))

        self.models = dict()
        self.tasks = tasks
        self.input_space = input_space

        for key in tasks:
                self.models[ key ] = GP(input_space.num_dims, **tasks[ key ].options)
                self.models[ key ].params['ls'].set_value(np.ones(input_space.num_dims) * 0.25 * input_space.num_dims)

                params = dict()
                params['hypers'] = dict()

                for hp in self.models[ key ].params:
                    params['hypers'][ hp ] = self.models[ key ].params[ hp ].value

                params['chain length'] = 0.0

		# We sample given the specified hyper-params

                samples = self.models[ key ].sample_from_prior_given_hypers(X)
                self.models[ key ].fit(X, samples, hypers = params, fit_hypers = False)

#	def compute_function(gp):
#		def f(x):
#			return gp.predict(x)[ 0 ]
#		return f

#	self.funs = dict()

#	for key in self.models:
#		self.funs[ key ] = compute_function(self.models[ key ])

        self.funs = { key : sample_gp_with_random_features(self.models[ key ], NUM_RANDOM_FEATURES) for key in self.models }

        np.random.set_state(state)

    def meshgrid_to_2d_grid(self, X, Y):
        final_piece = np.vstack((X[0,0].repeat(len(X[0])),Y[:,0])).T
        for i in range(len(X[0])-1):
            final_piece = np.concatenate((final_piece,np.vstack((X[0,i+1].repeat(len(X[0])),Y[:,0])).T))
        return final_piece

    def get_optimum(self, seed):
        grid_x = np.linspace(0.0, 1.0, 100)
        grid_y = np.linspace(0.0, 1.0, 100)
        X, Y = np.meshgrid(grid_x, grid_y)
        grid = self.meshgrid_to_2d_grid(X, Y)
        best_result = self.funs['o1'](grid[0], gradient = False)
        for point in grid:
            result = self.funs['o1'](point, gradient = False)
            if result < best_result:
                best_result = result
        f = open("optimums/best_result_" + str(seed) + ".txt", "w")
        f.write(str(best_result))
        f.close()

    def action_call(self):
        if exists("outputs/action.txt"):
            f = open("outputs/action.txt", "r")
            return f.read()
        else:
            return NO_ACTION 
    
    def get_params(self):
        result = np.array([])
        if exists("outputs/params_is.txt"):
            f = open("outputs/params_is.txt", "r")
            result = f.read().split(" ")
            result = np.array([float(r) for r in result])
        return result 

    def send_result(self, y):
        if not exists("outputs/result_ts.txt"):
            f = open("outputs/result_ts.txt", "w")
            f.write(str(y))
            f.close()
        if not exists("outputs/action_core.txt"):
            f = open("outputs/action_core.txt", "w")
            f.write(str(QUERY_SYN_PROBLEM))
            f.close()

    def sleep_until_call(self):
        action = NO_ACTION
        while(action == NO_ACTION):
            time.sleep(0.1)
            action = self.action_call()
            if(action != NO_ACTION):
                params = self.get_params()
                if len(params)>0:
                    y = self.funs['o1'](params, gradient = False)
                    self.send_result(y)
                    action = NO_ACTION
                    time.sleep(0.2)
                else:
                    action = FINISHED
    
    def f(self, x):

        values = np.zeros(len(x))

        i = 0
        for name in x:
            values[ i ] = x[ name ]
            i += 1

        if len(values.shape) <= 1:
            values = values.reshape((1, len(values)))

        evaluation = dict()

        for key in self.funs:
            evaluation[ key ] = self.funs[ key ](values, gradient = False)

        return evaluation

    def f_noisy(self, x):

        values = np.zeros(len(x))

        i = 0
        for name in x:
            values[ i ] = x[ name ]
            i +=1

        if len(values.shape) <= 1:
            values = values.reshape((1, len(values)))

        evaluation = dict()

        for key in self.funs:
            evaluation[ key ] = self.funs[ key ](values, gradient = False) + np.random.normal() * np.sqrt(1.0 / 100)

        return evaluation

    def plot(self, l_bound, h_bound):

        assert(self.input_space.num_dims == 2 or self.input_space.num_dims == 1)

        size = 50
        x = np.linspace(l_bound, h_bound, size)
        y = np.linspace(l_bound, h_bound, size)
        X, Y = np.meshgrid(x, y)

        if self.input_space.num_dims == 2:
            k = 0
            for key in self.models:

                Z = np.zeros((size, size))
                for i in range(size):
                    for j in range(size):
                        params = self.input_space.from_unit(np.array([ X[ i, j ], Y[ i, j ]])).flatten()
                        Z[ i, j ] = self.f(paramify_no_types(self.input_space.paramify(params)))[ key ]
	
                plt.figure()
                im = plt.imshow(Z, interpolation = 'bilinear', origin = 'lower', cmap = cm.gray, extent = (l_bound, h_bound, l_bound, h_bound))
                CS = plt.contour(X, Y, Z)
                plt.clabel(CS, inline = 1, fontsize = 10)
                plt.title('Objective function')
                plt.show()
                k += 1
        else:
            k = 0
            for key in self.models:

                Z = np.zeros(size)
                for i in range(size):
                    params = self.input_space.from_unit(np.array([ x[ i ]])).flatten()
                    Z[ i ] = self.f(paramify_no_types(self.input_space.paramify(params)))[ key ]

                plt.figure()
                plt.plot(x, Z, color='red', marker='.', markersize=1)
                plt.title(str(key))
                plt.show()
                k += 1



