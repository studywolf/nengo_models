import warnings

import numpy as np

import nengo
from nengo.dists import Choice, Uniform
import nengo.utils.function_space
reload(nengo.utils.function_space)

nengo.dists.Function = nengo.utils.function_space.Function
nengo.FunctionSpace = nengo.utils.function_space.FunctionSpace

num_samples = 10
domain = np.linspace(-1, 1, num_samples**2)
min_var = .01
max_var = .5
# define your function
def gauss_func(mean, var):
    var = min(max(var, min_var), max_var)
    return np.exp(-(domain-mean)**2/(2*var))

gauss_dist = nengo.dists.Function(gauss_func,
                       mean=nengo.dists.Uniform(-1, 1),
                       var=nengo.dists.Uniform(0.01, 0.5))

# build the function space
gauss_fs = nengo.FunctionSpace(gauss_dist, n_basis=20)

def make_func_ens(model, n_neurons, func, dist, fs, stim_vals, domain, plot_2D=False):
    with model: 
        ens = nengo.Ensemble(n_neurons=n_neurons, dimensions=fs.n_basis)
        ens.encoders = fs.project(dist)
        ens.eval_points = fs.project(dist)

        stimulus1 = fs.make_stimulus_node(func, n_params=len(stim_vals))
        nengo.Connection(stimulus1, ens)

        stim_control1 = nengo.Node(stim_vals)
        nengo.Connection(stim_control1, stimulus1)

        make_plot = fs.make_plot_node if plot_2D is False else fs.make_2Dplot_node
        plot = make_plot(domain)
        nengo.Connection(ens, plot[:fs.n_basis])
    return ens

model = nengo.Network()

with model:

    ens1 = make_func_ens(model=model, n_neurons=2000, 
        func=gauss_func, dist=gauss_dist, fs=gauss_fs,
        stim_vals=[0, .02], domain=domain)

    ens2 = nengo.Ensemble(n_neurons=2000, dimensions=gauss_fs.n_basis)
    ens2.encoders = gauss_fs.project(gauss_dist)
    ens2.eval_points = gauss_fs.project(gauss_dist)

    make_plot = gauss_fs.make_plot_node 
    plot = make_plot(domain)
    nengo.Connection(ens2, plot[:gauss_fs.n_basis])

    def mult_func_space(x):
        fx = gauss_fs.reconstruct(x)
        fx *= 2
        return gauss_fs.project(fx)
    nengo.Connection(ens1, ens2, function=mult_func_space)
