import warnings

import numpy as np

import nengo
from nengo.dists import Choice, Uniform
from nengo.networks.ensemblearray import EnsembleArray
from nengo.solvers import NnlsL2nz


# connection weights from (Gurney, Prescott, & Redgrave, 2001)
mm = 1
mp = 1
me = 1
mg = 1
ws = 1
wt = 1
wm = 1
wg = 1
wp = 0.9
we = 0.3
e = 0.2
ep = -0.25
ee = -0.2
eg = -0.2
le = 0.2
lg = 0.2

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
stim_vals = [0, .02] # default input

gauss_dist = nengo.dists.Function(gauss_func,
                       mean=nengo.dists.Uniform(-1, 1),
                       var=nengo.dists.Uniform(0.01, 0.5))

# build the function space
func_space = nengo.FunctionSpace(gauss_dist, n_basis=20)

ea_params = {'n_neurons': 500,
             'dimensions': func_space.n_basis}

model = nengo.Network() # major network
model.config[nengo.Ensemble].neuron_type = nengo.Direct()
def gen_BG():
    net = nengo.Network() # basal ganglia subnetwork
    with net:

        net.strD1 = nengo.Ensemble(**ea_params)
        net.strD2 = nengo.Ensemble(**ea_params)
        net.stn = nengo.Ensemble(**ea_params)
        net.gpi = nengo.Ensemble(**ea_params)
        net.gpe = nengo.Ensemble(**ea_params)

        net.input = nengo.Ensemble(n_neurons=1, 
                dimensions=func_space.n_basis, 
                neuron_type=nengo.Direct(), label="input")
        net.output = nengo.Node(label="output", size_in=func_space.n_basis)

        def mult_func_space(x, w): 
            fx = func_space.reconstruct(x)
            fx *= w
            return func_space.project(fx)
        nengo.Connection(net.input, net.strD1, synapse=None,
                function=lambda x: mult_func_space(x, ws*(1+lg)))
        nengo.Connection(net.input, net.strD2, synapse=None,
                function=lambda x: mult_func_space(x, ws*(1-le)))
        nengo.Connection(net.input, net.stn, synapse=None,
                function=lambda x: mult_func_space(x, wt))

        # connect the striatum to the GPi and GPe (inhibitory)
        def str_func_space(x): 
            fx = func_space.reconstruct(x)
            fx[fx < e] = 0
            fx = (mm * (fx - e)) * -wm
            return func_space.project(fx)

        nengo.Connection(net.strD1, net.gpi, function=str_func_space,)
        nengo.Connection(net.strD2, net.gpe, function=str_func_space,)

        # connect the STN to GPi and GPe (broad and excitatory)
        def stn_func_space(x):
            fx = func_space.reconstruct(x)
            fx[fx < ep] = 0
            fx = (mp * (fx - ep)) * wp
            # now sum up fx and output ones * sum(fx)
            return func_space.project(np.ones(len(fx)) * np.sum(fx)) 

        nengo.Connection(net.stn, net.gpi, function=stn_func_space,)
        nengo.Connection(net.stn, net.gpe, function=stn_func_space,)

        # connect the GPe to GPi and STN (inhibitory)
        def gpe_func_space(x, w):
            fx = func_space.reconstruct(x)
            fx[fx < ee] = 0
            fx = (me * (fx - ee)) * w
            return func_space.project(fx)

        nengo.Connection(net.gpe, net.gpi, 
                function=lambda x: gpe_func_space(x, -we))
        nengo.Connection(net.gpe, net.stn, 
                function=lambda x: gpe_func_space(x, -wg))

        # connect GPi to output (inhibitory)
        def gpi_func_space(x, output_weight):
            fx = func_space.reconstruct(x)
            print output_weight
            fx[fx < eg] = 0
            fx = (mg * (fx - eg)) * output_weight
            return func_space.project(fx)

        nengo.Connection(net.gpi, net.output, synapse=None,
                function=lambda x: gpi_func_space(x, -3))
    return net

with model:

    BG = gen_BG()

    stimulus1 = func_space.make_stimulus_node(gauss_func, 
            n_params=len(stim_vals))
    nengo.Connection(stimulus1, BG.input)

    stim_control1 = nengo.Node(stim_vals)
    nengo.Connection(stim_control1, stimulus1)

    make_plot = func_space.make_plot_node 
    plot_input = make_plot(domain)
    nengo.Connection(BG.input, plot_input)

    make_plot = func_space.make_plot_node 
    plot_output = make_plot(domain, max_y=100, min_y=-100)
    nengo.Connection(BG.output, plot_output)
