import numpy as np

import nengo

model = nengo.Network()

with model: 

    node_in = nengo.Node(output=lambda x: np.sin(x*3))

    # calculate derivative
    deriv = nengo.Ensemble(n_neurons=500, dimensions=2)
    nengo.Connection(node_in, deriv[0])
    nengo.Connection(node_in, deriv[1])
    nengo.Connection(deriv[0], deriv[1], transform=-1, synapse=.1)

    # population representing what gets sent to the system
    output = nengo.Ensemble(n_neurons=100, dimensions=1)
    nengo.Connection(node_in, output)

    # population that doesn't respond to signals less that .2
    gate = nengo.Ensemble(n_neurons=100, dimensions=1, 
            encoders=[[1]]*100, intercepts=nengo.dists.Uniform(.2, 1))
    # connect it up to the derivative
    nengo.Connection(deriv[1], gate, function=lambda x: abs(x))
    # if the derivative goes above .2, inhibit the output signal
    nengo.Connection(gate, output.neurons, 
            transform=[[-5]]*output.n_neurons)

