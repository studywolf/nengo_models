import numpy as np

import nengo

model = nengo.Network()

with model: 

    node_in = nengo.Node(output=lambda x: [np.sin(x/3.0), np.cos(x/3.0)])
    bg = nengo.networks.BasalGanglia(4)

    encoders = [[1,1],[-1,1],[-1,-1],[1,-1]]
    enses = []
    for ii,encoder in enumerate(encoders):
        enses.append(nengo.Ensemble(n_neurons=100, dimensions=2,
            encoders=nengo.dists.Choice([encoder]),
            intercepts=nengo.dists.Uniform(.5,1)))
        nengo.Connection(node_in, enses[ii])
        nengo.Connection(enses[ii].neurons, bg.input[ii], 
                transform=np.ones((1,enses[ii].n_neurons))/enses[ii].n_neurons)

    

