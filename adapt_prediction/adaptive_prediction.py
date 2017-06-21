""" An example of adaptive prediction. The system learns to supplement
a linear prediction of the state n time steps into the future. """

import numpy as np

import nengo
from nengolib.signal import z

dt = 0.001
n = 500  # number of time steps ahead to predict state

model = nengo.Network()
with model:
    # system state is a cosine, second term is its velocity
    q_node = nengo.Node(lambda t: [np.cos(2*t), -2*np.sin(2*t)])
    q = nengo.Ensemble(n_neurons=500, dimensions=2)
    nengo.Connection(q_node, q, synapse=None)

    # for storing the predicted state of q
    pred_q = nengo.Ensemble(n_neurons=1, dimensions=1,
                            neuron_type=nengo.Direct())

    # transform for initial linear prediction of state in n time steps
    T = np.array([[1, n*dt]])
    learn_conn = nengo.Connection(
        q, pred_q[0], transform=T,
        # our learning rule acts on activity from n time steps ago
        learning_rule_type=nengo.PES(
            pre_synapse=z**(-n), learning_rate=1e-6))

    # compare the predicted state and the actual state
    compare = nengo.Ensemble(n_neurons=1, dimensions=3,
                             neuron_type=nengo.Direct())
    # the actual state
    nengo.Connection(q[0], compare[0])
    # the adaptive prediction
    nengo.Connection(pred_q, compare[1], synapse=z**(-n))
    # the linear prediction
    nengo.Connection(q, compare[2], transform=T, synapse=z**(-n))
    # send in the error for learning, -(target - actual)
    nengo.Connection(compare, learn_conn.learning_rule,
                     function=lambda x: x[1] - x[0])

    probe = nengo.Probe(compare, synapse=0.01)

sim_time = 100  # in seconds
half_time = int(sim_time / 2)
sim = nengo.Simulator(model)
sim.run(sim_time)

target = sim.data[probe][:, 0]
adaptive = sim.data[probe][:, 1]
linear = sim.data[probe][:, 2]
print('error during second half: ')
print('adaptive estimate: ',
      np.sqrt(np.sum((target[half_time:] - adaptive[half_time:])**2)))
print('linear estimate: ',
      np.sqrt(np.sum((target[half_time:] - linear[half_time:])**2)))

import matplotlib.pyplot as plt
plt.plot(sim.trange(), target, 'r--', lw=3)
plt.plot(sim.trange(), adaptive, lw=2)
plt.plot(sim.trange(), linear, lw=2)
plt.legend(['target', 'adaptive pred', 'linear pred'])
plt.show()
