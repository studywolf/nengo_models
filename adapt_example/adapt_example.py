import nengo
import numpy as np

def generate_model(targets, adaptation=False):

    model = nengo.Network()
    with model:

        model.target_index = 0

        model.gain = 1.0
        model.state = 0.0
        def state_func(t, u):
            # set up our system's state, it's just a basic
            # d_state = input, but with a scaling term that reduces the
            # strength of the control signal input as time moves forward
            if t > 0:
                model.gain -= 1e-4 * model.gain
            model.state += u * model.gain
            return model.state
        state_node = nengo.Node(state_func, size_in=1)

        # create a node that defines the target state
        model.count = 0
        model.target = .5 # initial target
        def target_func(t):
            if model.count % 3000 == 0:
                model.target = targets[model.target_index]
                model.target_index += 1
            model.count += 1
            return model.target
        target_node = nengo.Node(output=target_func)

        kp = 1e-2
        def control_func(t, x):
            # implement basic P controller
            model.state = x[0]
            model.target = x[1]
            return kp * (model.target - model.state)
        controller = nengo.Node(output=control_func, size_in=2)

        # set up input for the controller
        nengo.Connection(state_node, controller[0])
        nengo.Connection(target_node, controller[1])
        # set up connection from controller to state
        nengo.Connection(controller, state_node)

        # for target / state comparison
        output = nengo.Ensemble(n_neurons=1, dimensions=2,
                                neuron_type=nengo.Direct())
        nengo.Connection(state_node, output[0])
        nengo.Connection(target_node, output[1])

        # set up probes for recording data
        model.probe_output = nengo.Probe(output)
        model.probe_control = nengo.Probe(controller)

        if adaptation == True:
            # adaptive component
            adapt_ens = nengo.Ensemble(n_neurons=100, dimensions=1)
            nengo.Connection(state_node, adapt_ens)
            conn_learn = nengo.Connection(adapt_ens, state_node,
                                        function=lambda x: 0.0,
                                        learning_rule_type=nengo.PES(
                                            learning_rate=1e-4))
            nengo.Connection(controller, conn_learn.learning_rule,
                            transform=-1)

            model.probe_adapt = nengo.Probe(adapt_ens, synapse=.02)

    return model

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    targets = np.random.random(1000) * 4 - 2

    sim_time=20

    # run without adaptation
    model_wo = generate_model(targets=targets, adaptation=False)
    sim_wo = nengo.Simulator(model_wo)
    sim_wo.run(sim_time)
    state_wo = sim_wo.data[model_wo.probe_output][:, 0]
    target_wo = sim_wo.data[model_wo.probe_output][:, 1]
    control_wo = sim_wo.data[model_wo.probe_control]

    # run with adaptation
    model_w = generate_model(targets=targets, adaptation=True)
    sim_w = nengo.Simulator(model_w)
    sim_w.run(sim_time)
    state_w = sim_w.data[model_w.probe_output][:, 0]
    target_w = sim_w.data[model_w.probe_output][:, 1]
    control_w = sim_w.data[model_w.probe_control]
    adapt_w = sim_w.data[model_w.probe_adapt]

    # calculate rse
    print('total rse no adaptation: ', np.sqrt(np.sum((target_wo - state_wo)**2)))
    print('total rse w adaptation: ', np.sqrt(np.sum((target_w - state_w)**2)))

    t = sim_w.trange()
    plt.figure(figsize=(10, 10))
    plt.subplot(4, 1, 1)
    plt.plot(t, state_wo)
    plt.plot(t, state_w)
    plt.plot(t, target_wo)
    plt.legend(['no adaptation', 'adaptation', 'target'],
               bbox_to_anchor=[.25, 1])
    plt.ylim([-1, 1])
    plt.ylabel('state')
    plt.xlabel('time (s)')

    plt.subplot(4, 1, 2)
    plt.plot(t, target_wo - state_wo)
    plt.plot(t, target_w - state_w)
    plt.legend(['no adaptation', 'adaptation'],
               bbox_to_anchor=[.25, 1])
    plt.ylim([-1, 1])
    plt.ylabel('error')
    plt.xlabel('time (s)')

    plt.subplot(4, 1, 3)
    plt.plot(t, control_wo)
    plt.plot(t, control_w)
    plt.legend(['no adaptation', 'adaptation'],
               bbox_to_anchor=[.25, 1])
    plt.ylabel('control signal')
    plt.xlabel('time (s)')

    plt.subplot(4, 1, 4)
    plt.plot(t, adapt_w)
    plt.legend(['adaptive output'],
               bbox_to_anchor=[.25, 1])
    plt.xlabel('time (s)')

    plt.tight_layout()
    plt.show()
