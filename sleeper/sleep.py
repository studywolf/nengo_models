import nengo
import numpy as np
    
model = nengo.Network()
with model:
    
    driving_input = nengo.Node(output=np.sin)
    des_x = nengo.Ensemble(n_neurons=100, dimensions=1)
    
    # create our 'plant'
    def plant(t,x):
        return -x
    plant = nengo.Node(output=plant, size_in=1, size_out=1)
    def mode_switching(t):
        if (t % 100) < 50:
            return 0
        return 1
    BG_mode = nengo.Node(output=mode_switching)
    
    predicted_plant_output = nengo.Ensemble(n_neurons=100, dimensions=1)
    predicted_des_x = nengo.Ensemble(n_neurons=100, dimensions=1)
    
    def router_func(t,x):
        # input is 
        # 0: BG mode
        # 1: driving_input 
        # 2: learn_pop3 (from des_x)
        # 3: plant
        # 4: predicted_plant_output
        # 5: predicted_des_x

        # output is 
        # 0: des_x
        # 1: plant
        # 2: learn_pop1 (to predicted_plant_output)
        # 3: learn_conn1.learning_rule
        # 4: learn_pop2 (to predicted_des_x)
        # 5: learn_conn2.learning_rule
        # 6: predicted_plant_output
        # 7: learn_conn3.learning_rule
        if abs(x[0]) < .1: 
            return np.hstack([x[1], # driving_input to des_x
                              x[2], # generated_u (des_x) to plant
                              x[2], # generated_u (des_x) to learn_pop1 (predicted_plant_output)
                              x[4] - x[3], # predicted_plant_output - plant to learn_conn1
                              x[4], # predicted_plant_output to learn_pop2 (predicted_des_x)
                              x[5] - x[2], # predicted_des_x - des_x to learn_conn2
                              np.zeros(2), # 0 to the rest 
                              ])
        return np.hstack([x[4], # predicted_plant_output to des_x
                          0.0, # 0 to the plant
                          0.0, # 0 to predicted_plant_output through its learned connection
                          0.0, # 0 to learn_conn1 
                          x[4], # predicted_plant_output to predicted_des_x
                          0.0, # 0 to learn_conn2
                          x[1], # driving_input to predicted_plant_output
                          x[2] - x[5], # generated_u - predicted_des_x to learn_conn3
                          ])
    BG = nengo.Node(output=router_func, size_in=11, size_out=8)

    learn_pop1 = nengo.Ensemble(n_neurons=100, dimensions=1)
    learn_conn1 = nengo.Connection(learn_pop1, predicted_plant_output,
                                  learning_rule_type=nengo.PES(learning_rate=1e-5))
    learn_pop2 = nengo.Ensemble(n_neurons=100, dimensions=1)
    learn_conn2 = nengo.Connection(learn_pop2, predicted_des_x,
                                   learning_rule_type=nengo.PES(learning_rate=1e-5))
    generated_u = nengo.Ensemble(n_neurons=100, dimensions=1)
    learn_conn3 = nengo.Connection(des_x, generated_u, 
                                   learning_rule_type=nengo.PES(learning_rate=1e-5))

    # BG inputs
    nengo.Connection(BG_mode, BG[0])
    nengo.Connection(driving_input, BG[1])
    nengo.Connection(generated_u, BG[2])
    nengo.Connection(plant, BG[3])
    nengo.Connection(predicted_plant_output, BG[4])
    nengo.Connection(predicted_des_x, BG[5])
    # BG outputs
    nengo.Connection(BG[0], des_x)
    nengo.Connection(BG[1], plant)
    nengo.Connection(BG[2], learn_pop1)
    nengo.Connection(BG[3], learn_conn1.learning_rule)
    nengo.Connection(BG[4], learn_pop2)
    nengo.Connection(BG[5], learn_conn2.learning_rule)
    nengo.Connection(BG[6], predicted_plant_output)
    nengo.Connection(BG[7], learn_conn3.learning_rule)

#sim = nengo.Simulator(model)
