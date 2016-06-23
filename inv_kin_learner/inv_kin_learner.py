import numpy as np
import matplotlib as plt

import nengo

import arm; reload(arm)

arm = arm.Arm2Link()
arm.reset(q=[np.pi/5.5, np.pi/1.7], dq=[0, 0])

model = nengo.Network()
with model: 
     # create input nodes
    def arm_func(t, x):
        arm.q0 = x[0]
        arm.q1 = x[1]

        data = np.hstack([arm.q0, arm.q1, arm.x]) # data returned from node to model

        # visualization code -----------------------------------------------------
        scale = 30

        target_x = x[2] * scale + 50
        target_y = 100 - x[3] * scale

        len0 = arm.l1 * scale
        len1 = arm.l2 * scale
        
        angles = data[:3]
        angle_offset = np.pi/2
        x1 = 50
        y1 = 100
        x2 = x1 + len0 * np.sin(angle_offset-angles[0])
        y2 = y1 - len0 * np.cos(angle_offset-angles[0])
        x3 = x2 + len1 * np.sin(angle_offset-angles[0] - angles[1])
        y3 = y2 - len1 * np.cos(angle_offset-angles[0] - angles[1])

        arm_func._nengo_html_ = '''
        <svg width="100%" height="100%" viewbox="0 0 100 100">
            <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" style="stroke:black"/>
            <line x1="{x2}" y1="{y2}" x2="{x3}" y2="{y3}" style="stroke:black"/>
            <circle cx="{x3}" cy="{y3}" r="1.5" stroke="black" stroke-width="1" fill="black" />
            <circle cx="{target_x}" cy="{target_y}" r="1.5" stroke="red" stroke-width="1" fill="black" />
        </svg>
        '''.format(**locals())
        # end of visualization code ---------------------------------------------

        return data
    arm_node = nengo.Node(output=arm_func, size_in=4)

    xy_node = nengo.Node(output=[0, 1]) 
    ens_xy_to_angles = nengo.Ensemble(n_neurons=2000, dimensions=2, radius=4)
    nengo.Connection(xy_node, ens_xy_to_angles)

    # ens_answer = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
    # nengo.Connection(ens_answer, arm_node[:2],
    #         function=arm.inv_kinematics)
    # nengo.Connection(xy_node, ens_answer)

    nengo.Connection(ens_xy_to_angles, arm_node[:2], 
            function=arm.inv_kinematics, synapse=1)

    nengo.Connection(xy_node, arm_node[2:])

