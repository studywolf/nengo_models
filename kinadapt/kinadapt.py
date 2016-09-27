'''
Copyright (C) 2016 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np

import arm

import nengo

import importlib
importlib.reload(arm)

# set the initial position of the arm
dt = 1e-3
arm = arm.two_link(dt=dt)
arm.reset(q=[np.pi/5.5, np.pi/1.7], dq=[0, 0])

model = nengo.Network()
with model:

    # create input nodes
    hand = np.zeros(2)
    trail_data = np.zeros((100, 2))

    def arm_func(t, x):
        global hand, trail_data
        u = x[:2]
        arm.apply_torque(u)
        # data returned from node to model
        data = np.hstack([arm.q0, arm.q1, arm.dq0, arm.dq1, arm.x])
        hand = np.copy(arm.x)

        # visualization code --------------------------------------------------
        scale = 15
        len0 = arm.l1 * scale
        len1 = arm.l2 * scale

        angles = data[:3]
        angle_offset = np.pi/2
        x1 = 50
        y1 = 50
        x2 = x1 + len0 * np.sin(angle_offset - angles[0])
        y2 = y1 - len0 * np.cos(angle_offset - angles[0])
        x3 = x2 + len1 * np.sin(angle_offset - angles[0] - angles[1])
        y3 = y2 - len1 * np.cos(angle_offset - angles[0] - angles[1])
        target_x = x1 + x[2] * scale
        target_y = y1 - x[3] * scale
        # update trail every few steps
        if int(t * 1000) % 5 == 0:
            trail_data[:-1] = trail_data[1:]
            trail_data[-1] = [x3, y3]

        trail = '''xa'''
        for ii in range(trail_data.shape[0] - 1):
            trail += ('<line x1="%f" y1="%f" x2="%f" y2="%f" ' %
                      (trail_data[ii, 0], trail_data[ii, 1],
                       trail_data[ii+1, 0], trail_data[ii+1, 1]) +
                      'style="stroke:black"/>')

        arm_func._nengo_html_ = '''
        <svg width="100%" height="100%" viewbox="0 0 100 100">
            {trail}
            <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"
            style="stroke:black"/>
            <line x1="{x2}" y1="{y2}" x2="{x3}" y2="{y3}"
            style="stroke:black"/>
            <circle cx="{x3}" cy="{y3}" r="1.5" stroke="black"
            stroke-width="1" fill="black" />
            <circle cx="{target_x}" cy="{target_y}" r="1.5"
            stroke="red" stroke-width="1" fill="red" />
        </svg>
        '''.format(**locals())

        # end of visualization code -------------------------------------------

        return data
    arm_node = nengo.Node(output=arm_func, size_in=4, size_out=6)

    task = 'point to point'  # 'circle trace'
    if task == 'circle trace':
        # specify torque input to arm
        target_node = nengo.Node(lambda t: np.array(
            [np.cos(t + np.pi) * .5,
             np.sin(t + np.pi) * .5 + 2]))
    elif task == 'point to point':
        count = 0
        target = np.zeros(2)

        def target_func(t):
            global count, target
            if count % 3000 == 0:
                target = (np.random.random(2) *
                          np.array([3.5, 3]) +
                          np.array([-1.25, 1]))
            count += 1
            return target
        target_node = nengo.Node(output=target_func, size_out=2)

    target_filter = .1
    nengo.Connection(target_node, arm_node[arm.DOF:],
                     synapse=target_filter)

    input_ens = nengo.Ensemble(n_neurons=100, dimensions=arm.DOF)
    kin_adapt = nengo.Ensemble(n_neurons=1, dimensions=arm.DOF,
                               neuron_type=nengo.Direct())
    conn_learn_kin = nengo.Connection(
        input_ens, kin_adapt,
        function=lambda x: np.ones(2),
        learning_rule_type=nengo.PES(learning_rate=1e-3))

    Wk = np.zeros((2, arm.DOF))  # low pass filtered Yk matrix
    y = np.zeros(2)  # estimate of dx
    hand_lp = np.zeros(2)  # low pass filtered hand position
    # parameters from experiment 1 of cheah and slotine, 2005
    kp = 4
    kv = 10
    learn_rate_k = np.diag([0.04, 0.045]) * 1e-2
    learn_rate_d = .0005
    alpha = 1.2
    lamb = 200.0 * np.pi

    def mult_func(t, x):
        q = x[:arm.DOF]
        dq = x[arm.DOF:arm.DOF*2]
        hand = x[arm.DOF*2:arm.DOF*2+2]
        target = x[arm.DOF*2+2:arm.DOF*2+4]
        theta_k = x[arm.DOF*2+4:arm.DOF*3+4]

        # generate Jacobian approximation
        J = np.array([
            [-theta_k[0] * np.sin(q[0]) - theta_k[1] * np.sin(q[0] + q[1]),
             -theta_k[1] * np.sin(q[0] + q[1])],
            [theta_k[0] * np.cos(q[0]) + theta_k[1] * np.cos(q[0] + q[1]),
             theta_k[1] * np.cos(q[0] + q[1])]])

        # y is the estimate of dx calculated in error_func
        u_x = np.dot(kp, (hand - target)) + kv * y

        u = -np.dot(J.T, u_x)
        return u

    mult_node = nengo.Node(mult_func, size_in=10, size_out=arm.DOF)
    # connect up mult_node input
    nengo.Connection(arm_node, mult_node[:arm.DOF*2+2])
    nengo.Connection(target_node, mult_node[arm.DOF*2+2:arm.DOF*2+4],
                     synapse=target_filter)
    nengo.Connection(kin_adapt, mult_node[arm.DOF*2+4:arm.DOF*3+4])
    # connect up mult_node to the arm
    nengo.Connection(mult_node, arm_node[:arm.DOF])

    def error_func(t, x):
        global Wk, y, hand, hand_lp
        q = x[:arm.DOF]
        dq = x[arm.DOF:arm.DOF*2]
        delta_x = x[arm.DOF*2:arm.DOF*2+2]
        theta_k = x[arm.DOF*2+2:arm.DOF*3+2]

        # calculate dx using a low pass filter over x and taking
        # the difference from the current value of x
        hand_lp += (lamb * (hand - hand_lp)) * dt
        y = lamb * (hand - hand_lp)

        Yk = np.array([[-np.sin(q[0]) * dq[0],
                        -np.sin(q[0] + q[1]) * (dq[0] + dq[1])],
                       [np.cos(q[0]) * dq[0],
                        np.cos(q[0] + q[1]) * (dq[0] + dq[1])]])

        Wk += (lamb * (Yk - Wk)) * dt

        dtheta_k = np.dot(learn_rate_k,
                          (-np.dot(Wk.T,
                                   np.dot(kv,
                                          (np.dot(Wk, theta_k) - y))) +
                           np.dot(Yk.T,
                                  np.dot(kp + alpha * kv,
                                         delta_x))))
        dtheta_k[(theta_k + dtheta_k) < .01] = 0.0
        dtheta_k[(theta_k + dtheta_k) > 3] = 0.0
        # print('dtheta_k: ', dtheta_k)
        return dtheta_k

    error_node = nengo.Node(error_func, size_in=arm.DOF*3+2, size_out=arm.DOF)
    # send in q, dq, and x
    nengo.Connection(arm_node[:arm.DOF*2], error_node[:arm.DOF*2])
    nengo.Connection(arm_node[arm.DOF*2:arm.DOF*2+2],
                     error_node[arm.DOF*2:arm.DOF*2+2],
                     transform=-1)
    # subtract target from x to calculate delta_x
    nengo.Connection(target_node, error_node[arm.DOF*2:arm.DOF*2+2],
                     transform=1, synapse=target_filter)
    # send in theta_k
    nengo.Connection(kin_adapt, error_node[arm.DOF*2+2:arm.DOF*3+2])
    # connect error up to learning rule
    nengo.Connection(error_node, conn_learn_kin.learning_rule, transform=-1)

    # dyn_adapt = nengo.Ensemble(n_neurons=1000, dimensions=arm.DOF*2,
    #                            radius=10)
    # nengo.Connection(arm_node[:arm.DOF*2], dyn_adapt)
    # conn_learn_dyn = nengo.Connection(
    #     dyn_adapt, arm_node[:arm.DOF],
    #     function=lambda x: np.zeros(arm.DOF),
    #     learning_rule_type=nengo.PES(learning_rate=1e-4))
    # # connect up mult_node to training dyn_adapt connection
    # nengo.Connection(mult_node, conn_learn_dyn.learning_rule,
    #                  transform=-1)
