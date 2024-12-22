# 基于casadi实现一个简单的避障运动规划
# 随意生成一组初始解，然后用MPC来优化
TRAJ_NUM = 50
dt = 0.1
init_state = [0 ,0 ,5, 0] # 定义初始速度为5m/s
L = 3.089  # 轴距

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
import casadi.tools as ca_tools

import numpy as np
import time



def bezier_curve(p0, p1, p2, p3, t):
    """
    计算三次贝塞尔曲线上的点
    p0, p1, p2, p3 是控制点
    t 是参数，范围在 [0, 1]
    """
    return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3

def bezier_curve_points(p0, p1, p2, p3, num_points=TRAJ_NUM + 1):
    """
    生成三次贝塞尔曲线上的多个点
    p0, p1, p2, p3 是控制点
    num_points 是生成的点的数量
    """
    t_values = np.linspace(0, 1, num_points)
    curve_points = np.zeros((num_points, 2))
    
    for i, t in enumerate(t_values):
        curve_points[i] = bezier_curve(p0, p1, p2, p3, t)
    
    return curve_points


def generate_coarse_traj():
    p0 = np.array([0, 0])
    p1 = np.array([10, 0])
    p2 = np.array([10, 3])
    p3 = np.array([20, 3])
    curve_points = bezier_curve_points(p0, p1, p2, p3)
    return curve_points


if __name__ == '__main__':
    N = TRAJ_NUM
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    theta = ca.SX.sym('theta')
    v = ca.SX.sym('v')
    states = ca.vertcat([x,y,theta,v])
    n_states = states.size()[0]

   
    steer = ca.SX.sym('steer')
    acc = ca.SX.sym('acc')
    controls = ca.vertcat([steer,acc])
    n_controls = controls.size()[0]

    ## rhs
    rhs = ca.vertcat([v * ca.cos(theta), v * ca.sin(theta), v * ca.tan(steer) / L, acc])

    ## function
    f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

    ## for MPC
    U = ca.SX.sym('U', n_controls, N)
    X = ca.SX.sym('X', n_states, N+1)
    P = ca.SX.sym('P', 4 + 2 * N ) # 初始状态 x_ref y _ref


    ### define
    X[:, 0] = P[:4] # initial condiction

    #### define the relationship within the horizon
    for i in range(N):
        f_value = f(X[:, i], U[:, i])
        X[:, i+1] = X[:, i] + f_value * dt

    ff = ca.Function('ff', [U, P], [X], ['input_U', 'target_state'], ['horizon_states'])

    Q = np.array([[1.0, 0.0, 0.0, 0.0],[0.0, 5.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, ]])
    R = np.array([[0.5, 0.0], [0.0, 0.05]])
    #### cost function
    obj = 0 #### cost
    for i in range(N):
        # obj = obj + ca.mtimes([(X[:, i]-P[3:]).T, Q, X[:, i]-P[3:]]) + ca.mtimes([U[:, i].T, R, U[:, i]])
        # new type to calculate the matrix multiplication
        obj = obj + (X[:, i]-P[3:]).T @ Q @ (X[:, i]-P[3:]) + U[:, i].T @ R @ U[:, i]

    #### constrains
    g = [] # equal constrains
    for i in range(N+1):
        g.append(X[0, i])
        g.append(X[1, i])

    nlp_prob = {'f': obj, 'x': ca.reshape(U, -1, 1), 'p':P, 'g':ca.vcat(g)} # here also can use ca.vcat(g) or ca.vertcat(*g)
    opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6, }

    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)


    # Simulation
    # lbg = -2.0
    # ubg = 2.0
    # lbx = []
    # ubx = []
    # for _ in range(N):
    #     lbx.append(-v_max)
    #     ubx.append(v_max)
    #     lbx.append(-omega_max)
    #     ubx.append(omega_max)
    t0 = 0.0
    x0 = np.array([0.0, 0.0, 0.0]).reshape(-1, 1)# initial state
    xs = np.array([1.5, 1.5, 0.0]).reshape(-1, 1) # final state
    u0 = np.array([0.0, 0.0]*N).reshape(-1, 2)# np.ones((N, 2)) # controls
    x_c = [] # contains for the history of the state
    u_c = []
    t_c = [] # for the time
    xx = []
    sim_time = 20.0

    ## start MPC
    mpciter = 0
    start_time = time.time()
    index_t = []
    c_p = np.concatenate((x0, xs))
    init_control = ca.reshape(u0, -1, 1)
    res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
    lam_x_ = res['lam_x']
    ### inital test
    while(np.linalg.norm(x0-xs)>1e-2 and mpciter-sim_time/T<0.0 ):
        ## set parameter
        c_p = np.concatenate((x0, xs))
        init_control = ca.reshape(u0, -1, 1)
        t_ = time.time()
        res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx, lam_x0=lam_x_)
        lam_x_ = res['lam_x']
        # res = solver(x0=init_control, p=c_p,)
        # print(res['g'])
        index_t.append(time.time()- t_)
        u_sol = ca.reshape(res['x'], n_controls, N) # one can only have this shape of the output
        ff_value = ff(u_sol, c_p) # [n_states, N+1]
        x_c.append(ff_value)
        u_c.append(u_sol[:, 0])
        t_c.append(t0)
        t0, x0, u0 = shift_movement(T, t0, x0, u_sol, f)

        x0 = ca.reshape(x0, -1, 1)
        xx.append(x0.full())
        mpciter = mpciter + 1
    t_v = np.array(index_t)
    print(t_v.mean())
    print((time.time() - start_time)/(mpciter))
    draw_result = Draw_MPC_point_stabilization_v1(rob_diam=0.3, init_state=x0.full(), target_state=xs, robot_states=xx, export_fig=False)

