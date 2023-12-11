import numpy as np

sigma =  10
rho = 28
beta = 8/3

def get_derivative(pos_vector):
    x, y, z = pos_vector

    dx_dt = sigma*(y-x)
    dy_dt = x*(rho-z)-y
    dz_dt = x*y - beta*z

    return np.array([dx_dt, dy_dt, dz_dt])

def RK4(pos_vector, dt):
    k1 = dt * get_derivative(pos_vector)
    k2 = dt * get_derivative(pos_vector + k1/2)
    k3 = dt * get_derivative(pos_vector+ k2/2)
    k4 = dt * get_derivative(pos_vector + k3)

    return pos_vector + (k1 + 2*k2 + 2*k3 + k4) / 6

def eulers_method(pos_vector, dt):
    delta = dt * get_derivative(pos_vector)
    return pos_vector + delta