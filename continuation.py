import numpy
from shooting import *
from odeSolver import *
from math import nan
from scipy.optimize import fsolve

# results = continuation(myode,  # the ODE to use
#     x0,  # the initial state
#     par0,  # the initial parameters
#     vary_par=0,  # the parameter to vary
#     step_size=0.1,  # the size of the steps to take
#     max_steps=100,  # the number of steps to take
#     discretisation=shooting,  # the discretisation to use
#     solver=scipy.optimize.fsolve)  # the solver to use

def continuation(ode, x0, t0, step_size, max_steps, discretisation):
    if discretisation == shooting:
        def func(f, X):
            return np.array(shoot(f, X), dot_product(f, X))
        print(fsolve(lambda U, f: func(f, U), np.append(x0, t0), ode))
    


def dU_3d(U, t):
    u1, u2, u3 = U
    Beta = 8
    sigma = -1
    du1 = Beta*u1 - u2 + sigma*u1*(u1*u1 + u2*u2)
    du2 = u1 + Beta*u2 + sigma*u2*(u1*u1 + u2*u2)
    du3 = -u3
    return np.array([du1, du2, du3])

def dU_2d(U, t):
    print(U)
    u1, u2 = U
    sigma = 0
    Beta = 1
    du1 = Beta*u1 - u2 + sigma*u1*(u1*u1 + u2*u2)
    du2 = u1 + Beta*u2 + sigma*u2*(u1*u1 + u2*u2)
    return np.array([du1, du2])

def alg_cubic(x, t, c):
    f1 = x*x*x - x + c
    return f1

def secant(f, x0):
    t = x0[-1]
    x =  x0[:-1] 
    dx = f(x,t) - x
    return np.array(dx)

def x_tilde(f, x0):
    x =  x0[:-1]
    x_tilde = x + secant(f, x0)
    return np.array(x_tilde)

def dot_product(f, x0):
    x =  x0[:-1]
    return np.vdot(secant(f, x0), (x - x_tilde(f, x0)))

#dot_product(dU_2d, np.array([1,1,5]))

continuation(dU_3d,
    np.array([2,2,2]),
    5,
    0.01,
    max_steps = 200,
    discretisation = shooting)

# par0 = np.array([])
# vary_par = par0[0]