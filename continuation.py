import numpy
from shooting import *
from odeSolver import *
from math import nan
from scipy.optimize import fsolve, root
import matplotlib.pyplot as plt

# results = continuation(myode,  # the ODE to use
#     x0,  # the initial state
#     par0,  # the initial parameters
#     vary_par=0,  # the parameter to vary
#     step_size=0.1,  # the size of the steps to take
#     max_steps=100,  # the number of steps to take
#     discretisation=shooting,  # the discretisation to use
#     solver=scipy.optimize.fsolve)  # the solver to use

# defining continuation function
def continuation(ODE, X1, X2, step_size, max_steps, discretisation):
    """
    A function that uses numerical continuation to find limit cycle oscillations
    of an ordinary differential equation.

    Parameters
    ----------
    ODE : function
        The set of first order ODEs that represent the ODE wanting to be solved.
        Must return an numpy.array
    X1 : numpy.array
        First solution of ODE, format [u0, T, param] where T is the period and
        param is the parameter iterated through for parameter continuation.
    X2 : numpy.array
        Second solution of ODE, format [u0, T, param] where T is the period and
        param is the parameter iterated through for parameter continuation.
    param : numpy.array
        Parameter array containing every value for parameter over which to solve.
    Returns
    -------
    Returns a numpy.array containing the corrected initial values for the limit
    cycle oscillation. If numerical root finder fails, will return empty array.
    
    """
    if discretisation == shooting:
        def func(f, x1, x2, x):
            return np.array(dot_product(f, x1, x2))
        # sol = fsolve(lambda U, f: func(f, U), np.append(x0, t0), ODE)
        sol
        print(sol)

# def dU_3d(U, t):
#     u1, u2, u3 = U
#     Beta = 8
#     sigma = -1
#     du1 = Beta*u1 - u2 + sigma*u1*(u1*u1 + u2*u2)
#     du2 = u1 + Beta*u2 + sigma*u2*(u1*u1 + u2*u2)
#     du3 = -u3
#     return np.array([du1, du2, du3])

# def dU_2d(U, t):
#     u1, u2 = U
#     sigma = 0
#     Beta = 1
#     du1 = Beta*u1 - u2 + sigma*u1*(u1*u1 + u2*u2)
#     du2 = u1 + Beta*u2 + sigma*u2*(u1*u1 + u2*u2)
#     return np.array([du1, du2])

def alg_cubic(x, c):
    x = x[0]
    return np.array([x*x*x - x + c])

def secant(X1, X2):
    dX = X2 - X1
    return np.array(dX)

def x_tilde(X1, X2):
    x_tilde = X2 + secant(X1, X2)
    return np.array(x_tilde)

def dot_product(X1, X2):
    return np.vdot(secant(X1, X2), (X1 - x_tilde(X1, X2)))

def cont_alg(f, x1, param):
    c1 = x1[-1]
    x1 = x1[:-1]
    sol1 = fsolve(f, x1, c1)
    sol2 = sol1
    sols = np.array(sol1)
    def pseudo_arc(x, c):
        x2 = fsolve(alg_cubic, x, c)
        c2 = c
        X1 = np.append(x, c)
        X2 = np.append(x2, c2)
        return np.array(f(x, c), dot_product(X1, X2))
    for c in param:
        sol2 = fsolve(pseudo_arc, sol2, c)
        sols = np.append(sols, sol2)
    return sols

# Predict: generate the secant and the prediction from the last two points calculated (or the starting two points)
# Correct: update the predicted solution with the root finder to find the true solution **keeping the secant and original prediction 
# fixed from the prediction step** (the secant and original prediction are fed into the “f” function as additional fixed parameters, 
# they are not calculated within f – only the dot product is calculated within f).
# Repeat.

sols = cont_alg(alg_cubic, np.array([1.52137971,-2]),  np.linspace(-1.98, 2, 200))
plt.plot(np.linspace(-2, 2, 201), sols)
plt.xlabel('alpha')
plt.ylabel('x')
plt.show()






X1 = np.array([1.52137971,-2])
X2 = np.array([1.51800612, -1.98])
# print(dot_product(X1, X2))


def dot_product2(X1, X2):
    x1 = X1[:-1]
    x2 = X2[:-1]
    c1 = X1[-1]
    c2 = X2[-1]
    c = -1.96
    X = x_tilde(x1, x2) - (1/secant(x1, x2))*secant(c1, c2)*(c-x_tilde(c1, c2))
    return X    

print(dot_product2(X1, X2))
# continuation(dU_3d,
#     np.array([2,2,2]),
#     5,
#     0.01,
#     max_steps = 200,
#     discretisation = shooting)

#dot_product(dU_2d, np.array([1,1,5]))
# par0 = np.array([])
# vary_par = par0[0]