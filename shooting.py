#shooting
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from odeSolver import *

#initial conditions from isolated periodic orbit
X0 = [0.2,0.2]

def shooting():
    X_solution, t = solve_ode(dX, X0, 0, 100, 0.01, 'rk4')
    plt.plot(t, X_solution)
    plt.show()
    #root-finding
    sol = fsolve(lambda U, f: shoot(f, U), [1,1,18], dX) # need to make own root-finding
    U0 = sol[:-1]
    T = sol[-1]
    print('U0: ', U0)
    print('Period: ',T)
    X_solution, t = solve_ode(dX, U0, 0, T, 0.01, 'rk4')
    plt.plot(t, X_solution)
    plt.show()

#finding residue of integration
def integral_res(method, f, X0, t0, T):
    X_solution, t = solve_ode(dX, X0, t0, T, 0.001, method)
    return X_solution[-1] - X0

#phase condition
def phase(f, X0):
    return np.array([f(X0, 0)[0]])

#singular shot 
def shoot(f, X):
    X0 = X[:2]
    T = X[-1]
    return np.concatenate((integral_res('rk4', dX, X0, 0, T), phase(dX, X0)))

#modelling equations for predator prey equations; can replace with any ODE modelled as series of first order eqns
def dX(X, t):
    a = 1
    d = 0.1
    b = 0.2
    x, y = X
    dx = x*(1-x) - (a*x*y)/(d+x)
    dy = b*y*(1-y/x)
    dX = [dx,dy]
    return np.array(dX)

#newton-raphson method for root-finding
def root_finder(f, XO, dX):
    X = X0
    fX = f(X, t)[1]
    tol = 1e-6
    print(fX)

    for iteration in range(100):
        if abs(fX) < tol:
            return X
    
        fpX = dX(X, t)[1]
        if fpX < tol:
            break
        
        X = X - fX/fpX
        fX = f(X)[1]
    
    return X
            
root_finder(lambda U, f: shoot(f, U), [1,1,18], dX)
        