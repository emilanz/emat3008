#shooting
from ODEs import Hop_bif_2D, Hop_bif_3D
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from odeSolver import *

def shooting(ODE, X0, T0, solver, plot: bool=False):
    """
    A function that uses numerical shooting to find limit cycle oscillations of 
    any given ODE.

    Parameters
    ----------
    ODE : function
        The set of first order ODEs that represent the ODE wanting to be solved.
        Must return an numpy.array
    X0 : numpy.array
        The initial guess of the initial values for the limit cycle oscillation.
    T0 : 
        The initial guess of the initial period of oscillation.
    plot : Boolean
        If plot=True then function will plot isolated limit cycle.

        
    Returns
    -------
    Returns a numpy.array containing the corrected initial values for the limit
    cycle oscillation. If numerical root finder fails, will return empty array.
    
    """
    #if you want to see oscillations with input params, uncomment these 3 line
    # X_solution, t = solve_ode(ODE, X0, 0, 100, 0.01, 'rk4')
    # plt.plot(t, X_solution)
    # plt.show()
    
    # checking dimensions of initial conditions match ODE
    try: ODE(X0, T0).shape == X0.shape
    except: raise(ValueError('Inital values (X0) do not match the dimensions of your ODE'))
    
    
    #need to write script that checks convergence and isolation of limit cycle oscillation
    
    #root-finding
    sol = solver(lambda U, f: shoot(f, U), np.append(X0,T0), ODE) #need to make own root-finding
    U0 = sol[:-1]
    T = sol[-1]
    print('U0: ', U0)
    print('Period: ',T)
    X_solution, t = solve_ode(ODE, U0, 0, T, 0.01, 'rk4')
    if plot == True:
        plt.plot(t, X_solution)
        plt.xlabel('t')
        plt.ylabel('x')
        plt.show()
    return X_solution, t

#finding residue of integration
def integral_res(method, f, X0, t0, T):
    X_solution, t = solve_ode(f, X0, t0, T, 0.001, method)
    # print('res: ', X_solution[-1] - X0)
    return X_solution[-1] - X0

#phase condition
def phase(f, X0):
    # print('phase: ', np.array([f(X0, 0)[0]]))
    return np.array([f(X0, 0)[0]]) #can change index for which to set derivative equal to zero.

#singular shot
def shoot(f, X):
    X0 = X[:-1]
    T = X[-1]
    return np.concatenate((integral_res('rk4', f, X0, 0, T), phase(f, X0)))
