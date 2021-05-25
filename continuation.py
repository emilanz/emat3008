from ODEs import predprey
from algebraic_equations import alg_cubic
import numpy as np
from shooting import *
from odeSolver import *
from math import nan
from scipy.optimize import fsolve, root
import matplotlib.pyplot as plt

# for error handling 
class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """Exception raised for input errors. Specifically discretisation."""

# defining continuation function
def continuation(F, x1, param, discretisation=''):
    """
    A function that uses numerical continuation to find limit cycle oscillations
    of an ordinary differential equation.

    Parameters
    ----------
    ODE : function
        The set of first order ODEs that represent the ODE wanting to be solved.
        Must return an numpy.array
    x1 : numpy.array
        First solution of ODE, format [u0, T, param] where T is the period and
        param is the parameter iterated through for parameter continuation.
    param : nump.array
        List of parameter values through which to iterate for new solutions.
    discretisation : string
        String input to determine how to discretise solutions. Default is set to
        none, i.e using F to discretise it. Alternative is shooting.
    
    Returns 
    -------
    Returns a numpy.array containing a family of solutions to the family of 
    parameter values. 
    
    """
#   Def shooting(f, u, p):
#        ...
 
#   Def pseudoarclength(f, u):
#       Pseudo = ...
#       Shoot_eqn = shooting(f, u[:-1], u[-1])
#       Return np.concatenate((shoot_eqn, pseudo))
    if discretisation == 'shooting':
        sol1 = fsolve(lambda U, f: shoot(f, U), x1, F)
        print(sol1)
        print('x: ', x)
        def pseudo(f, x):
            print('pseudo: ', np.array(shoot(f, X1), dot_product(X1, X2, x)))
            return np.array(shoot(f, X1), dot_product(X1, X2, x))
        sol1 = fsolve(lambda U, f: pseudo(f, U), x, F)
        # sol2 = sol1
        # sols = np.array(sol1)
        # for p in param:
        #     sol1 = fsolve(lambda U, f: shoot(f, U), x1, F)
        #     sols = np.append(sols, sol2)
    elif discretisation == 'algebraic':
        c1 = x1[-1]
        x1 = x1[:-1]
        sol1 = fsolve(F, x1, c1)
        sol2 = sol1
        sols = np.array(sol1)
        def pseudo_arc(x, c):  #trying pseudo arclength for algebraic eqns
            x2 = fsolve(F, x, c)
            c2 = c
            X1 = np.append(x, c)
            X2 = np.append(x2, c2)
            return np.array(dot_product(X1, X2))
        for c in param:
            sol2 = fsolve(F, sol2, c)
            sols = np.append(sols, sol2)
        return sols
    else: 
        raise InputError('You have not selected a method of discretisation. Please type a string: "shooting" or "algebraic" as your input.')

def secant(X1, X2):
    dX = X2 - X1
    return np.array(dX)

def x_tilde(X1, X2):
    x_tilde = X2 + secant(X1, X2)
    return np.array(x_tilde)

def dot_product(X1, X2, x):
    return np.vdot(secant(X1, X2), (x - x_tilde(X1, X2)))

# Predict: generate the secant and the prediction from the last two points calculated (or the starting two points)
# Correct: update the predicted solution with the root finder to find the true solution **keeping the secant and original prediction 
# fixed from the prediction step** (the secant and original prediction are fed into the “f” function as additional fixed parameters, 
# they are not calculated within f – only the dot product is calculated within f).
# Repeat.
