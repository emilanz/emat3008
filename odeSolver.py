import numpy as np 
import matplotlib.pyplot as plt

# step function to for Euler method
def euler_step(f, xn, tn, delta_t):
    xn = xn + f(xn, tn)*delta_t #euler step
    tn += delta_t
    return xn, tn  #returning new x and t values

# step function for Runge-Kutta 4th order
def RK4_step(f, xn, tn, delta_t): 
    k1 = delta_t*f(xn, tn) 
    k2 = delta_t*f(xn + 0.5 * k1 , tn + 0.5 * delta_t) 
    k3 = delta_t*f(xn + 0.5 * k2 , tn + 0.5 * delta_t) 
    k4 = delta_t*f(xn + k3, tn + delta_t) 
   
    xn = xn + (1.0/6.0)*(k1 + 2 * k2 + 2 * k3 + k4) 
    tn += delta_t 
    return xn, tn

#function to do euler steps from t0 to tf
def solve_to(xn, tn, tf, delta_tmax,f, method):
    x = [xn]
    t = [tn]
    #euler or rk4
    if method in ('euler', 'Euler', 'e', 'E'):
        step_function = euler_step
    elif method in ('rk', 'RK', 'r', 'Runge-Kutta 4', 'RK4', 'runge-kutta4', 'rk4'):
        step_function = RK4_step
    
    #looping until x(tf) found
    while tn + delta_tmax < tf: 
        xn, tn = step_function(f, xn, tn, delta_tmax)
        x.append(xn)
        t.append(tn)
    else:         
        diff = tf - tn
        newdelta_tmax = diff
        xn, tn = step_function(f, xn, tn, newdelta_tmax)
        x.append(xn)
        t.append(tn)
        return x, t

#creating function to solve ode using Euler's or RK-4
def solve_ode(f, x0, t0, tf, delta_t, method='rk4', plot: bool=False):
    """
    A function that solves arbitrary ODEs of arbitrary dimension using Euler's 
    method or 4th Order Runge-Kutta.

    Parameters
    ----------
    f : function
        The set of first order ODEs that represent the ODE wanting to be solved.
        Must return a numpy.array
    x0 : numpy.array
        The value of x at t=0 where x is a numpy.array of each representation
        in the set of first order differential equations defining the ODE.
    t0 : float
        The starting t value from which to solve.
    tf : float
        The final t value for which to solve to.
    delta_t: float
        The time step. 
    method: string
        The stepping method to solve with. Either 'Euler' or 'Runge-Kutta'. 
        Default is Runge-Kutta
    plot: Boolean
        Boolean True/False to create basic plot of solution against time. 
        Default is False.    

        
    Returns
    -------
    Returns two arrays containing the values for x and the values for t.
    
    """
    estimation, t = solve_to(x0, t0, tf, delta_t, f, method)
    estimation = np.array(estimation)
    if plot==True:
        plt.plot(t,estimation)
        plt.xlabel('t')
        plt.ylabel('X')
        plt.show()
    return estimation, t


