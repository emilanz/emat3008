#importing necessary modules
import numpy as np
import matplotlib.pyplot as plt

#function to make single euler step
def euler_step(f, xn, tn, delta_t):
    '''f is the right hand side of first order ODE x_dot = f 
       xn is the inital x value
       t is the timestep
       tn is initial t value'''
    xn = xn + f(xn, tn)*delta_t #euler step
    tn += delta_t
    return xn, tn  #returning new x and t values

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
def solve_ode(f, x0, t0, tf, delta_t, method):
    estimation, t = solve_to(x0,t0,tf,delta_t,f, method)
    return estimation, t

#step function for Runge-Kuta 4th order
def RK4_step(f, xn, tn, delta_t): 
    k1 = delta_t*f(xn, tn) 
    k2 = delta_t*f(xn + 0.5 * k1 , tn + 0.5 * delta_t) 
    k3 = delta_t*f(xn + 0.5 * k2 , tn + 0.5 * delta_t) 
    k4 = delta_t*f(xn + k3, tn + delta_t) 
   
    xn = xn + (1.0/6.0)*(k1 + 2 * k2 + 2 * k3 + k4) 
    tn += delta_t 
    return xn, tn

#creating main() function that can read user inputs in from command line and call solve_ode
def main():
    if __name__ == "__main__":
        x0 = float(input('What is your initial x value?'))
        t0 = float(input('What is your initial t value?'))
        tf = float(input('What is your final t value?'))
        delta_t = float(input('What is your timestep?'))
        method = str(input('Would you like to use Euler method or Runge-Kutta 4?'))
        solve_ode(lambda x,t: x, x0, t0, tf, delta_t, method)    


#function to create timestep error plot of ode method for both euler and rk4
def error_delta_t(deltat_values):
    errors_euler = [] 
    errors_rk4 = []
    #looping through delta_t values
    for delta_t in deltat_values: 
        (xe, te) = solve_ode(lambda x,t: x, 1, 0, 1, delta_t, 'euler')
        error_euler = (np.exp(te[-1]) - xe[-1])/np.exp(te[-1]) * 100   #
        errors_euler.append(error_euler)  #appending error array with every error
    for delta_t in deltat_values: 
        (x_rk4, t_rk4) = solve_ode(lambda x,t: x, 1, 0, 1, delta_t, 'rk4')
        error_rk4 = (np.exp(t_rk4[-1]) - x_rk4[-1])/np.exp(t_rk4[-1]) * 100   #
        errors_rk4.append(error_rk4)  #appending error array with every error
    return errors_euler, errors_rk4

def solve2nd_ode(): 
    
    def dXdt(X,t):
        x,y = X
        dx = y
        dy = -x
        dX = [dx, dy]
        return np.array(dX)

    x0 = 0
    y0 = 1
    X0 = [x0, y0]

    # Actually compute the solution: 
    X_solution = solve_ode(dXdt, X0, 0, 1, 0.3, 'e')

    return X_solution

#function to solve predator prey eqns 
def predprey():
    #defining variables
    a = 1
    d = 0.1
    b = 0.2
    
    x0 = 0.52
    y0 = 0.35
    X0 = [x0, y0]
    #modelling equations
    def dX(X,t):
        x,y = X
        dx = x*(1-x) - (a*x*y)/(d+x)
        dy = b*y*(1-y/x)
        dX = [dx,dy]
        return np.array(dX)
    #solving
    X_solution, t = solve_ode(dX, X0, 0, 100, 0.01, 'rk4')
    X = np.array(X_solution)
    return X, t
