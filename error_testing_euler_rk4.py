from odeSolver import *
import numpy as np

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

t = np.arange(0.001, 1, 0.001)
(Euler_error, RK4_error) = error_delta_t(t)

#plotting the double log graph of error against delta_t
plt.loglog(t, Euler_error, c='b', label='Euler error')
plt.loglog(t, RK4_error, c='r', label='Runge-Kutta 4th Order error')
plt.xlabel('maximum delta_t')
plt.ylabel('Percentage Error')
plt.title('Double logarithmic graph of error percentage against max delta_t')
plt.legend(loc = 'best')
plt.show()

