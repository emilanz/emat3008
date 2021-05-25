import numpy as np
from odeSolver import *
from ODEs import *


# solving for both euler and rk4 to compare both methods
(x, t) = solve_ode(lambda x, t: x, 1, 0, 1, 0.05, 'euler')
(x2, t2) = solve_ode(lambda x, t: x, 1, 0, 1, 0.05)

# exact answer
t_exact = np.linspace(0,1,1000)
x_exact = np.exp(t_exact)

#plotting approximations
plt.plot(t_exact, x_exact, c='g', label='Exact')
plt.plot(t,x, c='b', label = 'Euler')
plt.plot(t2,x2, c='r', label = 'Runge-Kutta 4')
plt.ylabel('x')
plt.xlabel('t')
plt.title('Euler and Runge-Kutta 4 approximations')
plt.legend(loc = 'best')
plt.show()


#function to solve predator prey eqns 
def solve_predprey():
    #initial conditions
    X0 = [0.52, 0.35]
    
    #solving
    X, t = solve_ode(predprey, X0, 0, 100, 0.01, 'rk4', plot=True)
    return X, t

def hopf_bif():
    U0 = [1,1,1]
    X, t = solve_ode(Hop_bif_3D, U0, 0, 20, 0.01, 'rk4')
    return X, t



X, t = solve_predprey()
#plotting predator-prey against time
plt.plot(t, X[:,0], 'g', label='prey')
plt.plot(t, X[:,1], 'r', label='predator')
plt.legend()
plt.xlabel('time')
plt.ylabel('population')
plt.show()

# #plotting predator against prey
# plt.plot(X[:,0], X[:,1])

X, t = hopf_bif()
#plotting predator-prey against time
plt.plot(t, X[:,0], 'g', label='u1')
plt.plot(t, X[:,1], 'r', label='u2')
plt.plot(t, X[:,2], 'b', label='u3')
plt.legend()
plt.xlabel('time')
plt.ylabel('U')
plt.show()

#plotting predator against prey
plt.plot(X[:,0], X[:,1])
