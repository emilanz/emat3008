from odeSolver import *
(x, t) = solve_ode(lambda x, t: x, 1, 0, 1, 0.15, 'euler')
(x2, t2) = solve_ode(lambda x, t: x, 1, 0, 1, 0.15, 'rk4')

#plotting approximations
plt.plot(t,x, label = 'Euler')
plt.plot(t2,x2, label = 'Runge-Kutta 4')
plt.ylabel('x')
plt.xlabel('t')
plt.title('Euler and Runge-Kutta 4 approximations')
plt.legend(loc = 'best')
plt.show()

t = np.arange(0.001, 1, 0.001)
(Euler_error, RK4_error) = error_delta_t(t)

#plotting the double log graph of error against delta_t
plt.loglog(t, Euler_error, label = 'Euler error')
plt.loglog(t, RK4_error, label = 'Runge-Kutta 4th Order error')
plt.xlabel('delta_t')
plt.ylabel('percentage error')
plt.title('Double logarithmic graph of error percentage against max delta_t')
plt.legend(loc = 'best')
plt.show()

solve2nd_ode()

X, t = predprey()

#plotting predator-prey against time
plt.plot(t, X[:,0], 'g', label='prey')
plt.plot(t, X[:,1], 'r', label='predator')
plt.legend()
plt.xlabel('time')
plt.show()

#plotting predator against prey
plt.plot(X[:,0], X[:,1])

