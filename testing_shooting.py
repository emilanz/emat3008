from shooting import *

#modelling equations for predator prey equations; can replace with any ODE modelled as series of first order eqns
def dX(X, t):
    a = 1
    d = 0.1
    b = 0.2
    x, y = X
    dx = x*(1-x) - (a*x*y)/(d+x)
    dy = b*y*(1-y/x)
    dX = [dx,dy]
    return np.array(dX), t

#shooting(dX, np.array([1,1]), 18)

def exact_U(t):
    Beta = 8
    phase = 0
    u1 = np.sqrt(Beta)*np.cos(t + phase)
    u2 = np.sqrt(Beta)*np.sin(t + phase)
    return np.array([u1, u2])

def dU_2d(U, t):
    u1, u2 = U
    Beta = 8
    sigma = -1
    du1 = Beta*u1 - u2 + sigma*u1*(u1*u1 + u2*u2)
    du2 = u1 + Beta*u2 + sigma*u2*(u1*u1 + u2*u2)
    return np.array([du1, du2])

def dU_3d(U, t):
    u1, u2, u3 = U
    Beta = 8
    sigma = -1
    du1 = Beta*u1 - u2 + sigma*u1*(u1*u1 + u2*u2)
    du2 = u1 + Beta*u2 + sigma*u2*(u1*u1 + u2*u2)
    du3 = -u3
    return np.array([du1, du2, du3])

shooting(dU_3d, [1,1,1], 6, fsolve)
#dU_3d doesnt converge to single oscillation when T0 ~ 8, and coverges to two full oscillations when 

#error testing for accuracy of shooting, passed
# result = shooting(dU_2d, [1,1], 5, fsolve) 
# exact = exact_U(result[1])
# res = np.subtract(result[0], exact)
# if np.all(abs(res<1e-6)):
#     print('Passed test')  #shows results are accurate to resolution of python
# else:
#     print('Test failed')

def test_ode(X):
    x1, x2 = X
    return np.array([x1, 2*x2])

#shooting(dU_2d, [1,1], 5, fsolve)