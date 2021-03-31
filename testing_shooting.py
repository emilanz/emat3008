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
    return np.array(dX)

shooting(dX, np.array([1,1]), 18)
