import numpy as np 

def dXdt(X,t):
    x,y = X
    dx = y
    dy = -x
    dX = [dx, dy]
    return np.array(dX)


# predator prey
def predprey(X, t):
    #defining variables
    a = 1
    d = 0.1
    b = 0.2
    x,y = X
    dx = x*(1-x) - (a*x*y)/(d+x)
    dy = b*y*(1-y/x)
    dX = [dx,dy]
    return np.array(dX)

def Hop_bif_2D(U, t):
    u1, u2 = U
    Beta = 2
    sigma = -1
    du1 = Beta*u1 - u2 + sigma*u1*(u1*u1 + u2*u2)
    du2 = u1 + Beta*u2 + sigma*u2*(u1*u1 + u2*u2)
    return np.array([du1, du2])

def Hop_bif_3D(U, t):
    u1, u2, u3 = U
    Beta = 2
    sigma = -1
    du1 = Beta*u1 - u2 + sigma*u1*(u1*u1 + u2*u2)
    du2 = u1 + Beta*u2 + sigma*u2*(u1*u1 + u2*u2)
    du3 = -u3
    return np.array([du1, du2, du3])
