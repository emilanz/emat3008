# simple forward Euler solver for the 1D heat equation
#   u_t = kappa u_xx  0<x<L, 0<t<T
# with zero-temperature boundary conditions
#   u=0 at x=0,L, t>0
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0

import numpy as np
from numpy import linalg
from numpy.core.fromnumeric import transpose
from numpy.ma import flatten_mask
import pylab as pl
from math import pi


# Set problem parameters/functions
kappa = 1.0   # diffusion constant
L=1.0         # length of spatial domain
T=0.5         # total time to solve for

# Set numerical parameters
mx = 10     # number of gridpoints in space
mt = 1000   # number of gridpoints in time

# Set up the numerical environment variables
x = np.linspace(0, L, mx+1)     # mesh points in space
t = np.linspace(0, T, mt+1)     # mesh points in time
deltax = x[1] - x[0]            # gridspacing in x
deltat = t[1] - t[0]            # gridspacing in t
lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
print("deltax=",deltax)
print("deltat=",deltat)
print("lambda=",lmbda)

# Set up the solution variables
u_j = np.zeros(x.size)        # u at current time step
u_jp1 = np.zeros(x.size)      # u at next time step


def main():
    # Forward Euler
    u = FE_neumann(u_j, mx, mt, lmbda)
    print(u)
    # Plot the final result and exact solution
    pl.plot(x,u,'ro',label='num')
    xx = np.linspace(0,L,250)
    pl.plot(xx,u_exact(xx,T),'b-',label='exact')
    pl.xlabel('x')
    pl.ylabel('u(x,0.5)')
    pl.legend(loc='upper right')
    pl.show()
    
def u_I(x):
    y = np.sin(pi*x/L)
    return y

def u_exact(x,t):
    # the exact solution
    kappa = 1.0   # diffusion constant     
    L=1.0    # length of spatial domain
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

def FE(u_j, mx, mt, lmbda):
    # Creating matrix 
    A_FE = np.zeros(shape=(mx-1, mx-1))
    np.fill_diagonal(A_FE, 1-2*lmbda)
    np.fill_diagonal(A_FE[1:], lmbda)
    np.fill_diagonal(A_FE[:,1:], lmbda)

    # Set initial condition
    for i in range(0, mx+1):
        u_j[i] = u_I(x[i])

    # Solve the PDE: matrix multiplications
    for i in range(0, mt):
        u_jp1[1:-1] = np.matmul(A_FE, transpose(u_j[1:-1]))
        
        # Boundary conditions
        u_jp1[0] = 0; u_jp1[mx] = 0

        # Save u_j at time t[j+1]
        u_j = u_jp1
    return u_j

def FE_dirichlet(u_j, mx, mt, lmbda):
    # Creating matrix 
    A_FE = np.zeros(shape=(mx-1, mx-1))
    np.fill_diagonal(A_FE, 1-2*lmbda)
    np.fill_diagonal(A_FE[1:], lmbda)
    np.fill_diagonal(A_FE[:,1:], lmbda)

    # Set initial condition
    for i in range(0, mx+1):
        u_j[i] = u_I(x[i])

    # setting boundary conditions, constant for now. can change to function
    p_j = 0.0002
    q_j = 0.0003
    bound = np.zeros(shape=(9,1))
    bound[0] = p_j
    bound[-1] = q_j
    # Solve the PDE: matrix multiplications
    for i in range(0, mt):
        u_jp1[1:-1] = np.matmul(A_FE, transpose(u_j[1:-1])) + lmbda*transpose(bound)
        
        # Boundary conditions
        # u_jp1[0] = p_j ; u_jp1[mx] = q_j

        # Save u_j at time t[j+1]
        u_j = u_jp1
    return u_j

def FE_neumann(u_j, mx, mt, lmbda):
    # Creating matrix 
    A_FE = np.zeros(shape=(mx+1, mx+1))
    np.fill_diagonal(A_FE, 1-2*lmbda)
    np.fill_diagonal(A_FE[1:], lmbda)
    np.fill_diagonal(A_FE[:,1:], lmbda)

    # Set initial condition
    for i in range(0, mx+1):
        u_j[i] = u_I(x[i])

    # setting boundary conditions, constant for now. can change to function
    P_j = 0.0002
    Q_j = 0.0003
    bound = np.zeros(shape=(11,1))
    bound[0] = -P_j
    bound[-1] = Q_j
    bound = transpose(bound)

    # Solve the PDE: matrix multiplications
    for i in range(0, mt):
        u_jp1 = np.matmul(A_FE, transpose(u_j)) + 2*deltax*lmbda*bound
        
        # Save u_j at time t[j+1]
        u_j = u_jp1[-1]

        # Boundary conditions
        u_j[0] = P_j ; u_j[mx] = Q_j
    return u_j

def BE(u_j, mx, mt, lmbda):
    # Creating matrix 
    A_FE = np.zeros(shape=(mx-1, mx-1))
    np.fill_diagonal(A_FE, 1+2*lmbda)
    np.fill_diagonal(A_FE[1:], -lmbda)
    np.fill_diagonal(A_FE[:,1:], -lmbda)

    # Set up the solution variables
    u_j = np.zeros(x.size)        # u at current time step
    u_jp1 = np.zeros(x.size)      # u at next time step

    # Set initial condition
    for i in range(0, mx+1):
        u_j[i] = u_I(x[i])

    # Solve the PDE: matrix multiplications
    for i in range(0, mt):
        u_jp1[1:-1] = linalg.solve(A_FE, u_j[1:-1])
    
        # Save u_j at time t[j+1]
        u_j = u_jp1 

    return u_j

def Crank_Nicholson(u_j, mx, mt, lmbda):
    # Creating matrix 
    A_cn = np.zeros(shape=(mx-1, mx-1))
    np.fill_diagonal(A_cn, 1+lmbda)
    np.fill_diagonal(A_cn[1:], -lmbda/2)
    np.fill_diagonal(A_cn[:,1:], -lmbda/2)

    B_cn = np.zeros(shape=(mx-1, mx-1))
    np.fill_diagonal(B_cn, 1-lmbda)
    np.fill_diagonal(B_cn[1:], lmbda/2)
    np.fill_diagonal(B_cn[:,1:], lmbda/2)

    # Set up the solution variables
    u_j = np.zeros(x.size)        # u at current time step
    u_jp1 = np.zeros(x.size)      # u at next time step

    # Set initial condition
    for i in range(0, mx+1):
        u_j[i] = u_I(x[i])

    # Solve the PDE: matrix multiplications
    for i in range(0, mt):
        u_jp1[1:-1] = linalg.solve(A_cn, np.matmul(B_cn, transpose(u_j[1:-1])))
    
        # Save u_j at time t[j+1]
        u_j = u_jp1 

    return u_j


if __name__ == '__main__':
    main()


