import numpy as np
from numpy import linalg
from numpy.core.fromnumeric import transpose
from numpy.ma import flatten_mask
import pylab as pl
from math import pi

# solver for the 1D heat equation
#   u_t = kappa u_xx  0<x<L, 0<t<T


def FE(u_j, mx, mt, lmbda):
    # Creating matrix 

    A_FE = np.zeros(shape=(mx-1, mx-1))
    np.fill_diagonal(A_FE, 1-2*lmbda)
    np.fill_diagonal(A_FE[1:], lmbda)
    np.fill_diagonal(A_FE[:,1:], lmbda)

    u_jp1 = np.zeros(u_j.size)

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

    u_jp1 = np.zeros(u_j.size)
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

    u_jp1 = np.zeros(u_j.size)

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

    u_jp1 = np.zeros(u_j.size) 

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
    u_jp1 = np.zeros(u_j.size)     # u at next time step
    
    # Solve the PDE: matrix multiplications
    for i in range(0, mt):
        u_jp1[1:-1] = linalg.solve(A_cn, np.matmul(B_cn, transpose(u_j[1:-1])))
    
        # Save u_j at time t[j+1]
        u_j = u_jp1 

    return u_j
def solve(kappa, L, T, mx, mt, method):
    """
    A function that uses finite differences method to solve 1D heat equation PDE.

    Parameters
    ----------
    kappa : float
        The diffusion constant
    L : float
        The length of the spatial domain.
    T : float
        Total time to solve for.
    mx : integer
        Number of gridpoints in spatial domain.
    mt : integer
        Number of gridpoints in time domain.
    method : 
        Method to solve with from list: BE, FE, FE_neumann, FE_dirichlet, Crank_Nicholson.


    Returns
    -------
    Returns a numpy.array containing the final row grid values for u_j. 
    i.e. the heat values at time T.

    """
    # Set up the numerical environment variables
    x = np.linspace(0, L, mx+1)     # mesh points in space
    t = np.linspace(0, T, mt+1)     # mesh points in time
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t

    def u_I(x):        #defining initial temperature
        y = np.sin(pi*x/L)
        return y
    # the exact solution
    def u_exact(x,t):
        y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
        return y
    # Set up the solution variables
    u_j = np.zeros(x.size)

    # Set initial condition
    for i in range(0, mx+1):
        u_j[i] = u_I(x[i])
    
    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
    print("deltax=",deltax)
    print("deltat=",deltat)
    print("lambda=",lmbda)

    # solving
    u = method(u_j, mx, mt, lmbda)
    # Plot the final result and exact solution
    pl.plot(x, u,'ro',label='num')
    xx = np.linspace(0,L,250)
    pl.plot(xx,u_exact(xx,T),'b-',label='exact')
    pl.title(str(method.__name__))
    pl.xlabel('x')
    pl.ylabel('u(x,0.5)')
    pl.legend(loc='upper right')
    pl.show()
    return u    

