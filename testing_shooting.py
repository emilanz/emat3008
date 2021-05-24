from ODEs import Hop_bif_2D, predprey, Hop_bif_3D
from shooting import *

# wrong dimensions prompt, comment to run rest of code
shooting(Hop_bif_3D, np.array([0.55, 1]), 20, fsolve, plot=True)

# # #3D Hopf Bifurcation
shooting(Hop_bif_3D, np.array([1, 1, 1]), 20, fsolve, plot=True)


# 2D predator prey ODE
X, t = shooting(predprey, np.array([0.55, 0.28]), 20, fsolve)
plt.plot(t, X[:,0], 'g', label='prey')
plt.plot(t, X[:,1], 'r', label='predator')
plt.legend()
plt.xlabel('time')
plt.ylabel('population')
plt.show()

# error testing for accuracy of shooting, passed
def exact_U(t):
    Beta = 2
    u1 = np.sqrt(Beta)*np.cos(t)
    u2 = np.sqrt(Beta)*np.sin(t)
    return np.array([u1, u2])

result = shooting(Hop_bif_2D, np.array([1,1]), 5, fsolve) 
exact = exact_U(result[1])
res = np.subtract(result[0], exact.T)
if np.all(abs(res<1e-6)):
    print('Passed test')  #shows results are accurate to resolution of python
else:
    print('Test failed')

