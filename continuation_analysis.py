from continuation import *
import numpy as np 
import matplotlib.pyplot as plt

# continuation for algebraic cubic function
sols = continuation(alg_cubic, np.array([1.52137971,-2]),  np.linspace(-1.98, 2, 200))
plt.plot(np.linspace(-2, 2, 201), sols)
plt.xlabel('alpha')
plt.ylabel('x')
plt.show()


# plot shows that when c reaches around 0.39 and x turns a corner, the solutions are no longer valid
# as they flatline at around 0.57, need 
