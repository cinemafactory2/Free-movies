import numpy as np
import matplotlib.pyplot as plt

z = np.arange(-5,5,.1)

sigma_fn = np.vectorize(lambda  z : 1.0/(1.0+np.exp(-z))) #for each element do this and return
#y = 1.0/(1.0+np.exp(-z))   
y = sigma_fn(z)


fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.plot(z,y)
ax.set_xlim([-5,5])
ax.set_ylim([-0.5,1.5])
ax.set_title('Sigmoid Function')
ax.set_xlabel('z')
ax.grid(True)

plt.show()