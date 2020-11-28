import numpy as np
import matplotlib.pyplot as plt


z = np.arange(-2,2,0.1)

zeros = np.zeros(len(z))

# relu = max(0,z) axis = 0 means row
y = np.max([zeros, z], axis=0)



def another_way():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(z,y)
    ax.set_ylim([-2,2])
    ax.set_xlim([-2,2])
    ax.set_xlabel('z')
    ax.set_title('Rectified linear unit')
    ax.grid(True)
    plt.show()


def one_way():
    plt.plot(z,y)
    plt.xlabel('z')
    plt.title('RELU')
    plt.ylim([-2,2])
    plt.xlim([-2,2])
    plt.grid(True)
    plt.show()

another_way()