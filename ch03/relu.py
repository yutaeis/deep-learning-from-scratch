import numpy as np
import pylab as plt

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5., 5., .1)
y = relu(x)

plt.plot(x,y)
#plt.ylim(-.1, 5.1)
plt.show()