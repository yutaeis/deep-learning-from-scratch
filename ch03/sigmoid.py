import numpy as np
import pylab as plt

def sigmoid(x):
    return 1. / (1. + np.exp(-x))


x = np.arange(-5., 5., .1)
y = sigmoid(x)

plt.plot(x,y)
plt.ylim(-.1, 1.1)
plt.show()
