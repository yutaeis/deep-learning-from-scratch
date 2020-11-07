import numpy as np
import pylab as plt


def step_func(x):
    return np.array(x > 0, dtype=np.int)

if __name__ == '__main__':
    print(step_func(1))
    print(step_func(-1))

x = np.arange(-5., 5., 0.1)
y = step_func(x)
plt.plot(x, y)
plt.ylim(-.1, 1.1)
plt.show()