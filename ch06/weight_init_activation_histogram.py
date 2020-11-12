import numpy as np
import pylab as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

input_data = np.random.randn(1000, 100)
node_num = 100
hidden_layer_size = 5
activations = {}

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
    
    #w = np.random.randn(node_num, node_num) * 1
    #w = np.random.randn(node_num, node_num) * 0.01
    #w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
    #w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)

    #Xavier
    #w = np.random.randn(node_num, node_num) / np.sqrt(node_num)

    #He
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num / 2.)
    
    #
    
    a = np.dot(x, w)

    #z = sigmoid(a)
    z = ReLU(a)
    #z = tanh(a)

    activations[i] = z

#ヒストグラム描画
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title("{}-layer".format(i+1))
    if i != 0: plt.yticks([], [])
    plt.hist(a.flatten(), 30, range=(0,1))
plt.savefig('weight_init_activation_histogram.png')    
plt.show()