import sys, os
sys.path.append(os.pardir)
import numpy as np 
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

#ハイパーパラメータ
iters_num = 100
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []
#1エポックあたりの繰り返し数
iter_per_epoch = max(train_size / batch_size, 1)


for i in range(iters_num):
    #ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #勾配計算
    grad = network.numerical_gradient(x_batch, t_batch)
    # grad = network.gradietn(x_batch, t_batch) #高速版

    #パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    #学習過程の記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("train acc: {:.2f}, test acc: {:.2f}".format(train_acc, test_acc))

plt.plot(train_loss_list)