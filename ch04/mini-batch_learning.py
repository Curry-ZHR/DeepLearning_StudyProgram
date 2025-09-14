import numpy as np
from reference.DeepLearningFromScratch.dataset.mnist import load_mnist
from TwoLayerNet import TwoLayerNet
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
train_loss_list = []

#hyperparameters
iters_num = 10000  # 迭代次数
train_size = x_train.shape[0]  # 60000
batch_size = 100  # mini-batch的大小
learning_rate = 0.1  # 学习率

#784是输入层的神经元个数 50是隐藏层神经元个数 10是输出层神经元个数
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

#每次从60000个训练数据中随机取出100个数据 对这个包含100个数据的mini-batch计算梯度 梯度法的更新次数为10000次
for i in range(iters_num):
    #获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)  # 从60000个数据中随机取出100个
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #计算梯度
    grad = network.numerical_gradient(x_batch, t_batch)

    #更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    #记录学习过程
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % 1000 == 0:
        print("迭代次数:", i, "loss:", loss)

# 绘制损失函数变化图
#把这个二维矩阵的值画出来
plt.plot(train_loss_list)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.show()