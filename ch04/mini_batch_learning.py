import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而
import numpy as np
from reference.DeepLearningFromScratch.dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000, 10)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)  # 从60000个数据中随机取出10个
#mini-batch 指定随机选出的索引 取出mini-batch 使用mini-batch计算损失函数
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]


#交叉熵误差 处理单个数据
def cross_entropy_error(y, t):
    #y的维度为1时，需要改变数据的形状
    if y.ndim == 1:
        t = t.reshape(1, t.size) 
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size  # 防止log(0)的情况

#mini-batch的 交叉熵误差 处理mini-batch数据
#计算预测概率y和真实标签t之间的交叉熵损失
def cross_entropy_error_mini_batch(y, t):
    #y的维度为1时，需要改变数据的形状 做鲁棒性处理
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
#np.arange(batch_size) 生成一个数组 [0,1,2,...,batch_size-1]
#y[np.arange(batch_size), t] 取出每个样本对应正确标签t的预测值

