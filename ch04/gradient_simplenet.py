import sys,os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from mean_squared_error import cross_entropy_error


#这个函数处理二维数组
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    #np.nditer是一个多维数组迭代器
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        it.iternext()   
        
    return grad

# #写一个softmax激活函数
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        # 写下标签对应的预测值大小
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

class simpleNet:
    def __init__(self):
        #用高斯分布进行初始化
        #W权重矩阵 是simpleNet自带的属性
        self.W = np.random.randn(2,3)  #从标准正态分布中随机生成2x3的权重

    #求预测值
    def predict(self, x):
        #将输入值与权重参数相乘 得到预测值
        return np.dot(x, self.W)  #x: (N,2) W:(2,3) 预测值y:(N,3)
    
    #计算损失函数 利用交叉熵误差
    def loss(self, x, t):
        z = self.predict(x)  #预测值
        y = softmax(z)  #通过softmax函数得到各类别的概率
        #t是正确标签
        loss = cross_entropy_error(y, t)  #计算交叉熵误差
        return loss

#创建一个简单的神经网络
net = simpleNet()
print("权重参数:",net.W)  #查看权重参数

x = np.array([0.6, 0.9])  #输入数据
p = net.predict(x)  #预测值
print("预测值：",p)  #查看预测值
print("预测最大值的索引：",np.argmax(p))  #查看预测值中最大值的索引 argmax返回最大值的索引

t = np.array([0,0,1])  #正确标签
print("损失函数：",net.loss(x, t))  #查看损失函数

#要计算损失函数的梯度
def f(W):
    return net.loss(x, t)  #损失函数只与W有关

#通过梯度下降法使损失函数最小化 net.W就是损失函数的自变量
dW = numerical_gradient(f, net.W)  #计算梯度
print("损失函数的梯度:",dW)  #查看梯度