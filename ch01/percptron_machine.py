import numpy as np
import matplotlib.pyplot as plt

#与门感知机
def AND1(x1,x2):
    w1,w2,theta = 0.5,0.5,0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
# print(AND(0,0))
# print(AND(1,0))
# print(AND(0,1))
# print(AND(1,1))

#与门感知机 使用偏置bias
def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
# print(AND(0,0))
# print(AND(1,0))
# print(AND(0,1))
# print(AND(1,1))
    
#bias 偏置 的 与非门
def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
# print(NAND(0,0))
# print(NAND(1,0))
# print(NAND(0,1))
# print(NAND(1,1))

#设置bias偏置 的 或门感知机
def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

#感知机无法实现异或门 则可以使用 多层感知机来实现
#仅通过感知机的叠加就可以实现 计算机的所有功能
def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)
    return y
print(XOR(0,0))
print(XOR(1,0))
print(XOR(0,1))
print(XOR(1,1))

#绘制阶跃函数
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0
    
#实现允许参数取numpy数组的阶跃函数
def step_function_np(x):
    y = x > 0
    return y.astype(np.int32)

x = np.arange(-5.0,5.0,0.1)
y = step_function_np(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
# plt.show()

#绘制sigmoid函数
def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.array([-1.0,1.0,2.0])

#激活函数必须使用非线性函数，因为如果是线性的，不管层数多深，都能用一个线性函数表示
#而非线性函数则可以表示更复杂的函数关系

#绘制ReLU函数
def ReLU(x):
    return np.maximum(0,x)

#用numpy数组可以很快速的完成 神经网络的构建
def neural_network():
    X = np.array([1.0,0.5])
    W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    B1 =np.array([0.1,0.2,0.3])
    A1 = np.dot(X,W1) + B1
    Z1 = sigmoid(A1)
    # print(A1)
    # print(Z1)

#恒等函数 回归问题最后一层会以恒等函数的形式输出
def identify_function(x):
    return x   

#分类问题最后一层会以softmax函数的形式输出
#使用softmax函数时可能会出现溢出问题
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

#进行softmax指数函数运算时，加上一个常数项 c 可以避免溢出问题，且不会影响运算的结果
#使用softmax函数不会影响各元素间的大小关系因为 y=exp(a)/sum(exp(a)) 其中exp是单调递增函数
def softmax_stable(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  #减去最大值 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    return network

#前向传播 表示从输入到输出方向的传递处理
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = identify_function(a3)
    return y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network,x)
print(y)