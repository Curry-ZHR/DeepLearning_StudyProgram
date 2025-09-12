import numpy as np
import matplotlib.pyplot as plt

#实现均方误差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)

t = [0,0,1,0,0,0,0,0,0,0] #正确解标签one-hot表示
y1 = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0] #神经网络的输出
print(mean_squared_error(np.array(y1), np.array(t)))

y2 = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0] #神经网络的输出
print(mean_squared_error(np.array(y2), np.array(t)))

#实现交叉熵误差
def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))

print(cross_entropy_error(np.array(y1), np.array(t)))
print(cross_entropy_error(np.array(y2), np.array(t)))