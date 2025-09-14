import numpy as np
import matplotlib.pylab as plt

class MultiplyLayer:
    def __init__(self):
        #用于保存正向传播的输入值
        self.x = None
        self.y = None

    #前向传播 用乘法节点
    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x * y
        return out
    
    #dout是上一层传递过来的梯度
    def backward(self,dout):
        dx = dout * self.y #x的梯度
        dy = dout * self.x #y的梯度
        return dx, dy
    
class AddLayer:
    #加法层不用存储正向传播的值
    def __init__(self):
        pass

    def forward(self,x,y):
        out = x + y
        return out
    
    def backward(self,dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

