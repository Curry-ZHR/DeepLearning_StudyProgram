import numpy as np
from common.functions import *
from common.util import im2col, col2im

#ReLu节点
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self,x):
        #设置布尔掩码 小于0时bool值为True
        self.mask = (x <= 0)
        out = x.copy()
        # 将小于等于0的部分置为0
        out[self.mask] = 0
        return out
    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout
        return dx
    
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self,x):
        out = sigmoid(x)
        self.out = out
        return out
    
    def backward(self,dout):
        #化简后的反向传播公式
        #dout是一个微分
        dx = dout * (1.0 - self.out) * self.out
        return dx

#做仿射变换
#偏置B是所有样本共享的全局参数，其梯度反应批次中所有样本的贡献
#因为我们要看的就是这个偏置B可以对模型的输出产生多大的影响
#反向传播的目的就是为了看各个参数节点对最终输出能产生多大的影响
class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b =b
        #self.x 用于保存输入的值
        self.x = None
        #original_x_shape 用于还原 x 的形状（用于卷积层的情况）
        self.original_x_shape = None
        #权重和偏置参数的导数
        self.dW = None
        self.db = None
    
    def forward(self,x):   
        #张量 四维
        self.x = x
        out = np.dot(x,self.W) + self.b
        return out
    
    def backward(self,dout):
        dx = np.dot(dout,self.W.T)
        self.dW = np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis=0)
        return dx
    
#softmax层将输入值正规化之后再输出.
#推理过程一般不使用softmax 学习过程使用
class SoftmaxWithLoss:
    