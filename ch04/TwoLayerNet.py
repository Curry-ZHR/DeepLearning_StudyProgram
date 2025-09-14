import sys,os
sys.path.append(os.pardir)  # 为了导入爷目录的文件而进行的设定
from reference.DeepLearningFromScratch.common.functions import *
# from DeepLearning_studyprogram.reference.DeepLearningFromScratch.dataset.mnist import load_mnist
#计算梯度
from gradient_simplenet import numerical_gradient
from collections import OrderedDict #有序字典
from common.layers import Affine, Sigmoid, SoftmaxWithLoss, Relu
#对于有序字典，可以记住向字典添加元素的顺序
#正向传播只需按照添加元素的顺序调用各层的forward()方法就可以实现
#反向传播只需要按照相反的顺序调用各层即可

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        
        #params变量保存该神经网络所需要的全部参数
        self.params ={}
        #weight_init_std 标准差 用于缩放随机生成的权重值 随机生成的权重值符合正态分布
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        #生成层
        self.layers = OrderedDict()  #有序字典
        #做矩阵的仿射变换
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        #最后一层 损失函数
        self.lastLayer = SoftmaxWithLoss() #softmax分类完再计算交叉熵损失函数



    #预测值的函数
    def predict(self, x):
        # W1, W2 = self.params['W1'], self.params['W2']
        # b1, b2 = self.params['b1'], self.params['b2']

        # a1 = np.dot(x, W1) + b1
        # z1 = sigmoid(a1)
        # a2 = np.dot(z1, W2) + b2
        # #通过softmax函数得到各类别的概率
        # y = softmax(a2)

        #从有序字典中取出各层
        for layer in self.layers.values():
            x = layer.forward(x)

        return x
    

    #计算损失值的函数
    def loss(self, x, t):
        y = self.predict(x)
        # return cross_entropy_error(y, t)

        #最后一层就是softmax和交叉熵误差
        return self.lastLayer.forward(y, t)
    
    #计算识别到的精度
    def accuracy(self, x, t):
        #得到每个类别的得分or概率
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        #t 为 ont-hot编码 argmax返回one-hot编码中1的索引
        t = np.argmax(t, axis=1)
        #计算正确的样本数除总样本数（x.shape[0]）
        if x.ndim != 1: 
            t = np.argmax(t, axis=1)
            accuracy = np.sum(y == t) / float(x.shape[0])
            return accuracy
    
    #类方法
    #利用数值微分计算梯度
    def numerical_gradient(self, x, t):
        #定义损失函数的计算
        loss_W = lambda W: self.loss(x, t)
        #字典型实例变量 保存了各个参数的梯度
        grads = {}
        #这里调用的是全局作用域的函数
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])     #权重矩阵梯度
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])     #偏置向量的梯度
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
    

    #利用计算图 反向传播法计算梯度
    def gradient(self,x,t):
        #forward
        self.loss(x,t)

        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        #layers是一个有序字典
        #反向传播是从最后一层开始的
        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        #把各个参数的梯度保存到grads中
        #两层神经网络的
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads