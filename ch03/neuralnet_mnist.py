import numpy as np
import matplotlib.pyplot as plt
import sys, os
os.sys.path.append(os.pardir)  # 父目录这一级 就可以使用reference目录下的模块了
from reference.DeepLearningFromScratch.dataset.mnist import load_mnist
import pickle
from ch01.percptron_machine import sigmoid, softmax

#normalize=True 是做归一化处理 数据的值在0~1.0 之间
#flatten=True 是将图像展开为一维数组
#one_hot_label=False 是不使用one-hot编码
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test


#pickle相当于一个对象存储工具 可以把内存中的复杂对象存储到文件中，也可以把存储的对象从文件中重新读取到内存
#将内存中的复杂数据结构“打包”成文件，随时“解包”回来
#权重文件 sample_weight.pkl 是训练后的权重文件
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

#加载现成的权重文件 并打印网络结构
def load_existing_weights():
    """加载现成的权重文件"""
    try:
        with open('sample_weight.pkl', 'rb') as f:
            network = pickle.load(f)
        
        print("权重文件加载成功!")
        print("网络结构:")
        for key, value in network.items():
            print(f"  {key}: {value.shape}")
        
        return network
    
    except FileNotFoundError:
        print("找不到 sample_weight.pkl 文件")
        return None
    except Exception as e:
        print(f"加载失败: {e}")
        return None

# 使用方法
network = load_existing_weights()


#测试集的单张图像进行识别
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y
#获取测试集数据
x, t = get_data()
# network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)  #取得最大值的索引
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))