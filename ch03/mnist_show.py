import sys,os
sys.path.append(os.pardir)  #父目录这一级 就可以使用reference目录下的模块了
import numpy as np
# 引入函数 load_mnist()
from reference.DeepLearningFromScratch.dataset.mnist import load_mnist
from PIL import Image  #PIL是Python Imaging Library的缩写，Python图像库

#手写数字集 展示
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))  #将NumPy数组转换为PIL图像
    pil_img.show()  #显示图像

#训练图像、训练标签
(x_train,t_train),(x_test,t_test) = load_mnist(flatten=True,normalize=False)  #读取数据集
img = x_train[0]  #取出第一张图像
label = t_train[0]  #取出第一张图像的标签
print(label)  #打印标签

#flatten=True时 读入的图像是以一维Numpy数组形式存储的，显示图像时需要将其还原为二维数组
print(img.shape)  #(784,) 28*28=784
img = img.reshape(28,28)  #将一维数组转为二维数组
print(img.shape)  #(28, 28)
img_show(img)  #显示图像