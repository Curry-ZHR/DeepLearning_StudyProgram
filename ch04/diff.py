import numpy as np
import matplotlib.pylab as plt

#差分

#前向差分
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h)-f(x)) / h

#中心差分
def numerical_diff_center(f, x):
    h = 1e-4
    return (f(x+h)-f(x-h)) / (2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
# plt.show()

print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))

#二元方程求偏导数
def function_2(x):
    return x[0]**2 + x[1]**2

#先固定为一元函数 再用数值微分求导
def function_tmp1(x0):
    return x0**2 + 4.0**2

def function_tmp2(x1):
    return 3.0**2 + x1**2

#全部变量的偏导数汇总的向量就是梯度
#这个梯度下降函数是用于一维数组的
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  #生成和x形状相同的数组
    #对每个变量求偏导数 每一个x变量
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)  #f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  #f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val  #还原值

    #返回一个偏导数的列表 向量
    return grad 

# print(numerical_gradient(function_2, np.array([3.0, 4.0])))
# print(numerical_gradient(function_2, np.array([0.0, 2.0])))
# print(numerical_gradient(function_2, np.array([3.0, 0.0])))

#梯度下降法
#init_x 初始值; lr 学习率; step_num 迭代次数;
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        #求偏导 找梯度
        grad = numerical_gradient(f, x)
        #沿梯度的反方向更新参数
        x -= lr * grad  #x = x - lr * grad
    return x

init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))
#学习率过大
print(gradient_descent(function_2,init_x=init_x,lr = 10.0,step_num=100))
#学习率过小
print(gradient_descent(function_2,init_x=init_x,lr=1e-10, step_num=100))

