
from layer_naive import AddLayer
from layer_naive import MultiplyLayer
import sys
import os

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1  

#有三个起点 和 四个节点 三个乘节点 一个加节点
#layer
mul_apple_layer = MultiplyLayer()
mul_orange_layer = MultiplyLayer()
add_orange_apple_layer = AddLayer()
mul_tax_layer = MultiplyLayer()

#forward
apple_price = mul_apple_layer.forward(apple,apple_num) #1
orange_price = mul_orange_layer.forward(orange, orange_num) #2
all_price = add_orange_apple_layer.forward(apple_price, orange_price) #3
total_price = mul_tax_layer.forward(all_price, tax) #4

#backward
dprice = 1 #反向传播起点 总价格的梯度
dall_price, dtax = mul_tax_layer.backward(dprice) #4
dapple_price,dorange_price = add_orange_apple_layer.backward(dall_price) #3
dorange, dorange_num = mul_orange_layer.backward(dorange_price) #2
dapple, dapple_num = mul_apple_layer.backward(dapple_price) #1

print(total_price)
print(dapple, dapple_num, dorange, dorange_num, dtax)
#dapple代表的含义是 苹果价格变动时 总价的变化
#dapple_num代表的含义是 苹果数量变动时 总价的变化
#dorange代表的含义是 橙子价格变动时 总价的变化
#dorange_num代表的含义是 橙子数量变动时 总价的变化
#dtax代表的含义是 税率变动时 总价的变化
#可以看出 橙子数量变动时 对总价的影响最大
#通过偏导数可以知道 该如何调整参数 让总价变化最大 (梯度下降法) 比起用数值微分来计算更高效