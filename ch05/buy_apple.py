from layer_naive import MultiplyLayer

apple = 100
apple_num = 2
tax = 1.1

#苹果和消费税 是两个起点
mul_apple_layer = MultiplyLayer()
mul_tax_layer = MultiplyLayer()

#forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price)

#注意正向传播和反向传播时 x和dx要对应 y和dy要对应

#backward
#反向传播起点为1
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple, dapple_num, dtax)
#dapple代表的含义是 苹果价格变动时 总价的变化
#dapple_num代表的含义是 苹果数量变动时 总价的变化
#dtax代表的含义是 税率变动时 总价的变化
#可以看出 税率变动时 对总价的影响最大

#通过偏导数可以知道 该如何调整参数 让总价变化最大 (梯度下降法) 比起用数值微分来计算更高效
#比如说 现在苹果价格变动1元时 总价变动2.2元
#苹果数量变动1个时 总价变动110元
#税率变动0.01时 总价变动2元
