import numpy as np
import math

# 仅使用Numpy库模拟单层神经网络，实现损失函数、梯度计算、反向传播过程
# y = a + b*x + c x^2 + d x^3模拟y=sin(x)

# Create random input and output data
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

a = np.random.randn() #生成[1,1)的随机数，前闭后开
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6

for t in range(10000):
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    loss = np.square(y_pred - y).sum()
    
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d
    if t % 200 == 0:
        print(t, loss)

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')
print(x)
print(y)
print(a + b * x + c * x ** 2 + d * x ** 3)
