import torch
import torch.nn as nn
import numpy as np

# 张量 list/array/tensor都可以生成张量
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data)   # 保留 x_data 的属性
print(f"Ones Tensor: \n {x_ones} \n")
x_rand = torch.rand_like(x_data, dtype=torch.float)   # 重写 x_data 的数据类型 int -> float
print(f"Random Tensor: \n {x_rand} \n")

shape = (2,3,) # 指定维度
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
print(rand_tensor.shape, rand_tensor.dtype, rand_tensor.device)
#

# 判断当前环境GPU是否可用, 然后将tensor导入GPU内运行
if torch.cuda.is_available():
    tensor = rand_tensor.to('cuda')
    print(123)

# torch.autograd是 PyTorch 的自动差分引擎，可为神经网络训练提供支持
# orch.autograd跟踪所有将其requires_grad标志设置为True的张量的操作。 对于不需要梯度的张量，将此属性设置为False会将其从梯度计算 DAG 中排除。
from torch import nn, optim
import torchvision
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters(): # 加载预训练模型，张量不进行梯度计算
    param.requires_grad = False
model.fc = nn.Linear(512, 10)    # 只求最后一层的张量的梯度
for name, parameters in model.named_parameters():
    print(name, ':', parameters.size(), parameters.requires_grad)


