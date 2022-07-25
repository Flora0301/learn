
import torch
from torch import nn

input = torch.Tensor([1, 2, 3])
target = torch.Tensor([1, 2, 5])

inputs = torch.reshape(input, (1, 1, 1, 3))
target = torch.reshape(target, (1, 1, 1, 3))

loss = nn.L1Loss()
loss1 = loss(inputs, target)

loss_mse = nn.MSELoss()  # 平方差
loss2 = loss_mse(inputs, target)

print(loss2)
print(loss1)

x = torch.tensor([0.1, 0.2, 0.3])  # torch,Tensor 是类torch.tensor的构造函数
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))  # batch_size, 3 类

loss_cross = nn.CrossEntropyLoss()
output = loss_cross(x, y)
print(output)