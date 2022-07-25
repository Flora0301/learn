# 最大池化的作用是保持输入的特征同时减小输入数据量
import torch
import torchvision
from torch import nn

input_tensor = torch.tensor([[1, 2, 0, 3, 1],
              [0, 1, 2, 3, 1],
              [1, 2, 1, 0, 0],
              [5, 2, 3, 1, 1],
              [2, 1, 0, 1, 1]], dtype=torch.float)  # 注意要设置数据类型为浮点数，否则会报错

input = torch.reshape(input_tensor, (-1, 1, 5, 5))

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.MaxPool = torch.nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        output = self.MaxPool(x)
        return output

module = Module()
output = module(input)
print(output)
