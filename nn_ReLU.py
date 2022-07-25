# 非线性激活
import torch
from torch import nn
from torch.nn import ReLU

input = torch.tensor([[1, -0.5],
                     [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))  # ReLU 函数对输入数据的要求是（batch_size, *）

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.output_relu = ReLU()

    def forward(self, x):
        output = self.output_relu(x)
        return output

module = Module()
output = module(input)
print(output)
