from torch import nn
import torch


class Module(nn.Module):

    def __init__(self):
        super(Module, self).__init__()  # 继承父类的初始化方法,super调用父类

    def forward(self, input):
        output = input + 1
        return output

module = Module()
x = torch.tensor(1.0)
output = module(x)
print(output)


