import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root="./datasets", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)  # return img, target

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.linear = Linear(196608, 10)

    def forward(self, x):
        output = self.linear(x)
        return output

module = Module()
for data in dataloader:
    img, target = data
    # input = torch.reshape(img, (1, 1, 1, -1))  # 将图片变化为一维向量
    input = torch.flatten(img)  # 将图片变化为一维向量
    output = module(input)
    print(output.shape)

