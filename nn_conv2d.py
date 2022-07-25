# 卷积
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ./ 代表当前所在目录，“../” 代表上级目录
dataset = torchvision.datasets.CIFAR10(root="./datasets", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)

class Module(nn.Module):

    def __init__(self):
        super(Module, self).__init__()
        self.conv1 = Conv2d(3, 6, 3, stride=1, padding=0)  # （in_channels, out_channels, kernel_size, stride=1, padding=0）

    def forward(self, input):  # 前向传播
        output = self.conv1(input)
        return output

module = Module()
# print(module)

writer = SummaryWriter("logs")
step = 0

# 从 dataloaer 中取数据
for data in dataloader:
    img, target = data
    output = module(img)
    # print(img.shape)  # torch.Size([64, 3, 32, 32])
    writer.add_images("input", img, step)

    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    # print(output.shape)  # torch.Size([64, 6, 30, 30])

    step = step + 1

writer.close()
