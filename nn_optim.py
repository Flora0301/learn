from torch import nn
import torch
import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root="datasets", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=1)

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024, 64)
        self.linear2 = nn.Linear(64, 10)

        self.module1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        output = self.module1(x)
        return output

loss = nn.CrossEntropyLoss()
module = Module()
optim = torch.optim.SGD(module.parameters(), lr=0.001)

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        img, target = data
        output = module(img)
        loss_result = loss(output, target)
        optim.zero_grad()  # 梯度下降
        loss_result.backward()  # 反向传播
        optim.step()  # 更新参数
        running_loss = loss_result + running_loss
        # print(loss_result)
        # print(output)
        # print(target)

    print(running_loss)

