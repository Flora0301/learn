import torchvision
import torch
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)  # 没有预训练过，初始参数

# 保存方式1，模型结构+模型参数
torch.save(vgg16, "module_methon1.pth")  # （模型，名称）一般都以pth结尾

# 保存方式2，模型参数，将网络模型的参数保存为字典格式 dict字典，不保存模型了（官方推荐）
torch.save(vgg16.state_dict(), "module_mathod2.pth")

class Moudle(nn.Module):
    def __init__(self):
        super(Moudle, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        output = self.conv1(x)
        return output

module = Moudle()
torch.save(module, "module_method3.pth")
