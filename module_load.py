import torch
import torchvision
# from module_save import *  实战中将所有class import进来

# 方式1 加载模型-》方式1 保存模型  该方法参数也被保存下来了
from torch import nn

module = torch.load("module_methon1.pth")
print(module)

# 方式2 加载模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("module_mathod2.pth"))
print(vgg16)

# 加载出来是字典形式的参数
# model = torch.load("vgg16_method2.pth")
# print(model)

# 方式1 陷阱：如果自己创建网络加载的时候就会报错-找不到该类，要将整个类copy过来
class Moudle(nn.Module):
    def __init__(self):
        super(Moudle, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        output = self.conv1(x)
        return output

# 不用实例化了
model = torch.load('module_method3.pth')
print(model)
