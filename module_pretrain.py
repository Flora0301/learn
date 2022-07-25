import torchvision

# train_data = torchvision.datasets.ImageNet("./imagedata", split='train', download)
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=True)

print(vgg16)

vgg16.add_module('add_linear', nn.Linear(1000, 10))
# vgg16.classifier.add_module(vgg16.classifier.add_module()) 在层级中加
print(vgg16)

# 直接修改层级
vgg16.classifier[6] = nn.Linear(4096, 10)