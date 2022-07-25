#
import torchvision
from torch.utils.tensorboard import SummaryWriter

# compose 用于将多个步骤整合到一起
dataset_trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
# 若数据集下载很慢，可以将下载链接拷贝至迅雷，迅雷提供一些镜像或者p2p加速
train_set = torchvision.datasets.CIFAR10(root="./datasets", transform=dataset_trans, train=True, download=True)  # CIFAR10 是一个数据集，root 存储路径
test_set = torchvision.datasets.CIFAR10(root="./datasets", transform=dataset_trans, train=False, download=True)  # transform将数据集中每张图片都转化为tensor数据类型

# print(test_set.classes)  # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#
# img, target = test_set[0]
# print(img)  # <PIL.Image.Image image mode=RGB size=32x32 at 0x255B1AF6F60>
# print(target)  # 3 说明这个图片对应的是 cat

# img.show()
# img_tensor = dataset_trans(img)  # 将图片转化为tensor格式

# print(test_set[0])
writer = SummaryWriter("logs")

for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()

