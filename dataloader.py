# 载入数据并将数据打包为 batch
import torchvision
from torch.utils.data import DataLoader

# 测试集
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10(root="./datasets", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)  # drop_last 最后剩余不够一个batch时是否舍弃

# 测试数据集中第一章图片
img, target = test_set[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
step = 0

for data in test_loader:
    imgs, target = data
    # print(imgs.shape)  # torch.Size([4, 3, 32, 32]) 4个图片为一组进行打包
    # print(target)  # tensor([8, 3, 1, 4]) 4个target进行打包，四张图片是随机取得sample没有进行设置
    writer.add_images("dataloader", imgs, step)
    step = step + 1

writer.close()
