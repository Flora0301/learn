import torchvision
import torch
from torch.utils.tensorboard import SummaryWriter

from model import *
from torch.utils.data import DataLoader

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./datasets", train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="./datasets", train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True)

# 获取数据集长度(图片个数)
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型，10分类的网络
model = Module()

# 损失函数
loss_fn = nn.CrossEntropyLoss()  # 交叉熵

# 优化器，随机梯度下降
# learning_rate = 0.01
# 1e-2 = 1 * (10)^(-2)
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter('logs_train')

for i in range(epoch):
    print("-----------第 {} 轮训练开始-------------".format(i+1))

    # 训练步骤开始
    model.train()  # 可以省略，只对特定的层有作用
    for data in train_dataloader:
        imgs, targets = data
        output = model(imgs)
        loss = loss_fn(output, targets)  # 预测的输出和真实的输出计算loss

        # 优化器优化模型
        optimizer.zero_grad()  # 利用优化器梯度清零
        loss.backward()  # 得到每个参数节点的梯度
        optimizer.step()  # 对参数进行优化

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))  # .item 变成真实的数字
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始 一般对整体测试数据计算正确率
    # 训练过后在测试数据集上进行测试看准确率或者loss，看是否训练好
    model.eval() # 非必须同model.train
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # 在with里边就没有了梯度，不会进行调优
        for data in test_dataloader:
            imgs, targets = data
            output = model(imgs)
            loss = loss_fn(output, targets)
            total_test_loss = loss.item() + total_test_loss
            accuracy = (output.argmax(1) == targets).sum()  # argmax找到对应类别的编号，1代表横向
            total_accuracy = total_accuracy + accuracy

        print("整体测试集上的loss：{}".format(total_test_loss))
        print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
        writer.add_scalar('test_loss', total_test_loss, total_test_step)
        writer.add_scalar('test_accuracy', total_accuracy, total_test_step)
        total_test_step = total_test_step + 1

        # 保存每一轮的模型
        torch.save(model, "model_{}.pth".format(i))

    writer.close()



