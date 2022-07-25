from torch.utils.data import Dataset
from PIL import Image
import os   #Python中的一个库

class MyData(Dataset):
    #初始化类，创建一个类的实例时就要运行的函数，为整个class提供全局变量
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)  #文件下的文件作为一个列表，每一个项都是该文件的名称

    # 获取每一个图片
    def __getitem__(self, idx):      #idx作为图片索引，通过列表索引获取图片
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)

root_dir = "dataset/train"
ants_label_dir = "ants"
bees_label_dir = "bees"

ants_dataset = MyData(root_dir, ants_label_dir)  #蚂蚁数据集 访问变量  ants_dataset[0]返回值有两个img, label
bees_dataset = MyData(root_dir, bees_label_dir)  #蜜蜂数据集

train_dataset = ants_dataset + bees_dataset  #数据集的集合，判断数据集的长度 len(train_dataset)  有时数据集不足就需要仿造数据集将真实数据集与仿造数据集进行拼接
                                             #常用数据集的表达，分为 ants_images 和 ants_labels (txt格式通常)



