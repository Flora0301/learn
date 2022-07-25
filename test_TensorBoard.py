from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter类
from PIL import Image
import numpy as np

img_path = "data/train/ants_image/0013035.jpg"
img = Image.open(img_path)  # 利用PIL库打开的图片格式为jpg格式
img_array = np.array(img)  # 将格式转为numpy类型

print(type(img_array))  # 查看图片类型
print(img_array.shape)  # 打印图片格式CHW

writer = SummaryWriter("logs")  # logs是文件名

writer.add_image("test", img_array, 1, dataformats="HWC")  # 要求文件类型是 Tensor或者numpy类型，需要添加参数dataformats确定shape中每个维度的含义
for i in range(100):
    # 打开logs文件 tensorboard --logdir=logs（事件文件所在文件夹名）--port=6007（指定端口号）
    writer.add_scalar("y=2x", 2 * i, i)  # 参数add_scalar(self, tag 图表标题, scalar_value 保存的数值y轴,global_step=None x轴,

writer.close()