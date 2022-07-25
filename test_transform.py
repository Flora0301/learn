# transform中最常用的类就是 totensor 将PLI类型或者numpy类型的文件转换为tensor
from PIL import Image
from torchvision import transforms
import cv2

# 通过 transforms.ToTensor 去看两个问题
# 若我们要调用 totensor 中的 __call__ 函数传入的参数是一个pic(PIL的image或者numpy

img_path = "data/train/ants_image/0013035.jpg"
img = Image.open(img_path)

# 1.transforms 如何使用,选择transform中的一个class进行实例的创建
tensor_trans = transforms.ToTensor()  # 创建实例
img_tensor = tensor_trans(img)

print(img_tensor)

'''2. 为什么我们需要 tensor 的数据类型
    tensor数据类型包装了反向神经网络理论基础所需要的一些参数
'''
img_cv  = cv2.imread(img_path)
print(type(img_cv))  # 可以看到类型是nump.array