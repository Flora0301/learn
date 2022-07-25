from PIL import Image
from torchvision import transforms

img = Image.open("dataset/train/ants/0013035.jpg")

# ToTensor
tensor_tran = transforms.ToTensor()
img_tensor = tensor_tran(img)

# Normalize 的使用：归一化 计算公式：output[channel] = (input[channel] - mean[channel]) / std[channel]
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 参数为均值，标准差，三个信道
img_nrom = trans_norm(img_tensor)

# Resize 输入是 PIL 格式
print(img.size)
trans_resize = transforms.Resize([512, 512])
img_resize = trans_resize(img)
img_resize = tensor_tran(img_resize)
print(img_resize)
