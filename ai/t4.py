

from models import snet
import torch
import numpy as np
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt

# 图像ImageNet标准化
MEAN_RGB=[0.485, 0.456, 0.406]
STED_RGB=[0.229, 0.224, 0.225]

# 指定类别名称
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# 指定硬件设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path):
    img = cv2.imread(image_path)
    return img

# 利用OpenCV和Numpy相关接口函数完成待测试图像的预处理
def preprocess(img, mean=MEAN_RGB, std=STED_RGB):
    assert isinstance(img, np.ndarray)
    # 图像尺寸变化
    img_rs = cv2.resize(img, dsize=(32, 32), interpolation = cv2.INTER_AREA)
    # 图像通道变换BGR转换为RGB
    im_rgb = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
    # 满足torch tensor的维度要求
    img_rs_arr_chw = img_rs.transpose(2,0,1)
    # 数据类型变换为float并归一化
    img_rs_tensor = torch.Tensor(img_rs_arr_chw).to(torch.float32)/255.
    #标准化处理
    img_norm_t = transforms.functional.normalize(img_rs_tensor, mean, std)
    #满足模型计算要求
    img_norm_t_b = img_norm_t.reshape(1, 3, 32, 32)
    return img_norm_t_b

## 加载中的预先训练模型或者赛题中指定的模型
# 初始化models.py中定义好的模型snet
model = snet
# 加载预训练模型权重
model.load_state_dict(torch.load('model.pth', map_location=device))

## 构建推理函数并完成指定图像的识别
def infer(image_path, model=model, device=device, label_names=label_names):
    img = load_image(image_path)
    # 完成图像的预处理过程
    img_t = preprocess(img)
    # 指定模型运行设备
    model.to(device)
    img_t = img_t.to(device)
    # 计算得到模型输出结果
    output = model(img_t)
    result = output.detach().cpu().numpy()  #阻断反向传播并将数据迁移至cpu上
    label_index = np.argmax(result) #求最大值的索引值
    label = label_names[label_index] #求标签
    print("分类结果为： {}".format(label))
    return label

## 完成images文件夹中任意五张图像的推理计算
if __name__ == '__main__':
    infer('images/bird1.png')
    infer('images/car1.png')
    infer('images/deer1.png')
    infer('images/horse.png')
    infer('images/ship1.png')