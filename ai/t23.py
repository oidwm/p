

import torch
from torch import nn
from torch.nn import Conv2d, Sequential, BatchNorm2d, ReLU, MaxPool2d, Flatten, Linear
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import ImageFolder
from matplotlib import pyplot as plt

MEAN_RGB = [0.485, 0.456, 0.406]
STED_RGB = [0.229, 0.224, 0.225]
img_height = 32
img_width = 32


## 训练函数
def train(model, dataloader, loss_fn, optimizer, device):
    model.train()
    losses = []
    # 通过循环加载图像数据及对应标签进行学习过程
    for i, (itdata, itlabel) in enumerate(dataloader):
        itdata = itdata.to(device)
        itlabel = itlabel.to(device)
        optimizer.zero_grad()
        outputs = model(itdata)
        loss = loss_fn(outputs, itlabel)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().item())
    losses_res = sum(losses) / len(losses)
    print('train process loss {}'.format(losses_res))
    return losses_res


## 测试函数
def test(model, dataloader, loss_fn, device):
    model.eval()
    losses = []
    # 通过循环加载图像数据及对应标签进行测试过程
    with torch.no_grad():
        for i, (itdata, itlabel) in enumerate(dataloader):
            itdata = itdata.to(device)
            itlabel = itlabel.to(device)
            output = model(itdata)
            loss = loss_fn(output, itlabel)
            losses.append(loss.cpu().item())
    losses_res = sum(losses) / len(losses)
    print('test process loss {}'.format(losses_res))
    return losses_res


## 准确率计算函数
def accuracy(model, dataloader, device):
    model.eval()
    outputsprd = []
    outputslbl = []
    # 通过循环加载图像数据进行计算得到预测值,与标签一起计算准确率
    with torch.no_grad():
        for i, (itdata, itlabel) in enumerate(dataloader):
            itdata = itdata.to(device)
            itlabel = itlabel.to(device)
            output = model(itdata)
            outputsprd.append(output.detach().cpu().numpy())
            outputslbl.append(itlabel.detach().cpu().numpy())
    outputsprd = np.concatenate(outputsprd)
    outputslbl = np.concatenate(outputslbl)
    acc = np.sum(np.equal(np.argmax(outputsprd, axis=1), outputslbl))
    return acc / len(outputslbl)


def main():
    # 任务二 数据集加载和预处理
    BATCH_SIZE = 32

    # 数据集路径
    TRAIN_DATA_DIR = "/data/VZ0XRK/cifar10_image/train"
    TEST_DATA_DIR = "/data/VZ0XRK/cifar10_image/test"

    ## 定义训练集的数据增强和预处理方法
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        # 随机水平翻转
        transforms.RandomHorizontalFlip(),
        # 随机垂直翻转
        transforms.RandomVerticalFlip(),
        # 图像格式变化为Torch Tensor及归一化
        transforms.ToTensor(),
        # 可以使用Imagenet标准化操作
        transforms.Normalize(
            mean=MEAN_RGB,
            std=STED_RGB
        )
    ])

    ## 定义测试集的预处理方法
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        # 图像格式变化为Torch Tensor及归一化
        transforms.ToTensor(),
        # 可以使用Imagenet标准化操作
        transforms.Normalize(
            mean=MEAN_RGB,
            std=STED_RGB
        )
    ])

    ## 使用ImageFolder将数据空间中的数据集加载加来，同时训练和测试集分别对应不同的预处理方法
    # 加载训练集将预处理方法作为参数输入
    trdataset = ImageFolder(TRAIN_DATA_DIR, transform=train_transform)
    # 加载测集并将预处理及增强的方法作为参数输入
    tsdataset = ImageFolder(TEST_DATA_DIR, transform=test_transform)
    ## 从Torch Dataset变化到DataLoader
    # 构建训练DataLoader
    traindataloader = torch.utils.data.DataLoader(trdataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # 构建测试DataLoader
    testdataloader = torch.utils.data.DataLoader(tsdataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 任务三 模型训练与测试

    ## 使用Sequential方法构造模型
    model = Sequential(
        # 综合使用卷积层，池化层，现行层，激活函数，批归一化等方法搭建模型
        Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),  # 64*32*32
        BatchNorm2d(64),
        ReLU(True),
        Conv2d(64, 128, (3, 3), 1, 1, ),  # 128*32*32
        MaxPool2d((2, 2)),  # 128*16*16
        BatchNorm2d(128),
        ReLU(True),
        Conv2d(128, 256, (3, 3), 1, 1),  # 256*16*16
        MaxPool2d((2, 2)),  # 256*8*8
        BatchNorm2d(256),
        ReLU(True),
        Conv2d(256, 512, (3, 3), 1, 1),  # 512*8*8
        MaxPool2d((2, 2)),  # 512*4*4
        BatchNorm2d(512),
        ReLU(True),
        MaxPool2d((4, 4)),  # 512*1*1
        Flatten(),
        Linear(512, 10)
    )

    # 初始化损失函数
    loss_function = nn.CrossEntropyLoss()
    # 定义学习率
    LEARNING_RATE = 1e-2
    # 初始化优化函数
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    # 指定GPU设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## 完成指定周期的训练并将损失函数值保存
    # 指定训练周期
    EPOCHS = 10
    train_losses = []
    test_losses = []
    # 指定模型训练设备
    model.to(device)

    for ep in range(EPOCHS):
        print("The number of epoch {}".format(ep))
        # 进行训练，返回损失值
        train_loss = train(model, traindataloader, loss_function, optimizer, device)
        # 进行测试，返回损失值
        test_loss = test(model, testdataloader, loss_function, device)
        # 记录训练损失值
        train_losses.append(train_loss)
        # 记录测试损失值
        test_losses.append(test_loss)

    ## 评估模型
    ## 绘制损失曲线
    plt.title('Training and Validation Loss')
    trloss, = plt.plot(train_losses, label='Training Loss')
    ttloss, = plt.plot(test_losses, label='Validation Loss')
    plt.legend(handles=[trloss, ttloss])
    plt.show()

    ## 计算测试集的准确率
    acc = accuracy(model, testdataloader, device)
    print("The accuracy {}".format(acc))

    # 利用PyTorch相关接口，完成模型的保存，请保存到个人空间/space/pytorch下
    # 如果/space下没有pytorch文件夹，请先创建
    torch.save(model.state_dict(), '/space/model.pth')


if __name__ == '__main__':
    main()