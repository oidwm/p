##任务一
import cv2
import glob

i = 0
for img in glob.glob(r'./tp/*.jpg'):
    img = cv2.imread(img)
    imgResize = cv2.resize(img, (256, 256))  ##将图片的长宽均缩小为 256 像素
    flipped = cv2.flip(imgResize, 1)  ##图像进行水平镜像翻转
    Cropped = flipped[0:255, 0:255]  ##图像进行左上角 250*250 像素进行裁剪
    dst = cv2.cvtColor(Cropped, cv2.COLOR_BGR2GRAY)  ##图像进行灰度化
    cv2.imwrite('./gtp/{}.jpg'.format(i), dst)
    i = i + 1