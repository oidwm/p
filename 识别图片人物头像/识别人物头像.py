import cv2


# 框出人脸
def face_test(img):
    # 联级分类器，haarcascade_frontalface_default.xml为储存了人脸特征的xml文件
    faces = cv2.CascadeClassifier('D:/人工智能培训/OpenCV/haarcascades/haarcascade_upperbody.xml')
    # 找出人脸的位置
    face = faces.detectMultiScale(img, 1.1, 5)
    # 坐标点
    for x, y, w, h in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)  # 画出框
    cv2.imshow('face', img)  # 显示

if __name__ == '__main__':
    video = cv2.imread('50.mp4')  # 打开摄像头
    while True:
        ret, img = video.read()  # 读取图片
        if ret is False: break
        face_test(img)  # 调用函数
        # 保持画面的连续，按esc键退出
        if cv2.waitKey(1) & 0xFF == 27:
            break
