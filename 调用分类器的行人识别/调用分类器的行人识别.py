import cv2
def face_test(img):
    faces = cv2.CascadeClassifier('./haarcascade_fullbody.xml')
    face = faces.detectMultiScale(img,1.1, 10)
    for x, y, w, h in face:
        cv2.rectangle(img,(x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('face', img)  # 显示

if __name__ == '__main__':
    video = cv2.VideoCapture('test.avi')
    while True:
        ret, img = video.read()  # 读取图片
        if ret is False: break
        face_test(img)  # 调用函数
        if cv2.waitKey(1) & 0xFF == 27:
            break
