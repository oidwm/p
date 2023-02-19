##任务一
import cv2

my = cv2.imread('../第三部分/img.jpg')
my = cv2.resize(my,None,fx=0.5,fy=0.5)
face_detector = cv2.CascadeClassifier("D:/人工智能培训/OpenCV/haarcascades/haarcascade_frontalface_default.xml")
faces = face_detector.detectMultiScale(my)
for x,y,w,h in faces:
    cv2.rectangle(my,pt1=(x,y),pt2=(x+w,y+h),color=[0,0,225],thickness=2)
cv2.imshow("my",my)
cv2.waitKey(0)
cv2.destroyAllWindows()

##任务二
tv = cv2.imread('50.mp4')
faces = cv2.CascadeClassifier('D:/人工智能培训/OpenCV/haarcascades/haarcascade_upperbody.xml')
face = faces.detectMultiScale(tv)
for x, y, w, h in face:
    cv2.rectangle(tv, (x, y), (x + w, y + h), (0, 0, 255), 1)  # 画出框
cv2.imshow('tv', tv)
cv2.waitKey(0)
cv2.destroyAllWindows()
