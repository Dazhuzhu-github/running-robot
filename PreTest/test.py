import cv2
import numpy as np
import sys 
from PyQt5 import QtWidgets,QtCore
#pyqt stuff

QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling,True)
#enable highdpi scaling
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps,True)
#use highdpi icon

def detect(readimg):
    img = cv2.cvtColor(readimg, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    c = 300
    coef = None
    if h > w:
        coef = w / c
        h = int(h / w * c)
        w = c
    else:
        coef = h / c
        w = int(w / h * c)
        h = c
    img = cv2.resize(img, (w, h))
    img = cv2.GaussianBlur(img, (5, 5), 2)
    circles = cv2.HoughCircles(img, dp=1.5, minDist=50, method=cv2.HOUGH_GRADIENT, minRadius=20, maxRadius=60, param1=25, param2=0.9)
    if circles is not None:
        x, y, r = circles[0, 0, :]
        return int(r*coef), int(x*coef), int(y*coef)
    return 0, 0, 0
#读入图片：默认彩色图，cv2.IMREAD_GRAYSCALE灰度图，cv2.IMREAD_UNCHANGED包含alpha通道
img = cv2.imread('2.png')
cv2.imshow('src',img)
# print(img.shape) # (h,w,c)
# print(img.size) # 像素总数目
# print(img.dtype)
# print(img)


r,x,y = detect(img)
# waitKey参数为0的时候窗口不会自动关闭，无限等待按键，假如设为10000，也就是10s，不做任何操作的情况# 下大概15s左右窗口自动关闭

r, g, b = 255, 0, 0

cv2.circle(img,(x,y),r,(b,g,r),10)
cv2.waitKey(0)
