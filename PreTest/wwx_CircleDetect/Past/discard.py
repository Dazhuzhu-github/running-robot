'''
    目前处于报废阶段，勿用
'''

import os
import sys
import cv2 as cv
import numpy as np
import matplotlib
import  matplotlib.pyplot as plt
import my

import sys
from PyQt5 import QtWidgets, QtCore, QtGui #pyqt stuff

QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True) #enable highdpi scaling
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True) #use highdpi icons

img_path = []
imgs = []
view_size = 300


def BGR2GRAY(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def BGR2H(img):
    return cv.cvtColor(img, cv.COLOR_BGR2HSV)[:, :, 0]

def _auto_scale(img, color=BGR2GRAY):
    if color is not None:
        img = color(img)
    h, w = None, None
    if len(img.shape) == 2:
        h, w = img.shape
    elif len(img.shape) == 3:
        h, w, _ = img.shape
    else:
        raise ValueError

    coef = min(h, w) / view_size
    h = int(h / coef)
    w = int(w / coef)
    img = cv.resize(img, (w, h))
    return img, coef


def _view(img, resize=True):
    img_ = None
    if resize:
        img_, _ = _auto_scale(img, color=None)
    else:
        img_ = img
    cv.imshow('view', img_)
    cv.waitKey(0)
    cv.destroyWindow('view')
    return


def transform(img, kernel=(5, 5), std=2, color=BGR2GRAY, **argw):
    img, coef = _auto_scale(img, color=color)
    if kernel is not None:
        img = cv.GaussianBlur(img, kernel, std)
    _view(img)
    return img, coef


def hough_detect(img, **argw):
    # img, method, dp, minDist, param1, param2, minRadius, maxRadius
    argw_ = dict(argw)
    trans = argw_.pop('transform')
    img, coef = trans['func'](img, **trans['argw'])
    # img, coef = transform(trans, **argw_)
    circles = cv.HoughCircles(img, **argw_)
    return circles if circles is None else circles * coef


# def detect_all0(img: np.ndarray):
#     # auto draw the circle
#     img, coef = transform(img, kernel=(3, 3), std=0)
#     circles = cv.HoughCircles(img, dp=3, minDist=50, method=cv.HOUGH_GRADIENT_ALT, minRadius=20, maxRadius=80, param1=50, param2=0.6)
#     if circles is not None:
#         for circle in circles[:, 0, :]:
#             x, y, r = circle
#             cv.circle(img, (x, y), int(r), (0, 0, 255), 4)
#             cv.circle(img, (x, y), 2, (0, 0, 255), 8)
#     _view(img)
#     return img


def prepare(path="./"):
    files = os.listdir(path)
    for file in files:
        if 'jpg' in file or 'png' in file:
            img_path.append(file)
    for p in img_path:
        img = cv.imread(os.path.join(path, p))
        imgs.append(img)


# def test(idx, **argw):
#     prepare()
#     for img in imgs:
#         res = eval(f'detect{idx}')(img, **argw)
#         if res is not None:
#             x, y, r = res
#             cv.circle(img, (x, y), int(r), (0, 0, 255), 2)
#             cv.circle(img, (x, y), 2, (0, 0, 255), 4)
#         _view(img)


# def test_all(idx, *args):
#     prepare()
#     for img in imgs:
#         img = eval(f'detect_all{idx}')(img)
#         _view(img)
#     return


def stack_detect(img, params):
    num = len(params)
    shape = img.shape[0:2]
    flag_img = np.zeros(shape)   # grey
    coef = min(shape) / view_size
    print('Detect...')
    for idx, param in enumerate(params):
        param_ = dict(param)
        method = param_.pop('detect')
        circles = method(img, **param_)
        if circles is None:
            continue
        tmp_img = np.zeros(shape)
        for circle in circles[0, :, :]:
            x, y, r = circle
            cv.circle(tmp_img, (x, y), 0, 1, int(r))
            print(f'{idx}: {x/coef},{y/coef},{r/coef}')
        flag_img += tmp_img
    flag_img /= num
    # if flag_img.max() >1e-5:
    #     print(flag_img.max())
    #     pass
    _view(img)  
    _view(flag_img)
    _view(img)        
        

if __name__ == '__main__':
    path = os.path.dirname(os.path.dirname(sys.argv[0])) + r'\data' 
    prepare(path)
    trans_h = {
        'func': transform,
        'argw': {
            #'kernel': None
            'color': BGR2H
        }
    }
    trans_gray = {
        'func': transform,
        'argw': {
        }
    }
    params = [
        #dict(detect=hough_detect, transform=trans_gray, dp=1, minDist=20, method=cv.HOUGH_GRADIENT, minRadius=20, maxRadius=90, param1=150, param2=35),
        dict(detect=hough_detect, transform=trans_h, dp=1, minDist=20, method=cv.HOUGH_GRADIENT, minRadius=20, maxRadius=90, param1=200, param2=25),

        # past:
        # dict(detect=hough_detect, dp=1.5, minDist=50, method=cv.HOUGH_GRADIENT_ALT, minRadius=20, maxRadius=80, param1=25, param2=0.9),
        # dict(detect=hough_detect, dp=1.5, minDist=50, method=cv.HOUGH_GRADIENT_ALT, minRadius=20, maxRadius=80, param1=25, param2=0.6),
        # dict(detect=hough_detect, transform=trans1, dp=1, minDist=20, method=cv.HOUGH_GRADIENT, minRadius=30, maxRadius=90, param1=50, param2=25),
        # dict(detect=hough_detect, transform=trans1, dp=1, minDist=20, method=cv.HOUGH_GRADIENT, minRadius=30, maxRadius=90, param1=75, param2=25),
        # dict(detect=hough_detect, transform=trans1, dp=1, minDist=20, method=cv.HOUGH_GRADIENT, minRadius=30, maxRadius=90, param1=100, param2=25),
        # dict(detect=hough_detect, transform=trans1, dp=1, minDist=20, method=cv.HOUGH_GRADIENT, minRadius=30, maxRadius=90, param1=150, param2=25),
    ]
    for idx, img in enumerate(imgs):
        # _view(img, resize=False)
        print(f'{img_path[idx]}')
        stack_detect(img, params)
