# path: /base

import os
import sys

LOCAL_PATH = os.path.join(os.path.dirname(sys.argv[0]), 'local')

if os.path.exists(LOCAL_PATH):
	LOCAL = True
	import cv2 as cv
	import numpy as np
	import cv2 as cv
	import numpy as np
	import matplotlib
	import  matplotlib.pyplot as plt
	from PyQt5 import QtWidgets, QtCore, QtGui #pyqt stuff
	QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True) #enable highdpi scaling
	QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True) #use highdpi icons
else:
	LOCAL = False

def transform(img):
	img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
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
	img = cv.resize(img, (w, h))
	img = cv.GaussianBlur(img, (5, 5), 2)
	return img, coef

def detect(img):
	img, coef = transform(img)
	circles = cv.HoughCircles(img, dp=1.5, minDist=50, method=cv.HOUGH_GRADIENT, minRadius=20, maxRadius=60, param1=25, param2=0.9)
	if circles is not None:
		x, y, r = circles[0, 0, :]
		return int(r*coef), int(x*coef), int(y*coef)
	return 0, 0, 0 
