import os
import sys
import cv2
import json
import numpy as np
import re
from PyQt5 import QtWidgets, QtCore, QtGui #pyqt stuff

QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True) #enable highdpi scaling
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True) #use highdpi icons

root_pt = os.path.dirname(sys.argv[0])
data_pt = os.path.join(root_pt, 'dataset')
bg_r_pt = os.path.join(data_pt, 'raw_background')	# 原始背景数据
ball_r_pt = os.path.join(data_pt, 'raw_ball')		# 原始球类数据


def view(img, name='view'):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyWindow(name)


def auto_padding(img, size=400):
	h, w = img.shape[0:2]
	border = np.abs((h - w) / 2).astype(np.int)
	if h > w:
		img = cv2.copyMakeBorder(img, 0, 0, border, border, cv2.BORDER_CONSTANT, value=(128,128,128))
	else:
		img = cv2.copyMakeBorder(img, border, border, 0, 0, cv2.BORDER_CONSTANT, value=(128,128,128))
	img = cv2.resize(img, (size, size))
	return img



def process_ball():
	json_pt = os.path.join(ball_r_pt, 'raw_ball.json')
	json_file = open(json_pt)
	ball_dict = json.load(json_file)
	json_file.close()
	size = 400
	for path in os.listdir(ball_r_pt):
		if '.json' in path:
			continue
		if path in ball_dict:
			continue
		else:
			img_pt = os.path.join(ball_r_pt, path)
			img = cv2.imread(img_pt)
			img = auto_padding(img, size)
			img_a = Annotate(img, path)
			if img_a.state == 3:
				break
			res = img_a.get_corner()
			if res is not None:
				ball_dict[path] = res
	json_str = json.dumps(ball_dict)
	json_str = re.sub(r'(, ")', ', \n\t\"', json_str)
	json_str = re.sub(r'{"', '{\n\t\"', json_str)
	json_str = re.sub(r']}', ']\n}', json_str)
	with open(json_pt, 'w') as json_file:
		json_file.write(json_str)


class Annotate:
	def __init__(self, img, name='default'):
		self.img = img
		self.name = name
		self.shape = img.shape[0:2]
		self.point1 = None
		self.point2 = None
		cv2.namedWindow(name)
		self.state = 0
		cv2.setMouseCallback(name, self.on_mouse)
		# cv2.startWindowThread()  # 加在这个位置
		cv2.imshow(name, img)
		while True:
			key = cv2.waitKey(0)
			print(key)
			if key == 13 or key == 32:	#按空格和回车键退出
				if self.state == 2:
					break
			elif key == 27:				# 按Esc键撤销
				self.state = 0
				self.point1 = None
				self.point2 = None
			elif key == 8:				# 按退格键强制停止标注过程
				self.state = 3
				break
			else:
				continue
		cv2.destroyWindow(self.name)

	def on_mouse(self, event, x, y, flags, param):
		img = self.img
		img2 = img.copy()
		if event == cv2.EVENT_MOUSEMOVE:
			if self.state == 0:
				max_h, max_w = self.shape
				cv2.line(img2, (x, 0), (x, max_h), (0, 255, 0), 2)
				cv2.line(img2, (0, y), (max_w, y), (0, 255, 0), 2)
				cv2.imshow(self.name, img2)
			elif self.state == 1:
				cv2.rectangle(img2, self.point1, (x, y), (255, 0, 0), thickness=2)
				cv2.imshow(self.name, img2)

		elif event == cv2.EVENT_LBUTTONDOWN:  # 左键点击,则在原图打点
			if self.state == 0:
				self.state = 1
				self.point1 = (x, y)
				cv2.circle(img2, self.point1, 3, (0, 255, 0), -1)
				cv2.imshow(self.name, img2)
			elif self.state == 1:
				self.point2 = (x, y)
				cv2.rectangle(img2, self.point1, self.point2, (0, 0, 255), thickness=2)
				cv2.imshow(self.name, img2)
				self.state = 2

	def get_corner(self):
		if self.state != 2:
			return None
		if self.point1 is None or self.point2 is None:
			return None
		x = (self.point1[0], self.point2[0])
		y = (self.point1[1], self.point2[1])
		return (min(x), min(y)), (max(x), max(y))


process_ball()


# cv2.copyMakeBorder()