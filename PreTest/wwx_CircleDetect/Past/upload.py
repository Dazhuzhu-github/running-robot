from ctypes import c_double
import os
import sys
from tokenize import Double
import cv2 as cv
import numpy as np
import matplotlib
import  matplotlib.pyplot as plt

import sys
from PyQt5 import QtWidgets, QtCore, QtGui #pyqt stuff

QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True) #enable highdpi scaling
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True) #use highdpi icons


def _view(img, resize=True, size=300):
	if resize:
		img = AutoScale(size)(img)
	cv.imshow('view', img)
	cv.waitKey(0)
	cv.destroyWindow('view')
	return


class Transform:
	def __init__(self):
		pass


class CvtColor(Transform):
	def __init__(self, type: str):
		super().__init__()
		self.type = type

	def __call__(self, img):
		"""
			根据type返回不同褪色方式的图片。
			Returns:
				img: 褪色的图片
		"""
		if self.type == 'BGR2GRAY':
			return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		elif self.type == 'BGR2B-G':
			img = img[:, :, 0] - img[:, :, 1]
			img[img<=0] = 0
			return img
		elif self.type == 'BGR2G-R':
			img = img[:, :, 1] - img[:, :, 2]
			img[img<=0] = 0
			return img
		elif self.type == 'BGR2R-B':
			img = img[:, :, 2] - img[:, :, 0]
			img[img<=0] = 0
			return img
		elif self.type == 'BGR2B':
			return img[:, :, 0]
		elif self.type == 'BGR2G':
			return img[:, :, 1]
		elif self.type == 'BGR2R':
			return img[:, :, 2]
		elif self.type == 'BGR2H':
			return cv.cvtColor(img, cv.COLOR_BGR2HSV)[:, :, 0] 
		elif self.type == 'BGR2S':
			return cv.cvtColor(img, cv.COLOR_BGR2HSV)[:, :, 1]
		elif self.type == 'BGR2V':
			return cv.cvtColor(img, cv.COLOR_BGR2HSV)[:, :, 2]
		else:
			raise NotImplementedError


class AutoScale(Transform):
	def __init__(self, size=300):
		super().__init__()
		self.size = size

	def __call__(self, img):
		if len(img.shape) == 2:
			h, w = img.shape
		elif len(img.shape) == 3:
			h, w, _ = img.shape
		else:
			raise ValueError
		coef = min(h, w) / self.size
		h = int(h / coef)
		w = int(w / coef)
		img = cv.resize(img, (w, h))
		return img

	@staticmethod
	def get_coef(img, size):
		# 计算图片缩放系数
		return min(img.shape) / size


class Blur(Transform):
	def __init__(self, type: str, **argw):
		self.type = type
		self.param = dict(argw)

	def __call__(self, img):
		if self.type == 'Gaussian':
			# ksize, sigmaX
			return cv.GaussianBlur(img, **self.param)
		else:
			raise TypeError


class Detect:
	def __init__(self):
		pass


class HoughCircle(Detect):
	def __init__(self, **argw):
		super().__init__()
		self.param = dict(argw)
		if 'auto' not in self.param:
			self.param['auto'] = False

	def __call__(self, img):
		if not self.param['auto']:
			return self._detect(img, **self.param)
		else:
			return self._auto(img)
	
	def _auto(self, img):
		raise NotImplementedError

	def _detect(self, img, method, dp, minDist, param1, param2, minRadius, maxRadius, **argw):
		# arg_list = ['method', 'dp', 'minDist', 'param1', 'param2', 'minRadius', 'maxRadius']
		return cv.HoughCircles(img, 
			method=method, 
			dp=dp, 
			minDist=minDist, 
			param1=param1, 
			param2=param2, 
			minRadius=minRadius, 
			maxRadius=maxRadius)


class Hook:
	def __init__(self, callback, **argw):
		"""
			Hook类：勾取forward运算过程中的信息；
			callback: 回调函数
			argw: 回调函数参数
		"""
		self.callback = callback
		self.params = argw

	def __call__(self, x):
		self.callback(x, **self.params)
		# 执行回调函数，但返回x本身
		return x


class ModuleList:
	def __init__(self, modules: list):
		self.modules = modules
		
	def __call__(self, img):
		for module in self.modules:
			img = module(img)
		return img

	def __getitem__(self, idx):
		return self.modules[idx]

	def __len__(self):
		return len(self.modules)


class Dataset:
	def __init__(self, path):
		self.path = path
		self.img_path: list
		self.imgs: list
	
	def load(self):
		self.img_path = []
		self.imgs = []
		files = os.listdir(self.path)
		for file in files:
			if 'jpg' in file or 'png' in file:
				self.img_path.append(file)
		for p in self.img_path:
			img = cv.imread(os.path.join(self.path, p))
			self.imgs.append(img)

	def __getitem__(self, idx):
		return self.imgs[idx]

	def __len__(self):
		return len(self.imgs)


class Paint:
	def __init__(self):
		pass


class CircleLayer(Paint):
	def __init__(self, circles=None):
		# self.circles.shape == (1, circle_num, 3)
		if isinstance(circles, CircleLayer):
			self.circles = np.ndarray(circles.circles)
		elif isinstance(circles, np.ndarray):
			assert(circles.shape[0] == 1)
			self.circles = circles
		elif circles is None:
			self.circles = None
		else:
			raise TypeError

	def __call__(self, shape, color=1):
		img = np.zeros(shape, dtype=np.float)
		if self.circles is None:
			return img
		assert(self.circles.shape[0] == 1)
		for circle in self.circles[0, :, :]:
			x, y, r = circle
			cv.circle(img, (x, y), int(r), color, -1)
		return img


class CircleModule(Paint):
	def __init__(self, layers=None):
		if layers is None:
			self.layers = []
		elif isinstance(layers, CircleModule):
			self.layers = layers
		elif isinstance(layers, list):
			self.layers = [CircleLayer(layer) for layer in layers]
		elif isinstance(layers, CircleLayer):
			self.layers = [layers]
		elif isinstance(layers, np.ndarray or np.array):
			self.layers = [CircleLayer(layers)]
		else:
			raise TypeError

	def __call__(self, shape, num=None):
		# 涂色的深浅
		color = 1/num if num is not None else 1/len(self.layers)
		shape = shape[0:2]
		img = np.zeros(shape)
		for layer in self.layers:
			layer: CircleLayer
			img += layer(shape, color)
		return img

	def __iadd__(self, obj):
		if isinstance(obj, CircleLayer):
			self.layers.append(obj)
		elif isinstance(obj, CircleModule):
			self.layers.extend(obj.layers)
		else:
			raise TypeError
		return self


class Classifier:
	def __init__(self, modules, **argw):
		self.modules = modules

	def __call__(self, img):
		shape = img.shape[0:2]
		for module in self.modules:
			if type(module) == AutoScale:
				coef = min(shape) / module.size
			img = module(img)
		# 返回绘制出的圆，原图片大小
		return CircleLayer() if img is None else CircleLayer(img * coef)


class ClassifierGroup:
	def __init__(self, classifiers: list, **argw):
		if isinstance(classifiers, list):
			self.classifiers = classifiers
		# elif isinstance(classifiers, Classifier):
		# 	self.classifiers = []
		else:
			raise TypeError

	def __call__(self, img):
		features = CircleModule()
		for classifier in self.classifiers:
			features += classifier(img)
		feature_img = features(img.shape)
		return feature_img


modules1 = ModuleList([
	AutoScale(),
	CvtColor('BGR2B-G'),
	Blur('Gaussian', ksize=(5, 5), sigmaX=2),
	#Hook(_view),
	HoughCircle(dp=1, minDist=150, method=cv.HOUGH_GRADIENT, minRadius=20, maxRadius=70, param1=200, param2=25)
])

modules2 = ModuleList([
	AutoScale(),
	CvtColor('BGR2G-R'),
	Blur('Gaussian', ksize=(5, 5), sigmaX=2),
	#Hook(_view),
	HoughCircle(dp=1, minDist=150, method=cv.HOUGH_GRADIENT, minRadius=20, maxRadius=70, param1=200, param2=25)
])

modules3 = ModuleList([
	AutoScale(),
	CvtColor('BGR2R-B'),
	Blur('Gaussian', ksize=(5, 5), sigmaX=2),
	#Hook(_view),
	HoughCircle(dp=1, minDist=150, method=cv.HOUGH_GRADIENT, minRadius=20, maxRadius=70, param1=200, param2=25)
])

modulesF = ModuleList([
	AutoScale(),
	CvtColor('BGR2GRAY'),
	Blur('Gaussian', ksize=(5, 5), sigmaX=2),
	#Hook(_view),
	HoughCircle(dp=1, minDist=150, method=cv.HOUGH_GRADIENT, minRadius=20, maxRadius=70, param1=200, param2=25)
])


def detect(readimg):
	cg = ClassifierGroup([
		Classifier(modules1),
		Classifier(modules2),
		Classifier(modules3)
	])
	vote = Classifier(ModuleList([
		AutoScale(),
		HoughCircle(dp=1, minDist=150, method=cv.HOUGH_GRADIENT, minRadius=20, maxRadius=70, param1=200, param2=25)
	]))
	# cg1 = ClassifierGroup([
	# 	Classifier(modules1)
	# ])
	# cg2 = ClassifierGroup([
	# 	Classifier(modules2)
	# ])
	# cg3 = ClassifierGroup([
	# 	Classifier(modules2)
	# ])
	# cgF = ClassifierGroup([
	# 	Classifier(modulesF)
	# ])
	# feature_img1 = cg1(readimg)
	# feature_img2 = cg2(readimg)
	# feature_img3 = cg3(readimg)
	# _view(feature_img1)
	# _view(feature_img2)
	# _view(feature_img3)
	# alpha = 0.5
	# beta = 0.5
	# gamma = 0
	# img_add = cv.addWeighted(feature_img1, alpha, feature_img2, beta, gamma)
	# img_add2 = cv.addWeighted(img_add, 0.67, feature_img3, 0.33, gamma)
	# _view(img_add2)
	feature_img = (cg(readimg) * 255).astype(np.uint8)
	# feature_img = np.array(img_add2*255, dtype = np.uint8)
	#final = img_add2*180
	feature_img[feature_img < feature_img.max()] = 0
	# circles = cv.HoughCircles(feature_img, dp=1.5, minDist=100, method=cv.HOUGH_GRADIENT, 
	# minRadius=5, maxRadius=100, param1=25, param2=9)
	circles = vote(feature_img).circles
	if circles is not None:
		x, y, r = circles[0, 0, :]
		print(r,x,y)
		#return r, x, y
		cv.circle(readimg,(x,y), int(r), (0,0,255), -1)
		cv.imshow('image', readimg)
		cv.waitKey (0) # 显示 10000 ms 即 10s 后消失
		cv.destroyAllWindows()

	
if __name__ == '__main__':
	path = os.path.dirname(os.path.dirname(sys.argv[0])) + r'\data' 
	path_list = os.listdir(path) #遍历整个文件夹下的文件name并返回一个列表
	print(path_list)
	dataset = Dataset(path)
	dataset.load()
	# cg = ClassifierGroup([
	# 	Classifier(modules1)
	# ])
	for i, img in enumerate(dataset):
		detect(img)
	# 	coef = feature_img.shape[0] / img.shape[0]
	# 	_view(feature_img)
	# 	_view(img)
	# 	# result = input()
	# 	# with open('DetectFilters.txt','a') as f:
	# 	# 	f.write(path_list[i]+' '+result+'\n')
	pass