# path: /formal

from abc import abstractmethod
import os
import sys
from abc import ABCMeta, abstractmethod, ABC

LOCAL_PATH = os.path.join(os.path.dirname(sys.argv[0]), 'local')
VIEW = 1

if os.path.exists(LOCAL_PATH):
	LOCAL = True
	import cv2 as cv
	import numpy as np
	import matplotlib
	import  matplotlib.pyplot as plt
	from PyQt5 import QtWidgets, QtCore, QtGui #pyqt stuff
	QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True) #enable highdpi scaling
	QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True) #use highdpi icons
else:
	LOCAL = False

def _view(img, name='view', resize=True, size=300):
	if not LOCAL or not VIEW:
		return
	if resize:
		img = AutoScale(size)(img)
	cv.imshow(name, img)
	cv.waitKey(0)
	cv.destroyWindow(name)
	return

class View:
	def __init__(self, type):
		global VIEW
		self.type = type
		self.last = VIEW

	def __enter__(self):
		global VIEW
		VIEW = self.type

	def __exit__(self, exc_type, exc_val, exc_tb):
		global VIEW
		VIEW = self.last

	def no_view():
		return View(0)


class Transform:
	def __init__(self):
		pass


class Identity:
	def __init__(self):
		pass
	
	def __call__(self, img):
		return img


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
		elif self.type == 'BGR2B-R':
			img = img[:, :, 0] - img[:, :, 2]
			img[img<=0] = 0
			return img
		elif self.type == 'BGR2G-R':
			img = img[:, :, 1] - img[:, :, 2]
			img[img<=0] = 0
			return img
		elif self.type == 'BGR2G-B':
			img = img[:, :, 1] - img[:, :, 0]
			img[img<=0] = 0
			return img
		elif self.type == 'BGR2R-B':
			img = img[:, :, 2] - img[:, :, 0]
			img[img<=0] = 0
			return img
		elif self.type == 'BGR2R-G':
			img = img[:, :, 2] - img[:, :, 1]
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


class Blur(Transform):
	def __init__(self, type: str, *args, **argw):
		self.type = type
		self.args = args
		self.argw = argw

	def __call__(self, img):
		if self.type == 'Gaussian':
			# ksize, sigmaX
			return cv.GaussianBlur(img, *self.args, **self.argw)
		elif self.type == 'BiFilter':
			# 双边滤波
			return cv.bilateralFilter(img, *self.args, **self.argw)
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
		img = img.astype(np.uint8)
		circles =  cv.HoughCircles(img, 
			method=method, 
			dp=dp, 
			minDist=minDist, 
			param1=param1, 
			param2=param2, 
			minRadius=minRadius, 
			maxRadius=maxRadius)
		return CircleLayer(circles)


class Hook:
	cache = {}
	last = None
	def __init__(self, callback, *args, **argw):
		"""
			Hook类：勾取forward运算过程中的信息；
			callback: 回调函数
			args, argw: 回调函数参数
		"""
		self.callback = callback if callback is not None else Identity()	# 回调函数
		self.args = args
		self.argw = argw			# 函数参数

	def __call__(self, x):
		x_ = x
		Hook.last = self.callback(x_, *self.args, **self.argw)
		# 执行回调函数，但返回x本身
		return x

	def call_x(self, x, *args, **argw):
		return x(*args, **argw)

	def hook(self, x, name=None, *args, **argw):
		if name is None:
			raise ValueError('"name" should not be None')
		Hook.cache[name] = x
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


class Paint(ABC):
	def __init__(self):
		pass


class CircleLayer(Paint):
	def __init__(self, circles=None):
		# self.circles.shape == (1, circle_num, 3)
		if isinstance(circles, CircleLayer):
			self.circles = circles.circles.copy()
		elif isinstance(circles, np.ndarray):
			assert(circles.shape[0] == 1)
			self.circles = circles
		elif circles is None:
			# self.circles = np.array([-1 ,-1, 0.]).reshape([1, 1, 3])
			self.circles = None
		else:
			raise TypeError

	def __call__(self, x, color=1):
		if isinstance(x, np.ndarray):
			shape = x.shape[0:2]
		elif isinstance(x, tuple):
			shape = x
		else:
			raise TypeError
		img = np.zeros(shape, dtype=np.float)
		if self.circles is None:
			return img
		assert(self.circles.shape[0] == 1)
		for circle in self.circles[0, :, :]:
			x, y, r = circle
			cv.circle(img, (int(x), int(y)), int(r), color, -1)
		return img

	def __mul__(self, times):
		if self.circles is None:
			return CircleLayer()
		else:
			return CircleLayer(self.circles * times)

	def __rmul__(self, times):
		if self.circles is None:
			return CircleLayer()
		else:
			return CircleLayer(self.circles * times)


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
		return img * coef

	@staticmethod
	def color_only(ctype, blur='Gaussian', preview=True):
		r'''
		Return a Classifier object, which serves as a baseline.

		Args:
			ctype(str): color type, to decide the CvtColor method
			preview(bool): whether to view the img after GaussianBlur
		'''
		if blur == 'Gaussian':
			blur_layer = Blur('Gaussian', ksize=(5, 5), sigmaX=2)
		elif blur == 'BiFilter':
			blur_layer = Blur('BiFilter', 9, 75, 75)
		else:
			raise ValueError(blur)
		return Classifier(
			ModuleList([
				AutoScale(),
				CvtColor(ctype),
				blur_layer,
				Hook(_view, ctype) if preview else Identity(),
				HoughCircle(dp=1, minDist=150, method=cv.HOUGH_GRADIENT, minRadius=20, maxRadius=70, param1=200, param2=25),
			])
		)


class ClassifierGroup:
	def __init__(self, classifiers: list, **argw):
		if isinstance(classifiers, list):
			self.classifiers = classifiers
		else:
			raise TypeError

	def __call__(self, img):
		features = CircleModule()
		for classifier in self.classifiers:
			features += classifier(img)
		feature_img = features(img.shape)
		return feature_img

	def __getitem__(self, idx):
		return self.classifiers[idx]

	def __len__(self):
		return len(self.classifiers)


def detect(img):
	classifiers = [
		Classifier.color_only('BGR2B-G'),
		Classifier.color_only('BGR2B-R'),
		Classifier.color_only('BGR2R-B'),
		Classifier.color_only('BGR2R-G'),
		Classifier.color_only('BGR2G-B'),
		Classifier.color_only('BGR2G-R'),
		Classifier.color_only('BGR2H'),
		Classifier.color_only('BGR2GRAY'),
	]

	
	cg = ClassifierGroup(classifiers)

	vote = Classifier(
		ModuleList([
			AutoScale(),
			HoughCircle(dp=1, minDist=150, method=cv.HOUGH_GRADIENT, minRadius=20, maxRadius=70, param1=50, param2=10)
		])
	)

	for c in cg:
		circles = c(img)
		img_ = circles(img)
		_view(img_, 'single layer')
	
	with View.no_view():
		feature_img = cg(img)
	_view(feature_img, 'before')
	feature_img[feature_img < feature_img.max()] = 0
	coef = 255 / feature_img.max()
	feature_img = (feature_img * coef).astype(np.uint8)
	_view(feature_img, 'after')
	circles = vote(feature_img).circles
	if circles is not None:
		print(circles.shape)
		x, y, r = circles[0, 0, :]
		return r, x, y
	return 0, 0, 0

	
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


def detect_base(img):
	img, coef = transform(img)
	circles = cv.HoughCircles(img, dp=1.5, minDist=50, method=cv.HOUGH_GRADIENT, minRadius=20, maxRadius=60, param1=100, param2=25)
	if circles is not None:
		x, y, r = circles[0, 0, :]
		return int(r*coef), int(x*coef), int(y*coef)
	return 0, 0, 0 


# if __name__ == '__main__' and LOCAL:
# 	path = os.path.dirname(os.path.dirname(sys.argv[0])) + r'\data' 
# 	path_list = os.listdir(path) #遍历整个文件夹下的文件name并返回一个列表
# 	print(path_list)
# 	dataset = Dataset(path)
# 	dataset.load()
# 	for i, img in enumerate(dataset):
# 		detect(img)
# 	pass