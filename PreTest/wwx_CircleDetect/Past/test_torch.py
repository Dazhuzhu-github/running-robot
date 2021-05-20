# path: /torch
# 用来测试一些奇思妙想的东西

import subprocess
from abc import ABCMeta, abstractmethod, ABC

def detect(img):
	a = lambda x: x
	try:
		b = 1
	except:
		pass
	shape = img.shape[0:2]
	r = min(shape) / 2
	return r/3, shape[0]*3/4, shape[1]*3/4