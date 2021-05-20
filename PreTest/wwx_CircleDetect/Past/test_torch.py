# path: /torch

import subprocess

def detect(img):
	shape = img.shape[0:2]
	r = min(shape) / 2
	return r/3, shape[0]*3/4, shape[1]*3/4