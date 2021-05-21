import os
import sys
import cv2 as cv
import numpy as np
import formal as my
import pandas as pd


if __name__ == '__main__':
	path = os.path.dirname(os.path.dirname(sys.argv[0])) + r'\data' 
	path_list = os.listdir(path) #遍历整个文件夹下的文件name并返回一个列表
	print(path_list)
	dataset = my.Dataset(path)
	dataset.load()
	for img in dataset:
		r, x, y = my.detect(img)
		cv.circle(img, (int(x), int(y)), int(r), (0, 0, 255), -1)
		my._view(img, 'result')
