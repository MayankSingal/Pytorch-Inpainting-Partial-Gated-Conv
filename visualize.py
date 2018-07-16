import cv2
import glob
import os

imList = glob.glob("samples/exp3/*.jpg")
imList.sort(key=os.path.getmtime)

for i in range(1,11):
	print(imList[-i])
	img = cv2.imread(imList[-i])
	img = img[...,::-1]
	cv2.imshow('test', img)
	cv2.waitKey(0) & 0xff
	

