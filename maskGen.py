import numpy as np
import cv2

def genMask(size, maxVertex, maxLength, maxBrushWidth, maxAngle, maxMasks):


	mask = np.zeros((size[0], size[1]), dtype=np.uint8)

	numMasks = np.random.randint(1,maxMasks+1)

	for j in range(numMasks):
		numVertex = np.random.randint(1, maxVertex+1)
		startX = int(np.random.randint(size[0]))
		startY = int(np.random.randint(size[1]))

		for i in range(numVertex):
			angle = np.random.randint(maxAngle+1) * np.pi/180
			if (i%2 == 0):
				angle = 2*np.pi - angle
			length = np.random.randint(1,maxLength+1)
			brushWidth = np.random.randint(1,maxBrushWidth+1)

			X = int(round(startX + length*np.sin(angle)))
			Y = int(round(startY + length*np.cos(angle)))

			cv2.line(mask, (startX, startY), (X,Y), (1,1,1), brushWidth)
			cv2.circle(mask, (startX, startY), int(brushWidth/2), (1,1,1), -1)

			startX = X
			startY = Y

	return np.reshape(mask,(size[0],size[1],1))



def test():
	for i in range(10):
		cv2.imshow('test',genMask((300,300), 8, 100, 40, 270,3))
		cv2.waitKey(0) & 0xff
