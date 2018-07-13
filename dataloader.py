import torch
import torch.utils.data as data
import numpy as np
import cv2
from PIL import Image
import glob
import os
import random
from maskGen import genMask

def populateTrainList(folderPath):
	folderList = [x[0] for x in os.walk(folderPath)]

	trainList = []

	for folder in folderList:
		imageList = sorted(glob.glob(folder + '/' + '*.jpg'))
		trainList += imageList

	return trainList


class inpaintLoader(data.Dataset):

	def __init__(self, folderPath):

		self.trainList = populateTrainList(folderPath)
		print("# of training samples:", len(self.trainList))

	def __getitem__(self, index):

		img_path = self.trainList[index]
		img = np.array(cv2.imread(img_path), dtype=np.float32)
		h,w,c = img.shape

		# Normalizing
		img = (img/127.5) - 1

		mask = np.array(genMask((h,w),8,100,30,270,5), dtype=np.float32)

		masked_img = img*(1-mask)
		masked_seg = img*mask

		img = torch.from_numpy(img.transpose((2,0,1)))
		mask = torch.from_numpy(mask.transpose((2,0,1)))
		masked_img = torch.from_numpy(masked_img.transpose((2,0,1)))
		masked_seg = torch.from_numpy(masked_seg.transpose((2,0,1)))

		return img, mask, masked_img, masked_seg

	def __len__(self):
		return len(self.trainList)

