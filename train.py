import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
from torch.autograd import Variable
import os
import sys
import dataloader
import network
import numpy as np


def train_val():

	UNet = network.UNet_inpaint().cuda()

	datafeeder = dataloader.inpaintLoader('/home/user/data/places365_standard/train')
	train_loader = torch.utils.data.DataLoader(datafeeder, batch_size=16, 
											  shuffle=True, num_workers=1,
											  pin_memory=True)

	criterion = nn.L1Loss().cuda()

	optimizer = torch.optim.Adam(UNet.parameters(), lr=0.0001)

	UNet.train()

	for epoch in range(5):
		for i, (img, mask, masked_img, masked_seg) in enumerate(train_loader):
			
			img_var = Variable(img).cuda()
			mask_var = Variable(mask).cuda()
			masked_img_var = Variable(masked_img).cuda()

			gen_img = UNet(masked_img_var, mask_var)
			
			loss = criterion(gen_img*(1-mask_var), img_var*(1-mask_var)) + 6*criterion(gen_img*mask_var, img_var*mask_var)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if ((i+1) % 10) == 0:
				print("Loss at iteration", i+1, ":", loss.item())
			if ((i+1) % 100) == 0:
				torchvision.utils.save_image((torch.cat((gen_img,img_var),0)+1)/2,'samples/'+ str(i+1) +'.jpg')
				#torchvision.utils.save_image((img_var+1)/2,'samples/'+ str(i+1) +'orig.jpg')
			




if __name__ == '__main__':
	train_val()