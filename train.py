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
	res50_model = torchvision.models.resnet50(pretrained=True)
	res50_conv = nn.Sequential(*list(res50_model.children())[:-1])
	res50_conv.cuda()

	for param in res50_conv.parameters():
		param.requires_grad = False

	#datafeeder = dataloader.inpaintLoader('/home/user/data/places365_standard/train')
	datafeeder = dataloader.inpaintLoader('/home/user/data/img_celeba')
	train_loader = torch.utils.data.DataLoader(datafeeder, batch_size=8, 
											  shuffle=True, num_workers=2,
											  pin_memory=True)

	criterion = nn.L1Loss().cuda()

	optimizer = torch.optim.Adam(UNet.parameters(), lr=0.0001)

	UNet.train()

	for epoch in range(5):
		for i, (img, mask, masked_img, masked_seg) in enumerate(train_loader):
			
			img_var = Variable(img).cuda()
			mask_var = Variable(mask).cuda()
			masked_img_var = Variable(masked_img).cuda()

			## Generate Image
			gen_img = UNet(masked_img_var, mask_var)

			## Compute Variables for Perceptual loss
			feat_img = res50_conv(img_var)
			feat_gen_img = res50_conv(gen_img)

			## Compute Variables for Style Loss
			b, c, h, w = feat_img.size()
			feat_img = feat_img.view(b, c, h*w)
			feat_gen_img = feat_gen_img.view(b, c, h*w)
			gramMatrix_feat_img = torch.bmm(feat_img, feat_img.transpose(1,2)) / (c*h*w)
			gramMatrix_feat_gen_img = torch.bmm(feat_gen_img, feat_gen_img.transpose(1,2)) / (c*h*w)

			## Compute Total Variation Loss
			# X Direction
			TV_img = torch.mean(torch.abs(img_var[:, :, :, :-1] - img_var[:, :, :, 1:])) + torch.mean(torch.abs(img_var[:, :, :-1, :] - img_var[:, :, 1:, :]))
			TV_gen_img = torch.mean(torch.abs(gen_img[:, :, :, :-1] - gen_img[:, :, :, 1:])) + torch.mean(torch.abs(gen_img[:, :, :-1, :] - gen_img[:, :, 1:, :]))

			
			loss = criterion(gen_img*(1-mask_var), img_var*(1-mask_var)) + 6*criterion(gen_img*mask_var, img_var*mask_var) + \
							0.05*criterion(feat_gen_img, feat_img) + \
							120*criterion(gramMatrix_feat_gen_img, gramMatrix_feat_img) + \
							0.1*criterion(TV_gen_img, TV_img)

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