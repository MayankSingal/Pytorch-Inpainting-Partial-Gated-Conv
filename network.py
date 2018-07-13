import torch
import torch.nn as nn
import math

class UNet_inpaint(nn.Module):

	def __init__(self):
		super(UNet_inpaint, self).__init__()

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.leakyRelu = nn.LeakyReLU(negative_slope=0.2)
		self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

		## Encoder 
		self.E_conv1 = nn.Conv2d(3,64,7,2,3,bias=False)
		self.E_conv1_mask = nn.Conv2d(3,1,7,2,3,bias=False)

		self.E_conv2 = nn.Conv2d(64,128,5,2,2,bias=False)
		self.E_conv2_mask = nn.Conv2d(64,1,5,2,2,bias=False)

		self.E_conv3 = nn.Conv2d(128,256,5,2,2,bias=False)
		self.E_conv3_mask = nn.Conv2d(128,1,5,2,2,bias=False)

		self.E_conv4 = nn.Conv2d(256,512,3,2,1,bias=False)
		self.E_conv4_mask = nn.Conv2d(256,1,3,2,1,bias=False)

		self.E_conv5 = nn.Conv2d(512,512,3,2,1,bias=False)
		self.E_conv5_mask = nn.Conv2d(512,1,3,2,1,bias=False)

		self.E_conv6 = nn.Conv2d(512,512,3,2,1,bias=False)
		self.E_conv6_mask = nn.Conv2d(512,1,3,2,1,bias=False)

		self.E_conv7 = nn.Conv2d(512,512,3,2,1,bias=False)
		self.E_conv7_mask = nn.Conv2d(512,1,3,2,1,bias=False)

		## Decoder
		self.D_conv1 = nn.Conv2d(1024,512,3,1,1,bias=False)
		self.D_conv1_mask = nn.Conv2d(1024,1,3,1,1,bias=False)

		self.D_conv2 = nn.Conv2d(1024,512,3,1,1,bias=False)
		self.D_conv2_mask = nn.Conv2d(1024,1,3,1,1,bias=False)

		self.D_conv3 = nn.Conv2d(1024,512,3,1,1,bias=False)
		self.D_conv3_mask = nn.Conv2d(1024,1,3,1,1,bias=False)

		self.D_conv4 = nn.Conv2d(512+256,256,3,1,1,bias=False)
		self.D_conv4_mask = nn.Conv2d(512+256,1,3,1,1,bias=False)

		self.D_conv5 = nn.Conv2d(256+128,128,3,1,1,bias=False)
		self.D_conv5_mask = nn.Conv2d(256+128,1,3,1,1,bias=False)

		self.D_conv6 = nn.Conv2d(128+64,64,3,1,1,bias=False)
		self.D_conv6_mask = nn.Conv2d(128+64,1,3,1,1,bias=False)

		self.D_conv7 = nn.Conv2d(64+3,3,3,1,1,bias=False)
		self.D_conv7_mask = nn.Conv2d(64+3,1,3,1,1,bias=False)

	def forward(self, masked_image, mask):
		sources = []
		sources.append(masked_image)
		### Encoder
		mask = self.E_conv1_mask(masked_image)
		mask = self.sigmoid(mask)
		x = self.E_conv1(masked_image)
		x = self.relu(x)
		x = x*mask
		sources.append(x)
		#print(x.size())

		mask = self.E_conv2_mask(x)
		mask = self.sigmoid(mask)
		x = self.E_conv2(x)
		x = self.relu(x)
		x = x*mask
		sources.append(x)
		#print(x.size())

		mask = self.E_conv3_mask(x)
		mask = self.sigmoid(mask)
		x = self.E_conv3(x)
		x = self.relu(x)
		x = x*mask
		sources.append(x)
		#print(x.size())

		mask = self.E_conv4_mask(x)
		mask = self.sigmoid(mask)
		x = self.E_conv4(x)
		x = self.relu(x)
		x = x*mask
		sources.append(x)
		#print(x.size())

		mask = self.E_conv5_mask(x)
		mask = self.sigmoid(mask)
		x = self.E_conv5(x)
		x = self.relu(x)
		x = x*mask
		sources.append(x)
		#print(x.size())

		mask = self.E_conv6_mask(x)
		mask = self.sigmoid(mask)
		x = self.E_conv6(x)
		x = self.relu(x)
		x = x*mask
		sources.append(x)
		#print(x.size())

		mask = self.E_conv7_mask(x)
		mask = self.sigmoid(mask)
		x = self.E_conv7(x)
		x = self.relu(x)
		x = x*mask
		#print(x.size())

		### Decoder
		x = self.upsample(x)
		x = torch.cat((x,sources[-1]),1)
		mask = self.D_conv1_mask(x)
		mask = self.sigmoid(mask)
		x = self.D_conv1(x)
		x = self.leakyRelu(x)
		x = x*mask
		#print(x.size())	

		x = self.upsample(x)
		x = torch.cat((x,sources[-2]),1)
		mask = self.D_conv2_mask(x)
		mask = self.sigmoid(mask)
		x = self.D_conv2(x)
		x = self.leakyRelu(x)
		x = x*mask
		#print(x.size())	

		x = self.upsample(x)
		x = torch.cat((x,sources[-3]),1)
		mask = self.D_conv3_mask(x)
		mask = self.sigmoid(mask)
		x = self.D_conv3(x)
		x = self.leakyRelu(x)
		x = x*mask
		#print(x.size())

		x = self.upsample(x)
		x = torch.cat((x,sources[-4]),1)
		mask = self.D_conv4_mask(x)
		mask = self.sigmoid(mask)
		x = self.D_conv4(x)
		x = self.leakyRelu(x)
		x = x*mask
		#print(x.size())

		x = self.upsample(x)
		x = torch.cat((x,sources[-5]),1)
		mask = self.D_conv5_mask(x)
		mask = self.sigmoid(mask)
		x = self.D_conv5(x)
		x = self.leakyRelu(x)
		x = x*mask
		#print(x.size())

		x = self.upsample(x)
		x = torch.cat((x,sources[-6]),1)
		mask = self.D_conv6_mask(x)
		mask = self.sigmoid(mask)
		x = self.D_conv6(x)
		x = self.leakyRelu(x)
		x = x*mask
		#print(x.size())

		x = self.upsample(x)
		x = torch.cat((x,sources[-7]),1)
		mask = self.D_conv7_mask(x)
		mask = self.sigmoid(mask)
		x = self.D_conv7(x)
		#x = self.leakyRelu(x)
		x = x*mask
		#print(x.size())

		return x




