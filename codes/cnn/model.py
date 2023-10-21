# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
class BatchNorm2d(nn.Module):
	# TODO START
	def __init__(self, num_features):
		super(BatchNorm2d, self).__init__()
		self.num_features = num_features

		# Parameters
		self.weight = nn.Parameter(torch.ones(num_features))
		self.bias = nn.Parameter(torch.zeros(num_features))

		# Store the average mean and variance
		self.register_buffer('running_mean', torch.ones(num_features))
		self.register_buffer('running_var', torch.zeros(num_features))
  
		self.alpha = 0.99
		
		# Initialize your parameter

	def forward(self, input):
		# input: [batch_size, num_feature_map, height, width]
		if self.training:
			# [1, num_feature_map, 1, 1]
			mean = torch.mean(input, (0,2,3), keepdim=True)
			# [1, num_feature_map, 1, 1]
			var = torch.var(input, (0,2,3), keepdim=True)
			output = (input - mean) / torch.sqrt(var + 1e-5)
			output = output * self.weight.reshape((1, self.weight.shape[0], 1, 1)) + self.bias.reshape((1, self.weight.shape[0], 1, 1))
			self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * mean.reshape((self.running_mean.shape[0],))
			self.running_var = self.alpha * self.running_var + (1 - self.alpha) * var.reshape((self.running_var.shape[0],))
		else:
			output = (input - self.running_mean.reshape((1, self.running_mean.shape[0], 1, 1))) / torch.sqrt(self.running_var.reshape((1, self.running_var.shape[0], 1, 1)) + 1e-5)
			output = self.weight.reshape((1, self.weight.shape[0], 1, 1)) * output + self.bias.reshape((1, self.bias.shape[0], 1, 1))
		return output
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
		if self.training:
			mask = torch.bernoulli(torch.full(input.shape, self.p))
			output = input.clone()
			output[mask == 1] = 0
			output = output / (1 - self.p)
			# input: [batch_size, num_feature_map, height, width]
			return output
		else:
			return input
	# TODO END

class Model(nn.Module):
	def __init__(self, H, W, channels, drop_rate=0.5):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		# x: [bs, 3, 32, 32]
		self.conv1 = nn.Conv2d(channels, 128, 5)
		# x: [bs, 128, 28, 28]
		self.bn1 = BatchNorm2d(128)
		self.relu1 = nn.ReLU()
		self.dropout1 = Dropout(drop_rate)
		# x: [bs, 128, 28, 28]
		self.maxp1 = nn.MaxPool2d(2)
		# x: [bs, 128, 14, 14]
		self.conv2 = nn.Conv2d(128, 128, 5)
		# x: [bs, 128, 10, 10]
		self.bn2 = BatchNorm2d(128)
		self.relu2 = nn.ReLU()
		self.dropout2 = Dropout(drop_rate)
		# x: [bs, 128, 10, 10]
		self.maxp2 = nn.MaxPool2d(2)
		# x: [bs, 128, 5, 5]
		self.fc = nn.Linear(128*5*5, 10)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):	
		# TODO START
		# the 10-class prediction output is named as "logits"
		y_hat = self.conv1(x)
		y_hat = self.bn1(y_hat)
		y_hat = self.relu1(y_hat)
		y_hat = self.dropout1(y_hat)
		y_hat = self.maxp1(y_hat)
		y_hat = self.conv2(y_hat)
		y_hat = self.bn2(y_hat)
		y_hat = self.relu2(y_hat)
		y_hat = self.dropout2(y_hat)
		y_hat = self.maxp2(y_hat)
		y_hat = y_hat.reshape((y_hat.shape[0], -1))
		y_hat = self.fc(y_hat)
		logits = y_hat
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc

class Model_noDrop(nn.Module):
	def __init__(self, H, W, channels, drop_rate=0.5):
		super(Model_noDrop, self).__init__()
		# TODO START
		# Define your layers here
		# x: [bs, 3, 32, 32]
		self.conv1 = nn.Conv2d(channels, 128, 5)
		# x: [bs, 128, 28, 28]
		self.bn1 = BatchNorm2d(128)
		self.relu1 = nn.ReLU()
		# self.dropout1 = Dropout(drop_rate)
		# x: [bs, 128, 28, 28]
		self.maxp1 = nn.MaxPool2d(2)
		# x: [bs, 128, 14, 14]
		self.conv2 = nn.Conv2d(128, 128, 5)
		# x: [bs, 128, 10, 10]
		self.bn2 = BatchNorm2d(128)
		self.relu2 = nn.ReLU()
		# self.dropout2 = Dropout(drop_rate)
		# x: [bs, 128, 10, 10]
		self.maxp2 = nn.MaxPool2d(2)
		# x: [bs, 128, 5, 5]
		self.fc = nn.Linear(128*5*5, 10)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):	
		# TODO START
		# the 10-class prediction output is named as "logits"
		y_hat = self.conv1(x)
		y_hat = self.bn1(y_hat)
		y_hat = self.relu1(y_hat)
		# y_hat = self.dropout1(y_hat)
		y_hat = self.maxp1(y_hat)
		y_hat = self.conv2(y_hat)
		y_hat = self.bn2(y_hat)
		y_hat = self.relu2(y_hat)
		# y_hat = self.dropout2(y_hat)
		y_hat = self.maxp2(y_hat)
		y_hat = y_hat.reshape((y_hat.shape[0], -1))
		y_hat = self.fc(y_hat)
		logits = y_hat
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
class Model_noBN(nn.Module):
	def __init__(self, H, W, channels, drop_rate=0.5):
		super(Model_noBN, self).__init__()
		# TODO START
		# Define your layers here
		# x: [bs, 3, 32, 32]
		self.conv1 = nn.Conv2d(channels, 128, 5)
		# x: [bs, 128, 28, 28]
		# self.bn1 = BatchNorm2d(128)
		self.relu1 = nn.ReLU()
		self.dropout1 = Dropout(drop_rate)
		# x: [bs, 128, 28, 28]
		self.maxp1 = nn.MaxPool2d(2)
		# x: [bs, 128, 14, 14]
		self.conv2 = nn.Conv2d(128, 128, 5)
		# x: [bs, 128, 10, 10]
		# self.bn2 = BatchNorm2d(128)
		self.relu2 = nn.ReLU()
		self.dropout2 = Dropout(drop_rate)
		# x: [bs, 128, 10, 10]
		self.maxp2 = nn.MaxPool2d(2)
		# x: [bs, 128, 5, 5]
		self.fc = nn.Linear(128*5*5, 10)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):	
		# TODO START
		# the 10-class prediction output is named as "logits"
		y_hat = self.conv1(x)
		# y_hat = self.bn1(y_hat)
		y_hat = self.relu1(y_hat)
		y_hat = self.dropout1(y_hat)
		y_hat = self.maxp1(y_hat)
		y_hat = self.conv2(y_hat)
		# y_hat = self.bn2(y_hat)
		y_hat = self.relu2(y_hat)
		y_hat = self.dropout2(y_hat)
		y_hat = self.maxp2(y_hat)
		y_hat = y_hat.reshape((y_hat.shape[0], -1))
		y_hat = self.fc(y_hat)
		logits = y_hat
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc

# 先过激活函数再过BN
class Model_switch1(nn.Module):
	def __init__(self, H, W, channels, drop_rate=0.5):
		super(Model_switch1, self).__init__()
		# TODO START
		# Define your layers here
		# x: [bs, 3, 32, 32]
		self.conv1 = nn.Conv2d(channels, 128, 5)
		# x: [bs, 128, 28, 28]
		self.bn1 = BatchNorm2d(128)
		self.relu1 = nn.ReLU()
		self.dropout1 = Dropout(drop_rate)
		# x: [bs, 128, 28, 28]
		self.maxp1 = nn.MaxPool2d(2)
		# x: [bs, 128, 14, 14]
		self.conv2 = nn.Conv2d(128, 128, 5)
		# x: [bs, 128, 10, 10]
		self.bn2 = BatchNorm2d(128)
		self.relu2 = nn.ReLU()
		self.dropout2 = Dropout(drop_rate)
		# x: [bs, 128, 10, 10]
		self.maxp2 = nn.MaxPool2d(2)
		# x: [bs, 128, 5, 5]
		self.fc = nn.Linear(128*5*5, 10)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):	
		# TODO START
		# the 10-class prediction output is named as "logits"
		y_hat = self.conv1(x)
		y_hat = self.relu1(y_hat)
		y_hat = self.bn1(y_hat)
		y_hat = self.dropout1(y_hat)
		y_hat = self.maxp1(y_hat)
		y_hat = self.conv2(y_hat)
		y_hat = self.relu2(y_hat)
		y_hat = self.bn2(y_hat)
		y_hat = self.dropout2(y_hat)
		y_hat = self.maxp2(y_hat)
		y_hat = y_hat.reshape((y_hat.shape[0], -1))
		y_hat = self.fc(y_hat)
		logits = y_hat
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc

# 先过Maxpool再过Dropout
class Model_switch2(nn.Module):
	def __init__(self, H, W, channels, drop_rate=0.5):
		super(Model_switch2, self).__init__()
		# TODO START
		# Define your layers here
		# x: [bs, 3, 32, 32]
		self.conv1 = nn.Conv2d(channels, 128, 5)
		# x: [bs, 128, 28, 28]
		self.bn1 = BatchNorm2d(128)
		self.relu1 = nn.ReLU()
		self.dropout1 = Dropout(drop_rate)
		# x: [bs, 128, 28, 28]
		self.maxp1 = nn.MaxPool2d(2)
		# x: [bs, 128, 14, 14]
		self.conv2 = nn.Conv2d(128, 128, 5)
		# x: [bs, 128, 10, 10]
		self.bn2 = BatchNorm2d(128)
		self.relu2 = nn.ReLU()
		self.dropout2 = Dropout(drop_rate)
		# x: [bs, 128, 10, 10]
		self.maxp2 = nn.MaxPool2d(2)
		# x: [bs, 128, 5, 5]
		self.fc = nn.Linear(128*5*5, 10)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):	
		# TODO START
		# the 10-class prediction output is named as "logits"
		y_hat = self.conv1(x)
		y_hat = self.bn1(y_hat)
		y_hat = self.relu1(y_hat)
		y_hat = self.maxp1(y_hat)
		y_hat = self.dropout1(y_hat)
		y_hat = self.conv2(y_hat)
		y_hat = self.bn2(y_hat)
		y_hat = self.relu2(y_hat)
		y_hat = self.maxp2(y_hat)
		y_hat = self.dropout2(y_hat)
		y_hat = y_hat.reshape((y_hat.shape[0], -1))
		y_hat = self.fc(y_hat)
		logits = y_hat
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
