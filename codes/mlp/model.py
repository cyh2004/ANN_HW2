# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
class BatchNorm1d(nn.Module):
	# TODO START
	def __init__(self, num_features):
		super(BatchNorm1d, self).__init__()
		self.num_features = num_features

		# Parameters   
		self.weight = nn.Parameter(torch.ones(num_features))
		self.bias = nn.Parameter(torch.zeros(num_features))

		# Store the average mean and variance
		self.register_buffer('running_mean', torch.zeros(num_features))
		self.register_buffer('running_var', torch.ones(num_features))
		self.alpha = 0.99
		
		# Initialize your parameter

	def forward(self, input):
		if self.training:
			mean = torch.mean(input, dim=1, keepdim=True)
			var = torch.var(input, dim=1, keepdim=True)
			output = (input - mean) / torch.sqrt(var + 1e-5)
			output = self.weight.unsqueeze(0) * output + self.bias
			self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * mean
			self.running_var = self.alpha * self.running_var + (1 - self.alpha) * var
		else:
			output = self.weight.unsqueeze(0) * (input - self.running_mean) / torch.sqrt(self.running_var + 1e-5) + self.bias
		# input: [batch_size, num_feature_map * height * widh]
		return output
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		if self.training:
			mask = torch.bernoulli(torch.full(input.shape, self.p))
			output = input.clone()
			output[mask == 1] = 0
			output = output / (1 - self.p)
			return output
		else:
			return input
	# TODO END

class Model(nn.Module):
	def __init__(self, in_num, nclasses, drop_rate=0.5):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		self.fc1 = nn.Linear(in_num, 1024)
		self.bn1 = BatchNorm1d(1024)
		self.relu1 = nn.ReLU()
		self.dropout1 = Dropout(drop_rate)
		self.fc2 = nn.Linear(1024, nclasses)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):
		# TODO START
		# the 10-class prediction output is named as "logits"
		y_hat = self.fc1(x)
		y_hat = self.bn1(y_hat)
		y_hat = self.relu1(y_hat)
		y_hat = self.dropout1(y_hat)
		logits = self.fc2(y_hat)
		
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc

class Model_noDrop(nn.Module):
	def __init__(self, in_num, nclasses, drop_rate=0.5):
		super(Model_noDrop, self).__init__()
		# TODO START
		# Define your layers here
		self.fc1 = nn.Linear(in_num, 1024)
		self.bn1 = BatchNorm1d(1024)
		self.relu1 = nn.ReLU()
		# self.dropout1 = Dropout(drop_rate)
		self.fc2 = nn.Linear(1024, nclasses)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):
		# TODO START
		# the 10-class prediction output is named as "logits"
		y_hat = self.fc1(x)
		y_hat = self.bn1(y_hat)
		y_hat = self.relu1(y_hat)
		# y_hat = self.dropout1(y_hat)
		logits = self.fc2(y_hat)
		
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc

class Model_noBN(nn.Module):
	def __init__(self, in_num, nclasses, drop_rate=0.5):
		super(Model_noBN, self).__init__()
		# TODO START
		# Define your layers here
		self.fc1 = nn.Linear(in_num, 1024)
		# self.bn1 = BatchNorm1d(1024)
		self.relu1 = nn.ReLU()
		self.dropout1 = Dropout(drop_rate)
		self.fc2 = nn.Linear(1024, nclasses)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):
		# TODO START
		# the 10-class prediction output is named as "logits"
		y_hat = self.fc1(x)
		# y_hat = self.bn1(y_hat)
		y_hat = self.relu1(y_hat)
		y_hat = self.dropout1(y_hat)
		logits = self.fc2(y_hat)
		
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
