import numpy as np
from random import gauss
import random
import matplotlib.pyplot as plt
random.seed(10)

class LinearRegression():
	def __init__(self,X_train,y_train,p=1):
		self.X_train = X_train
		self.y_train = y_train
		self.p = p

	def fit(self):
		self.X_bar = np.concatenate((self.X_train, np.ones((self.X_train.shape[0],1))), axis=1)
		self.weights = ((np.linalg.inv(self.X_bar.transpose().dot(self.X_bar))).dot(self.X_bar.transpose())).dot(self.y_train)

	def predict(self,X_test):
		y_predict = np.zeros((X_test.shape))
		for i in range(X_test.shape[0]):
			y_predict[i][0] += self.weights[0][0]*X_test[i][0] + self.weights[1][0]*X_test[i][1] + self.weights[2][0]
		return y_predict

	def mse(self,y_predict,y_true):
		error = 0
		for i in range(y_predict.shape[0]):
			error += pow(y_predict[i][0] - y_true[i][0],2)
		return error/y_predict.shape[0]

x = np.array([[-30, -40, -20, -10, -5 ,0,  10, 15, 20, 30, 40]])
x1 = x
x2 = x + gauss(0, 1) * 10
# print(x1)
# print(x2)
y = np.zeros((1, x1.shape[1]))
for i in range(x.shape[1]):
	y[0][i] = x1[0][i]*0.5 - x2[0][i]*0.75 + 10 + gauss(0, 1) * 10
print(y)
x = np.concatenate((x1.transpose(), x2.transpose()), axis=1)
lr = LinearRegression(x,y.transpose(),1)
lr.fit()
print(lr.weights)
print(y)
