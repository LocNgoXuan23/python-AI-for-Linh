import numpy as np
from random import gauss
import random
import matplotlib.pyplot as plt
random.seed(10)
np.random.seed(4)



class LinearRegression():
	def __init__(self, X_train, y_train, p=1, f=1):
		self.X_train = X_train
		self.y_train = y_train
		self.p = p


	def fit(self):
		# self.X_bar = np.concatenate((self.X_train, np.ones(self.X_train.shape)), axis=1)
		self.X_bar = np.ones(self.X_train.shape)
		for i in range(self.p):
			self.X_bar = np.concatenate(((self.X_train**(i+1)), self.X_bar), axis=1)
			# self.X_bar = np.concatenate(((pow(self.X_train, i+1)), self.X_bar), axis=1)
		# self.X_bar = self.X_bar.astype(float)

		self.weights = ((np.linalg.inv(self.X_bar.transpose().dot(self.X_bar))).dot(self.X_bar.transpose())).dot(self.y_train)


	def predict(self, X_test):
		# Nx1
		y_predict = np.zeros((X_test.shape))
		for i in range(X_test.shape[0]):
			for j in range(self.p, -1, -1):
				y_predict[i][0] += self.weights[j][0]*pow(X_test[i][0], self.p - j)
		return y_predict


	def mse(self, y_predict, y_true):
		# print(y_predict.shape)
		# print(y_true.shape)
		error = 0
		for i in range(y_predict.shape[0]):
			error += pow(y_predict[i][0] - y_true[i][0], 2)
		return error / y_predict.shape[0]

	def imshow(self):
		a = np.linspace(-2.5, 7, 1000)
		b = 0
		# b = pow(a, 2)*self.weights[0][0] + a*self.weights[1][0] + self.weights[2][0] 
		for j in range(self.p, -1, -1):
			b += self.weights[j][0]*pow(a, self.p - j)
		plt.plot(self.X_train.transpose()[0], self.y_train.transpose()[0], 'ro')
		plt.plot(a, b)
		plt.axis([-4, 10, np.amin(self.y_train.transpose()) - 50, np.amax(self.y_train.transpose()) + 50])
		plt.show()





# # Bai 1 : ve parabol vs X và y 
# x = np.array([[-30, -40, -20, -10, -5 ,0,  10, 15, 20, 30, 40]])
# # y = x^2*-3 + x*6 + 4
# y = np.zeros((1, x.shape[1]))
# for i in range(x.shape[1]):
# 	y[0][i] = pow(x[0][i], 2)*-3 + x[0][i]*6 + 4 + gauss(0, 1) * 100


N = 30
x = np.random.rand(N, 1)*5
y = 3*(x -2) * (x - 3)*(x-4) +  1*np.random.randn(N, 1)
x = x.transpose()
y = y.transpose()

lr = LinearRegression(x.transpose(), y.transpose(), p=30)
lr.fit()
y_predict = lr.predict(x.transpose())
mse = lr.mse(y_predict, y.transpose())
print(mse)
lr.imshow()
# 139
# 69
