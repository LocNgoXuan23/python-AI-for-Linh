import numpy as np
import matplotlib.pyplot as plt
np.random.seed(10)

class GradientDescent:
	def __init__(self, X_train, y_train, X_val, y_val, epoch=50, learning_rate=0.001):
		self.X_train = X_train
		self.y_train = y_train
		self.X_val = X_val
		self.y_val = y_val
		self.epoch = epoch
		self.learning_rate = learning_rate
		self.weight = np.random.randn(2, 1)

	def fit(self):
		self.X_train = np.concatenate((self.X_train, np.ones(self.X_train.shape)), axis=1)
		self.train_loss_history = []
		self.val_loss_history = []
		for i in range(self.epoch):
			self.weight = self.weight - self.learning_rate*self.gradient(self.weight)

			y_pred = self.predict(self.X_train[:, 0].reshape(-1, 1))
			loss_train = self.loss(y_pred, self.y_train)
			self.train_loss_history.append(loss_train)

			y_pred = self.predict(self.X_val)
			loss_val = self.loss(y_pred, self.y_val)
			self.val_loss_history.append(loss_val)
			

	def predict(self, X_test):
		y_predict = np.zeros(self.y_train.shape)
		for i in range(self.y_train.shape[0]):
			y_predict[i][0] = self.weight[0][0] * X_test[i][0] + self.weight[1][0]
		return y_predict	


	def gradient(self, weight):
		result = (2/self.X_train.shape[1]) * self.X_train.transpose().dot(self.X_train.dot(weight) - self.y_train)
		return result

	def loss(self, y_predict, y_true):
		mse = 0
		for i in range(y_predict.shape[0]):
			mse += pow(y_predict[i][0] - y_true[i][0], 2)
		return mse / y_predict.shape[0]

	def mse(self, y_predict, y_true):
		mse = 0
		for i in range(y_predict.shape[0]):
			mse += pow(y_predict[i][0] - y_true[i][0], 2)
		return mse / y_predict.shape[0]

	def imshow(self):
		pass


X = 2 * np.random.rand(100, 1)
y = 4  + 3 * X + np.random.rand(100, 1)
X_val = 2 * np.random.rand(100, 1)
y_val = 4  + 3 * X_val + np.random.rand(100, 1)

X_test = 2 * np.random.rand(100, 1)
y_test = 4  + 3 * X_test + np.random.rand(100, 1)
# y = 3x + 4
X = X.transpose()
y = y.transpose()
X_val = X_val.transpose()
y_val = y_val.transpose()
X_test = X_test.transpose()
y_test = y_test.transpose()




gd = GradientDescent(X.transpose(), y.transpose(), X_val.transpose(), y_val.transpose(), 1000, 0.001)
gd.fit()
# y_predict = gd.predict(X.transpose())
# print(y_predict.transpose()[0][2])
# print(y[0][2])


# plt.plot(gd.train_loss_history)
# plt.plot(gd.val_loss_history)
# plt.show()
y_predict = gd.predict(X_test.transpose())
print(gd.mse(y_predict.transpose(), y_test))


