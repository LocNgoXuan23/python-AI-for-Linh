import numpy as np
import matplotlib.pyplot as plt
np.random.seed(10)

# B1 Tao class Gradient decent ve duong thang bac 2, ve train_loss_history, val_loss_history danh gia mo hinh sau khi training vs tap test
m = 100
X_train = 6 * np.random.rand(m, 1) - 3
y_train = 0.5 * X_train**2 + 2 + np.random.randn(m, 1)

X_val = 6 * np.random.rand(m, 1) - 3
y_val = 0.5 * X_val**2 + 2 + np.random.randn(m, 1)

X_test = 6 * np.random.rand(m, 1) - 3
y_test = 0.5 * X_test**2 + 2 + np.random.randn(m, 1)


plt.plot(X_train, y_train, 'bo')
plt.plot(X_val, y_val, 'ro')
plt.show()

# B2 Tao class Gradient decent ve duong thang bac 3, ve train_loss_history, val_loss_history danh gia mo hinh sau khi training vs tap test
m = 100
X_train = 6 * np.random.rand(m, 1) - 3
y_train = -0.7 * X_train**3 + 0.8 * X_train**2 + 6 + np.random.randn(m, 1)

X_val = 6 * np.random.rand(m, 1) - 3
y_val = -0.7 * X_val**3 + 0.8 * X_val**2 + 6 + np.random.randn(m, 1)

X_test = 6 * np.random.rand(m, 1) - 3
y_test = -0.7 * X_test**3 + 0.8 * X_test**2 + 6 + np.random.randn(m, 1)

plt.plot(X_train, y_train, 'bo')
plt.plot(X_val, y_val, 'ro')
plt.show()