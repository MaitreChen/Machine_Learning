import numpy as np
from matplotlib import pyplot as plt

# make data
X = np.arange(0, 10, 0.02)
Z = [21 + 5 * x for x in X]
Y = np.array([np.random.normal(z, 0.5) for z in Z])

# data pre-processing
X = X.reshape(X.shape[0], 1)
Y = np.array(Y).reshape(Y.shape[0], 1)

# initialize params
w, b = 1, 1
alpha = 0.01  # learning rate
epochs = 1000

# training
num_of_training = X.shape[0]
loss_set = []
for epoch in range(epochs):
    dw = np.sum(-(2 / num_of_training) * X * (Y - w * X - b))
    db = np.sum(-(2 / num_of_training) * (Y - w * X - b))

    loss = np.sum((Y - w * X - b) ** 2) / num_of_training
    loss_set.append(loss)

    w = w - alpha * dw
    b = b - alpha * db
    print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss))
print(f"w={w:.2f}  b={b:.2f}")
print(f"loss={loss:.6f}")

# plot training loss
plt.figure(figsize=(10, 6))
plt.rc('font', family='Times New Roman')
plt.title("Training loss", fontsize=20)
X_a = [_ for _ in range(epochs)]
plt.plot(X_a, loss_set)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('MSE', fontsize=20)
plt.xticks(fontsize=20), plt.yticks(fontsize=20)
plt.show()

# plot the result of fitting
x_set = [0, 10]
y_set = [w * x + b for x in x_set]
plt.figure(figsize=(10, 6))
plt.rc('font', family='Times New Roman')
plt.title("Linear regression", fontsize=20)
plt.xlabel("X", fontsize=20)
plt.ylabel("Y", fontsize=20)
plt.xticks(fontsize=20), plt.yticks(fontsize=20)
plt.plot(X, Y, 'bo', markersize='2')
plt.plot(x_set, y_set, 'r', linewidth=4)
plt.show()

