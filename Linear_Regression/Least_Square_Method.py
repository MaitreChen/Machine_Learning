import numpy as np
import matplotlib.pyplot as plt

# line exp: y=5x+21
X = np.arange(0, 10, 0.02)
Z = [21 + 5 * x for x in X]
Y = [np.random.normal(z, 1) for z in Z]
plt.plot(X, Y, 'bo', markersize='2')

# LST
n = len(X)
sum_xy = sum(X * Y)
sum_xx = sum(X * X)
sum_x = sum(X)
sum_y = sum(Y)

tmp = n * sum_xx - sum_x ** 2
k = (n * sum_xy - sum_x * sum_y) / tmp
b = (sum_xx * sum_y - sum_x * sum_xy) / tmp
# 损失
error = np.sum((k * X + b - Y) ** 2)
print(f"k={k:.2f}  b={b:.2f}")
print(f"The error is: {error:.2f}")

x_set = [0, 10]
y_set = [k * x + b for x in x_set]
plt.title("Least Square Method ")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(X, Y, 'bo', markersize='2')
plt.plot(x_set, y_set, 'r', linewidth=4)
plt.show()


