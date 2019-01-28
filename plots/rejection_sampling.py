import matplotlib.pyplot as plt
import numpy as np

n = 10000
x = 2 * np.random.rand(n, 2) - 1
r = np.linalg.norm(x, ord=2, axis=1)
x = x[r <= 1]
plt.scatter(x=x[:, 0], y=x[:, 1])
plt.show()
