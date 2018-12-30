import seaborn as sns; sns.set(font_scale=2)
import matplotlib.pyplot as plt
import numpy as np


def scaled_softsign(z):
    return np.divide(z + 1 + np.abs(z), 2 * (1 + np.abs(z)))


X = 100
scale = np.arange(start=1, stop=6)/5

x = np.linspace(start=-X, stop=+X, num=1000)
sigmoid = 1/(1 + np.exp(-x))

plt.plot(x, sigmoid, label=r"$\sigma$")

for s in reversed(scale):
    plt.plot(x, scaled_softsign(s * x), label=r"$\hat{s}_{" + str(s) + r"}$")

plt.legend(loc='upper left', prop={'size': 22})
plt.tight_layout()
plt.show()
