import seaborn as sns; sns.set(font_scale=2)
import matplotlib.pyplot as plt
import numpy as np

EPS = .001
scales = np.arange(start=1, stop=6)/5
scales = [10, 20, 30]


def pdf_sigmoid(y, mu=0, std=1):
    var = std**2
    numerator = np.exp(-np.divide(np.square(np.log(y/(1-y)) - mu), 2 * var))
    denominator = np.sqrt(2 * np.pi * var) * (y - np.square(y))
    return np.divide(numerator, denominator)


def pdf_parametric(y, scale, mu=0, std=1):
    var = std**2

    # y smaller than 1/2
    numerator_below = np.exp(-np.divide(np.square(np.divide(2 * y - 1,
                                                            2 * scale * y) - mu),
                                        2 * var))
    denominator_below = np.sqrt(2 * np.pi * var) * 2 * scale * np.square(y)
    below = np.divide(numerator_below, denominator_below)

    # y larger than 1/2
    numerator_above = np.exp(-np.divide(np.square(np.divide(2 * y - 1,
                                                            2 * scale * (1 - y)) - mu),
                                        2 * var))
    denominator_above = np.sqrt(2 * np.pi * var) * 2 * scale * np.square(1 - y)
    above = np.divide(numerator_above, denominator_above)

    return np.where(y < .5,
                    below,
                    above)


yy = np.linspace(start=EPS, stop=1-EPS, num=100)
fy = pdf_sigmoid(yy)
plt.plot(yy, pdf_sigmoid(yy), label=r"$\sigma$")

for scale in reversed(scales):
    plt.plot(yy, pdf_parametric(yy, scale=scale), label=r"$\hat{s}_{" + str(scale) + r"}$")

plt.xlabel(r"$x$")
plt.ylabel(r"$f_X(x)$")
plt.legend(loc='upper left', prop={'size': 22})
plt.tight_layout()
plt.show()
