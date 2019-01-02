import seaborn as sns; sns.set(font_scale=2)
import matplotlib.pyplot as plt
import numpy as np

EPS = .001
scales = np.arange(start=1, stop=6)/5
MU = 0
STD = 1
A, B = -1, 1


def sigmoid(x):
    return np.reciprocal(1 + np.exp(-x))


def pdf_sigmoid_uniform(y, a=A, b=B):
    return np.where(np.logical_and(sigmoid(a) <= y, y <= sigmoid(b)),
                    1/(b - a) * 1/(y - np.square(y)),
                    np.zeros_like(y))


def pdf_parametric_uniform(y, scale, a=A, b=B):

    # y smaller than 1/2
    below_lb = -1/(2 * (scale * a - 1))
    below_ub = 1/2
    below_v = 1/(b - a) * 1/(2 * scale * np.square(y))
    below = np.where(np.logical_and(below_lb <= y, y <= below_ub), below_v, np.zeros_like(y))

    # y larger than 1/2
    above_lb = 1/2
    above_ub = (2 * b * scale + 1)/(2 * (b * scale + 1))
    above_v = 1/(b - a) * 1/(2 * scale * np.square(1 - y))
    above = np.where(np.logical_and(above_lb <= y, y <= above_ub), above_v, np.zeros_like(y))

    return below + above


def pdf_sigmoid_normal(y, mu=MU, std=STD):
    var = std**2
    numerator = np.exp(-np.divide(np.square(np.log(y/(1-y)) - mu), 2 * var))
    denominator = np.sqrt(2 * np.pi * var) * (y - np.square(y))
    return np.divide(numerator, denominator)


def pdf_parametric(y, scale, mu=MU, std=STD):
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
plt.plot(yy, pdf_sigmoid_uniform(yy), label=r"$\sigma(\cdot)$")

for scale in reversed(scales):
    plt.plot(yy, pdf_parametric_uniform(yy, scale=scale), label=r"$\hat{s}_{" + str(scale) + r"}$")

plt.xlabel(r"$x$")
plt.ylabel(r"$f_X(x)$")
plt.legend(loc='upper left', prop={'size': 22})
plt.tight_layout()
plt.show()
