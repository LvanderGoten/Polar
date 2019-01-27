import seaborn as sns; sns.set(font_scale=2)
import matplotlib.pyplot as plt
import numpy as np

MEDIUM_SIZE = 25
BIGGER_SIZE = 30

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


EPS = .001
scales = [0.1, 0.25, 0.5, 1.0, 2.0]
MU = 0
STD = 1
A, B = -1, 1


def sigmoid(x):
    return np.reciprocal(1 + np.exp(-x))


def pdf_sigmoid_uniform(y, scale, a=A, b=B):
    return np.where(np.logical_and(sigmoid(scale * a) <= y, y <= sigmoid(scale * b)),
                    1/((b - a)*scale) * 1/(y - np.square(y)),
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


fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True, gridspec_kw = {'width_ratios': [1, 1]})
plt.subplots_adjust(wspace=.05)
yy = np.linspace(start=EPS, stop=1-EPS, num=100)

sigmoid_handles, softsign_handles = [], []
for scale in reversed(scales):
    sigmoid_handles.append(ax[0].plot(yy, pdf_sigmoid_uniform(yy, scale=scale,), linewidth=3)[0])
    softsign_handles.append(ax[1].plot(yy, pdf_parametric_uniform(yy, scale=scale), linewidth=3)[0])

ax[0].set_title("Sigmoid family")
ax[0].set_aspect(1/20, adjustable="box")
ax[1].set_aspect(1/20, adjustable="box")
ax[1].set_title("Scaled softsign family")
labels = [r"$\lambda = {}$".format(s) for s in scales]
plt.legend(sigmoid_handles + softsign_handles, labels, loc='upper right')

ax[0].set_ylabel("Density")
ax[0].set_xlabel(r"$y$")
ax[1].set_xlabel(r"$z$")
plt.show()
