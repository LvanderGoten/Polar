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


x = np.linspace(start=-20, stop=20, num=200)
scales = [0.1, 0.25, 0.5, 1.0, 2.0]


def scaled_sigmoid(x, scale):
    return np.divide(1, 1 + np.exp(-scale * x))


def scaled_softsign(x, scale):
    return np.divide(scale * x + 1 + np.abs(scale * x), 2 * (1 + np.abs(scale * x)))


fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True, gridspec_kw={'width_ratios': [1, 1]})
plt.subplots_adjust(wspace=.075)

sigmoid_handles, softsign_handles = [], []
for scale in reversed(scales):
    sigmoid_handles.append(ax[0].plot(x, scaled_sigmoid(x, scale=scale), linewidth=3)[0])
    softsign_handles.append(ax[1].plot(x, scaled_softsign(x, scale=scale), linewidth=3)[0])

ax[0].set_xlabel(r"$x$")
ax[1].set_xlabel(r"$x$")
ax[0].set_ylabel(r"$\sigma_\lambda(x)$")
ax[1].set_ylabel(r"$\hat{s}_\lambda(x)$")
ax[0].set_title("Sigmoid family")
ax[1].set_title("Scaled softsign family")
ax[0].set_aspect(40, adjustable="box")
ax[1].set_aspect(40, adjustable="box")
labels = [r"$\lambda = {}$".format(s) for s in scales]
plt.legend(sigmoid_handles + softsign_handles, labels, loc='lower right')
plt.show()
