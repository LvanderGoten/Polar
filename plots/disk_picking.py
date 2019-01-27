import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
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

n = 2000
r = np.random.rand(n)
phi = np.pi * (2 * np.random.rand(n) - 1)

x_root = np.sqrt(r) * np.cos(phi)
y_root = np.sqrt(r) * np.sin(phi)
x_incorrect = r * np.cos(phi)
y_incorrect = r * np.sin(phi)

fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(20, 10))
plt.subplots_adjust(wspace=.05)
ax[0].set_aspect("equal")
ax[1].set_aspect("equal")
ax[0].scatter(x=x_incorrect, y=y_incorrect)
ax[1].scatter(x=x_root, y=y_root)
ax[0].set_xlabel("x")
ax[1].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_title(r"$[X, Y] = R \cdot [\cos{\Phi}, \sin{\Phi}]$")
ax[1].set_title(r"$[X, Y] = \sqrt{R} \cdot [\cos{\Phi}, \sin{\Phi}]$")
ax[0].grid(False)
ax[1].grid(False)
sns.despine(left=True, bottom=True, right=True)
plt.savefig("disk_picking.png", transparent=True)
