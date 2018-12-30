import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D     # loading has side effects
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np

# Number of points
N = 1000

alpha = np.linspace(start=-np.pi, stop=np.pi, num=N)
beta = np.linspace(start=-np.pi, stop=np.pi, num=N)
alpha_v, beta_v = np.meshgrid(alpha, beta)
z = np.abs(np.arctan2(np.sin(alpha_v - beta_v), np.cos(alpha_v - beta_v)))/np.pi

sns.set(font_scale=3)   # Seaborn style
fig, ax = plt.subplots(nrows=2, ncols=2, gridspec_kw={"height_ratios": [1, 19],
                                                      "width_ratios": [1, 1],
                                                      "hspace": .1,
                                                      "wspace": .4})
contour = ax[1, 0].contourf(alpha_v, beta_v, z, levels=200, cmap=cm.Greens, vmin=0, vmax=1, antialiased=True)
ax[1, 0].set_xlabel(r'$\alpha$')
ax[1, 0].set_ylabel(r'$\beta$')
norm = colors.Normalize(vmin=contour.cvalues.min(), vmax=contour.cvalues.max())
sm = plt.cm.ScalarMappable(norm=norm, cmap=contour.cmap)
sm.set_array([])
cb = plt.colorbar(sm, cax=ax[0, 0], orientation="horizontal")
cb.ax.xaxis.set_ticks_position('top')
cb.ax.xaxis.set_label_position('top')
ax[0, 1].axis("off")

gamma = np.linspace(start=0, stop=2*np.pi, num=100)
y = np.where(gamma <= np.pi, gamma, 2 * np.pi - gamma)
ax[1, 1].plot(gamma, y)
ax[1, 1].set_xlabel(r"$\gamma$")
ax[1, 1].set_ylabel(r"$\mathcal{G}(\gamma)$")
plt.tight_layout()
plt.show()
