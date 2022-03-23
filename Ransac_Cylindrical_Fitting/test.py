import numpy as np
from scipy.optimize import leastsq
import pandas as pd
from matplotlib.pyplot import title, xlabel, ylabel
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties



fig = plt.figure()
ax = fig.gca(projection='3d')

delta = np.linspace(-np.pi, np.pi, 20)
z = np.linspace(-10, 10, 20)

Delta, Z = np.meshgrid(delta, z)
X = 2 * np.cos(Delta)
Y = 2 * np.sin(Delta)

ax.plot_surface(X, Y, Z, alpha=0.2)
plt.show()