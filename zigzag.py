import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

import functions

functionClass = functions.Rosenbrock

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
interval = functionClass.interval
x = np.arange(interval[0][0], interval[0][1], 0.05)
y = np.arange(interval[1][0], interval[1][1], 0.05)
X, Y = np.meshgrid(x, y)
Z = functions.Rosenbrock.getZMeshGrid(X, Y)
ax.view_init(elev=functionClass.camera[0], azim=functionClass.camera[1])

ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
