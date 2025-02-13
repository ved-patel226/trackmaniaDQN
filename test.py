import numpy as np
from matplotlib import pyplot as plt


data = np.load("map2.npy")
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(data[:, 0], data[:, 1], data[:, 2])
plt.show()
