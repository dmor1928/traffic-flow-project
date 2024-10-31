import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np


fig, ax = plt.subplots()

for i, sigma in enumerate([0.25, 0.15, 0.05]):
    np.random.seed(42)
    init_v = np.clip(np.random.normal(loc=0.5, scale=sigma, size=100), 0, 1)
    mean = np.mean(init_v)
    print(mean)
    ax.hist(init_v, bins=15, color=get_cmap('viridis')(i / 2))

    ax.vlines(x=mean, ymin=0, ymax=15, colors=get_cmap('viridis')(i/2), linewidths=4, linestyle='dashed')

plt.savefig('42_dsitributions.png', dpi=300)
