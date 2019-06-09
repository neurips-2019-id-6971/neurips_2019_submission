# Copyright (c) 2019
# Copyright holder of the paper "Noisy-Input Entropy Search for Efficient Robust Bayesian Optimization" submitted to NeurIPS 2019 for review.
# All rights reserved.

import matplotlib.pyplot as plt
import numpy as np


def gp_pred_with_bounds(x_plot, mean, variance, color='C0', label=None, std_factor=1.0, lw=1.5):
    x_plot = x_plot.squeeze()
    mean = mean.squeeze()
    std = np.sqrt(variance).squeeze()

    plt.plot(x_plot, mean, color=color, label=label, lw=lw)
    plt.fill_between(x_plot, mean + std_factor*std, mean - std_factor*std, color=color, alpha=0.3)
