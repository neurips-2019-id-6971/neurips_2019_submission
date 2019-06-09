# Copyright (c) 2019
# Copyright holder of the paper "Noisy-Input Entropy Search for Efficient Robust Bayesian Optimization" submitted to NeurIPS 2019 for review.
# All rights reserved.

import numpy as np
from .acquisition_base import AcquisitionBase


class UCB(AcquisitionBase):
    def __init__(self, domain, n_restarts=10, exploration_factor=2.0):
        super(UCB, self).__init__(domain=domain, n_restarts=n_restarts)
        self.exploration_factor = exploration_factor
        self.__name__ = 'ucb'

    def next_point(self, gp):
        self.gp = gp
        x_next = self._optimize_acq()
        return x_next

    def _f_acq(self, x):
        x = np.atleast_2d(x)
        mu, var = self.gp.predict(x)
        std = np.sqrt(var)[:, None]
        return mu + self.exploration_factor * std
