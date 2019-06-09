# Copyright (c) 2019
# Copyright holder of the paper "Noisy-Input Entropy Search for Efficient Robust Bayesian Optimization" submitted to NeurIPS 2019 for review.
# All rights reserved.

import numpy as np
from core.acquisitions.acquisition_base import AcquisitionBase
from core.util.misc import optimize_gp_2
from scipy.stats import norm


class EI(AcquisitionBase):
    def __init__(self, domain, n_restarts=10):
        super(EI, self).__init__(domain=domain, n_restarts=n_restarts)
        self.f_max = None  # Maximum of GP mean
        self.__name__ = 'ei'

    def next_point(self, gp):
        self.gp = gp
        _, self.f_max = optimize_gp_2(gp, self.domain)
        x_next = self._optimize_acq()
        return x_next

    def _f_acq(self, x):
        x = np.atleast_2d(x)
        mu, var = self.gp.predict(x)
        std = np.sqrt(var)[:, None]
        gamma = (self.f_max - mu) / std
        return std * ((norm.cdf(gamma) - 1) * gamma + norm.pdf(gamma))
