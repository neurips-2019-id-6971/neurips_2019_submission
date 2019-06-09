# Copyright (c) 2019
# Copyright holder of the paper "Noisy-Input Entropy Search for Efficient Robust Bayesian Optimization" submitted to NeurIPS 2019 for review.
# All rights reserved.

import numpy as np
from core.acquisitions.acquisition_base import AcquisitionBase
from core.util.misc import optimize_gp_2, project_in_domain
from scipy.stats import norm


class UEI(AcquisitionBase):
    """
    Implements the Unscented Expected Improvement acquisition function:
    Source: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7759310
    """
    def __init__(self, domain, input_var, n_restarts=10):
        super(UEI, self).__init__(domain=domain, n_restarts=n_restarts)
        self.f_max = None  # Maximum of GP mean
        self.k = 1.0
        self.d = domain.lb.shape[0]
        self.input_var = input_var
        self.w_sig = self._calc_sigma_weights()
        assert np.isclose(sum(self.w_sig), 1.0)
        self.__name__ = 'uei'

    def next_point(self, gp):
        self.gp = gp
        _, self.f_max = optimize_gp_2(gp, self.domain)
        x_next = self._optimize_acq()
        return x_next

    def _f_acq(self, x):
        x = np.atleast_2d(x)

        # For each input in x, calculate the sigma points
        x_sig = self._calc_sigma_points(x)

        # UEI is the weighted sum of EI evaluated at the sigma points
        res = []
        for xi_sig, wi_sig in zip(x_sig, self.w_sig):
            res.append(self._ei(xi_sig) * wi_sig)
        res = sum(res)
        return res

    def _calc_sigma_points(self, x):
        x_sig = [x.copy() for _ in range(2*self.d + 1)]
        for i in range(self.d):
            ei = np.zeros(x.shape)
            ei[:, i] = np.sqrt((self.d + self.k) * self.input_var)
            x_sig[i] += ei
            x_sig[len(x_sig) - 1 - i] -= ei

        x_sig = [project_in_domain(xi_sig, self.domain) for xi_sig in x_sig]
        return x_sig

    def _calc_sigma_weights(self):
        w0 = [self.k / (self.d + self.k)]
        wi = [0.5 / (self.d + self.k)] * self.d
        return wi + w0 + wi

    def _ei(self, x):
        mu, var = self.gp.predict(x)
        std = np.sqrt(var)[:, None]
        gamma = (self.f_max - mu) / std
        return std * ((norm.cdf(gamma) - 1) * gamma + norm.pdf(gamma))
