# Copyright (c) 2019
# Copyright holder of the paper "Noisy-Input Entropy Search for Efficient Robust Bayesian Optimization" submitted to NeurIPS 2019 for review.
# All rights reserved.

import numpy as np
from scipy.optimize import minimize
import sobol_seq
from tqdm import tqdm

from core.util import misc


class AcquisitionBase:
    def __init__(self, domain, n_restarts=10):
        self.domain = domain
        self.n_restarts = n_restarts
        self.gp = None

    def next_point(self, gp):
        raise NotImplementedError

    def _optimize_acq(self):
        dim = self.gp.kernel.input_dim
        x0_candidates = self.domain.lb + (self.domain.ub - self.domain.lb) * \
                        sobol_seq.i4_sobol_generate(dim, self.n_restarts) + \
                        np.random.randn(self.n_restarts, dim)
        x_opt_candidates = np.empty((self.n_restarts, dim))
        f_opt = np.empty((self.n_restarts,))
        for i, x0 in enumerate(tqdm(x0_candidates, disable=True)):
            res = minimize(fun=misc.neg(self._f_acq), x0=x0, bounds=self.domain)
            x_opt_candidates[i] = res['x']
            f_opt[i] = -1 * res['fun']

        x_opt = x_opt_candidates[np.argmax(f_opt)]
        return x_opt

    def _f_acq(self, x):
        raise NotImplementedError
