# Copyright (c) 2019
# Copyright holder of the paper "Noisy-Input Entropy Search for Efficient Robust Bayesian Optimization" submitted to NeurIPS 2019 for review.
# All rights reserved.

import GPy
import numpy as np
import pickle
import os
from tqdm import tqdm

import core.util.gp as gp_module
import core.util.misc as misc
from core.util.stats import calc_sigma_points_and_weights


def run_bo_nes(acq, objective, param, x_init, y_init, run_idx, res_dir,
               hyper_opt=False, hyper_opt_iter=5):
    while True:
        try:
            # Set up the Gaussian processes
            k_f = GPy.kern.RBF(input_dim=param['input_dim'], variance=param['signal_var'],
                               lengthscale=param['lengthscale'], ARD=True)
            gp = gp_module.GP(k_f, x_init, y_init, param['noise_var'], normalize_Y=True)

            # Store evaluated points
            X, Y = x_init, y_init
            x_belief = np.empty((param['max_iter'], param['input_dim']))
            g_belief = np.empty((param['max_iter'],))

            for it in range(param['max_iter']):
                if hyper_opt and not np.fmod(it, hyper_opt_iter):
                    print("Optimize hypers in iteration {}".format(it))

                    k_hyper = GPy.kern.RBF(input_dim=param['input_dim'], ARD=True)
                    gp_hyper = GPy.models.GPRegression(X, Y, kernel=k_hyper, noise_var=param['noise_var'])
                    gp_hyper.likelihood.constrain_fixed(param['noise_var'], warning=False)
                    gp_hyper.kern.variance.constrain_positive(warning=False)

                    ell_prior_1 = GPy.priors.LogGaussian(np.log(np.sqrt(param['input_var'][0])), 0.07)
                    ell_prior_2 = GPy.priors.LogGaussian(np.log(np.sqrt(param['input_var'][1])), 0.07)
                    gp_hyper.kern.lengthscale[[0]].set_prior(ell_prior_1, warning=False)
                    gp_hyper.kern.lengthscale[[1]].set_prior(ell_prior_2, warning=False)
                    gp_hyper.optimize()

                    gp.kernel.lengthscale[:] = gp_hyper.kern.lengthscale[:]
                    gp.kernel.variance[:] = gp_hyper.kern.variance[:]
                    gp.set_xy(X, Y)

                    print("GP lengthscales are: {}".format(gp.kernel.lengthscale))

                x_next = acq.next_point(gp)
                y_next = objective(x_next, param['noise_var'])
                X = np.vstack((X, x_next))
                Y = np.vstack((Y, y_next))
                gp.set_xy(X, Y)

                # Calculate current belief of optimum
                ngp = gp_module.NoisyInputGP.from_gp(gp, param['input_var'])
                x_guess, g_guess = misc.optimize_gp_2(ngp, acq.domain, n_restarts=100)
                x_belief[it] = x_guess
                g_belief[it] = g_guess

            sub_res_dir = res_dir + acq.__name__
            if not os.path.exists(sub_res_dir):
                os.mkdir(sub_res_dir)
            res = {'x_belief': x_belief, 'g_belief': g_belief, 'x_eval': X, 'f_eval': Y.squeeze()}
            pickle.dump(res, open(sub_res_dir + "/res_{}.pkl".format(run_idx), "wb"))

            return x_belief, g_belief, X, Y.squeeze()
        except ValueError:
            print("Warning: Need to restart due to error.")
            pass


def run_bo_uu(acq, objective, param, x_init, y_init, run_idx, res_dir):
    """
    Bayesian Optimization under Uncertainty. All these methods only depend on
    the NoisyGP. Thus we can simply exchange the acquisition function.
    """
    # Set up the Gaussian processes
    k_f = GPy.kern.RBF(input_dim=param['input_dim'], variance=param['signal_var'],
                       lengthscale=param['lengthscale'], ARD=True)
    gp = gp_module.GP(k_f, x_init, y_init, param['noise_var'])
    ngp = gp_module.NoisyInputGP.from_gp(gp, param['input_var'])

    # Store evaluated points
    X, Y = x_init, y_init
    x_belief = np.empty((param['max_iter'], param['input_dim']))
    g_belief = np.empty((param['max_iter'],))
    for it in range(param['max_iter']):
        x_next = acq.next_point(ngp)
        y_next = objective(x_next, param['noise_var'])
        X = np.vstack((X, x_next))
        Y = np.vstack((Y, y_next))
        gp.set_xy(X, Y)

        # Calculate current belief of optimum
        ngp.set_xy(X, Y)
        x_guess, g_guess = misc.optimize_gp_2(ngp, acq.domain, n_restarts=100)
        x_belief[it] = x_guess
        g_belief[it] = g_guess

    sub_res_dir = res_dir + acq.__name__ + "_uu"
    if not os.path.exists(sub_res_dir):
        os.mkdir(sub_res_dir)
    res = {'x_belief': x_belief, 'g_belief': g_belief, 'x_eval': X, 'f_eval': Y.squeeze()}
    pickle.dump(res, open(sub_res_dir + "/res_{}.pkl".format(run_idx), "wb"))

    return x_belief, g_belief, X, Y.squeeze()


def run_bo_unsc(acq, objective, param, x_init, y_init, run_idx, res_dir):
    """
    Bayesian Optimization with unscented transformation.
    """
    # Set up the Gaussian process
    k_f = GPy.kern.RBF(input_dim=param['input_dim'], variance=param['signal_var'],
                       lengthscale=param['lengthscale'], ARD=True)
    gp = gp_module.GP(k_f, x_init, y_init, param['noise_var'])
    ngp = gp_module.NoisyInputGP.from_gp(gp, param['input_var'])

    # Store evaluated points
    X, Y = x_init, y_init
    x_belief = np.empty((param['max_iter'], param['input_dim']))
    g_belief = np.empty((param['max_iter'],))
    for it in tqdm(range(param['max_iter']), disable=True):
        x_next = acq.next_point(gp)
        y_next = objective(x_next, param['noise_var'])
        X = np.vstack((X, x_next))
        Y = np.vstack((Y, y_next))
        gp.set_xy(X, Y)

        # Calculate current belief of optimum
        ngp.set_xy(X, Y)
        x_guess, g_guess = misc.optimize_gp_unsc(gp, acq.domain, acq, n_restarts=100)
        x_belief[it] = x_guess
        g_belief[it] = g_guess

    sub_res_dir = res_dir + acq.__name__
    if not os.path.exists(sub_res_dir):
        os.mkdir(sub_res_dir)
    res = {'x_belief': x_belief, 'g_belief': g_belief, 'x_eval': X, 'f_eval': Y.squeeze()}
    pickle.dump(res, open(sub_res_dir + "/res_{}.pkl".format(run_idx), "wb"))

    return x_belief, g_belief, X, Y.squeeze()


def run_bo_vanilla(acq, objective, param, x_init, y_init, run_idx, res_dir,
                   hyper_opt=False, hyper_opt_iter=5):
    """
    Vanilla Bayesian Optimization, i.e., no robustness considered.
    """
    # Set up the Gaussian process
    k_f = GPy.kern.RBF(input_dim=param['input_dim'], variance=param['signal_var'],
                       lengthscale=param['lengthscale'], ARD=True)
    gp = gp_module.GP(k_f, x_init, y_init, param['noise_var'], normalize_Y=True)

    # Store evaluated points
    X, Y = x_init, y_init
    x_belief = np.empty((param['max_iter'], param['input_dim']))
    g_belief = np.empty((param['max_iter'],))
    for it in tqdm(range(param['max_iter']), disable=True):
        if hyper_opt and not np.fmod(it, hyper_opt_iter):
            print("Optimize hypers in iteration {}".format(it))

            k_hyper = GPy.kern.RBF(input_dim=param['input_dim'], ARD=True)
            gp_hyper = GPy.models.GPRegression(X, Y, kernel=k_hyper, noise_var=param['noise_var'])
            gp_hyper.likelihood.constrain_fixed(param['noise_var'], warning=False)
            gp_hyper.kern.variance.constrain_positive(warning=False)

            ell_prior_1 = GPy.priors.LogGaussian(np.log(np.sqrt(param['input_var'][0])), 0.07)
            ell_prior_2 = GPy.priors.LogGaussian(np.log(np.sqrt(param['input_var'][1])), 0.07)
            gp_hyper.kern.lengthscale[[0]].set_prior(ell_prior_1, warning=False)
            gp_hyper.kern.lengthscale[[1]].set_prior(ell_prior_2, warning=False)
            gp_hyper.optimize()

            gp.kernel.lengthscale[:] = gp_hyper.kern.lengthscale[:]
            gp.kernel.variance[:] = gp_hyper.kern.variance[:]
            gp.set_xy(X, Y)

            print("GP lengthscales are: {}".format(gp.kernel.lengthscale))

        x_next = acq.next_point(gp)
        y_next = objective(x_next, param['noise_var'])
        X = np.vstack((X, x_next))
        Y = np.vstack((Y, y_next))
        gp.set_xy(X, Y)

        # Calculate current belief of optimum
        x_guess, g_guess = misc.optimize_gp_2(gp, acq.domain, n_restarts=100)
        x_belief[it] = x_guess
        g_belief[it] = g_guess

    sub_res_dir = res_dir + acq.__name__ + '_vanilla'
    if not os.path.exists(sub_res_dir):
        os.mkdir(sub_res_dir)
    res = {'x_belief': x_belief, 'g_belief': g_belief, 'x_eval': X, 'f_eval': Y.squeeze()}
    pickle.dump(res, open(sub_res_dir + "/res_{}.pkl".format(run_idx), "wb"))

    return x_belief, g_belief, X, Y.squeeze()


def run_bo_pseudo(acq, objective, param, x_init, y_init, run_idx, res_dir):
    """
    Bayesian Optimization on Noisy Input GP with pseudo observations.
    """
    def pseudo_objective(x, objective, noise_var):
        """Function generating pseudo-observations of the robust objective."""
        x = np.atleast_2d(x)
        zi = []
        for xi in x:
            x_sig, w_sig = calc_sigma_points_and_weights(xi, k_factor=1.0, input_var=param['input_var'], domain=acq.domain)
            zi.append(np.sum([wi_sig * objective(xi_sig, noise_var) for (wi_sig, xi_sig) in zip(w_sig, x_sig)]))

        return np.array(zi)[:, None]

    # Generate initial pseudo-observations
    z_init = pseudo_objective(x_init, objective, param['noise_var'])

    # Set up the Gaussian process
    k_f = GPy.kern.RBF(input_dim=param['input_dim'], variance=param['signal_var'],
                       lengthscale=param['lengthscale'], ARD=True)
    k_g, _ = gp_module.create_noisy_input_rbf_kernel(k_f, param['input_var'])
    gp = gp_module.GP(k_g, x_init, z_init, param['noise_var'])
    ngp = gp_module.NoisyInputGP.from_gp(gp, param['input_var'])

    # Each pseudo-observations uses n_sig_points evaluations of the objective.
    # For fair comparison to other methods, this needs to be considered in the
    # sense that only every n_sig_points observations, the GP is updated.
    # Thus, the regret curves will have plateaus of width n_sig_points.
    n_sig_points = param['input_dim'] * 2 + 1
    n_iterations = int(param['max_iter'] // n_sig_points)
    n_iterations = n_iterations + 1 if np.fmod(param['max_iter'], n_sig_points) else n_iterations

    # Store evaluated points
    X, Z = x_init, z_init
    x_belief = np.empty((n_iterations, param['input_dim']))
    g_belief = np.empty((n_iterations,))
    for it in tqdm(range(n_iterations), disable=True):
        x_next = acq.next_point(gp)
        z_next = pseudo_objective(x_next, objective, param['noise_var'])
        X = np.vstack((X, x_next))
        Z = np.vstack((Z, z_next))
        gp.set_xy(X, Z)

        # Calculate current belief of optimum
        ngp.set_xy(X, Z)
        x_guess, g_guess = misc.optimize_gp_2(gp, acq.domain, n_restarts=100)
        x_belief[it] = x_guess
        g_belief[it] = g_guess

    # Repeat the belief since we needed multiple evaluations of f(x) for one pseudo observation
    x_belief = np.repeat(x_belief, n_sig_points, axis=0)[:param['max_iter'], :]
    g_belief = np.repeat(g_belief, n_sig_points, axis=0)[:param['max_iter']]
    X = np.repeat(X, n_sig_points, axis=0)[:param['max_iter'] + param['n_init'], :]
    Z = np.repeat(Z, n_sig_points, axis=0)[:param['max_iter'] + param['n_init'], :]

    sub_res_dir = res_dir + acq.__name__ + '_pseudo'
    if not os.path.exists(sub_res_dir):
        os.mkdir(sub_res_dir)
    res = {'x_belief': x_belief, 'g_belief': g_belief, 'x_eval': X, 'f_eval': Z.squeeze()}
    pickle.dump(res, open(sub_res_dir + "/res_{}.pkl".format(run_idx), "wb"))

    return x_belief, g_belief, X, Z.squeeze()
