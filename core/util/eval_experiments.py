# Copyright (c) 2019
# Copyright holder of the paper "Noisy-Input Entropy Search for Efficient Robust Bayesian Optimization" submitted to NeurIPS 2019 for review.
# All rights reserved.

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from tqdm import tqdm

from core.util import misc


def plot_convergence(res_dir, objective):

    # Processing the data may be time-consuming, thus save processed data
    res_dir_proc = res_dir[:-1] + '_proc/'
    processed_data_exists = False
    if os.path.exists(res_dir_proc):
        processed_data_exists = True
        print("There exists an already processes version of this directory.")

    method_keys = ['nes_rs',
                   'nes_ep',
                   'ucb_uu',
                   'ei_uu',
                   'uei',
                   'ei_vanilla',
                   'ei_pseudo',
                   'ucb_vanilla']

    lw = 5
    label_args = {method_keys[0]: dict(color='C0', label='NES-RS', ls='--', lw=lw),
                  method_keys[1]: dict(color='C2', label='NES-EP', ls='--', lw=lw),
                  method_keys[2]: dict(color='C4', label='UCB-UU', ls='-.', lw=lw),
                  method_keys[3]: dict(color='C5', label='EI-UU', ls='-.', lw=lw),
                  method_keys[4]: dict(color='C6', label='Unsc. EI', ls='-.', lw=lw),
                  method_keys[5]: dict(color='C7', label='Vanilla EI', ls='-', lw=lw),
                  method_keys[6]: dict(color='C8', label='Pseudo EI', ls=':', lw=lw),
                  method_keys[7]: dict(color='C9', label='Vanilla UCB', ls='--', lw=lw)}

    # Get some general parameters for notational convenience
    param = pickle.load(open(res_dir + 'param.pkl', 'rb'))
    g_opt = param['g_opt']
    filter_width = np.sqrt(param['input_var'])
    x_opt = np.array(param['x_opt'])
    # if objective == objectives.synthetic_1d_01:
    #     x_opt = np.array([0.31111868])
    # elif objective == objectives.rkhs_synth:
    #     x_opt = np.array([0.3157128])
    # elif objective == objectives.gmm_2d:
    #     x_opt = np.array([0.20029798, 0.20022463])
    # elif objective == objectives.synth_poly_2d_norm or objective == objectives.synth_poly_2d:
    #     x_opt = np.array([0.2672007, 0.67455068])
    # elif objective == objectives.hartmann_3d:
    #     x_opt = np.array([0.11728554, 0.56940675, 0.83030156])
    # else:
    #     raise ValueError("Error, error...")

    # Wrapper for robust objective (depending on dimensionality, this takes some time to evaluate)
    def objective_filtered(x):
        return misc.conv_wrapper(objective, x, filter_width, 201, param['input_dim'])

    # Unpickle data if exist
    res = {}
    for method_key in method_keys:
        file_name = res_dir + 'res_' + method_key + '.pkl'
        if os.path.isfile(file_name):
            tmp = pickle.load(open(file_name, 'rb'))
            if not (tmp['x_belief'] == 0.0).all():
                res[method_key] = pickle.load(open(file_name, 'rb'))
        else:
            if method_key in os.listdir(res_dir):
                sub_res_dir = res_dir + method_key + "/"

                x_belief = np.zeros((param['max_iter'], param['input_dim'], param['n_runs']))
                g_belief = np.zeros((param['max_iter'], param['n_runs']))
                x_eval = np.zeros((param['max_iter'] + param['n_init'], param['input_dim'], param['n_runs']))
                f_eval = np.zeros((param['max_iter'] + param['n_init'], param['n_runs']))
                for i, res_file_name in enumerate(os.listdir(sub_res_dir)):
                    tmp = pickle.load(open(sub_res_dir + res_file_name, 'rb'))
                    x_belief[:, :, i] = tmp['x_belief']
                    g_belief[:, i] = tmp['g_belief']
                    x_eval[:, :, i] = tmp['x_eval']
                    f_eval[:, i] = tmp['f_eval']
                res[method_key] = {'x_belief': x_belief, 'g_belief': g_belief,
                                   'x_eval': x_eval, 'f_eval': f_eval}

    if processed_data_exists:
        gx_belief = pickle.load(open(res_dir_proc + "gx_belief.pkl", "rb"))
    else:
        # Calculate relevant statistics for plotting for each method
        gx_belief = {}
        for method_key, method_res in tqdm(res.items()):
            # Evaluate robust objective at current belief of the optimum
            gx_belief[method_key] = np.zeros((param['max_iter'], param['n_runs']))
            for i in range(param['n_runs']):
                gx_belief[method_key][:, i] = objective_filtered(method_res['x_belief'][:, :, i])

        os.mkdir(res_dir_proc)
        pickle.dump(gx_belief, open(res_dir_proc + "gx_belief.pkl", "wb"))

    # Calculate inference regret and corresponding uncertainty bounds
    inf_regret, inf_regret_bounds = {}, {}
    dx_regret, dx_regret_bounds = {}, {}
    for method_key in gx_belief.keys():
        # Absolute distance to true robust optimum gives robust inference regret
        inf_regret[method_key] = np.abs(gx_belief[method_key] - g_opt)

        xn = res[method_key]['x_belief']
        dx_regret[method_key] = np.linalg.norm(xn.transpose((0, 2, 1)) - x_opt, axis=2)

        # Calculate confidence bounds for plotting
        percentiles = [25, 75]
        inf_regret_bounds[method_key] = np.percentile(inf_regret[method_key], percentiles, axis=1).T
        dx_regret_bounds[method_key] = np.percentile(dx_regret[method_key], percentiles, axis=1).T

    #############################################
    ############### VISUALIZATION ###############
    #############################################

    # plot_type = 'mean_pm_stderr'
    plot_type = 'median_pm_2575'
    alpha = 0.3
    n = np.arange(1, param['max_iter']+1)
        
    #######################################################################
    # ROBUST INFERENCE REGRET
    #######################################################################

    fig = plt.figure(figsize=(5.0, 5.0))
    plt.subplot(121)
    plt.title("Inference regret")
    for method_key in inf_regret.keys():
        if plot_type == 'mean_pm_stderr':
            y = np.mean(inf_regret[method_key], axis=1)
            std = np.std(inf_regret[method_key], axis=1) / np.sqrt(param['n_runs'])
            lower = y - std
            upper = y + std
        elif plot_type == 'median_pm_2575':
            y = np.median(inf_regret[method_key], axis=1)
            lower = inf_regret_bounds[method_key][:, 0]
            upper = inf_regret_bounds[method_key][:, 1]
        else:
            raise ValueError

        plt.fill_between(n, lower, upper, color=label_args[method_key]['color'], alpha=alpha)
        plt.plot(n, y, **label_args[method_key])
    ax = fig.gca()
    ax.set_yscale('log')
    plt.xlabel('# Function evaluations')
    plt.tight_layout()
    
    #######################################################################
    # DISTANCE REGRET
    #######################################################################

    plt.subplot(122)
    plt.title("Distance to optimum")
    for method_key in dx_regret.keys():
        if plot_type == 'mean_pm_stderr':
            y = np.mean(dx_regret[method_key], axis=1)
            std = np.std(dx_regret[method_key], axis=1) / np.sqrt(param['n_runs'])
            lower = y - std
            upper = y + std
        elif plot_type == 'median_pm_2575':
            y = np.median(dx_regret[method_key], axis=1)
            lower = dx_regret_bounds[method_key][:, 0]
            upper = dx_regret_bounds[method_key][:, 1]
        else:
            raise ValueError

        plt.fill_between(n, lower, upper, color=label_args[method_key]['color'], alpha=alpha)
        plt.plot(n, y, **label_args[method_key])
    ax = fig.gca()
    ax.set_yscale('log')
    plt.legend()
    plt.xlabel('# Function evaluations')
    plt.tight_layout()

    plt.show()
