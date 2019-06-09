import GPy
# import GPyOpt
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import optimize
from scipy.integrate import ode
from tqdm import tqdm

from core.util import gp as gp_module


class Planet(object):
    def __init__(self, pos, mass, name):
        self.pos = pos
        self.mass = mass
        self.name = name


class WorldModel(object):
    def __init__(self):
        self.home_planet = Planet(np.array([0.0, 0.0]), 1.0, 'A')
        self.target_planet = Planet(np.array([10.0, 0.0]), 10.0, 'D')
        self.planets = [self.home_planet, self.target_planet]

        # Planets for sling-shot maneuver
        self.planets.append(Planet(np.array([4.0, 0.2]), 2.0, 'B'))
        self.planets.append(Planet(np.array([5.0, -1.5]), 8.0, 'C'))

        self.G = 1.0

    def ode_rhs(self, _, x):
        pos = x[:2]
        vel = x[2:]

        forces = []
        for planet in self.planets:
            r = planet.pos - pos
            d = np.linalg.norm(r)
            f = self.G * planet.mass * r / d ** 3
            forces.append(f)

        dpos = vel
        dvel = sum(forces)

        dx = np.hstack((dpos, dvel))
        return dx


class WorldSimulation(object):
    def __init__(self, ode_rhs):
        self.solver = ode(ode_rhs)
        self.solver.set_integrator('vode')

    def run(self, x0, dt, n_steps):
        self.solver.set_initial_value(x0, 0)

        t = np.zeros(n_steps+1)
        x = np.zeros((n_steps+1, 4))
        x[0, :] = x0
        i = 0
        while self.solver.successful() and self.solver.t < dt * n_steps:
            self.solver.integrate(self.solver.t + dt)
            t[i] = self.solver.t
            x[i] = self.solver.y
            i += 1

        return x, t


def main():
    from core.util.objectives import rocket_simulation
    from core.util import misc

    # Bounds for configuration
    alpha_bounds = np.deg2rad([-10, 45])
    speed_bounds = [4.1, 5.0]

    input_var = np.array([np.deg2rad(3.0), 0.05]) ** 2
    filter_width = np.sqrt(input_var)

    def objective_filtered(x):
        return misc.conv_wrapper(rocket_simulation, x, filter_width, 21, 2)

    n_grid = 100
    alpha_linspace = np.linspace(*alpha_bounds, n_grid)
    speed_linspace = np.linspace(*speed_bounds, n_grid)
    alpha_grid, speed_grid = np.meshgrid(alpha_linspace, speed_linspace)


    # alpha_bounds_training = np.deg2rad([15, 50])
    # speed_bounds_training = np.array([4.5, 4.9])
    # # alpha_bounds_training = alpha_bounds
    # # speed_bounds_training = speed_bounds
    #
    # alpha_linspace_training = np.linspace(*alpha_bounds_training, n_grid)
    # speed_linspace_training = np.linspace(*speed_bounds_training, n_grid)
    # alpha_grid_training, speed_grid_training = np.meshgrid(alpha_linspace_training, speed_linspace_training)
    #
    # training_min = [alpha_bounds_training[0], speed_bounds_training[0]]
    # training_max = [alpha_bounds_training[1], speed_bounds_training[1]]
    # x_training, x_training_shape = misc.create_flattened_meshgrid_lin(training_min, training_max, 20, 2)
    # y_training = rocket_simulation(x_training)
    #
    # eval_min = [alpha_bounds[0], speed_bounds[0]]
    # eval_max = [alpha_bounds[1], speed_bounds[1]]
    # x_eval, eval_shape = misc.create_flattened_meshgrid_lin(eval_min, eval_max, n_grid, 2)
    #
    # from mpl_toolkits.mplot3d import Axes3D
    # from matplotlib import cm
    #
    # import json
    # with open('../Experiments/cfg/param_rocket_sim.json') as f:
    #     param_obj = json.load(f)
    #
    # with open('../Experiments/cfg/exp_params_{}d.json'.format(param_obj['input_dim'])) as f:
    #     param_exp = json.load(f)
    # param = {**param_exp, **param_obj}
    # print(param)
    #
    # # Set up the Gaussian process
    # k_f = GPy.kern.RBF(input_dim=param['input_dim'], variance=param['signal_var'],
    #                    lengthscale=param['lengthscale'], ARD=True)
    # k_g, _ = gp_module.create_noisy_input_rbf_kernel(k_f, param['input_var'])
    # gp = gp_module.GP(k_g, x_training, y_training, param['noise_var'], normalize_Y=True)
    # ngp = gp_module.NoisyInputGP.from_gp(gp, param['input_var'])
    # nmu, _ = ngp.predict(x_eval)
    # nmu = nmu.reshape(eval_shape)
    #
    # from scipy.optimize import Bounds
    # domain = Bounds(np.array(training_min), np.array(training_max))
    # x_opt, g_opt = misc.optimize_gp_2(ngp, domain, n_restarts=100)
    # x_opt_viz = np.array([np.deg2rad(32.5), 4.71])
    # print("g_opt from ngp optimization: {}".format(g_opt))
    # print("g_opt from evaluating x_opt: {}".format(objective_filtered(x_opt)))
    # print("g_opt at visual optimum:     {}".format(objective_filtered(x_opt_viz)))
    # print("x_opt: {}".format(x_opt))
    #
    # f = np.load('rocket_sim_cost.npy')
    #
    # plt.figure()
    # plt.contourf(np.rad2deg(alpha_grid), speed_grid, nmu, levels=50)
    # plt.colorbar()
    # # plt.plot(x_eval[:, 0], x_eval[:, 1], 'ko')
    # plt.plot(np.rad2deg(x_training[:, 0]), x_training[:, 1], 'kx')
    # plt.plot(np.rad2deg(x_opt[0]), x_opt[1], 'C3x')
    # plt.plot(np.rad2deg(x_opt_viz[0]), x_opt_viz[1], 'bx')
    #
    #
    # plt.show()
    # return

    # f = np.empty((n_grid, n_grid))
    # for i, a0 in enumerate(tqdm(alpha_linspace)):
    #     for j, s0 in enumerate(speed_linspace):
    #         f[i, j] = rocket_simulation(np.array([a0, s0]))
    # np.save('rocket_sim_cost.npy', f)
    f = np.load('rocket_sim_cost.npy')

    world = WorldModel()
    sim = WorldSimulation(world.ode_rhs)
    alphas = np.deg2rad(([1., 29.5, 12.8]))
    speeds = np.array([4.32, 4.75, 4.49])
    traj_colors = ['C4', 'C1', 'C3']
    traj_shapes = ['X', 'o', 's']

    file_name_pdf_cost_function = "/home/flr2lr/PhD/Paper/2019_NeurIPS/draft/figures/application_cost_function.pdf"
    file_name_pdf_cost_function_cropped = "/home/flr2lr/PhD/Paper/2019_NeurIPS/draft/figures/application_cost_function_cropped.pdf"
    file_name_pdf_trajectories = "/home/flr2lr/PhD/Paper/2019_NeurIPS/draft/figures/application_trajectories.pdf"
    file_name_pdf_trajectories_cropped = "/home/flr2lr/PhD/Paper/2019_NeurIPS/draft/figures/application_trajectories_cropped.pdf"
    file_name_pdf_colorbar = "/home/flr2lr/PhD/Paper/2019_NeurIPS/draft/figures/application_cost_function_colorbar.pdf"

    #######################################################################
    # OBJECTIVE FUNCTION
    #######################################################################

    plt.figure(figsize=(5.0, 5.0))
    plt.xlabel("Initial Angle [deg]")
    plt.ylabel("Initial Speed [x]")

    levels = np.linspace(-1.26, -0.93, 50)
    plt.contourf(np.rad2deg(alpha_grid), speed_grid, f.T, levels=levels, alpha=1.0)
    for alpha0, speed0, color, shape in zip(alphas, speeds, traj_colors, traj_shapes):
        plt.plot(np.rad2deg(alpha0), speed0, color + shape)
    plt.colorbar()
    plt.title("Objective function\\xLim: {:.1f}, {:.1f}, xLim: {:.3e}, {:.3e}".format(*plt.xlim(), *plt.ylim()))
    plt.savefig(file_name_pdf_cost_function, format='pdf')

    fig = plt.figure(figsize=(3.0, 3.0), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    plt.contourf(np.rad2deg(alpha_grid), speed_grid, f.T, levels=levels, alpha=1.0)
    for alpha0, speed0, color, shape in zip(alphas, speeds, traj_colors, traj_shapes):
        plt.plot(np.rad2deg(alpha0), speed0, color + shape)
    # plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.savefig(file_name_pdf_cost_function_cropped, format='pdf')

    fig = plt.figure(figsize=(0.3, 5.0), frameon=False)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    mpl.colorbar.ColorbarBase(ax)
    plt.savefig(file_name_pdf_colorbar, format='pdf')

    #######################################################################
    # EXAMPLE TRAJECTORIES
    #######################################################################

    plt.figure(figsize=(5.0, 5.0))
    plt.xlabel("x-position in space")
    plt.ylabel("y-position in space")
    # plt.axis('equal')
    for planet in world.planets:
        plt.plot(planet.pos[0], planet.pos[1], 'ko', markersize=5)
        plt.text(planet.pos[0], planet.pos[1] + 0.4, planet.name)

    plt.plot(world.home_planet.pos[0], world.home_planet.pos[1], 'ko', markersize=10)
    plt.plot(world.target_planet.pos[0], world.target_planet.pos[1], 'ko', markersize=10)

    for alpha0, speed0, color, shape in zip(alphas, speeds, traj_colors, traj_shapes):
        vel0 = speed0 * np.array([np.cos(alpha0), np.sin(alpha0)])
        pos0 = np.array([0.1, 0.0])

        x0 = np.hstack((pos0, vel0))
        dt = 0.001
        n_steps = 4000
        x, t = sim.run(x0, dt, n_steps)

        d = np.linalg.norm(x[:, :2] - world.target_planet.pos, axis=1)
        idx_min = np.argmin(d)

        x = np.vstack((x0, x))

        plt.plot(x[:idx_min, 0], x[:idx_min, 1], color)
        plt.plot(x[500:idx_min:600, 0], x[500:idx_min:600, 1], color + shape)

    plt.ylim(-3, 3)
    plt.title("Example trajectories\\xLim: {:.1f}, {:.1f}, xLim: {:.3e}, {:.3e}".format(*plt.xlim(), *plt.ylim()))
    plt.tight_layout()
    plt.savefig(file_name_pdf_trajectories, format='pdf')

    fig = plt.figure(figsize=(3.0, 3.0), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    for planet in world.planets:
        plt.plot(planet.pos[0], planet.pos[1], 'ko', markersize=5)
        plt.text(planet.pos[0], planet.pos[1] + 0.3, planet.name, size=15)

    plt.plot(world.home_planet.pos[0], world.home_planet.pos[1], 'ko', markersize=10)
    plt.plot(world.target_planet.pos[0], world.target_planet.pos[1], 'ko', markersize=10)

    for alpha0, speed0, color, shape in zip(alphas, speeds, traj_colors, traj_shapes):
        vel0 = speed0 * np.array([np.cos(alpha0), np.sin(alpha0)])
        pos0 = np.array([0.1, 0.0])

        x0 = np.hstack((pos0, vel0))
        dt = 0.001
        n_steps = 4000
        x, t = sim.run(x0, dt, n_steps)

        d = np.linalg.norm(x[:, :2] - world.target_planet.pos, axis=1)
        idx_min = np.argmin(d)

        x = np.vstack((x0, x))

        plt.plot(x[:idx_min, 0], x[:idx_min, 1], color)
        plt.plot(x[500:idx_min:600, 0], x[500:idx_min:600, 1], color + shape)

    plt.ylim(-3, 3)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(file_name_pdf_trajectories_cropped, format='pdf')

    plt.show()


if __name__ == '__main__':
    main()

