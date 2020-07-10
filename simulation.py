"""
Implementation of parabolic replicators on a 2D grid using the basic idea that elements can share the same grid square.
"""

import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
from gillespie import Grid


def single_run_trajectory():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    grid_size = 25
    a_rate = 1e-1
    b_rate = 1e-2
    c_rate = 1e-4
    frac_occupied = 0.3
    frac_x = 0.25
    phi = 1e-3  # 1e-4
    diffusion_param = 0.1
    sim_time = 5000000

    grid = Grid(size=grid_size,
                a=a_rate,
                b=b_rate,
                c=c_rate,
                fraction_occupied=frac_occupied,
                fraction_x=frac_x,
                phi=phi,
                diffusion=diffusion_param)

    x_counts, times, plots = grid.gillespie(sim_time, anim=False)


    # Write x_counts to file
    # print(f'seed {seed}, exchange rate {phi} diffusion {diffusion_param}')
    # with open(f'results/4_14_{seed}_{phi}_{diffusion_param}_{grid_size}.txt', 'w+') as f:
    #     f.write(f'{seed} {a_rate} {b_rate} {c_rate} {frac_occupied} {phi} {diffusion_param} '
    #             f'{grid_size} {sim_time}')
    #     for i in range(len(x_counts)):
    #         x = x_counts[i]
    #         t = times[i]
    #         f.write(f'{t} {x}\n')
    # fractional_x_counts = [i / grid_size ** 2 for i in x_counts]
    # print(x_counts[:10])
    # print(fractional_x_counts[:10])
    plt.plot(times, x_counts)
    plt.xlabel('timesteps')
    plt.ylabel('Number of T elements')
    plt.title('Trajectories of T counts')
    plt.show()
    # plt.savefig(f'old_sim_plots/{frac_occupied}_{phi}.png')


def single_run_animation():
    grid_size = 40
    a_rate = 1e-1
    b_rate = 1e-2
    c_rate = 1e-4
    frac_occupied = 0.3
    frac_x = 0.25
    sim_time = 30000000 
    phi = 0.001 
    diffusion = 0.001
    seed = 3

#    for seed in range(2, 10):
#        random.seed(seed)
#        np.random.seed(seed)
#        for phi in [0.0001, 5e-5, 1e-5, 5e-6, 1e-6]:
#            for diffusion in [0.01, 0.001, 0.0001, 1e-5, 1e-6]:
    fig = plt.figure()
    grid = Grid(size=grid_size,
                a=a_rate,
                b=b_rate,
                c=c_rate,
                fraction_occupied=frac_occupied,
                fraction_x=frac_x,
                phi=phi,
                diffusion=diffusion)

    x_counts, times, plots = grid.gillespie(sim_time, anim=False, seed=seed)
    pickle.dump(grid.grid, open(f'pickled_fourier_grids/{seed}_{grid_size}_{phi}_{diffusion}_{sim_time}', 'wb'))

    # Write x_counts to file
#    print(f'seed {seed}, phi {phi}, diffusion {diffusion}')
#    with open(f'sim_data/{sim_time}_{phi}_{diffusion}_{grid_size}_{seed}.txt', 'w+') as f:
#        f.write(f'{seed} {a_rate} {b_rate} {c_rate} {frac_occupied} {phi} {diffusion} '
#                f'{grid_size} {sim_time}\n')
#        for i in range(len(x_counts)):
#            x = x_counts[i]
#            t = times[i]
#            f.write(f'{t} {x}\n')

#            g = grid_copy @ [0, 0.5, 1]
#            filtered_grid = g - np.mean(g)
#            ftimage = np.fft.fft2(filtered_grid)
#            # ftimage = np.fft.fftshift(ftimage)
#            absftimage = np.abs(ftimage)

def iterated_runs():
    # seed = 0
    # random.seed(seed)
    # np.random.seed(seed)

    grid_size = 25
    a_rate = 1e-1
    b_rate = 1e-2
    c_rate = 1e-4
    frac_occupied = 0.3
    frac_x = 0.25
    sim_time = 5000000
    for seed in [1]:
        random.seed(seed)
        np.random.seed(seed)
        for phi in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 5e-3, 5e-4, 5e-5, 5e-6, 5e-7]:
            for diffusion in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
                print(phi, diffusion)
                grid = Grid(size=grid_size,
                            a=a_rate,
                            b=b_rate,
                            c=c_rate,
                            fraction_occupied=frac_occupied,
                            fraction_x=frac_x,
                            phi=phi,
                            diffusion=diffusion)

                x_counts, times, plots = grid.gillespie(sim_time, anim=False)
                # g = copy.deepcopy(grid.grid)
                # g = g @ [0, 0.5, 1]
                # zero_grid = np.zeros(g.shape)
                # ones_grid = np.ones(g.shape)
                # filtered_grid = np.where(g != 0, zero_grid, ones_grid)
                # plt.clf()
                # plt.imshow(filtered_grid)
                # plt.title(f'Filtered final grid: $\\varphi$={phi}, D={diffusion}')
                # # plt.savefig(f'fourier_plots/{phi}_{diffusion}_grid.png')
                # ftimage = np.fft.fft2(filtered_grid)
                # ftimage = np.fft.fftshift(ftimage)
                # plt.clf()
                # plt.imshow(np.abs(ftimage))
                # plt.colorbar()
                # plt.title(f'Fourier transform of final g rid with $\\varphi$={phi}, D={diffusion}')
                # plt.savefig(f'fourier_plots/{phi}_{diffusion}_fourier_inv.png')
                # # Write x_counts to file
                # with open(f'sim_data/5mil_{phi}_{diffusion}_seed{seed}.txt', 'w+') as f:
                #     f.write(f'{seed} {a_rate} {b_rate} {c_rate} {frac_occupied} {phi} {diffusion} '
                #             f'{frac_x} {grid_size} {sim_time}\n')
                #     for i in range(len(x_counts)):
                #         x = x_counts[i]
                #         t = times[i]
                #         f.write(f'{t} {x}\n')


def main():
    single_run_animation()


if __name__ == '__main__':
    main()
