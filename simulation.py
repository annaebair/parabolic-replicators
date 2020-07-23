"""
Implementation of parabolic replicators on a 2D grid where elements can interact with nearest neighbors.
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
    grid_size = 25
    a_rate = 1e-1
    b_rate = 1e-2
    c_rate = 1e-4
    d_rate = 1e-5
    e_rate = 1e-6
    frac_occupied = 0.3
    frac_x = 0.25
    w = 0.8
    sim_time = 5000000
    for phi in [1e-3, 1e-4, 1e-5]:
        for diffusion in [1e-4, 1e-5]:
            
            random.seed(seed)
            np.random.seed(seed)

            grid = Grid(size=grid_size,
                        a=a_rate,
                        b=b_rate,
                        c=c_rate,
                        d=d_rate,
                        e=e_rate,
                        fraction_occupied=frac_occupied,
                        fraction_x=frac_x,
                        phi=phi,
                        diffusion=diffusion,
                        w=w)

            x_counts, times, plots = grid.gillespie(time=sim_time, seed=seed)

            print(f'seed {seed}, phi {phi} diffusion {diffusion}')
            with open(f'results/{seed}_{phi}_{diffusion}_{w}_{grid_size}.txt', 'w+') as f:
                f.write(f'{seed} {a_rate} {b_rate} {c_rate} {d_rate} {e_rate} {frac_occupied} {phi} {diffusion} '
                        f'{w} {grid_size} {sim_time}\n')
                for i in range(len(x_counts)):
                    t, x = times[i], x_counts[i]
                    f.write(f'{t} {x}\n')


def main():
    single_run_trajectory()


if __name__ == '__main__':
    main()
