"""
Implementation of parabolic replicators on a 2D grid using the basic idea that elements can share the same grid square.
"""

import math
import random
import time
import numpy as np
from scipy.stats import expon
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Grid:
    def __init__(self, size, a, b, c, f, m):
        self.size = size

        self.a = a
        self.b = b
        self.c = c
        self.m = m

        self.num_types = 3
        self.a_idx = 0
        self.t_idx = 1
        self.d_idx = 2
        self.a_value = 0.25
        self.t_value = 0.5
        self.d_value = 1
        self.square_max = 1

        self.weights = [self.a_value, self.t_value, self.d_value]
        self.grid = np.zeros((self.size, self.size, self.num_types))
        self.val_to_idx = {self.a_value: self.a_idx, self.t_value: self.t_idx, self.d_value: self.d_idx}

        self.tuples = [(int(math.floor(s / size)), s % size) for s in np.arange(size ** 2)]

        self.vn_neighbors = {}

        self.time = 0

        self.initialize_grid(f)

    def initialize_grid(self, f):
        """
        Place correct initial ratios of elements onto the grid. Indicate presence of an element as +1. Multiply
        by weights vector to determine how much space is left on each square given the size of each element.
        """
        frac_t = 0.1
        n_max = (1/min(self.weights)) * self.size ** 2  # count of 4N, i.e. total mass
        n = round(f * n_max)  # number (out of of 4N) (mass) that is occupied
        t_0 = round(n * frac_t * self.t_value)  # initial T count
        a_0 = round(n - (self.t_value / self.a_value) * t_0)  # initial A count

        for i in range(t_0):
            occupancy = self._occupancy()
            possibilities = np.nonzero(occupancy <= (self.square_max - self.t_value))
            if len(possibilities[0]) > 0:
                location = np.random.randint(0, len(possibilities[0]))
                index = (possibilities[0][location], possibilities[1][location], self.t_idx)
                self.grid[index] += 1
        for i in range(a_0):
            occupancy = self._occupancy()
            possibilities = np.nonzero(occupancy <= (self.square_max - self.a_value))
            if len(possibilities[0]) > 0:
                location = np.random.randint(0, len(possibilities[0]))
                index = (possibilities[0][location], possibilities[1][location], self.a_idx)
                self.grid[index] += 1

    def _counts(self):
        a = int(np.sum(self.grid[:, :, self.a_idx]))
        t = int(np.sum(self.grid[:, :, self.t_idx]))
        d = int(np.sum(self.grid[:, :, self.d_idx]))
        return a, t, d

    def _occupancy(self):
        return self.grid @ self.weights

    def rxn_choice(self, r):
        element_locs = {'a': np.nonzero(self.grid[:, :, self.t_idx] == 2),
                        'b': np.nonzero(self.grid[:, :, self.d_idx] == 1),
                        'c': np.nonzero((self.grid[:, :, self.a_idx] == 2) & (self.grid[:, :, self.t_idx] == 1)),
                        'exchange': None
                        }
        rxn_counts = np.array([len(element_locs['a'][0]), len(element_locs['b'][0]),  len(element_locs['c'][0])])
        rxn_ratios = [self.a, self.b, self.c] * rxn_counts
        ratios = np.concatenate((rxn_ratios, [r]))
        r_tot = np.sum(ratios)
        probs = ratios / r_tot
        rxn = np.random.choice(['a', 'b', 'c', 'exchange'], p=probs)
        locs = element_locs[rxn]
        return rxn, r_tot, locs

    def von_neumann_neighborhood(self, x, y):
        if (x, y) in self.vn_neighbors:
            return self.vn_neighbors[(x, y)]
        else:
            options = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
            neighbors = [(i % self.size, j % self.size) for (i, j) in options]
            self.vn_neighbors[(x, y)] = neighbors
            return neighbors

    def motion(self, timesteps):
        # Number of steps depends on dt, diffusion rate, and grid size
        steps = timesteps * self.m * self.size ** 2
        occupancy = self._occupancy()
        for s in range(math.floor(steps)):
            a_count, t_count, d_count = self._counts()
            tot = a_count + t_count + d_count
            probs = np.array([a_count, t_count, d_count])/tot
            obj = np.random.choice([self.a_value, self.t_value, self.d_value], p=probs)
            idx = self.val_to_idx[obj]
            option_grid = self.grid[:, :, idx]
            if np.sum(option_grid) == 0:
                continue
            flattened_option_grid = option_grid.flatten() / np.sum(option_grid)
            nums = np.arange(0, self.size ** 2)
            location = np.random.choice(nums, p=flattened_option_grid)
            x = int(math.floor(location / self.size))
            y = location % self.size
            neighbors = self.von_neumann_neighborhood(x, y)
            neighbor_occupancy = np.array([occupancy[n] for n in neighbors])
            options = np.array(neighbors)[np.nonzero(neighbor_occupancy <= 1 - obj)[0]]
            if len(options) > 0:
                new_square = random.choice(options)
                # Make objects move probabilistically at rate inversely proportional to size
                if random.random() < 1 / (4 * obj):
                    self.grid[x, y, idx] -= 1
                    self.grid[new_square[0], new_square[1], idx] += 1
                    occupancy = self._occupancy()

    def gillespie(self, time, r, anim):
        new_vectors = {'a': [0, 0, 1], 'b': [0, 2, 0], 'c': [0, 0, 1]}
        a_count, t_count, d_count = self._counts()
        print(f'Starting element counts: A: {a_count}, T: {t_count}, D: {d_count}')
        a_count, t_count, d_count = 0, 0, 0
        interval = 0
        x_count = []
        times = []
        plots = []
        cmaplist = [(1, 1, 1, 1), (0, 0.8, 0.2, 1), (0, 0, 1, 1), (0.7, 0.2, 0, 1), (1, 0.8, 0, 1)]
        no_d_cmaplist = [(1, 1, 1, 1), (0, 0.8, 0.2, 1), (0, 0, 1, 1)]
        my_cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, len(cmaplist))
        no_d_cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', no_d_cmaplist, len(no_d_cmaplist))
        while self.time < time:
            interval += 1
            if interval % 1000 == 0:
                print(f'{int(round(self.time / time, 2)* 100)}% completed')
            if interval % 1000 == 0 and anim:
                _, _, d_count = self._counts()
                anim_grid = im_frame(self.grid, self.size, self.a_idx, self.t_idx, self.d_idx)
                cmap = no_d_cmap if d_count == 0 else my_cmap
                p = plt.imshow(anim_grid, animated=True, cmap=cmap)
                ax = plt.gca()
                ax.set_xticks(np.arange(0, 2*self.size))
                ax.set_yticks(np.arange(0, 2*self.size))

                ax.set_xticks(np.arange(-.5, 2 * self.size, 2), minor=True)
                ax.set_yticks(np.arange(-.5, 2 * self.size, 2), minor=True)

                ax.set_xticklabels([''])
                ax.set_yticklabels([''])
                plt.grid(which='minor', color='k', linewidth=1)
                plots.append([p])
                print(self._counts())
            rxn, r_tot, locs = self.rxn_choice(r)
            if rxn in {'a', 'b', 'c'}:
                choice_idx = np.random.randint(0, len(locs[0]))
                self.grid[locs[0][choice_idx], locs[1][choice_idx]] = new_vectors[rxn]
            else:  # Exchange reaction

                # Removing elements
                a_count, t_count, d_count = self._counts()
                counts_on_grid = np.array([a_count, t_count, d_count])
                count_sum = np.sum(counts_on_grid)
                p_a, p_t, p_d = a_count / count_sum, t_count / count_sum, d_count / count_sum
                set_removal_ratios = [p_d, p_t ** 2, p_t * p_a ** 2, p_a ** 4]
                set_removal_probs = set_removal_ratios/np.sum(set_removal_ratios)
                set_to_remove = np.random.choice(['D', 'TT', 'TAA', 'AAAA'], p=set_removal_probs)
                if set_to_remove == 'D':
                    elements = [self.d_idx]
                elif set_to_remove == 'TT':
                    elements = [self.t_idx, self.t_idx]
                elif set_to_remove == 'TAA':
                    elements = [self.t_idx, self.a_idx, self.a_idx]
                else:
                    elements = [self.a_idx] * 4
                for elt in elements:
                    elt_grid = self.grid[:, :, elt]
                    flat_elt_grid = elt_grid.flatten()
                    flat_grid_probs = flat_elt_grid / np.sum(flat_elt_grid)
                    nums = np.arange(0, self.size ** 2)
                    to_remove = np.random.choice(nums, p=flat_grid_probs)
                    x, y = int(math.floor(to_remove / self.size)), to_remove % self.size
                    self.grid[x, y, elt] -= 1

                # Adding elements
                t_bulk = 0.02
                d_bulk = 0.08
                a_bulk = 0.9
                bulk_probs = np.array([d_bulk, t_bulk ** 2, t_bulk * a_bulk ** 2, a_bulk ** 4])
                occupancy = self._occupancy()
                occ_25 = len(np.nonzero(occupancy <= 0.75)[0])
                occ_50 = len(np.nonzero(occupancy <= 0.5)[0])
                occ_1 = len(np.nonzero(occupancy == 1)[0])
                occupancy_ratios = [occ_1, occ_50 ** 2, occ_50 * occ_25 ** 2, occ_25 ** 2]
                occupancy_probs = occupancy_ratios / np.sum(occupancy_ratios)
                probs = bulk_probs * occupancy_probs
                normalized_probs = probs / np.sum(probs)
                elt_set_to_add = np.random.choice(['D', 'TT', 'TAA', 'AAAA'], p=normalized_probs)
                if elt_set_to_add == 'D':
                    elements = [self.d_value]
                elif elt_set_to_add == 'TT':
                    elements = [self.t_value, self.t_value]
                elif elt_set_to_add == 'TAA':
                    elements = [self.t_value, self.a_value, self.a_value]
                else:
                    elements = [self.a_value] * 4
                for elt_val in elements:
                    occupancy = self._occupancy()
                    open_spots = np.nonzero(occupancy <= (1-elt_val))
                    num_open_spots = len(open_spots[0])
                    if num_open_spots == 0:
                        print(elements)
                        print(occupancy)
                    else:  # As long as there is at least one possible open spot to place the element
                        # if there is no space, the element will just not get added.
                        location = random.randint(0, num_open_spots-1)
                        x, y = open_spots[0][location], open_spots[1][location]
                        self.grid[x, y, self.val_to_idx[elt_val]] += 1

            # Advance time
            dt = expon(scale=r_tot).rvs()
            self.time += dt
            self.motion(dt)

            # Keep track of x over time
            a_count, t_count, d_count = self._counts()
            x_count.append(t_count + 2 * d_count)
            times.append(self.time)
        return a_count, t_count, d_count, x_count, times, plots


def im_frame(grid, size, a_idx, t_idx, d_idx):
    a_grid = grid[:, :, a_idx]
    t_grid = grid[:, :, t_idx]
    d_grid = grid[:, :, d_idx]
    new_grid = np.zeros((2 * size, 2 * size))
    ax, ay = np.where(a_grid > 0)
    tx, ty = np.where(t_grid > 0)
    dx, dy = np.where(d_grid > 0)
    if len(dx) > 0:
        for i in range(len(dx)):
            x = dx[i]
            y = dy[i]
            new_grid[2 * x + 1, 2 * y + 1] = 4
            new_grid[2 * x + 1, 2 * y] = 4
            new_grid[2 * x, 2 * y] = 4
            new_grid[2 * x, 2 * y + 1] = 4
    if len(tx) > 0:
        for i in range(len(tx)):
            x = tx[i]
            y = ty[i]
            new_grid[2 * x, 2 * y] = 2
            new_grid[2 * x + 1, 2 * y] = 2
            if t_grid[x, y] == 2:
                new_grid[2 * x, 2 * y + 1] = 2
                new_grid[2 * x + 1, 2 * y + 1] = 2
    if len(ax) > 0:
        for i in range(len(ax)):
            x = ax[i]
            y = ay[i]
            for a_to_place in range(int(a_grid[x, y])):
                if new_grid[2 * x, 2 * y] == 0:
                    new_grid[2 * x, 2 * y] = 1
                elif new_grid[2 * x + 1, 2 * y] == 0:
                    new_grid[2 * x + 1, 2 * y] = 1
                elif new_grid[2 * x, 2 * y + 1] == 0:
                    new_grid[2 * x, 2 * y + 1] = 1
                else:
                    new_grid[2 * x + 1, 2 * y + 1] = 1
    return new_grid

# TODO: refactor into: single set of params (plot trajectory), iterated runs, and single set (animation)
def main():
    fig = plt.figure()
    seeds = [0, 1, 2]
    for s in seeds:
        print('S:', s)
        random.seed(s)
        np.random.seed(s)
        a_rate = 1e-1
        b_rate = 1e-2
        c_rate = 1e-4
        frac_occupied = 0.3
        for c, d in [(0.001, 0.01), (0.001, 0.1)]:
            chemostat_rate = c
            diffusion_param = d
            grid_size = 50
            sim_time = 10000000
            start = time.time()
            grid = Grid(size=grid_size,
                        a=a_rate,
                        b=b_rate,
                        c=c_rate,
                        f=frac_occupied,
                        m=diffusion_param)

            a, t, d, x_counts, times, plots = grid.gillespie(sim_time, chemostat_rate, anim=False)
            end = time.time()
            # print('Net time: ', end - start)
            # print(f'Final element counts: A: {a}, T: {t}, D: {d}')
            # Write x_counts to file
            print(f'seed {s}, exchange rate {chemostat_rate} diffusion {diffusion_param}')
            with open(f'results/4_2_{s}_{chemostat_rate}_{diffusion_param}_{grid_size}.txt', 'w+') as f:
                f.write(f'{s} {a_rate} {b_rate} {c_rate} {frac_occupied} {chemostat_rate} {diffusion_param} '
                        f'{grid_size} {sim_time}')
                for i in range(len(x_counts)):
                    x = x_counts[i]
                    t = times[i]
                    f.write(f'{t} {x}\n')

        # Histogram of T_counts over trajectories

        # plt.hist(x_counts, bins=50, alpha=0.5)
        # plt.xlabel('Number of T')
        # plt.ylabel('Timesteps with number of T present')
        # plt.title('Histogram of counts of T for 3 trajectories')

        # x_count trajectories

        # plt.plot(times, x_counts)
        # plt.xlabel('timesteps')
        # plt.ylabel('Number of T elements')
        # plt.title(f'Trajectories of T counts for {round(frac_occupied * 100)}% occupied')
    #
    # plt.show()

    # Animation

    # ani = animation.ArtistAnimation(fig, plots, interval=400, blit=True, repeat_delay=2000)
    # ani.save('a.html')
    # plt.grid(color='k')
    # plt.show()


if __name__ == '__main__':
    main()
