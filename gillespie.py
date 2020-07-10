"""
Implementation of parabolic replicators on a 2D grid using the basic idea that elements can share the same grid square.
"""

import math
import random
import numpy as np
from scipy.stats import expon
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle


def _im_frame(grid, size, a_idx, t_idx, d_idx):
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


class Grid:
    def __init__(self, size, a, b, c, fraction_occupied, fraction_x, phi, diffusion):
        self.size = size

        self.a = a
        self.b = b
        self.c = c
        self.phi = phi
        self.diffusion = diffusion

        self.a_idx = 0
        self.t_idx = 1
        self.d_idx = 2
        self.a_value = 0.25
        self.t_value = 0.5
        self.d_value = 1
        self.square_max = 1

        self.weights = [self.a_value, self.t_value, self.d_value]
        self.grid = np.zeros((self.size, self.size, len(self.weights)))
        self.reaction_counts = np.zeros((self.size, self.size, len(self.weights)))
        self.val_to_idx = {self.a_value: self.a_idx, self.t_value: self.t_idx, self.d_value: self.d_idx}
        self.tuples = [(int(math.floor(s / size)), s % size) for s in np.arange(size ** 2)]
        self.vn_neighbors = {}
        self.time = 0
        self.a_probs = []
        self.b_probs = []

        self.q = self.initialize_grid(fraction_occupied, fraction_x)

    def initialize_grid(self, fraction_occupied, fraction_x):
        """
        Place correct initial ratios of elements onto the grid. Indicate presence of an element as +1. Multiply
        by weights vector to determine how much space is left on each square given the size of each element.
        """
        grid_space = (1/min(self.weights)) * self.size ** 2  # 4L^2
        occupied_grid_mass = round(fraction_occupied * grid_space)  # 4L^2 * 0.3
        x_mass = round(fraction_x * occupied_grid_mass)
        num_x = round(x_mass / 2)

        # subdivide x_mass using X = 2D + T, (a/b)T**2 = D, and quadratic formula, where T and D are count variables
        num_t = round((self.b / (4 * self.a)) * (math.sqrt(1 + (8 * self.a * num_x) / self.b) - 1))
        num_d = round(num_x - num_t)
        num_a = occupied_grid_mass - 2 * num_t - 4 * num_d

        for num, val, idx in [(num_d, self.d_value, self.d_idx),
                              (num_t, self.t_value, self.t_idx),
                              (num_a, self.a_value, self.a_idx)]:
            for i in range(num):
                occupancy = self._occupancy()
                possibilities = np.nonzero(occupancy <= (self.square_max - val))
                if len(possibilities[0]) > 0:
                    location = np.random.randint(0, len(possibilities[0]))
                    index = (possibilities[0][location], possibilities[1][location], idx)
                    self.grid[index] += 1
                else:
                    print('can\'t add')
        return (num_a + num_t + num_d) / occupied_grid_mass

    def _counts(self):
        a = int(np.sum(self.grid[:, :, self.a_idx]))
        t = int(np.sum(self.grid[:, :, self.t_idx]))
        d = int(np.sum(self.grid[:, :, self.d_idx]))
        return a, t, d

    def _occupancy(self):
        return self.grid @ self.weights

    def _von_neumann_neighborhood(self, x, y):
        if (x, y) in self.vn_neighbors:
            return self.vn_neighbors[(x, y)]
        else:
            options = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
            neighbors = [(i % self.size, j % self.size) for (i, j) in options]
            self.vn_neighbors[(x, y)] = neighbors
            return neighbors

    def _move(self, value):
        """
        Single step in a random walk for a uniformly randomly chosen element of the given value.
        Does nothing if there are no elements with the given value on the grid.
        """
        occupancy = self._occupancy()
        idx = self.val_to_idx[value]
        option_grid = self.grid[:, :, idx]
        if np.sum(option_grid) > 0:
            flattened_option_grid = option_grid.flatten() / np.sum(option_grid)
            nums = np.arange(0, self.size ** 2)
            location = np.random.choice(nums, p=flattened_option_grid)
            x, y = int(math.floor(location / self.size)), location % self.size
            neighbors = self._von_neumann_neighborhood(x, y)
            neighbor_occupancy = np.array([occupancy[n] for n in neighbors])
            options = np.array(neighbors)[np.nonzero(neighbor_occupancy <= 1 - value)[0]]
            if len(options) > 0:
                new_square = random.choice(options)
                self.grid[x, y, idx] -= 1
                self.grid[new_square[0], new_square[1], idx] += 1

    def _add_elt(self, value):
        """
        Adds an element of the given value to a randomly chosen available spot.
        """
        occupancy = self._occupancy()
        flattened = occupancy.flatten()
        ones = np.ones(occupancy.flatten().shape)
        available = ones - flattened
        proportion_available = np.floor(available / value)
        if np.sum(proportion_available) > 0:
            available_probs = proportion_available / np.sum(proportion_available)
            nums = np.arange(0, self.size ** 2)
            to_add = np.random.choice(nums, p=available_probs)
            x, y = int(math.floor(to_add / self.size)), to_add % self.size
            self.grid[x, y, self.val_to_idx[value]] += 1

    def _remove_elt(self, idx):
        """
        Remove an element of the given index uniformly at random from the grid
        """
        elt_grid = self.grid[:, :, idx]
        flat_elt_grid = elt_grid.flatten()
        if np.sum(flat_elt_grid) > 0:
            flat_grid_probs = flat_elt_grid / np.sum(flat_elt_grid)
            nums = np.arange(0, self.size ** 2)
            to_remove = np.random.choice(nums, p=flat_grid_probs)
            x, y = int(math.floor(to_remove / self.size)), to_remove % self.size
            self.grid[x, y, idx] -= 1

    def reaction(self, indicate_count=False):
        """
        Select a reaction based on predefined rates and current state of the grid. Carry out the chosen reaction.
        Returns r_tot.
        """
        element_locs = {'a': np.nonzero(self.grid[:, :, self.t_idx] == 2),
                        'b': np.nonzero(self.grid[:, :, self.d_idx] == 1),
                        'c': np.nonzero((self.grid[:, :, self.a_idx] == 2) & (self.grid[:, :, self.t_idx] == 1))
                        }
        """
        1. How many squares there are that can undergo an a, b, or c reaction.
        - For move, try adding total number of that element
        - For remove, again total number of that element.
        - For add, maybe try the number of places the element could go.
        - But then this runs into the issue of sort of diluting the effect of phi and q. Or is this okay because
        it's what happens in Gillespie?
        - Also if we do this then it won't allow reactions that can't happen to be chosen.
        - Do we want to count how many (for instance) T elements there are? Or do we want to use the mass of T? I think
        we should just stick to counts because rxn_ratios accounts for relative size differences.
        """

        num_a, num_t, num_d = self._counts()
        occupancy = self._occupancy()
        empty_size_a = sum(np.floor((1 - occupancy) / self.a_value).flatten())
        empty_size_t = sum(np.floor((1 - occupancy) / self.t_value).flatten())
        empty_size_d = sum(np.floor((1 - occupancy) / self.d_value).flatten())
        rxn_counts = np.concatenate(([len(element_locs['a'][0]), len(element_locs['b'][0]), len(element_locs['c'][0])],
                                    [empty_size_a, empty_size_t, empty_size_d],  # b_db: add to surface
                                    [num_a, num_t, num_d],  # db_b: remove from surface
                                    [num_a, num_t, num_d]))  # move
        rxn_ratios = np.concatenate(([self.a, self.b, self.c],
                                 [self.phi, self.phi, self.phi],
                                 [self.phi * self.q, self.phi * self.q * 0.5, self.phi * self.q * 0.25],
                                 [self.diffusion, 0.5 * self.diffusion, 0.25 * self.diffusion]))
        ratios = rxn_ratios * rxn_counts
        r_tot = np.sum(ratios)
        probs = ratios / r_tot
        self.a_probs.append(probs[0])
        self.b_probs.append(probs[1])
        reaction_options = ['a', 'b', 'c',
                            'b_db_a', 'b_db_t', 'b_db_d',
                            'db_b_a', 'db_b_t', 'db_b_d',
                            'move_a', 'move_t', 'move_d']
        rxn = np.random.choice(reaction_options, p=probs)
        if rxn in {'a', 'b', 'c'}:  # association, dissociation, or replication
            locs = element_locs[rxn]
            new_vectors = {'a': [0, 0, 1], 'b': [0, 2, 0], 'c': [0, 0, 1]}
            choice_idx = np.random.randint(0, len(locs[0]))
            self.grid[locs[0][choice_idx], locs[1][choice_idx]] = new_vectors[rxn]
            if indicate_count:
                rxn_loc = {'a': 0, 'b': 1, 'c': 2}
                self.reaction_counts[locs[0][choice_idx], locs[1][choice_idx], rxn_loc[rxn]] += 1
        elif rxn == 'b_db_a':  # add an A to surface
            self._add_elt(self.a_value)
        elif rxn == 'b_db_t':  # add a T to surface
            self._add_elt(self.t_value)
        elif rxn == 'b_db_d':  # add a D to surface
            self._add_elt(self.d_value)
        elif rxn == 'db_b_a':  # remove an A from surface
            self._remove_elt(self.a_idx)
        elif rxn == 'db_b_t':  # remove a T from surface
            self._remove_elt(self.t_idx)
        elif rxn == 'db_b_d':  # remove a D from surface
            self._remove_elt(self.d_idx)
        elif rxn == 'move_a':  # always move a A a single step in a random walk
            self._move(self.a_value)
        elif rxn == 'move_t':  # always move a T a single step in a random walk
            self._move(self.t_value)
        elif rxn == 'move_d':  # always move a D a single step in a random walk
            self._move(self.d_value)
        return r_tot

    def gillespie(self, time, anim, seed=None):
        interval = 0
        x_count = []
        times = []
        plots = []
        cmaplist = [(1, 1, 1, 1), (0, 0.8, 0.2, 1), (0, 0, 1, 1), (0.7, 0.2, 0, 1), (1, 0.8, 0, 1)]
        no_d_cmaplist = [(1, 1, 1, 1), (0, 0.8, 0.2, 1), (0, 0, 1, 1)]
        my_cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, len(cmaplist))
        no_d_cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', no_d_cmaplist, len(no_d_cmaplist))
        plt_count = 0
        while self.time < time:
            interval += 1
            if interval % 100000 == 0:
                if interval % 100000 == 0:
                    print(f'{int(round(self.time / time, 2)* 100)}% completed')
                    if seed:
                        pickle.dump(self.grid, open(f'sample_stationarity_analysis_/{seed}_{interval}_{self.phi}_{self.diffusion}_{time}', 'wb'))

                if anim:
                    _, _, d_count = self._counts()
                    anim_grid = _im_frame(self.grid, self.size, self.a_idx, self.t_idx, self.d_idx)
                    cmap = no_d_cmap if d_count == 0 else my_cmap
                    plt.clf()
                    p = plt.imshow(anim_grid, animated=True, cmap=cmap)
                    ax = plt.gca()
                    ax.set_xticks(np.arange(0, 2*self.size))
                    ax.set_yticks(np.arange(0, 2*self.size))

                    ax.set_xticks(np.arange(-.5, 2 * self.size, 2), minor=True)
                    ax.set_yticks(np.arange(-.5, 2 * self.size, 2), minor=True)

                    ax.set_xticklabels([''])
                    ax.set_yticklabels([''])
                    plt.grid(which='minor', color='k', linewidth=1)
                    plt.title(f'$\\varphi$={self.phi}, D={self.diffusion}, grid {plt_count}')
                    # plt.savefig(f'snapshot_images/{self.phi}_{self.diffusion}_{interval}.png')
                    plt.clf()
                    plt.imshow(self.reaction_counts[:, :, 0])
                    plt.colorbar()
                    plt.title(f'Reaction a, grid {plt_count}')
                    # plt.savefig(f'snapshot_images/{self.phi}_{self.diffusion}_{interval}_a.png')
                    plt.clf()
                    plt.imshow(self.reaction_counts[:, :, 1])
                    plt.colorbar()
                    plt.title(f'Reaction b, grid {plt_count}')
                    # plt.savefig(f'snapshot_images/{self.phi}_{self.diffusion}_{interval}_b.png')
                    plt.clf()
                    plt.imshow(self.reaction_counts[:, :, 2])
                    plt.colorbar()
                    plt.title(f'Reaction c, grid {plt_count}')
                    # plt.savefig(f'snapshot_images/{self.phi}_{self.diffusion}_{interval}_c.png')
                    plt_count += 1
                    self.reaction_counts = np.zeros((self.size, self.size, len(self.weights)))
            r_tot = self.reaction(indicate_count=True)
            dt = expon(scale=r_tot).rvs()
            self.time += dt  # Advance time

            a_count, t_count, d_count = self._counts()
            x_count.append(t_count + 2 * d_count)
            times.append(self.time)
        print("a and b avg probs: ", np.mean(self.a_probs), np.mean(self.b_probs))
        return x_count, times, plots
