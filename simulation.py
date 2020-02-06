"""
Implementation of parabolic replicators on a 2D grid using the basic idea that elements can share the same grid square.
"""

import math
import random
import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt


class Grid:
    def __init__(self, size, a, b, c, f):
        self.size = size
        self.num_types = 3
        self.A_idx = 0
        self.T_idx = 1
        self.D_idx = 2
        self.A_value = 0.25
        self.T_value = 0.5
        self.D_value = 1
        self.square_max = 1
        self.weights = [self.A_value, self.T_value, self.D_value]
        self.grid = np.zeros((self.size, self.size, self.num_types))

        self.a = a
        self.b = b
        self.c = c

        self.time = 0
        self.initialize_grid(f)

    def initialize_grid(self, f):
        """
        Place correct initial ratios of elements onto the grid. Indicate presence of an element as +1. Multiply
        by weights vector to determine how much space is left on each square given the "size" of each element.
        """
        frac_T = 0.1
        N_max = (1/min(self.weights)) * self.size ** 2
        N = round(f * N_max)
        T_0 = round(N * frac_T * 0.5)
        A_0 = N - 2 * T_0
        print(T_0, A_0, N)

        for i in range(T_0):
            availability = self.grid @ self.weights
            possibilities = np.nonzero(availability <= (self.square_max - self.T_value))
            if len(possibilities[0]) > 0:
                location = np.random.randint(0, len(possibilities[0]))
                index = (possibilities[0][location], possibilities[1][location], self.T_idx)
                self.grid[index] += 1
        for i in range(A_0):
            availability = self.grid @ self.weights
            possibilities = np.nonzero(availability <= (self.square_max - self.A_value))
            if len(possibilities[0]) > 0:
                location = np.random.randint(0, len(possibilities[0]))
                index = (possibilities[0][location], possibilities[1][location], self.A_idx)
                self.grid[index] += 1
            else: print('here')

        A_count = int(np.sum(self.grid[:, :, self.A_idx]))
        T_count = int(np.sum(self.grid[:, :, self.T_idx]))

    def rxn_choice(self, element_locs):
        rxn_counts = np.array([len(element_locs['a'][0]), len(element_locs['b'][0]),  len(element_locs['c'][0])])
        ratios = [self.a, self.b, self.c] * rxn_counts
        r_tot = np.sum(ratios)
        if r_tot == 0:
            return None, 0
        probs = ratios / r_tot
        rxn = np.random.choice(['a', 'b', 'c'], p=probs)
        return rxn, r_tot

    def von_neumann_neighborhood(self, x, y):
        options = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        return [(i % self.size, j % self.size) for (i, j) in options]

    def motion(self):
        # TODO: should larger elements move slower?
        val_to_idx = {self.A_value: self.A_idx, self.T_value: self.T_idx, self.D_value: self.D_idx}
        grid_squares = np.arange(self.size ** 2)
        tuples = [(int(math.floor(s / 10)), s % 10) for s in grid_squares]
        random.shuffle(tuples)
        for tup in tuples:
            x, y = tup
            vec = self.grid[tup]
            objects = [self.A_value] * int(vec[self.A_idx]) + [self.T_value] * int(vec[self.T_idx]) + [self.D_value] * int(vec[self.D_idx])
            neighbors = self.von_neumann_neighborhood(x, y)
            for obj in objects:
                options = []
                occupancy = self.grid @ self.weights
                for n in neighbors:
                    n_x, n_y = n
                    if occupancy[n_x, n_y] + obj <= 1:
                        options.append(n)
                if len(options) > 0:
                    new_square = random.choice(options)
                    object_idx = val_to_idx[obj]
                    self.grid[x, y, object_idx] -= 1
                    self.grid[new_square[0], new_square[1], object_idx] += 1

    def gillespie(self, time):
        rxns = {'a': 0, 'b': 0, 'c': 0}
        A_count = int(np.sum(self.grid[:, :, self.A_idx]))
        T_count = int(np.sum(self.grid[:, :, self.T_idx]))
        D_count = int(np.sum(self.grid[:, :, self.D_idx]))
        print(f'Starting element counts: A: {A_count}, T: {T_count}, D: {D_count}')
        A_count, T_count, D_count = 0, 0, 0
        interval = 0
        # checkpoints = [1000, 5000, 7500, 10000, 20000, 30000, 40000]
        # pointer = 0
        x_count = []
        times = []
        while self.time < time:
            interval += 1
            # if interval % 1000 == 0:
            #     print(f'{round(self.time / time, 2)* 100}% completed')
            #     print(f'T + 2A squares: {len(np.nonzero((self.grid[:, :, 0] == 2) & (self.grid[:, :, 1] == 1))[0])}, '
            #           f'2T squares: {len(np.nonzero(self.grid[:, :, 1] == 2)[0])}, '
            #           f'D squares: {len(np.nonzero(self.grid[:, :, 2] == 1)[0])}')
            #     A_count = int(np.sum(self.grid[:, :, self.A_idx]))
            #     T_count = int(np.sum(self.grid[:, :, self.T_idx]))
            #     D_count = int(np.sum(self.grid[:, :, self.D_idx]))
            #     print(f'num A: {A_count}, num T: {T_count} num D: {D_count}')
            element_locs = {'a': np.nonzero((self.grid[:, :, self.T_idx] == 2)),
                            'b': np.nonzero(self.grid[:, :, self.D_idx] == 1),
                            'c': np.nonzero((self.grid[:, :, self.A_idx] == 2) & (self.grid[:, :, self.T_idx] == 1))
                            }
            new_vectors = {'a': [0, 0, 1], 'b': [0, 2, 0], 'c': [0, 0, 1]}

            self.motion()
            rxn, r_tot = self.rxn_choice(element_locs)
            if rxn is None:
                # TODO: Modify this behavior once diffusion is added
                print(f'No valid reactions at time={self.time}')
                break
            locs = element_locs[rxn]
            choice_idx = np.random.randint(0, len(locs[0]))
            self.grid[locs[0][choice_idx], locs[1][choice_idx]] = new_vectors[rxn]

            dt = expon(scale=r_tot).rvs()
            self.time += dt

            A_count = int(np.sum(self.grid[:, :, self.A_idx]))
            T_count = int(np.sum(self.grid[:, :, self.T_idx]))
            D_count = int(np.sum(self.grid[:, :, self.D_idx]))
            rxns[rxn] += 1
            x_count.append(T_count + 2 * D_count)
            times.append(self.time)
            # if pointer < len(checkpoints):
            #     if self.time > checkpoints[pointer]:
            #         print(checkpoints[pointer], ': ', 'A:', A_count, 'T:', T_count, 'D:', D_count)
            #         pointer += 1

        print(f'Total number of each reaction: {rxns}')
        print('num of T + 2A squares:', len(np.nonzero((self.grid[:, :, 0] == 2) & (self.grid[:, :, 1] == 1))[0]))
        print('num of 2T squares:', len(np.nonzero(self.grid[:, :, 1] == 2)[0]))
        return A_count, T_count, D_count, x_count, times


def main():
    for s in range(5):
        print('S:', s)
        random.seed(s)
        np.random.seed(s)
        a_rate = 1e-1
        b_rate = 1e-2
        c_rate = 1e-4
        frac_occupied = 0.8
        grid = Grid(size=10,
                    a=a_rate,
                    b=b_rate,
                    c=c_rate,
                    f=frac_occupied)
        A, T, D, x_counts, times = grid.gillespie(5000)
        print(f'Final element counts: A: {A}, T: {T}, D: {D}')
        plt.plot(times, np.sqrt(x_counts))
        plt.xlabel('timesteps')
        # plt.ylim((0, max(x_counts)+ 10))
        plt.ylabel('Number of T elements')
        plt.title('Trajectories for 30% occupied')
    plt.show()


if __name__ == '__main__':
    main()
