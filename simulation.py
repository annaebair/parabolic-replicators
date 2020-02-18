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
        self.vn_neighbors = {}
        self.val_to_idx = {self.A_value: self.A_idx, self.T_value: self.T_idx, self.D_value: self.D_idx}

        self.tuples = [(int(math.floor(s / size)), s % size) for s in np.arange(size ** 2)]

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
        T_0 = round(N * frac_T * self.T_value)
        A_0 = round(N - (self.T_value / self.A_value) * T_0)
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

    def rxn_choice(self):
        element_locs = {'a': np.nonzero((self.grid[:, :, self.T_idx] == 2)),
                        'b': np.nonzero(self.grid[:, :, self.D_idx] == 1),
                        'c': np.nonzero((self.grid[:, :, self.A_idx] == 2) & (self.grid[:, :, self.T_idx] == 1))
                        }
        rxn_counts = np.array([len(element_locs['a'][0]), len(element_locs['b'][0]),  len(element_locs['c'][0])])
        ratios = [self.a, self.b, self.c] * rxn_counts
        r_tot = np.sum(ratios)
        if r_tot == 0:
            return None, 0
        probs = ratios / r_tot
        rxn = np.random.choice(['a', 'b', 'c'], p=probs)
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
        steps = math.floor(timesteps)
        random_nums = np.random.random(size=round(self.size ** 2 * (1/self.A_value) * steps))
        rand_idx = 0
        occupancy = self.grid @ self.weights
        for s in range(steps):
            random.shuffle(self.tuples)
            for tup in self.tuples:
                x, y = tup
                # vector of counts of each element on the square
                vec = self.grid[tup]
                # objects is a list of values of objects
                objects = [self.A_value] * int(vec[self.A_idx]) + [self.T_value] * int(vec[self.T_idx]) + \
                          [self.D_value] * int(vec[self.D_idx])
                neighbors = self.von_neumann_neighborhood(x, y)
                neighbor_occupancy = np.array([occupancy[n] for n in neighbors])
                # TODO: Is it worth adding something here to stop searching if there are no valid locations for
                #  any remaining objects?
                for obj in objects:
                    options = np.array(neighbors)[np.nonzero(neighbor_occupancy <= 1 - obj)[0]]
                    if len(options) > 0:
                        new_square = random.choice(options)
                        object_idx = self.val_to_idx[obj]
                        # Make objects move probabilistically at rate inversely proportional to size
                        if random_nums[rand_idx] < 1/(4 * obj):
                            self.grid[x, y, object_idx] -= 1
                            self.grid[new_square[0], new_square[1], object_idx] += 1
                            occupancy = self.grid @ self.weights
                            neighbor_occupancy = np.array([occupancy[n] for n in neighbors])
                        rand_idx += 1
                break

    def gillespie(self, time):
        rxns = {'a': 0, 'b': 0, 'c': 0}
        new_vectors = {'a': [0, 0, 1], 'b': [0, 2, 0], 'c': [0, 0, 1]}
        A_count = int(np.sum(self.grid[:, :, self.A_idx]))
        T_count = int(np.sum(self.grid[:, :, self.T_idx]))
        D_count = int(np.sum(self.grid[:, :, self.D_idx]))
        print(f'Starting element counts: A: {A_count}, T: {T_count}, D: {D_count}')
        A_count, T_count, D_count = 0, 0, 0
        interval = 0
        x_count = []
        times = []
        while self.time < time:
            interval += 1
            if interval % 100 == 0:
                print(f'{int(round(self.time / time, 2)* 100)}% completed')

            rxn, r_tot, locs = self.rxn_choice()
            if rxn is None:
                # Does this ever happen? Maybe at low densities? Should this continue with more motion until
                # something is possible or time runs out?
                print(f'No valid reactions at time={self.time}')
                break
            choice_idx = np.random.randint(0, len(locs[0]))
            self.grid[locs[0][choice_idx], locs[1][choice_idx]] = new_vectors[rxn]

            dt = expon(scale=r_tot).rvs() * 100
            self.time += dt
            self.motion(dt)

            A_count = int(np.sum(self.grid[:, :, self.A_idx]))
            T_count = int(np.sum(self.grid[:, :, self.T_idx]))
            D_count = int(np.sum(self.grid[:, :, self.D_idx]))
            rxns[rxn] += 1
            x_count.append(T_count + 2 * D_count)
            times.append(self.time)

        # print(f'Total number of each reaction: {rxns}')
        # print('num of T + 2A squares:', len(np.nonzero((self.grid[:, :, 0] == 2) & (self.grid[:, :, 1] == 1))[0]))
        # print('num of 2T squares:', len(np.nonzero(self.grid[:, :, 1] == 2)[0]))
        return A_count, T_count, D_count, x_count, times


def main():
    for s in range(3):
        print('S:', s)
        random.seed(s)
        np.random.seed(s)
        a_rate = 1e-1
        b_rate = 1e-2
        c_rate = 1e-4
        frac_occupied = 0.3
        grid = Grid(size=10,
                    a=a_rate,
                    b=b_rate,
                    c=c_rate,
                    f=frac_occupied)
        A, T, D, x_counts, times = grid.gillespie(100000)
        with open('counts.txt', 'w+') as f:
            for i in range(len(x_counts)):
                x = x_counts[i]
                t = times[i]
                f.write(f'{t} {x}\n')
        # plt.hist(x_counts, bins=50, alpha=0.5)
        # plt.xlabel('Number of T')
        # plt.ylabel('Timesteps with number of T present')
        # plt.title('Histogram of counts of T for 3 trajectories')
        # plt.show()
        print(f'Final element counts: A: {A}, T: {T}, D: {D}')
        # plt.plot(times, [i for i in x_counts])
        # plt.xlabel('timesteps')
        # plt.ylabel('Number of T elements')
        # plt.title('Trajectories of T counts for 80% occupied')
        # plt.show()
        # TODO: average trajectories?
        plt.plot(times, [i ** 2 for i in x_counts])
        plt.xlabel('timesteps')
        plt.ylabel('Squared number of T elements')
        plt.title('Trajectories of T**2 counts for 80% occupied')
    plt.show()


if __name__ == '__main__':
    main()
