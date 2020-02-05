"""
Implementation of parabolic replicators on a 2D grid using the basic idea that elements can share the same grid square.
"""

import numpy as np
from scipy.stats import expon

np.random.seed(0)


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
        T_max = f * N_max
        T_0 = round(T_max * frac_T)
        A_0 = round(self.T_value/self.A_value * (T_max - T_0))

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

    def rxn_choice(self, element_locs):
        rxn_counts = np.array([len(element_locs['a'][0]), len(element_locs['b'][0]),  len(element_locs['c'][0])])
        ratios = [self.a, self.b, self.c] * rxn_counts
        r_tot = np.sum(ratios)
        if r_tot == 0:
            return None, 0
        probs = ratios / r_tot
        rxn = np.random.choice(['a', 'b', 'c'], p=probs)
        return rxn, r_tot

    def gillespie(self, time):
        rxns = {'a': 0, 'b': 0, 'c': 0}
        A_count, T_count, D_count = 0, 0, 0
        while self.time < time:
            element_locs = {'a': np.nonzero((self.grid[:, :, self.T_idx] == 2)),
                            'b': np.nonzero(self.grid[:, :, self.D_idx] == 1),
                            'c': np.nonzero((self.grid[:, :, self.A_idx] == 2) & (self.grid[:, :, self.T_idx] == 1))
                            }
            new_vectors = {'a': [0, 0, 1],
                           'b': [0, 2, 0],
                           'c': [0, 0, 1]}

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
        print(f'Total number of each reaction: {rxns}')
        return A_count, T_count, D_count


def main():
    a_rate = 1e-1
    b_rate = 1e-2
    c_rate = 1e-3
    frac_occupied = 0.4
    grid = Grid(size=15,
                a=a_rate,
                b=b_rate,
                c=c_rate,
                f=frac_occupied)
    A, T, D = grid.gillespie(100)
    print(f'Final element counts: A: {A}, T: {T}, D: {D}')


if __name__ == '__main__':
    main()
