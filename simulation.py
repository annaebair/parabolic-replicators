"""
Implementation of parabolic replicators on a 2D grid using the basic idea that elements can share the same grid square.
"""

import numpy as np

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

        self.initialize_grid(f)

    def initialize_grid(self, f):
        """
        Place correct initial ratios of elements onto the grid. Indicate presence of an element as +1. Multiply
        by weights vector to determine how much space is left on each square given the "size" of each element.
        """
        N_max = 4 * self.size ** 2
        T_max = f * N_max
        T_0 = round(T_max / 10)
        A_0 = round((9 * T_max) / 5)

        for i in range(A_0):
            availability = self.grid @ self.weights
            possibilities = np.nonzero(availability <= (self.square_max - self.A_value))
            if len(possibilities[0]) > 0:
                location = np.random.randint(0, len(possibilities[0]))
                index = (possibilities[0][location], possibilities[1][location], self.A_idx)
                self.grid[index] += 1
        for i in range(T_0):
            availability = self.grid @ self.weights
            possibilities = np.nonzero(availability <= (self.square_max - self.T_value))
            if len(possibilities[0]) > 0:
                location = np.random.randint(0, len(possibilities[0]))
                index = (possibilities[0][location], possibilities[1][location], self.T_idx)
                self.grid[index] += 1


def main():
    a_rate = 1e-1
    b_rate = 1e-2
    c_rate = 1e-3
    frac_occupied = 0.3
    grid = Grid(size=10,
                a=a_rate,
                b=b_rate,
                c=c_rate,
                f=frac_occupied)


if __name__ == '__main__':
    main()
