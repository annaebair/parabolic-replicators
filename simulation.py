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
import matplotlib.ticker
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'


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
        self.N, self.T_0, self.A_0 = self.initialize_grid(f)

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
        return N, T_0, A_0

    def rxn_choice(self, r):
        occupancy = self.grid @ self.weights
        element_locs = {'a': np.nonzero(self.grid[:, :, self.T_idx] == 2),
                        'b': np.nonzero(self.grid[:, :, self.D_idx] == 1),
                        'c': np.nonzero((self.grid[:, :, self.A_idx] == 2) & (self.grid[:, :, self.T_idx] == 1)),
                        'exchange': (np.nonzero(occupancy == 0), np.nonzero(occupancy == 1))
                        }
        # put some amt of stuff on an empty square, take all the stuff off a full square - maybe better ways to do this
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
                # Seems to break here for density ~< 0.3
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

    def gillespie(self, time, r):
        new_vectors = {'a': [0, 0, 1], 'b': [0, 2, 0], 'c': [0, 0, 1]}
        A_count = int(np.sum(self.grid[:, :, self.A_idx]))
        T_count = int(np.sum(self.grid[:, :, self.T_idx]))
        D_count = int(np.sum(self.grid[:, :, self.D_idx]))
        print(f'Starting element counts: A: {A_count}, T: {T_count}, D: {D_count}')
        A_count, T_count, D_count = 0, 0, 0
        interval = 0
        x_count = []
        times = []
        plots = []

        cmaplist = [(1, 1, 1, 1), (0, 0.8, 0.2, 1), (0, 0, 1, 1), (0.7, 0.2, 0, 1), (0.8, 0, 0.1, 1)]
        no_d_cmaplist = [(1, 1, 1, 1), (0, 0.8, 0.2, 1), (0, 0, 1, 1)]
        N = len(cmaplist)
        no_d_N = len(no_d_cmaplist)
        my_cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, N)
        no_d_cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', no_d_cmaplist, no_d_N)

        while self.time < time:
            interval += 1
            if interval % 100 == 0:
                print(f'{int(round(self.time / time, 2)* 100)}% completed')
            if interval % 100 == 0:

                A_count = int(np.sum(self.grid[:, :, self.A_idx]))
                T_count = int(np.sum(self.grid[:, :, self.T_idx]))
                D_count = int(np.sum(self.grid[:, :, self.D_idx]))
                if D_count == 0:
                    p = plt.imshow(im_frame(self.grid), animated=True, cmap=no_d_cmap)
                    plt.grid(True, animated=True)
                    plt.axis('off')
                    plots.append([p])
                else:
                    p1 = plt.imshow(im_frame(self.grid), animated=True, cmap=my_cmap)
                    plt.grid(True, animated=True)
                    plt.axis('off')
                    plots.append([p1])
            rxn, r_tot, locs = self.rxn_choice(r)
            if rxn is None:
                # Does this ever happen? Maybe at low densities? Should this continue with more motion until
                # something is possible or time runs out?
                print(f'No valid reactions at time={self.time}')
                break
            elif rxn in {'a', 'b', 'c'}:
                choice_idx = np.random.randint(0, len(locs[0]))
                self.grid[locs[0][choice_idx], locs[1][choice_idx]] = new_vectors[rxn]
            else:  # rxn is exchange:
                available, full = locs
                # check if neither are len zero?
                idx_to_place = np.random.randint(0, len(available[0]))
                idx_to_remove = np.random.randint(0, len(full[0]))
                # D, TT, AAAA, TTA
                vecs_to_add = [[0, 0, 1], [0, 2, 0], [4, 0, 0], [2, 1, 0]]
                # ratios of numbers of elements such that volume adds up to 1: 0.01, 0.02, 0.9
                # should we account for size? i.e. T_bulk = 0.02, D_bulk = 0.08? This is for ratios of what elements
                #  to select
                # TODO: Also these are hardcoded so fix this eventually
                T_bulk = 0.02
                D_bulk = 0.08
                A_bulk = 0.9
                probs = np.array([D_bulk,
                                  T_bulk ** 2,
                                  A_bulk ** 4,
                                  T_bulk * A_bulk ** 2])
                normalized_probs = probs / sum(probs)
                vec = vecs_to_add[np.random.choice([0, 1, 2, 3], p=normalized_probs)]
                self.grid[available[0][idx_to_place], available[1][idx_to_place], :] += vec
                self.grid[full[0][idx_to_remove], full[1][idx_to_remove], :] = 0

            dt = expon(scale=r_tot).rvs() * 100
            self.time += dt
            self.motion(dt)

            A_count = int(np.sum(self.grid[:, :, self.A_idx]))
            T_count = int(np.sum(self.grid[:, :, self.T_idx]))
            D_count = int(np.sum(self.grid[:, :, self.D_idx]))
            x_count.append(T_count + 2 * D_count)
            times.append(self.time)
        return A_count, T_count, D_count, x_count, times, plots


def im_frame(grid):
    a_grid = grid[:, :, 0]
    t_grid = grid[:, :, 1]
    d_grid = grid[:, :, 2]
    new_grid = np.zeros((20, 20))
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


def main():
    fig = plt.figure()
    iterations = 1
    for s in range(iterations):
        print('S:', s)
        random.seed(s)
        np.random.seed(s)
        a_rate = 1e-1
        b_rate = 1e-2
        c_rate = 1e-4
        frac_occupied = 0.8
        chemostat_rate = 0.001
        grid = Grid(size=10,
                    a=a_rate,
                    b=b_rate,
                    c=c_rate,
                    f=frac_occupied)

        A, T, D, x_counts, times, plots = grid.gillespie(5000, chemostat_rate)

        # with open(f'counts_chemostat_million_steps_{s}.txt', 'w+') as f:
        #     for i in range(len(x_counts)):
        #         x = x_counts[i]
        #         t = times[i]
        #         f.write(f'{t} {x}\n')
        # plt.hist(x_counts, bins=50, alpha=0.5)
        # plt.xlabel('Number of T')
        # plt.ylabel('Timesteps with number of T present')
        # plt.title('Histogram of counts of T for 3 trajectories')
        # plt.show()
        print(f'Final element counts: A: {A}, T: {T}, D: {D}')
        plt.plot(times, x_counts)
        plt.xlabel('timesteps')
        plt.ylabel('Number of T elements')
        plt.title('Trajectories of T counts for 80% occupied')
        plt.show()
        # plt.plot(times, [i ** 2 for i in x_counts])
        # plt.xlabel('timesteps')
        # plt.ylabel('Squared number of T elements')
        # plt.title('Trajectories of T**2 counts for 80% occupied')
    # plt.show()

    # ani = animation.ArtistAnimation(fig, plots, interval=500, blit=True,
    #                                 repeat_delay=2000)
    # ani.save('thing.html')
    # plt.show()


if __name__ == '__main__':
    main()
