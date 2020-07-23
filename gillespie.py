"""
Implementation of parabolic replicators on a 2D grid using the basic idea that elements can share the same grid square.
"""

import math
import random
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import expon


class Grid:
    def __init__(self, size, a, b, c, d, e, fraction_occupied, fraction_x, phi, diffusion, w):
        self.size = size

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.phi = phi
        self.diffusion = diffusion
        self.w = w  # Concentration of A elements

        self.t_idx = 0 
        self.d_idx = 1 

        self.grid = np.zeros((self.size, self.size, 2))
        self.adj = np.zeros((self.size**2, self.size**2))
        self.reaction_counts = np.zeros((self.size, self.size, 2))
        self.val_to_idx = {'t': self.t_idx, 'd': self.d_idx}
        self.vn_neighbors = {}
        self.time = 0

        self.relationship = {(0, 0): 2,
                            (0, 1): 3, (1, 0): 3,
                            (0, 2): 4, (2, 0): 4,
                            (1, 1): 5,
                            (1, 2): 6, (2, 1): 6,
                            (2, 2): 7
                            }

        self.q = self.initialize_grid(fraction_occupied, fraction_x)
        self.initialize_adjacency_matrix()
        self.a_count = 0
        self.b_count = 0
        self.c_count = 0
        self.d_count = 0
        self.e_count = 0
        self.m_count = 0

    def initialize_grid(self, fraction_occupied, fraction_x):
        """
        Place correct initial ratios of elements onto the grid. Indicate presence of an element as +1. Multiply
        by weights vector to determine how much space is left on each square given the size of each element. If
        too many elements are given than can fit, silently doesn't add them. 
        """
        grid_space = self.size ** 2
        occupied_grid_mass = round(fraction_occupied * grid_space)  
        num_x = round(fraction_x * occupied_grid_mass)
        
        # TODO: decide whether we want to allocate T and D like this, or in another way, or just start with all T   
        # Subdivide x_mass using X = 2D + T, (a/b)T**2 = D, and quadratic formula, where T and D are count variables
        num_t = round((self.b / (4 * self.a)) * (math.sqrt(1 + (8 * self.a * num_x) / self.b) - 1))
        num_d = round(num_x - num_t)

        for num, idx in [(num_d, self.d_idx), (num_t, self.t_idx)]:
            for i in range(num):
                occupancy = self._occupancy() # Returns a binary grid indicating where any elements are
                possibilities = np.nonzero(occupancy==0)
                if len(possibilities[0]) > 0:
                    location = random.randint(0, len(possibilities[0]))
                    index = (possibilities[0][location], possibilities[1][location], idx)
                    self.grid[index] += 1
        return (num_t + num_d) / occupied_grid_mass

    def initialize_adjacency_matrix(self):
        g = self.grid @ [1, 2]
        flat = g.flatten()
        for idx in range(self.size**2):
            i_type = int(flat[idx])
            x, y = int(math.floor(idx / self.size)), int(idx % self.size)
            n = self.neighbors(x, y)
            for nn in n:
                nx, ny = nn
                n_type = int(g[nn])
                jdx = nx * self.size + ny
                link = self.relationship[(i_type, n_type)]
                self.adj[idx, jdx] = link

    def neighbors(self, x, y):
        """
        von Neumann neighbors
        """
        options = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        return [(i % self.size, j % self.size) for (i, j) in options]

    def _counts(self):
        t = int(np.sum(self.grid[:, :, self.t_idx]))
        d = int(np.sum(self.grid[:, :, self.d_idx]))
        return t, d

    def _occupancy(self):
        return self.grid @ [1, 1] # "size" of each element is 1

    def reaction(self):
        """
        Select a reaction based on predefined rates and current state of the grid. Carry out the chosen reaction.
        Returns r_tot.
        Uses types: 0 = empty, 1 = T, 2 = D as referenced in self.relationship and in self.grid @ [1, 2]
        """
        num_t, num_d = self._counts()
        rxn_options = ['a', 'b', 'c', 'd', 'e', 'm']
        rxn_counts = np.array([np.count_nonzero(self.adj == 5), np.count_nonzero(self.adj == 4), num_t * self.w ** 2,
                     num_t, np.count_nonzero(self._occupancy() == 0), np.count_nonzero(self.adj == 3)])
        rxn_rates = np.array([self.a, self.b, self.c, self.d, self.e, self.diffusion])
        ratios = rxn_counts * rxn_rates
        r_tot = np.sum(ratios)
        probs = ratios / r_tot
        rxn = np.random.choice(rxn_options, p=probs)

        def multi_element(x1, y1, x2, y2, new_pt_1_type, new_pt_2_type):
            pt_1_neighbors = self.neighbors(x1, y1)
            pt_2_neighbors = self.neighbors(x2, y2)
            for n in pt_1_neighbors:
                n_type = (self.grid @ [1, 2])[n]
                new_relationship = self.relationship[(n_type, new_pt_1_type)]
                self.adj[x1 * self.size + y1, n[0] * self.size + n[1]] = new_relationship
                self.adj[n[0] * self.size + n[1], x1 * self.size + y1] = new_relationship
            for n in pt_2_neighbors:
                n_type = (self.grid @ [1, 2])[n]
                new_relationship = self.relationship[(n_type, new_pt_2_type)]
                self.adj[x2 * self.size + y2, n[0] * self.size + n[1]] = new_relationship
                self.adj[n[0] * self.size + n[1], x2 * self.size + y2] = new_relationship

        def single_element(x, y, new_type):
            pt_neighbors = self.neighbors(x, y)
            for n in pt_neighbors:
                n_type = (self.grid @ [1, 2])[n]
                new_relationship = self.relationship[(n_type, new_pt_type)]
                self.adj[x * self.size + y, n[0] * self.size + n[1]] = new_relationship
                self.adj[n[0] * self.size + n[1], x * self.size + y] = new_relationship

        if rxn == 'a':
            # TODO: rename these to like pt_1_options instead of using x and y to reference adj
            x_adj_locs, y_adj_locs = np.nonzero(self.adj == 5) # x values and y values for where self.adj == 5
            loc = random.randint(0, len(x_adj_locs)-1) # pick one location at random
            pt_1 = x_adj_locs[loc] # first elt (x in adj)
            pt_2 = y_adj_locs[loc] # second elt (y in adj)
            x1, y1 = int(math.floor(pt_1 / self.size)), pt_1 % self.size # (x, y) in grid
            x2, y2 = int(math.floor(pt_2 / self.size)), pt_2 % self.size # (x, y) in grid
            assert self.grid[x1, y1, 0] == 1
            assert self.grid[x1, y1, 1] == 0
            assert self.grid[x2, y2, 0] == 1
            assert self.grid[x2, y2, 1] == 0
            self.grid[x1, y1, 0] = 0 # Remove a T
            self.grid[x1, y1, 1] = 1 # Add a D
            self.grid[x2, y2, 0] = 0 # Remove a T (leaves behind an empty spot)
            new_pt_1_type = 2 # becomes TT
            new_pt_2_type = 0 # becomes empty 
            multi_element(x1, y1, x2, y2, new_pt_1_type, new_pt_2_type)
            self.a_count += 1

        if rxn == 'b':
            x_adj_locs, y_adj_locs = np.nonzero(self.adj == 4)
            loc = random.randint(0, len(x_adj_locs)-1)
            pt_1 = x_adj_locs[loc]
            pt_2 = y_adj_locs[loc]
            x1, y1 = int(math.floor(pt_1 / self.size)), pt_1 % self.size # (x, y) in grid
            x2, y2 = int(math.floor(pt_2 / self.size)), pt_2 % self.size # (x, y) in grid
            self.grid[x1, y1, 0] = 1
            self.grid[x1, y1, 1] = 0
            self.grid[x2, y2, 0] = 1
            self.grid[x2, y2, 1] = 0
            new_pt_1_type = 1 # becomes T
            new_pt_2_type = 1 # becomes T 
            multi_element(x1, y1, x2, y2, new_pt_1_type, new_pt_2_type)
            self.b_count += 1

        if rxn == 'm':
            x_adj_locs, y_adj_locs = np.nonzero(self.adj == 3)
            loc = random.randint(0, len(x_adj_locs)-1)
            pt_1 = x_adj_locs[loc]
            pt_2 = y_adj_locs[loc]
            x1, y1 = int(math.floor(pt_1 / self.size)), pt_1 % self.size 
            x2, y2 = int(math.floor(pt_2 / self.size)), pt_2 % self.size 
            if self.grid[x1, y1, 0] == 1:
                assert self.grid[x2, y2, 0] == 0
                assert self.grid[x2, y2, 1] == 0
                self.grid[x1, y1, 0] = 0
                self.grid[x2, y2, 0] = 1
                new_pt_1_type = 0 # becomes open 
                new_pt_2_type = 1 # becomes T 
            elif self.grid[x2, y2, 0] == 1:
                assert self.grid[x1, y1, 0] == 0
                assert self.grid[x1, y1, 1] == 0
                self.grid[x2, y2, 0] = 0
                self.grid[x1, y1, 0] = 1
                new_pt_1_type = 1 # becomes T
                new_pt_2_type = 0 # becomes open 
            multi_element(x1, y1, x2, y2, new_pt_1_type, new_pt_2_type)
            self.m_count += 1

        if rxn == 'c':
            x_locs, y_locs = np.nonzero(self.grid[:, :, 0]) # Any T's i.e. grid[:, :, 0] is nonzero
            loc = random.randint(0, len(x_locs)-1)
            x, y = x_locs[loc], y_locs[loc]
            assert self.grid[x, y, 1] == 0
            self.grid[x, y, 0] = 0
            self.grid[x, y, 1] = 1
            new_pt_type = 2 # TT
            single_element(x, y, new_pt_type)
            self.c_count += 1

        if rxn == 'd':
            x_locs, y_locs = np.nonzero(self.grid[:, :, 0]) # Any T's i.e. grid[:, :, 0] is nonzero
            loc = random.randint(0, len(x_locs)-1)
            x, y = x_locs[loc], y_locs[loc]
            self.grid[x, y, 0] = 0
            self.grid[x, y, 1] = 0
            new_pt_type = 0 # empty 
            single_element(x, y, new_pt_type)
            self.d_count += 1

        if rxn == 'e':
            x_locs, y_locs = np.nonzero((self.grid @ [1, 1]) == 0) # empty spaces 
            loc = random.randint(0, len(x_locs)-1)
            x, y = x_locs[loc], y_locs[loc]
            self.grid[x, y, 0] = 1
            new_pt_type = 1 # T
            single_element(x, y, new_pt_type)
            self.e_count += 1

        return r_tot


    def gillespie(self, time, seed=None):
        interval = 0
        x_count = []
        times = []
        plots = []
        plt_count = 0
        while self.time < time:
            interval += 1
            if interval % 10000 == 0:
                print(f'{int(round(self.time / time, 2)* 100)}% completed')
            if interval % 100000 == 0:
                if seed:
                    pickle.dump(self.grid, open(f'stationarity_analysis/{seed}_{interval}_{self.phi}_{self.diffusion}_{time}', 'wb'))
            r_tot = self.reaction()
            dt = expon(scale=r_tot).rvs()
            self.time += dt 
            t_count, d_count = self._counts()
            x_count.append(t_count + 2 * d_count)
            times.append(self.time)
        print(self.a_count, self.b_count, self.c_count, self.d_count, self.e_count, self.m_count)
        return x_count, times, plots

