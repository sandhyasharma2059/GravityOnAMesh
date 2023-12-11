'''
Name: spher_sym_testcase.py
Description: This file contains the test case for the spherically symmetric particle distribution.
Author: Greta Goldberg, Jason Li, Sandhya Sharma
Last Modified: December 10, 2023

'''

import gravity 
import numpy as np
import matplotlib.pyplot as plt

# creating a spherically symmetric particle distribution
x , y, z = gravity.spherical_distribution((16,16,16), 32**3, 16)

# computing the density field
particles = np.stack((x,y,z), axis = -1)
density_field = gravity.compute_density_field(particles, grid_res=32)

# solving the Poisson equation
density_field = gravity.expand_meshgrid(density_field, 64)
g = gravity.green_function(64)
phi = gravity.solve_poisson_green(density_field, g, 32)

gravity.plot_2d_slice(phi, "Potential Cross Section", axis='z', slice_index=None)
gravity.plot_2d_slice(phi, "Potential Cross Section", axis='x', slice_index=None)
gravity.plot_2d_slice(phi, "Potential Cross Section", axis='y', slice_index=None)

# verifying plot
gravity.plot_potential_vs_radius(phi, 'Spherical', 16)
