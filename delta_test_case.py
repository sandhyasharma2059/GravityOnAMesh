'''
Name: delta_test_case.py
Description: This file contains the test case for the delta source.
Author: Greta Goldberg, Jason Li, Sandhya Sharma
Last Modified: December 10, 2023

'''

import gravity 
import numpy as np
import matplotlib.pyplot as plt

#using old spherical distribution just to put one particle on the grid
def spherically_sym_particles(center, num_of_particles, r):
    '''
    The function returns a spherically symmetric distribution of particles of a given number. 

    Parameters:
        - center: center of the distribution
        - num_of_particles: number of particles in the distribution 

    Returns:
        - 3 1D arrays of length num_particles (x, y, z)
          (A spherically symmetric distribution of particles of a given number)
    '''

    # r = np.random.uniform(0, 1, num_of_particles)
    theta = np.random.uniform(0, np.pi, num_of_particles)
    phi = np.random.uniform(0, 2*np.pi, num_of_particles)

    x = center[0] + r * np.sin(theta) * np.cos(phi)
    y = center[1] + r * np.sin(theta) * np.sin(phi)
    z = center[2] + r * np.cos(theta)

    return x, y, z

# creating a spherically symmetric particle distribution
x, y, z = spherically_sym_particles((16,16,16), 1, 0)

gravity.plot_particles(x,y,z)

# computing the density field
particles = np.stack((x,y,z), axis = -1)
density_field = gravity.compute_density_field(particles, grid_res=32)

gravity.plot_2d_slice(density_field,"Density Field",axis="z",slice_index=None)

# solving the Poisson equation
density_field = gravity.expand_meshgrid(density_field, 64)
g = gravity.green_function(64)
phi = gravity.solve_poisson_green(density_field, g, 32)

# verifying plot
gravity.plot_potential_vs_radius(phi)

gravity.plot_2d_slice(phi, "Potential Cross Section", axis='z', slice_index=None)
gravity.plot_2d_slice(phi, "Potential Cross Section", axis='x', slice_index=None)
gravity.plot_2d_slice(phi, "Potential Cross Section", axis='y', slice_index=None)
