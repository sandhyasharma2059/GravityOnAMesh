'''
Name: gaussian_testcases.py
Description: This file contains the test case for Gaussian distribution of particles.
Author: Greta Goldberg, Jason Li, Sandhya Sharma
Last Modified: December 4, 2023

'''

import particle_distribution
import plot_particles
import representation2D
import avgpotential_vs_radius

import density_field
import greens_function

import numpy as np

# creating a spherically symmetric particle distribution
x,y,z = particle_distribution.gaussian_particles((0,0,0), 0.2, 0.3, 0.4,  32**3)
plot_particles.plot_particles(x,y,z)

# computing the density field
particles = np.column_stack([x,y,z])
density_field = density_field.compute_density_field(particles, grid_res=32)
representation2D.plot_2d_slice(density_field, 'Density', axis='z', slice_index=None)

# solving the Poisson equation
density_field = greens_function.expand_meshgrid(density_field, 64)
g = greens_function.green_function(64)
phi = greens_function.solve_poisson_green(density_field, g)
representation2D.plot_2d_slice(phi[:32,:32,:32], 'Potential', axis='z', slice_index=None)

avgpotential_vs_radius.plot_potential_vs_radius(phi[:32,:32,:32])



