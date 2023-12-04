'''
Name: spher_sym_testcase.py
Description: This file contains the test case for the spherically symmetric particle distribution.
Author: Greta Goldberg, Jason Li, Sandhya Sharma
Last Modified: December 3, 2023

'''

import particle_distribution
import plot_particles
import representation2D
import solve_poisson_equation
import density_field
import numpy as np

# creating a spherically symmetric particle distribution
x,y,z = particle_distribution.spherically_sym_particles((0,0,0), 32**3, 3)
plot_particles.plot_particles(x,y,z)

# computing the density field
particles = np.column_stack([x,y,z])
density_field = density_field.compute_density_field(particles, grid_res=32)
representation2D.plot_2d_slice(density_field, 'Density', axis='z', slice_index=None)

# solving the Poisson equation
potential_field = solve_poisson_equation.solve_poisson_fft(density_field)
representation2D.plot_2d_slice(potential_field, 'Potential', axis='z', slice_index=None)


