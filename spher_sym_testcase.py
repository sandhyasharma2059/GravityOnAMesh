'''
Name: spher_sym_testcase.py
Description: This file contains the test case for the spherically symmetric particle distribution.
Author: Greta Goldberg, Jason Li, Sandhya Sharma
Last Modified: December 4, 2023

'''

import gravity 
import numpy as np

# creating a spherically symmetric particle distribution
x,y,z = gravity.spherically_sym_particles((0,0,0), 32**3, 2)
# gravity.plot_particles(x,y,z)

# computing the density field
particles = np.column_stack([x,y,z])
density_field = gravity.compute_density_field(particles, grid_res=32)
# gravity.plot_2d_slice(density_field, 'Density', axis='z', slice_index=None)

# solving the Poisson equation
density_field = gravity.expand_meshgrid(density_field, 64)
g = gravity.green_function(64)
phi = gravity.solve_poisson_green(density_field, g)
phi = phi[:32,:32,:32]
# gravity.plot_2d_slice(phi, 'Potential', axis='z', slice_index=None)

# gravity.plot_potential_vs_radius(phi)

tend = 2 
time_step = 1.0

init_vel = np.zeros((32, 32, 32, 3))

#initial conditions 
positions = particles
velocities = init_vel
density = density_field

for t in range(tend):
    positions, velocities, density = gravity.ver(positions, velocities, density, g, time_step)

print(positions.shape)
print(velocities.shape)
print(positions)
print(velocities)

