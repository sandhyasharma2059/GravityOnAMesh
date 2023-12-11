'''
Name: spher_sym_testcase.py
Description: This file contains the test case for the spherically symmetric particle distribution.
Author: Greta Goldberg, Jason Li, Sandhya Sharma
Last Modified: December 4, 2023

'''

import gravity
import numpy as np
import matplotlib.pyplot as plt

# creating a gaussian particle distribution
x , y, z = gravity.gaussian_particles((16,16,16), .2, .3, .4, 32**3)

# computing the density field
particles = np.stack((x,y,z), axis = -1)
density_field = gravity.compute_density_field(particles, grid_res=32)

# solving the Poisson equation
density_field = gravity.expand_meshgrid(density_field, 64)
g = gravity.green_function(64)
phi = gravity.solve_poisson_green(density_field, g, 32)

# verifying plot
gravity.plot_potential_vs_radius(phi)

gravity.plot_2d_slice(density_field[:32,:32,:32],"Density Field",axis="z",slice_index=None)

gravity.plot_2d_slice(phi, "Potential Cross Section", axis='z', slice_index=None)
gravity.plot_2d_slice(phi, "Potential Cross Section", axis='x', slice_index=None)
gravity.plot_2d_slice(phi, "Potential Cross Section", axis='y', slice_index=None)

# initial conditions 
N = len(particles)
positions = particles
vx = np.zeros(shape = (N,))
vy = np.zeros(shape = (N,))
vz = np.zeros(shape = (N,))

density = density_field
tend = 10 
time_step = 0.1

pos_array = [positions]

#integrating in time
for t in range(tend):
    positions, vx, vy, vz, density = gravity.ver(positions, vx, vy, vz, density, g, time_step)
    pos_array.append(positions.copy())

print(positions.shape)
print(vx.shape)

all_position_array = np.array(pos_array)

print(all_position_array.shape)
print(all_position_array)


# ----------- PlOTTING -------------

# Extract x positions of the first particle at each time step
x_positions_first_particle = all_position_array[:, 0, 0]

# Create a time array for x-axis
time_steps = range(1, 11)  # Assuming there are 10 time steps (for N timesteps use range(1,N+1)

# Plotting x position of the first particle over time
plt.plot(time_steps, x_positions_first_particle, marker='o')
plt.title('X Position of First Particle Over Time')
plt.xlabel('Time Step')
plt.ylabel('X Position')
plt.grid(True)
plt.show()

positions = all_position_array

# Get the number of particles
num_particles = positions.shape[0]

# Initialize the figure and axis for plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot trajectories of all particles
for i in range(num_particles-32600):
    x_positions = positions[:, i, 0]
    y_positions = positions[:, i, 1]
    z_positions = positions[:, i, 2]
    ax.plot(x_positions, y_positions, z_positions)

# Set labels and title
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('Trajectories of All Particles Over Time')

# Show the plot
plt.show()
