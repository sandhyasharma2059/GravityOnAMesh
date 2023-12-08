'''
Name: spher_sym_testcase.py
Description: This file contains the test case for the spherically symmetric particle distribution.
Author: Greta Goldberg, Jason Li, Sandhya Sharma
Last Modified: December 4, 2023

'''

import gravity 
import numpy as np
import matplotlib.pyplot as plt

# creating a spherically symmetric particle distribution
x,y,z = gravity.spherically_sym_particles((0,0,0), 32**3, 2)

# computing the density field
# particles = np.column_stack([x,y,z])
particles = np.stack((x,y,z), axis = -1)
print(particles.shape)
density_field = gravity.compute_density_field(particles, grid_res=32)
print(density_field.shape)

# solving the Poisson equation
density_field = gravity.expand_meshgrid(density_field, 64)
g = gravity.green_function(64)
phi = gravity.solve_poisson_green(density_field, g)
phi = phi[:32,:32,:32]

#initial conditions 
positions = particles
vx = np.zeros(shape =(32768,))
vy = np.zeros(shape =(32768,))
vz = np.zeros(shape =(32768,))

density = density_field
tend = 2 
time_step = 0.1

pos_array = []

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
time_steps = range(1, 3)  # Assuming there are 5 time steps

# Plotting x position of the first particle over time
plt.plot(time_steps, x_positions_first_particle, marker='o')
plt.title('X Position of First Particle Over Time')
plt.xlabel('Time Step')
plt.ylabel('X Position')
plt.grid(True)
plt.show()

positions = all_position_array

# Get the number of particles
num_particles = positions.shape[1]

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