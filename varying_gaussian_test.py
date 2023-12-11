from gravity import particle_distribution
#from gravity import plot_particles
from gravity import representation2D
from gravity import density_field
from gravity import solve_poisson_equation
from gravity import numpy as np

# x1,y1,z1 = particle_distribution.gaussian_particles((0,0,0), 2, 4, 5, 32**3)
# plot_particles.plot_particles(x1,y1,z1)
# particles = np.column_stack([x1,y1,z1])
# density_field1 = density_field.compute_density_field(particles, grid_res=32)
# representation2D.plot_2d_slice(density_field1, 'Density', axis='z', slice_index=None)
# potential_field = solve_poisson_equation.solve_poisson_fft(density_field1)
# representation2D.plot_2d_slice(potential_field, 'Potential', axis='z', slice_index=None)

x2,y2,z2 = particle_distribution.gaussian_particles((0,0,0), 0.001, 0.2, 0.4, 32**3)
# plot_particles.plot_particles(x2,y2,z2)
particles = np.column_stack([x2,y2,z2])
density_field2 = density_field.compute_density_field(particles, grid_res=32)
# representation2D.plot_2d_slice(density_field2, 'Density', axis='z', slice_index=None)
potential_field = solve_poisson_equation.solve_poisson_fft(density_field2)
representation2D.plot_2d_slice(potential_field, 'Potential', axis='z', slice_index=None)