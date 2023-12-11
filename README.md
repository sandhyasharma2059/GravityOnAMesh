# GravityOnAMesh

## Managing Python Scripts for Gravity in a Mesh 

Functions within gravity.py
1. gaussian_particles: function distribute_particles returns a 32x32x32 array with 3D Gaussian distributions of particles
2. spherical_distribution: generates a spherical distribution of particles in 3D space
3. compute_density_field: function returns the density field of the particles
4. plot_particles: uses particle_distribution to generate 3D Gaussian distribution of particles and plots them in 3D
5. plot_potential_vs_radius: function plots the average potential vs. the radius
6. plot_2d_slice: plots a 2D slice of a 3D quantity (example: density field, potential field, etc.)
7. density_field: uses particle_distribution to generate 3D Gaussian distribution of particles and returns a 32x32x32 array of density values 
8. solve_poisson_fft: takes a 3D array of density field and uses it to solve the poisson equation by using FFT and return the potential field
9. solve_poisson_green: function returns the potential of the density field by solving the Poisson equation using the Green's function
10. green_function: returns the Green's function in a 3D grid, symmetric across the eight octants
11. ver: integrates the PDE over time according to the verlet method
12. get_acceleration: (the non-commented out version) uses interpolation to assign acclerations to each particle
13. expand_meshgrid: function returns the potential function in a 3D grid, shifted to the correct spot to convolve with your Green's function
14. plot_trajectory: plots the trajectory of a single particle given its positions at different times

Tests
1. spher_sym_testcase: generates potential field for a spherically symmetric particle distribution 
2. varying_gaussian_test: generates potential field for varying a,b,c values in a Gaussian distribution of particles
3. delta_test_case : generates potential field for a delta distribution
4. gaussian_test_case: generates potential field for a gaussian distribution
