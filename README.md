# GravityOnAMesh

## Managing Python Scripts for Gravity in a Mesh 


1. particle_distribution: function distribute_particles returns a 32x32x32 array with 3D Gaussian and spherical distributions of particles 
2. plot_particles: uses particle_distribution to generate 3D Gaussian distribution of particles and plots them in a 3D plot
3. density_field: uses particle_distribution to generate 3D Gaussian distribution of particles and returns a 32x32x32 array of density values 
4. 2D_representation: takes a 3D quantity (example: density field in a 3D Cartesian coordinate) and returns its 2D representation (a slice) 
5. solve_poisson_equation: takes a 3D array of density field and uses it to solve the poisson equation by using FFT and return the potential field 
6. delta_source: defines a delta source, uses its distribution to solve poisson’s equation and plots the 2D representation of the (periodic) potential field
7. greens_function: 
8. spher_sym_testcase: generates potential field for a spherically symmetric particle distribution 
9. varying_gaussian_test: generates potential field for varying a,b,c values in a Gaussian distribution of particles 
