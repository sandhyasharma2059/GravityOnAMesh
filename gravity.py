'''
Name: gravity.py
Description: This file contains all the functions required for the gravity simulation.
Author: Greta Goldberg, Jason Li, Sandhya Sharma
Last Modified: December 4, 2023

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn

def gaussian_particles(center, a, ba, ca, num_particles=32**3):
    '''
        The function returns a 3D multivariate Gaussian distribution of particles of a given number. 

        Parameters:
            - center: center of the distribution
            - a: semi-major axis
            - ba: axis ratio b/a
            - ca: axis ratio c/a
            - num_particles: number of particles in the distribution 

        Returns:
            - 3 1D arrays of length num_particles (x, y, z)
            (A 3D multivariate Gaussian distribution of particles of a given number)
    '''
    sigma = np.diag([a**2, (ba * a)**2, (ca * a)**2])
    coords = np.random.multivariate_normal(center, sigma, num_particles)
    
    return coords[:,0], coords[:,1], coords[:,2] 

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

def plot_particles(x,y,z): 
    '''
        The function plots the 3D particle distribution.

        Parameters:
            - None

        Returns:
            - None (plots the 3D particle distribution)
    
    '''

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=1, c='b', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Particle Distribution')

    plt.show()
    
    return

def compute_density_field(particles, grid_res=32):
    '''
        The function returns the density field of the particles.

        Parameters:
        particles: the particles example: particles = np.column_stack([x,y,z])
        grid_res: the resolution of the grid

        Returns:
        The density field of the particles (32x32x32 array)
    '''
    min_coords = particles.min(axis=0) - 0.5
    max_coords = particles.max(axis=0) + 0.5
    cell_size = (max_coords - min_coords) / grid_res

    x = np.linspace(min_coords[0], max_coords[0], grid_res, endpoint=False)
    y = np.linspace(min_coords[1], max_coords[1], grid_res, endpoint=False)
    z = np.linspace(min_coords[2], max_coords[2], grid_res, endpoint=False)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')

    density = np.zeros((grid_res, grid_res, grid_res))

    for p in particles:
        particle_min = p - 0.5
        particle_max = p + 0.5

        overlap_min_x = np.maximum(grid_x, particle_min[0])
        overlap_min_y = np.maximum(grid_y, particle_min[1])
        overlap_min_z = np.maximum(grid_z, particle_min[2])

        overlap_max_x = np.minimum(grid_x + cell_size[0], particle_max[0])
        overlap_max_y = np.minimum(grid_y + cell_size[1], particle_max[1])
        overlap_max_z = np.minimum(grid_z + cell_size[2], particle_max[2])

        overlap_size_x = np.maximum(0, overlap_max_x - overlap_min_x)
        overlap_size_y = np.maximum(0, overlap_max_y - overlap_min_y)
        overlap_size_z = np.maximum(0, overlap_max_z - overlap_min_z)

        overlap_volume = overlap_size_x * overlap_size_y * overlap_size_z

        density += overlap_volume

    return density

def plot_2d_slice(quantity, name, axis='z', slice_index=None):

    """
    Plot a 2D slice of a 3D quantity (example: density field, potential field, etc.)
    
    Parameters:
    - quantity: 3D numpy array of quantity values.
    - axis: The axis to take the slice along ('x', 'y', or 'z').
    - slice_index: The index of the slice along the chosen axis. If None, it defaults to the middle of the grid.
    """
    N = quantity.shape[0]
    
    if slice_index is None:
        slice_index = N // 2
    
    if axis == 'x':
        slice_2d = quantity[slice_index, :, :]
    elif axis == 'y':
        slice_2d = quantity[:, slice_index, :]
    elif axis == 'z':
        slice_2d = quantity[:, :, slice_index]
    else:
        raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")
    
    plt.figure(figsize=(8, 6))
    plt.imshow(slice_2d, extent=[0, N, 0, N], origin='lower', cmap='viridis')
    plt.colorbar(label=name)
    plt.xlabel(f'{axis} = {slice_index}')
    plt.ylabel('Grid Index')
    plt.title(f'2D Slice along {axis}-axis')
    plt.show()

    return 

def solve_poisson_fft(density_field):

    '''
        The function returns the potential of the density field by solving the Poisson 
        equation using the FFT.

        Parameters:
            - density_field: the density field (32x32x32 array)

        Returns:
            - phi: the potential of the density field (32x32x32 array)
    '''

    N = density_field.shape[0]
    m = np.fft.fftfreq(N) * N
    kx, ky, kz = np.meshgrid(2*np.pi*m,2*np.pi*m,2*np.pi*m, indexing='ij')
    
    rho_hat = np.fft.fftn(density_field)

    #avoid division by zero
    denominator = np.where(np.cos(2*np.pi*m/ N)-1 == 0, 1, (np.cos(2*np.pi*m/N)-1)*(kx**2+ky**2+kz**2))

    phi_hat = 4*np.pi*rho_hat/denominator

    #inverse Fourier transform to get the potential
    phi = np.fft.ifftn(phi_hat).real

    return phi

def green_function(N):
    '''
    This function returns the Green's function in a 3D grid, symmetric across the eight octants.

    Parameters:
    N (int): The number of grid points in each dimension.

    Returns:
    numpy.ndarray: The Green's function values in a 3D N x N x N grid.
    '''
    # Define the meshgrid
    x = np.linspace(0, N, N, endpoint=False)
    y = np.linspace(0, N, N, endpoint=False)
    z = np.linspace(0, N, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Calculate the distance from the origin
    r = np.sqrt(X**2 + Y**2 + Z**2)

    # Calculate the Green's function values
    g = np.where(r != 0, 1/r, 0)
    g[0, 0, 0] = 1  # Handle the special case at the origin

    # Mirror the values to other octants to maintain symmetry
    for i in range(N):
        for j in range(N):
            for k in range(N):
                g[i, j, k] = g[min(i, N - 1 - i), min(j, N - 1 - j), min(k, N - 1 - k)]

    return g

def expand_meshgrid(grid, M):

    N = grid.shape[0]

    # Check if the original grid is N x N x N
    if grid.shape != (N, N, N):
        raise ValueError("The original grid must be a cube (N x N x N)")
    
    convol_grid = np.zeros((M, M, M))
    convol_grid[:N, :N, :N] = grid

    return convol_grid

def solve_poisson_green(density, g):
    '''
        The function returns the potential of the density field by solving the Poisson equation using the Green's function.

        Parameters:
        density: the density field
        g: the Green's function

        Returns:
        The potential of the density field. 
    '''
    
    density_hat = fftn(density)
    g_hat = fftn(g)

    phi_hat = density_hat * g_hat

    phi = ifftn(phi_hat).real
    return phi

def plot_potential_vs_radius(phi):

    '''
        The function plots the average potential vs. the radius.

        Parameters:
            - phi: the potential (32x32x32 array)

        Returns:
            - None (plots the average potential vs. the radius)
    
    '''

    N = phi.shape[0]
    center = N // 2

    x, y, z = np.indices((N, N, N)) - center
    r = np.sqrt(x**2 + y**2 + z**2)
 
    r_flat = r.flatten()
    phi_flat = phi.flatten()

    unique_r = np.unique(r_flat)
    average_phi = np.array([phi_flat[r_flat == radius].mean() for radius in unique_r])

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(unique_r, average_phi, marker='o')
    plt.xlabel('Radius (r)')
    plt.ylabel('Average Potential (phi)')
    plt.title('Average Potential vs. Radius')
    plt.grid(True)
    plt.show()

def ver(positions, velocities, density, g, time_step,grid_size=32):
    potential = solve_poisson_green(density, g)
    potential = potential[:grid_size,:grid_size,:grid_size]

    # get values of force (3 components) on each particle at the current positions
    force = np.gradient(potential)
    # force = np.array(force)
    force = np.transpose(force, (1, 2, 3, 0))
    print(force.shape)
    print(velocities.shape)

    # calculate v(t + half step)
    vx, vy, vz = np.split(velocities, 3, axis=3)
    fx, fy, fz = np.split(force, 3, axis=3)
    new_vx = vx + 0.5*time_step*fx
    new_vy = vy + 0.5*time_step*fy
    new_vz = vz + 0.5*time_step*fz

    # new_velocities = velocities + 0.5*time_step*np.array(force)
    new_velocities = np.concatenate([new_vx, new_vy, new_vz], axis=1)

    #calculate x(t + step)
    new_x = positions[:,0] + time_step*new_vx
    new_y = positions[:,1] + time_step*new_vy
    new_z = positions[:,2] + time_step*new_vz
    new_positions = np.column_stack([new_x, new_y, new_z])

    # calculate density at x(t + step)
    new_density = compute_density_field(new_positions, grid_res=grid_size)

    return new_positions, new_velocities, new_density
