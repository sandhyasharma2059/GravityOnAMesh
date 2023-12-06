import particle_distribution
import density_field
import avgpotential_vs_radius

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn

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

# x,y,z = particle_distribution.spherically_sym_particles((0,0,0), 32**3, 2)
# particles = np.column_stack([x,y,z])
# density = density_field.compute_density_field(particles, 32)

def expand_meshgrid(grid, M):

    N = grid.shape[0]

    # Check if the original grid is N x N x N
    if grid.shape != (N, N, N):
        raise ValueError("The original grid must be a cube (N x N x N)")
    
    convol_grid = np.zeros((M, M, M))
    convol_grid[:N, :N, :N] = grid

    return convol_grid

# density = expand_meshgrid(density, 64)
# g = green_function(32)
# g = expand_meshgrid(g, 64)
# phi = solve_poisson_green(density, g)
# print(phi)
# print(phi.shape)

# avgpotential_vs_radius.plot_potential_vs_radius(phi)
# plt.plot(phi)
# plt.show()
