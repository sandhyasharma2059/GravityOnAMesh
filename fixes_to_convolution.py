#fixes to green's function

import numpy as np
from scipy.fft import fftn, ifftn

def green_function(NG):
    '''
    The function returns the Green's function.

    Parameters:
    NG: the number of grid points

    Returns:
    The Green's function.
    '''
    #Expand the grid size for the Green's function
    N = NG * 2

    #Create a meshgrid centered at the expanded grid
    x = np.linspace(-N/2, N/2, N, endpoint=False)
    y = np.linspace(-N/2, N/2, N, endpoint=False)
    z = np.linspace(-N/2, N/2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Calculate the radial distance
    r = np.sqrt(X**2 + Y**2 + Z**2)
    r[N//2, N//2, N//2] = 1

    #Define the Green's function
    g = np.where(r <= N//2, 1/r, 0)
    g[0, 0, 0] = 1

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
    #Ensure both density and Green's function have the same shape
    target_shape = density.shape
    if g.shape != target_shape:
        g = resize_green_function(g, target_shape)

    #Perform FFT on both density and Green's function
    density_hat = fftn(density)
    g_hat = fftn(g)

    #Perform element-wise multiplication in the frequency domain
    phi_hat = density_hat * g_hat

    # Perform inverse FFT to obtain the potential
    phi = ifftn(phi_hat).real

    return phi

def resize_green_function(g, target_shape):
    '''
    Resize the Green's function to match the target shape.

    Parameters:
    g: the Green's function
    target_shape: the target shape

    Returns:
    Resized Green's function.
    '''
    resized_g = np.zeros(target_shape)
    slices = [slice(0, min(dim, target_dim)) for dim, target_dim in zip(g.shape, target_shape)]
    resized_g[slices] = g[slices]
    return resized_g
