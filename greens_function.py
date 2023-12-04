import numpy as np

def green_function(N):
    '''
        The function returns the Green's function.

        Parameters:
        N: the number of grid points

        Returns:
        The Green's function.
    '''
    
    x = np.linspace(-N/2, N/2, N, endpoint=True)
    y = np.linspace(-N/2, N/2, N, endpoint=False)
    z = np.linspace(-N/2, N/2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    r = np.sqrt(X**2 + Y**2 + Z**2)
    r[N//2, N//2, N//2] = 1
    greens_function = np.where(r <= N//2, 1/r, 0) 
    greens_function[0,0,0] = 1 
    return greens_function

def solve_poisson_green(density, g):
    '''
    The function returns the potential of the density field by solving the Poisson equation using the Green's function.

    Parameters:
    density: the density field
    g: the Green's function

    Returns:
    The potential of the density field. 
    '''
    density_hat = fftn(np.fft.ifftshift(density))
    g_hat = fftn(np.fft.ifftshift(g))

    phi_hat = density_hat * g_hat

    phi = np.fft.fftshift(ifftn(phi_hat).real)
    return phi
