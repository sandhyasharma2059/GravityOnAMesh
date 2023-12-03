# SOLVING THE POISSON EQUATION
def solve_poisson_fft(density_field):
    '''
        The function returns the potential of the density field by solving the Poisson equation using the FFT.

        Parameters:
        density_field: the density field

        Returns:
        The potential of the density field. 
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