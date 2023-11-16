import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.fft import fftn, ifftn
from mpl_toolkits.mplot3d import Axes3D

# PARTICLE DISTRIBUTION
def distribute_particles(center, a, ba, ca, num_particles=32**3):
    '''
        The function returns a 3D multivariate Gaussian distribution of particles of a given number. 

        Parameters:
        center: center of the distribution
        a: semi-major axis
        ba: axis ratio b/a
        ca: axis ratio c/a
        num_particles: number of particles in the distribution 

        Returns:
        A 3D multivariate Gaussian distribution of particles of a given number
    '''
    sigma = np.diag([a**2, (ba * a)**2, (ca * a)**2])
    coords = np.random.multivariate_normal(center, sigma, num_particles)
    
    return coords[:,0], coords[:,1], coords[:,2]

center = (0, 0, 0)  
a = 5
ba = 0.7
ca = 0.6
x, y, z = distribute_particles(center, a, ba, ca)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=1, c='b', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Particle Distribution')

plt.show()

# COMPUTING DENSITY FIELD
particles = np.column_stack([x,y,z])

def compute_density_field(particles, grid_res=32):
    '''
        The function returns the density field of the particles.

        Parameters:
        particles: the particles
        grid_res: the resolution of the grid

        Returns:
        The density field of the particles.
    '''

    min_coords = particles.min(axis=0) - 0.5  
    max_coords = particles.max(axis=0) + 0.5
    cell_size = (max_coords - min_coords) / grid_res
    
    density = np.zeros((grid_res, grid_res, grid_res))
    
    cell_particle_map = {}
    
    # Created map to compute computations, created a map of "relevant particles" so the distance is not computed for particles far off
    for p in particles:
        particle_min = p - 0.5
        particle_max = p + 0.5

        cell_min_idx = np.floor((particle_min - min_coords) / cell_size).astype(int)
        cell_max_idx = np.ceil((particle_max - min_coords) / cell_size).astype(int)

        for i in range(cell_min_idx[0], cell_max_idx[0]):
            for j in range(cell_min_idx[1], cell_max_idx[1]):
                for k in range(cell_min_idx[2], cell_max_idx[2]):
                    if (i, j, k) not in cell_particle_map:
                        cell_particle_map[(i, j, k)] = []
                    cell_particle_map[(i, j, k)].append(p)
                    
    for i in range(grid_res):
        for j in range(grid_res):
            for k in range(grid_res):
                cell_min = min_coords + cell_size * np.array([i, j, k])
                cell_max = cell_min + cell_size
                
                relevant_particles = cell_particle_map.get((i, j, k), [])
                for p in relevant_particles:
                    particle_min = p - 0.5
                    particle_max = p + 0.5
                    
                    overlap_min = np.maximum(cell_min, particle_min)
                    overlap_max = np.minimum(cell_max, particle_max)
                    overlap_size = np.maximum(0, overlap_max - overlap_min)
                    overlap_volume = overlap_size[0] * overlap_size[1] * overlap_size[2]
                    
                    density[i, j, k] += overlap_volume
                    
    return density

density_field = compute_density_field(particles)

#plotting the density field
x, y, z = np.meshgrid(np.arange(density_field.shape[0]), np.arange(density_field.shape[1]), np.arange(density_field.shape[2]))

print(x.shape, y.shape, z.shape)

x = x.flatten()
y = y.flatten()
z = z.flatten()

density_values = density_field.flatten()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, c=density_values, cmap='viridis', vmin=15, vmax=25, alpha = 0.1)
plt.colorbar(sc, label='Density')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Density Plot')
plt.show()

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
    k = np.fft.fftfreq(N) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    
    rho_hat = np.fft.fftn(density_field)

    denominator = np.where(kx**2 + ky**2 + kz**2 == 0, 1, kx**2 + ky**2 + kz**2)
    phi_hat = (4 * np.pi * rho_hat) / denominator

    # Inverse Fourier transform to get the potential
    phi = np.fft.ifftn(phi_hat).real

    return phi

def delta_source(N): 
    '''
        The function returns a delta source at the center of the grid.

        Parameters:
        N: the number of grid points

        Returns:
        A delta source at the center of the grid.
    '''
    delta_source = np.zeros((N, N, N))
    delta_source[N//2, N//2, N//2] = 1
    return delta_source 

def plot_potential(phi, title):
    '''
        The function plots the potential for a given potential.

        Parameters:
        phi: the potential
        title: the title of the plot

        Returns:
        A plot of the potential in the form of a 3D scatter plot.
    '''
    N = phi.shape[0]
    x, y, z = np.indices((N, N, N))

    fig = plt.figure(figsize=(10, 8))  
    ax = fig.add_subplot(111, projection='3d')

    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    phi = phi.flatten()

    # Normalizing potential values for color mapping and transparency
    phi_norm = (phi - phi.min()) / (phi.max() - phi.min())
    cmap = plt.cm.viridis
    # Points with phi_norm <= 0.1 are almost invisible
    alpha = np.where(phi_norm > 0.05, 1, 0)  

    sc = ax.scatter(x, y, z, c=phi_norm, cmap=cmap, alpha=alpha, marker='o', s=10)

    plt.colorbar(sc)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    plt.show()
    return 

#solving the Poisson equation for a delta source
point_source = delta_source(32)
phi_point_source = solve_poisson_fft(point_source)
plot_potential(phi_point_source, 'Potential for a Delta Source')

def green_function(N):
    '''
        The function returns the Green's function.

        Parameters:
        N: the number of grid points

        Returns:
        The Green's function.
    '''
    x = np.linspace(-N/2, N/2, N, endpoint=False)
    y = np.linspace(-N/2, N/2, N, endpoint=False)
    z = np.linspace(-N/2, N/2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    r = np.sqrt(X**2 + Y**2 + Z**2)
    g = np.where((np.abs(X) < 1) & (np.abs(Y) < 1) & (np.abs(Z) < 1), 1/r, 0)
    g[N//2, N//2, N//2] = 1  

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

N = 64
point_source = delta_source(64)
g = green_function(64)
phi = solve_poisson_green(point_source, g)
plot_potential(phi, 'Potential for a Delta Source using Green Function') 