import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fftn, ifftn

# TODO: PART 2
def distribute_particles(center, a, ba, ca, num_particles=32**3):

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

plt.show()

particles = np.column_stack([x,y,z])

def compute_density_field(particles, grid_res=32):

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

# TODO: PART 3
def solve_poisson_fft(density_field):
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
    delta_source = np.zeros((N, N, N))
    delta_source[N//2, N//2, N//2] = 1
    return delta_source 

point_source = delta_source(32)
phi_point_source = solve_poisson_fft(point_source)

# Plotting
def plot_potential(phi):
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

    # Colorbar
    plt.colorbar(sc)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    return 

plot_potential(phi_point_source)

def green_function(N):
    x = np.linspace(-N/2, N/2, N, endpoint=False)
    y = np.linspace(-N/2, N/2, N, endpoint=False)
    z = np.linspace(-N/2, N/2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    r = np.sqrt(X**2 + Y**2 + Z**2)
    g = np.where((np.abs(X) < 1) & (np.abs(Y) < 1) & (np.abs(Z) < 1), 1/r, 0)
    g[N//2, N//2, N//2] = 1  

    return g

def solve_poisson_green(density, g):
    density_hat = fftn(density)
    g_hat = fftn(g)

    phi_hat = density_hat * g_hat

    phi = ifftn(phi_hat).real
    return phi

N = 64
point_source = delta_source(64)
g = green_function(64)
phi = solve_poisson_green(point_source, g)
plot_potential(phi)