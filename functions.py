import numpy as np
import matplotlib.pyplot as plt

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

def compute_density_field(particles, grid_res=32):

    min_coords = particles.min(axis=0) - 0.5  # accounting for the 1x1x1 cube around the particle
    max_coords = particles.max(axis=0) + 0.5
    cell_size = (max_coords - min_coords) / grid_res
    
    # Initialize the density field to zeros
    density = np.zeros((grid_res, grid_res, grid_res))
    
    # Iterate over all grid cells
    for i in range(grid_res):
        for j in range(grid_res):
            for k in range(grid_res):
                # Determine the spatial extent of this cell
                cell_min = min_coords + cell_size * np.array([i, j, k])
                cell_max = cell_min + cell_size
                
                # Check overlap with each particle's cube and sum contributions
                for p in particles:
                    particle_min = p - 0.5
                    particle_max = p + 0.5
                    
                    # Compute the overlapping volume between the grid cell and particle cube
                    overlap_min = np.maximum(cell_min, particle_min)
                    overlap_max = np.minimum(cell_max, particle_max)
                    overlap_size = np.maximum(0, overlap_max - overlap_min)
                    overlap_volume = overlap_size[0] * overlap_size[1] * overlap_size[2]
                    
                    # Add the overlap volume to the density of this cell
                    density[i, j, k] += overlap_volume
                    
    return density

# Compute the density field for the given particles
density_field = compute_density_field(particles)

