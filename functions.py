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

particles = np.column_stack(x,y,z]

def compute_density_field(particles, grid_res=32):

    min_coords = particles.min(axis=0) - 0.5  
    max_coords = particles.max(axis=0) + 0.5
    cell_size = (max_coords - min_coords) / grid_res
    
    density = np.zeros((grid_res, grid_res, grid_res))
    
    cell_particle_map = {}

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

