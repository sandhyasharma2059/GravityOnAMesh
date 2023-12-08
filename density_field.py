'''
Name: density_field.py
Description: This file contains the function to compute the density field of the particles.
Author: Greta Goldberg, Jason Li, Sandhya Sharma
Last Modified: December 3, 2023

'''

import particle_distribution
import numpy as np

# def compute_density_field(particles, grid_res=32):
#     '''
#         The function returns the density field of the particles.

#         Parameters:
#         particles: the particles example: particles = np.column_stack([x,y,z])
#         grid_res: the resolution of the grid

#         Returns:
#         The density field of the particles (32x32x32 array)
#     '''

#     min_coords = particles.min(axis=0) - 0.5  
#     max_coords = particles.max(axis=0) + 0.5
#     cell_size = (max_coords - min_coords) / grid_res
    
#     density = np.zeros((grid_res, grid_res, grid_res))
    
#     cell_particle_map = {}
    
#     # Created map to compute computations, created a map of "relevant particles" so the distance is not computed for particles far off
#     for p in particles:
#         particle_min = p - 0.5
#         particle_max = p + 0.5

#         cell_min_idx = np.floor((particle_min - min_coords) / cell_size).astype(int)
#         cell_max_idx = np.ceil((particle_max - min_coords) / cell_size).astype(int)

#         for i in range(cell_min_idx[0], cell_max_idx[0]):
#             for j in range(cell_min_idx[1], cell_max_idx[1]):
#                 for k in range(cell_min_idx[2], cell_max_idx[2]):
#                     if (i, j, k) not in cell_particle_map:
#                         cell_particle_map[(i, j, k)] = []
#                     cell_particle_map[(i, j, k)].append(p)
                    
#     for i in range(grid_res):
#         for j in range(grid_res):
#             for k in range(grid_res):
#                 cell_min = min_coords + cell_size * np.array([i, j, k])
#                 cell_max = cell_min + cell_size
                
#                 relevant_particles = cell_particle_map.get((i, j, k), [])
#                 for p in relevant_particles:
#                     particle_min = p - 0.5
#                     particle_max = p + 0.5
                    
#                     overlap_min = np.maximum(cell_min, particle_min)
#                     overlap_max = np.minimum(cell_max, particle_max)
#                     overlap_size = np.maximum(0, overlap_max - overlap_min)
#                     overlap_volume = overlap_size[0] * overlap_size[1] * overlap_size[2]
                    
#                     density[i, j, k] += overlap_volume
                    
#     return density


def compute_density_field(particles, grid_res=32):
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
