'''
Name: particle_distribution.py
Description: This file contains the functions to create the following particle distributions:
                - 3D multivariate Gaussian distribution
                - Spherically symmetric distribution
Author: Greta Goldberg, Jason Li, Sandhya Sharma
Last Modified: December 3, 2023

'''

import numpy as np

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



def spherically_sym_particles(center, num_of_particles):
    '''
    The function returns a spherically symmetric distribution of particles of a given number. 

    Parameters:
        - center: center of the distribution
        - num_of_particles: number of particles in the distribution 

    Returns:
        - 3 1D arrays of length num_particles (x, y, z)
          (A spherically symmetric distribution of particles of a given number)
    '''

    r = np.random.uniform(0, 1, num_of_particles)
    theta = np.random.uniform(0, np.pi, num_of_particles)
    phi = np.random.uniform(0, 2*np.pi, num_of_particles)

    x = center[0] + r * np.sin(theta) * np.cos(phi)
    y = center[1] + r * np.sin(theta) * np.sin(phi)
    z = center[2] + r * np.cos(theta)

    return x, y, z
