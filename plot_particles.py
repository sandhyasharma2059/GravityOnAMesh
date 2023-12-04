'''
Name: plot_particles.py
Description: This file contains the function to plot the 3D particle distribution for different distributions.
Author: Greta Goldberg, Jason Li, Sandhya Sharma
Last Modified: December 3, 2023

'''

import particle_distribution
import numpy as np
import matplotlib.pyplot as plt

def plot_particles(x,y,z): 
    '''
        The function plots the 3D particle distribution.

        Parameters:
            - None

        Returns:
            - None (plots the 3D particle distribution)
    
    '''

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=1, c='b', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Particle Distribution')

    plt.show()
    
    return