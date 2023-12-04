'''
Name: delta_source.py
Description: This file contains the function to create a delta source at the center of 
             the grid and plot the 2D slices of the potential along the x-z plane.
Author: Greta Goldberg, Jason Li, Sandhya Sharma
Last Modified: December 3, 2023

'''
import solve_poisson_equation
import numpy as np
import matplotlib.pyplot as plt

def delta_source(N): 
    '''
        The function returns a delta source at the center of the grid.
        Given its nature, its density distribution is the same as its 
        particle distribution.

        Parameters:
        N: the number of grid points

        Returns:
        A delta source at the center of the grid.
    '''

    delta_source = np.zeros((N, N, N))
    delta_source[N//2, N//2, N//2] = 1
    return delta_source 

point_source = delta_source(32)
potential_point_source = solve_poisson_equation.solve_poisson_fft(point_source)

def plot_slices_2d(potential, N):
    """
    Plot 2D slices of the potential along the y-axis.

    Parameters:
    potential: 3D potential array
    N: Size of the potential array

    Returns:
    None (plots the 2D slices)
    """
    fig, axs = plt.subplots(1, N, figsize=(15, 5), sharey=True)
    for i in range(N):
        im = axs[i].imshow(potential[:, i, :], extent=(0, 1, 0, 1), origin='lower')  # Hard-coded L = 1.0
        axs[i].set_title(f'Potential at y = {i}')

    fig.suptitle('2D Slices of Potential from a Point Source')
    fig.text(0.5, 0.04, 'x', ha='center')
    fig.text(0.04, 0.5, 'z', va='center', rotation='vertical')

    colorbar = fig.colorbar(im, ax=axs.ravel().tolist())  # Add a single colorbar for all subplots
    colorbar.set_label('Potential')
    plt.show()

plot_slices_2d(potential_point_source,5)