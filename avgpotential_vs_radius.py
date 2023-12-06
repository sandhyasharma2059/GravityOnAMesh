'''
Name: avgpotential_vs_radius.py
Description: This file contains the function to plot the average potential vs. the radius.
Author: Greta Goldberg, Jason Li, Sandhya Sharma
Last Modified: December 3, 2023

'''

import numpy as np
import matplotlib.pyplot as plt

def plot_potential_vs_radius(phi):
    '''
        The function plots the average potential vs. the radius.

        Parameters:
            - phi: the potential (32x32x32 array)

        Returns:
            - None (plots the average potential vs. the radius)
    
    '''

    N = phi.shape[0]
    center = N // 2

    x, y, z = np.indices((N, N, N)) - center
    r = np.sqrt(x**2 + y**2 + z**2)

    r_flat = r.flatten()
    phi_flat = phi.flatten()

    unique_r = np.unique(r_flat)
    average_phi = np.array([phi_flat[r_flat == radius].mean() for radius in unique_r])

    r = np.linspace(0.1, 10, 100)
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(unique_r, average_phi, marker='o')
    # plt.plot(r_flat, 1 / r_flat + 300, label='1/r', color='red')
    # plt.plot(r, 10/ r + 300, label='1/r', color='red')
    plt.xlabel('Radius (r)')
    plt.ylabel('Average Potential (phi)')
    plt.title('Average Potential vs. Radius')
    plt.grid(True)
    plt.show()

