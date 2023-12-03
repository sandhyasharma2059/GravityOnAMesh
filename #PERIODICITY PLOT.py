#AVERAGE POTENTIAL VS RADIUS

def plot_potential_vs_radius(phi):
    N = phi.shape[0]
    center = N // 2

    x, y, z = np.indices((N, N, N)) - center
    r = np.sqrt(x**2 + y**2 + z**2)

    r_flat = r.flatten()
    phi_flat = phi.flatten()

    unique_r = np.unique(r_flat)
    average_phi = np.array([phi_flat[r_flat == radius].mean() for radius in unique_r])

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(unique_r, average_phi, marker='o')
    plt.plot(unique_r, 1 / unique_r, label='1/r', color='red')
    plt.xlabel('Radius (r)')
    plt.ylabel('Average Potential (phi)')
    plt.title('Average Potential vs. Radius')
    plt.grid(True)
    plt.show()

