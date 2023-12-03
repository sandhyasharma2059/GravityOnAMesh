#PLOTTING 2D SLICE

def plot_2d_slice(phi, name, axis='z', slice_index=None):
    """
    Plot a 2D slice of the 3D quantity.
    
    Parameters:
    - phi: 3D numpy array of quantity values.
    - axis: The axis to take the slice along ('x', 'y', or 'z').
    - slice_index: The index of the slice along the chosen axis. If None, it defaults to the middle of the grid.
    """
    N = phi.shape[0]
    
    if slice_index is None:
        slice_index = N // 2
    
    if axis == 'x':
        slice_2d = phi[slice_index, :, :]
    elif axis == 'y':
        slice_2d = phi[:, slice_index, :]
    elif axis == 'z':
        slice_2d = phi[:, :, slice_index]
    else:
        raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")
    
    plt.figure(figsize=(8, 6))
    plt.imshow(slice_2d, extent=[0, N, 0, N], origin='lower', cmap='viridis')
    plt.colorbar(label=name)
    plt.xlabel(f'{axis} = {slice_index}')
    plt.ylabel('Grid Index')
    plt.title(f'2D Slice along {axis}-axis')
    plt.show()

    return 
