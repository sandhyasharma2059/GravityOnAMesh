#DELTA SOURCE

def delta_source(N): 
    '''
        The function returns a delta source at the center of the grid.

        Parameters:
        N: the number of grid points

        Returns:
        A delta source at the center of the grid.
    '''
    delta_source = np.zeros((N, N, N))
    delta_source[N//2, N//2, N//2] = 1
    return delta_source 