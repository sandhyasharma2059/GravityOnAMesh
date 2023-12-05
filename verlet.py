import numpy as np

def verlet_function(x0=None, y0=None, z0=None, vx0=None, vy0=None, vz0=None, tend=None, dt=1.e-5, potential_array=None):
    """Integrate forward in time
    
    Parameters
    ----------
    
    dt : np.float32
        time step
        
    potential_array : np.ndarray 
        3D array containing the potential field
    
    x0, y0, z0 : np.float32
        initial positions
    
    vx0, vy0, vz0 : np.float32
        initial velocities
        
    tend : np.float32
        time to integrate to
        
    Returns
    -------
    
    positions : list of np.ndarray
        List containing arrays of positions at each time step
    times : list of np.float32
        List containing times at each time step
    """
    positions = []
    times = []
    
    x = x0
    y = y0
    z = z0
    vx = vx0
    vy = vy0
    vz = vz0
    t = 0.0
    
    while t < tend: 
        # Update positions at half-integer step
        x = x + 0.5 * vx * dt
        y = y + 0.5 * vy * dt
        z = z + 0.5 * vz * dt
        t = t + 0.5 * dt
        
        # Save positions and times at each half-integer step
        positions.append(np.array([x, y, z]))
        times.append(t)
        
        # Calculate the gradient of the potential at the given position
        gradient = -np.array(np.gradient(potential_array, x, y, z))
        
        # Update velocities at integer step
        vx = vx + dt * gradient[0]
        vy = vy + dt * gradient[1]
        vz = vz + dt * gradient[2]
        t = t + dt
        
        # Update positions at half-integer step
        x = x + 0.5 * vx * dt
        y = y + 0.5 * vy * dt
        z = z + 0.5 * vz * dt
        t = t + 0.5 * dt
    
    return positions, times


#courant condition one 

import numpy as np

def verlet_function(x0=None, y0=None, z0=None, vx0=None, vy0=None, vz0=None, tend=None, dt=1.e-5, potential_array=None):
    """Integrate forward in time with adaptive time stepping based on the CFL condition
    
    Parameters
    ----------
    
    dt : np.float32
        Initial time step
        
    potential_array : np.ndarray 
        3D array containing the potential field
    
    x0, y0, z0 : np.float32
        Initial positions
    
    vx0, vy0, vz0 : np.float32
        Initial velocities
        
    tend : np.float32
        Time to integrate to
        
    Returns
    -------
    
    positions : list of np.ndarray
        List containing arrays of positions at each time step
    times : list of np.float32
        List containing times at each time step
    """
    positions = []
    times = []
    
    x = x0
    y = y0
    z = z0
    vx = vx0
    vy = vy0
    vz = vz0
    t = 0.0
    
    while t < tend: 
        # Update positions at half-integer step
        x = x + 0.5 * vx * dt
        y = y + 0.5 * vy * dt
        z = z + 0.5 * vz * dt
        t = t + 0.5 * dt
        
        # Save positions and times at each half-integer step
        positions.append(np.array([x, y, z]))
        times.append(t)
        
        # Calculate the gradient of the potential at the given position
        gradient = -np.array(np.gradient(potential_array, x, y, z))
        
        # Calculate the maximum velocity for CFL condition
        max_velocity = np.max(np.abs([vx, vy, vz]))
        
        # Update velocities at integer step with adaptive time step
        dt = min(dt, 0.1 / max_velocity)
        vx = vx + dt * gradient[0]
        vy = vy + dt * gradient[1]
        vz = vz + dt * gradient[2]
        t = t + dt
        
        # Update positions at half-integer step
        x = x + 0.5 * vx * dt
        y = y + 0.5 * vy * dt
        z = z + 0.5 * vz * dt
        t = t + 0.5 * dt
    
    return positions, times
