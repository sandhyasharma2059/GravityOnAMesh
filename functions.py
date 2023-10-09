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

