
import particle_distribution
import numpy as np
import matplotlib.pyplot as plt


x,y,z = particle_distribution.spherically_sym_particles((0,0,0), 32**3)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=1, c='b', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Particle Distribution')

plt.show()

