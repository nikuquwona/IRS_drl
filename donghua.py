
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


# Load data from file
with open('filename_41.txt', 'r') as f:
    data = np.array(eval(f.read()))

# Select first three columns
data = data[:,:3]
print(len(data))
reduced_data = np.vstack([data[::1000], data[-1]])
print(reduced_data[0])
print(len(reduced_data))
# input()
# Assuming 'reduced_data' is your data
data = reduced_data 

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize the line
line, = ax.plot([], [], [], lw=2)

# Set the limits
ax.set_xlim([min(data[:,0]), max(data[:,0])])
ax.set_ylim([min(data[:,1]), max(data[:,1])])
ax.set_zlim([min(data[:,2]), max(data[:,2])])

# Initialization function 
def init():
    line.set_data([], [])
    line.set_3d_properties([])
    return line,

# Animation function. This is called sequentially
def animate(i):
    line.set_data(data[:i, 0], data[:i, 1])
    line.set_3d_properties(data[:i, 2])
    return line,

# Create an animation
ani = animation.FuncAnimation(fig, animate, frames=len(data), init_func=init, blit=True)

# Save the animation
ani.save('3d_trajectory.mp4', writer='ffmpeg')

plt.show()
