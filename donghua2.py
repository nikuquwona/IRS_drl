import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


with open('filename_41.txt', 'r') as f:
    data = np.array(eval(f.read()))
data = data[:,:3]
reduced_data = np.vstack([data[::100], data[-1]])
# Assuming 'reduced_data' is your data
data = reduced_data 

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize the line and scatter
line, = ax.plot([], [], [], lw=2, color='blue')
scat = ax.scatter([], [], [], lw=2, color='blue')

# Set the limits
ax.set_xlim([min(data[:,0]), max(data[:,0])])
ax.set_ylim([min(data[:,1]), max(data[:,1])])
ax.set_zlim([min(data[:,2]), max(data[:,2])])

# Initialization function 
def init():
    line.set_data([], [])
    line.set_3d_properties([])
    scat._offsets3d = ([], [], [])
    return line, scat,

# Animation function. This is called sequentially
def animate(i):
    line.set_data(data[:i, 0], data[:i, 1])
    line.set_3d_properties(data[:i, 2])
    x = data[:i, 0]
    y = data[:i, 1]
    z = data[:i, 2]
    scat._offsets3d = (x, y, z)
    # Mark the start and end points
    if i == 0:
        ax.scatter(*data[0], color='g', s=100, label='Start')
    if i == len(data) - 1:
        ax.scatter(*data[-1], color='r', s=100, label='End')
        ax.legend()
    return line, scat,

# Create an animation
ani = animation.FuncAnimation(fig, animate, frames=len(data), init_func=init, blit=False)

# Save the animation
ani.save('3d_trajectory_line_and_points.mp4', writer='ffmpeg')

plt.show()

# # Create a figure and a 3D axis
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Initialize scatter
# scat = ax.scatter([], [], [], lw=2)

# # Set the limits
# ax.set_xlim([min(data[:,0]), max(data[:,0])])
# ax.set_ylim([min(data[:,1]), max(data[:,1])])
# ax.set_zlim([min(data[:,2]), max(data[:,2])])

# # Initialization function 
# def init():
#     scat._offsets3d = ([], [], [])
#     return scat,

# # Animation function. This is called sequentially
# def animate(i):
#     x = data[:i, 0]
#     y = data[:i, 1]
#     z = data[:i, 2]
#     scat._offsets3d = (x, y, z)
#     # Mark the start and end points
#     if i == 0:
#         ax.scatter(*data[0], color='g', s=100, label='Start')
#     if i == len(data) - 1:
#         ax.scatter(*data[-1], color='r', s=100, label='End')
#         ax.legend()
#     return scat,

# # Create an animation
# ani = animation.FuncAnimation(fig, animate, frames=len(data), init_func=init, blit=False)

# # Save the animation
# ani.save('3d_trajectory_points2.mp4', writer='ffmpeg')

# plt.show()
