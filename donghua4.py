import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Assuming 'reduced_data' is your data
with open('filename_44.txt', 'r') as f:
    data = np.array(eval(f.read()))
data = data[:,:3]
reduced_data = np.vstack([data[::5], data[-1]])
data = reduced_data 

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize the scatter with 'none' facecolors to make hollow points
scat = ax.scatter([], [], [], facecolors='none', edgecolors='blue')

# Set the limits
ax.set_xlim([min(data[:,0]), max(data[:,0])])
ax.set_ylim([min(data[:,1]), max(data[:,1])])
ax.set_zlim([0, max(data[:,2])])  # Set zlim from 0

ax.set_xlim([0,50])
ax.set_ylim([-10,10])
ax.set_zlim([0, 10])  # Set zlim from 0


# Add a reference point
ref_point = np.array([40, 3, 0])
ax.scatter(*ref_point, color='y', s=100, label='User')
# ,[44, 0, 0]
ref_point = np.array([44, 0, 0])
ax.scatter(*ref_point, color='y', s=100, label='User')

ref_point = np.array([0, 0, 2])
ax.scatter(*ref_point, color='b', s=100, label='AP')
# Initialization function 
def init():
    scat._offsets3d = ([], [], [])
    return scat,

# Animation function. This is called sequentially
def animate(i):
    x = data[:i, 0]
    y = data[:i, 1]
    z = data[:i, 2]
    scat._offsets3d = (x, y, z)
    return scat,

# Create an animation with repeat=False
ani = animation.FuncAnimation(fig, animate, frames=len(data), init_func=init, blit=False, repeat=False)

# Mark the start and end points outside the animation function
ax.scatter(*data[0], color='g', s=100, label='Start')
ax.scatter(*data[-1], color='r', s=100, label='End')
ax.legend()

# Save the animation
ani.save('3d_trajectory_hollow_points_reference_zlim_no_repeat_no_duplicate_legend4.mp4', writer='ffmpeg')

plt.show()
