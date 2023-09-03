import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D, art3d

# Assuming 'reduced_data' is your data
with open('filename_45.txt', 'r') as f:
    data = np.array(eval(f.read()))
data = data[:,:3]
reduced_data = np.vstack([data[::100], data[-1]])
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

# Add a reference point
# ref_point = np.array([48, 3, 0])
# ax.scatter(*ref_point, color='y', s=100, label='Reference')

# Users data
users = np.array([(48,-3,0),(49,5,0),(46,1,0),(30,10,0),(70,-20,0),(39,-18,0),(55,19,0),(70,11,0),(37,8,0),(26,-12.5,0)])

# Add users
ax.scatter(users[:,0], users[:,1], users[:,2], color='purple', s=50, label='Users')

# Create a circle with the convex hull of the users points
hull = ConvexHull(users[:,:2])
center = np.mean(hull.points[hull.vertices, :], axis=0)
radius = max(np.linalg.norm(hull.points[hull.vertices, :] - center, axis=1))
circle = Circle((center[0], center[1]), radius, color='lightblue', fill=True, alpha=0.2)

# Add the circle to the plot
ax.add_patch(circle)
# Transform circle to match the 3D plot
art3d.pathpatch_2d_to_3d(circle, z=0, zdir="z")

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
ani.save('3d_trajectory_hollow_points_reference_zlim_no_repeat_no_duplicate_legend_users5.mp4', writer='ffmpeg')

plt.show()
