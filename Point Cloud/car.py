import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Set random seed for reproducibility
np.random.seed(42)

# Function to create a basic car shape using point clouds
def create_car_point_cloud(num_points=5000):
    # Car dimensions
    car_length = 4.5  # length in meters
    car_width = 2.0   # width in meters
    car_height = 1.5  # height in meters
    
    # Main body (cuboid)
    body_points = np.random.rand(int(num_points * 0.7), 3)
    body_points[:, 0] *= car_length        # x-axis (length)
    body_points[:, 1] *= car_width         # y-axis (width)
    body_points[:, 2] *= car_height * 0.7  # z-axis (height)
    body_points[:, 1] -= car_width / 2     # Center the car on y-axis
    
    # Roof (slightly narrower than body)
    roof_points = np.random.rand(int(num_points * 0.15), 3)
    roof_points[:, 0] *= car_length * 0.6      # shorter than body
    roof_points[:, 0] += car_length * 0.15     # shifted backwards
    roof_points[:, 1] *= car_width * 0.8       # narrower than body
    roof_points[:, 1] -= car_width * 0.4       # Center the roof
    roof_points[:, 2] *= car_height * 0.3      # roof height
    roof_points[:, 2] += car_height * 0.7      # above the body
    
    # Wheels (4 clusters of points)
    wheel_radius = 0.35
    wheel_width = 0.2
    wheel_points = []
    
    wheel_positions = [
        [car_length * 0.2, car_width * 0.5, wheel_radius],       # front-right
        [car_length * 0.2, -car_width * 0.5, wheel_radius],      # front-left
        [car_length * 0.8, car_width * 0.5, wheel_radius],       # rear-right
        [car_length * 0.8, -car_width * 0.5, wheel_radius]       # rear-left
    ]
    
    for wheel_pos in wheel_positions:
        wheel_pts = np.random.rand(int(num_points * 0.0375), 3)
        # Convert to cylindrical coordinates for wheel shape
        theta = 2 * np.pi * wheel_pts[:, 0]
        radius = wheel_radius * np.sqrt(wheel_pts[:, 1])
        
        wx = wheel_pos[0] + wheel_width * (wheel_pts[:, 2] - 0.5)
        wy = wheel_pos[1] + radius * np.cos(theta)
        wz = radius * np.sin(theta)
        
        wheel = np.column_stack((wx, wy, wz))
        wheel_points.append(wheel)
    
    # Combine all points
    car_points = np.vstack([body_points, roof_points] + wheel_points)
    
    # Add color information (RGB)
    # Body is blue, roof is darker blue, wheels are black
    colors = np.ones((car_points.shape[0], 3)) * 0.2  # Initialize with dark gray
    
    # Body points are blue
    body_indices = range(body_points.shape[0])
    colors[body_indices] = np.array([0.1, 0.3, 0.8])  # Blue
    
    # Roof points are darker blue
    roof_indices = range(body_points.shape[0], body_points.shape[0] + roof_points.shape[0])
    colors[roof_indices] = np.array([0.05, 0.15, 0.5])  # Darker blue
    
    # Wheels are black
    wheel_indices = range(body_points.shape[0] + roof_points.shape[0], car_points.shape[0])
    colors[wheel_indices] = np.array([0.1, 0.1, 0.1])  # Black
    
    return car_points, colors

# Create the car point cloud
car_points, car_colors = create_car_point_cloud(8000)

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the car point cloud
scatter = ax.scatter(
    car_points[:, 0], car_points[:, 1], car_points[:, 2],
    c=car_colors, s=5, alpha=0.8
)

# Set plot limits and labels
ax.set_xlim(0, 5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(0, 2)
ax.set_xlabel('X (length)')
ax.set_ylabel('Y (width)')
ax.set_zlabel('Z (height)')
ax.set_title('3D Car Point Cloud Visualization')

# Set equal aspect ratio
ax.set_box_aspect([5/3, 3/3, 2/3])  # Based on the plot limits

# Function to update the plot for animation
def update(frame):
    ax.view_init(elev=20, azim=frame)
    return scatter,

# Create animation
ani = FuncAnimation(fig, update, frames=range(0, 360, 2), interval=50, blit=True)

plt.tight_layout()
plt.show()

# If you want to save the animation (requires ffmpeg or imagemagick)
# ani.save('car_point_cloud_rotation.gif', writer='pillow', fps=20, dpi=100)
