import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set random seed for reproducibility
np.random.seed(42)

# Function to create a basic car shape using point clouds
def create_car_point_cloud(num_points=8000):
    # Car dimensions
    car_length = 4.5  # length in meters
    car_width = 2.0   # width in meters
    car_height = 1.5  # height in meters
    
    # Main body (cuboid)
    body_points = np.random.rand(int(num_points * 0.6), 3)
    body_points[:, 0] *= car_length        # x-axis (length)
    body_points[:, 1] *= car_width         # y-axis (width)
    body_points[:, 2] *= car_height * 0.7  # z-axis (height)
    body_points[:, 1] -= car_width / 2     # Center the car on y-axis
    body_colors = ['rgba(30, 100, 200, 0.8)'] * len(body_points)  # blue
    body_part = ['Body'] * len(body_points)
    
    # Windshield (front)
    windshield_points = np.random.rand(int(num_points * 0.05), 3)
    windshield_points[:, 0] = car_length * 0.25 + windshield_points[:, 0] * car_length * 0.15
    windshield_points[:, 1] = (windshield_points[:, 1] - 0.5) * car_width * 0.8
    windshield_points[:, 2] = car_height * 0.6 + windshield_points[:, 2] * car_height * 0.3
    windshield_colors = ['rgba(150, 200, 255, 0.7)'] * len(windshield_points)  # light blue transparent
    windshield_part = ['Windshield'] * len(windshield_points)
    
    # Rear window
    rear_window_points = np.random.rand(int(num_points * 0.05), 3)
    rear_window_points[:, 0] = car_length * 0.75 - rear_window_points[:, 0] * car_length * 0.15
    rear_window_points[:, 1] = (rear_window_points[:, 1] - 0.5) * car_width * 0.8
    rear_window_points[:, 2] = car_height * 0.6 + rear_window_points[:, 2] * car_height * 0.3
    rear_window_colors = ['rgba(150, 200, 255, 0.7)'] * len(rear_window_points)
    rear_window_part = ['Rear Window'] * len(rear_window_points)
    
    # Roof (slightly narrower than body)
    roof_points = np.random.rand(int(num_points * 0.15), 3)
    roof_points[:, 0] *= car_length * 0.5      # shorter than body
    roof_points[:, 0] += car_length * 0.25     # shifted to middle
    roof_points[:, 1] *= car_width * 0.8       # narrower than body
    roof_points[:, 1] -= car_width * 0.4       # Center the roof
    roof_points[:, 2] *= car_height * 0.3      # roof height
    roof_points[:, 2] += car_height * 0.7      # above the body
    roof_colors = ['rgba(20, 70, 150, 0.9)'] * len(roof_points)  # darker blue
    roof_part = ['Roof'] * len(roof_points)
    
    # Hood
    hood_points = np.random.rand(int(num_points * 0.05), 3)
    hood_points[:, 0] *= car_length * 0.25     # front quarter of car
    hood_points[:, 1] *= car_width * 0.9      
    hood_points[:, 1] -= car_width * 0.45      
    hood_points[:, 2] *= car_height * 0.05     # thin layer
    hood_points[:, 2] += car_height * 0.55     # at height of hood
    hood_colors = ['rgba(40, 110, 210, 0.85)'] * len(hood_points)  # slightly different blue
    hood_part = ['Hood'] * len(hood_points)
    
    # Trunk
    trunk_points = np.random.rand(int(num_points * 0.05), 3)
    trunk_points[:, 0] *= car_length * 0.25    # rear quarter of car
    trunk_points[:, 0] += car_length * 0.75    # start from 3/4 mark
    trunk_points[:, 1] *= car_width * 0.9
    trunk_points[:, 1] -= car_width * 0.45
    trunk_points[:, 2] *= car_height * 0.05    # thin layer
    trunk_points[:, 2] += car_height * 0.55    # at height of trunk
    trunk_colors = ['rgba(40, 110, 210, 0.85)'] * len(trunk_points)
    trunk_part = ['Trunk'] * len(trunk_points)
    
    # Headlights
    headlights_points = []
    headlights_colors = []
    headlights_part = []
    
    # Left headlight
    left_headlight = np.random.rand(int(num_points * 0.02), 3)
    left_headlight[:, 0] *= car_length * 0.05
    left_headlight[:, 1] = (left_headlight[:, 1] - 0.5) * car_width * 0.25 - car_width * 0.25
    left_headlight[:, 2] = car_height * 0.4 + left_headlight[:, 2] * car_height * 0.15
    headlights_points.append(left_headlight)
    headlights_colors.extend(['rgba(255, 255, 200, 0.9)'] * len(left_headlight))  # yellow
    headlights_part.extend(['Headlight'] * len(left_headlight))
    
    # Right headlight
    right_headlight = np.random.rand(int(num_points * 0.02), 3)
    right_headlight[:, 0] *= car_length * 0.05
    right_headlight[:, 1] = (right_headlight[:, 1] - 0.5) * car_width * 0.25 + car_width * 0.25
    right_headlight[:, 2] = car_height * 0.4 + right_headlight[:, 2] * car_height * 0.15
    headlights_points.append(right_headlight)
    headlights_colors.extend(['rgba(255, 255, 200, 0.9)'] * len(right_headlight))
    headlights_part.extend(['Headlight'] * len(right_headlight))
    
    headlights_points = np.vstack(headlights_points) if headlights_points else np.empty((0, 3))
    
    # Wheels (4 clusters of points)
    wheel_radius = 0.35
    wheel_width = 0.25
    wheel_points = []
    wheel_colors = []
    wheel_part = []
    
    wheel_positions = [
        [car_length * 0.2, car_width * 0.5, wheel_radius],       # front-right
        [car_length * 0.2, -car_width * 0.5, wheel_radius],      # front-left
        [car_length * 0.8, car_width * 0.5, wheel_radius],       # rear-right
        [car_length * 0.8, -car_width * 0.5, wheel_radius]       # rear-left
    ]
    
    for i, wheel_pos in enumerate(wheel_positions):
        # Create denser wheel points for better visualization
        wheel_pts = np.random.rand(int(num_points * 0.04), 3)
        # Convert to cylindrical coordinates for wheel shape
        theta = 2 * np.pi * wheel_pts[:, 0]
        radius = wheel_radius * np.sqrt(wheel_pts[:, 1])
        
        wx = wheel_pos[0] + wheel_width * (wheel_pts[:, 2] - 0.5)
        wy = wheel_pos[1] + radius * np.cos(theta)
        wz = radius * np.sin(theta)
        
        wheel = np.column_stack((wx, wy, wz))
        wheel_points.append(wheel)
        wheel_colors.extend(['rgba(40, 40, 40, 0.9)'] * len(wheel))  # black
        
        position = ["Front-Right", "Front-Left", "Rear-Right", "Rear-Left"][i]
        wheel_part.extend([f'{position} Wheel'] * len(wheel))
    
    wheel_points = np.vstack(wheel_points)
    
    # Combine all points, colors and part labels
    car_points = np.vstack([
        body_points, 
        roof_points, 
        windshield_points,
        rear_window_points,
        hood_points,
        trunk_points,
        headlights_points,
        wheel_points
    ])
    
    car_colors = (
        body_colors + 
        roof_colors + 
        windshield_colors +
        rear_window_colors +
        hood_colors +
        trunk_colors +
        headlights_colors +
        wheel_colors
    )
    
    car_parts = (
        body_part + 
        roof_part + 
        windshield_part +
        rear_window_part +
        hood_part +
        trunk_part +
        headlights_part +
        wheel_part
    )
    
    return car_points, car_colors, car_parts

# Create the car point cloud
car_points, car_colors, car_parts = create_car_point_cloud(12000)

# Create the interactive 3D plot with Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=car_points[:, 0],
    y=car_points[:, 1],
    z=car_points[:, 2],
    mode='markers',
    marker=dict(
        size=2.5,
        color=car_colors,
        opacity=0.8
    ),
    hoverinfo='text',
    text=car_parts,
    name='Car Point Cloud'
)])

# Configure the layout for better visualization
fig.update_layout(
    title='Interactive 3D Car Point Cloud Visualization',
    scene=dict(
        xaxis_title='X (Length)',
        yaxis_title='Y (Width)',
        zaxis_title='Z (Height)',
        aspectmode='data',  # Maintains proportions
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=0.8),  # Initial camera position
            up=dict(x=0, y=0, z=1)
        ),
        xaxis=dict(range=[0, 5]),
        yaxis=dict(range=[-1.5, 1.5]),
        zaxis=dict(range=[0, 2])
    ),
    width=900,
    height=700,
    margin=dict(l=0, r=0, b=0, t=30)
)

# Add sliders for different views
camera_positions = {
    'Front View': dict(eye=dict(x=-2, y=0, z=0.5)),
    'Top View': dict(eye=dict(x=0, y=0, z=3)),
    'Side View': dict(eye=dict(x=0, y=-2.5, z=0.5)),
    'Iso View': dict(eye=dict(x=1.5, y=1.5, z=0.8))
}

# Add buttons for camera control
button_list = []
for view_name, cam_pos in camera_positions.items():
    button = dict(
        label=view_name,
        method='relayout',
        args=['scene.camera', cam_pos]
    )
    button_list.append(button)

fig.update_layout(
    updatemenus=[
        dict(
            type='buttons',
            showactive=False,
            buttons=button_list,
            x=0.05,
            y=1.15,
            xanchor='left',
            yanchor='top'
        )
    ]
)

# Add annotations and instructions
fig.add_annotation(
    text="Drag to rotate | Scroll to zoom | Double-click to reset view",
    xref="paper", yref="paper",
    x=0.5, y=1.06,
    showarrow=False,
    font=dict(size=14)
)

# Show the figure
fig.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation

# # Set random seed for reproducibility
# np.random.seed(42)

# # Function to create a basic car shape using point clouds
# def create_car_point_cloud(num_points=5000):
#     # Car dimensions
#     car_length = 4.5  # length in meters
#     car_width = 2.0   # width in meters
#     car_height = 1.5  # height in meters
    
#     # Main body (cuboid)
#     body_points = np.random.rand(int(num_points * 0.7), 3)
#     body_points[:, 0] *= car_length        # x-axis (length)
#     body_points[:, 1] *= car_width         # y-axis (width)
#     body_points[:, 2] *= car_height * 0.7  # z-axis (height)
#     body_points[:, 1] -= car_width / 2     # Center the car on y-axis
    
#     # Roof (slightly narrower than body)
#     roof_points = np.random.rand(int(num_points * 0.15), 3)
#     roof_points[:, 0] *= car_length * 0.6      # shorter than body
#     roof_points[:, 0] += car_length * 0.15     # shifted backwards
#     roof_points[:, 1] *= car_width * 0.8       # narrower than body
#     roof_points[:, 1] -= car_width * 0.4       # Center the roof
#     roof_points[:, 2] *= car_height * 0.3      # roof height
#     roof_points[:, 2] += car_height * 0.7      # above the body
    
#     # Wheels (4 clusters of points)
#     wheel_radius = 0.35
#     wheel_width = 0.2
#     wheel_points = []
    
#     wheel_positions = [
#         [car_length * 0.2, car_width * 0.5, wheel_radius],       # front-right
#         [car_length * 0.2, -car_width * 0.5, wheel_radius],      # front-left
#         [car_length * 0.8, car_width * 0.5, wheel_radius],       # rear-right
#         [car_length * 0.8, -car_width * 0.5, wheel_radius]       # rear-left
#     ]
    
#     for wheel_pos in wheel_positions:
#         wheel_pts = np.random.rand(int(num_points * 0.0375), 3)
#         # Convert to cylindrical coordinates for wheel shape
#         theta = 2 * np.pi * wheel_pts[:, 0]
#         radius = wheel_radius * np.sqrt(wheel_pts[:, 1])
        
#         wx = wheel_pos[0] + wheel_width * (wheel_pts[:, 2] - 0.5)
#         wy = wheel_pos[1] + radius * np.cos(theta)
#         wz = radius * np.sin(theta)
        
#         wheel = np.column_stack((wx, wy, wz))
#         wheel_points.append(wheel)
    
#     # Combine all points
#     car_points = np.vstack([body_points, roof_points] + wheel_points)
    
#     # Add color information (RGB)
#     # Body is blue, roof is darker blue, wheels are black
#     colors = np.ones((car_points.shape[0], 3)) * 0.2  # Initialize with dark gray
    
#     # Body points are blue
#     body_indices = range(body_points.shape[0])
#     colors[body_indices] = np.array([0.1, 0.3, 0.8])  # Blue
    
#     # Roof points are darker blue
#     roof_indices = range(body_points.shape[0], body_points.shape[0] + roof_points.shape[0])
#     colors[roof_indices] = np.array([0.05, 0.15, 0.5])  # Darker blue
    
#     # Wheels are black
#     wheel_indices = range(body_points.shape[0] + roof_points.shape[0], car_points.shape[0])
#     colors[wheel_indices] = np.array([0.1, 0.1, 0.1])  # Black
    
#     return car_points, colors

# # Create the car point cloud
# car_points, car_colors = create_car_point_cloud(8000)

# # Create 3D plot
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot the car point cloud
# scatter = ax.scatter(
#     car_points[:, 0], car_points[:, 1], car_points[:, 2],
#     c=car_colors, s=5, alpha=0.8
# )

# # Set plot limits and labels
# ax.set_xlim(0, 5)
# ax.set_ylim(-1.5, 1.5)
# ax.set_zlim(0, 2)
# ax.set_xlabel('X (length)')
# ax.set_ylabel('Y (width)')
# ax.set_zlabel('Z (height)')
# ax.set_title('3D Car Point Cloud Visualization')

# # Set equal aspect ratio
# ax.set_box_aspect([5/3, 3/3, 2/3])  # Based on the plot limits

# # Function to update the plot for animation
# def update(frame):
#     ax.view_init(elev=20, azim=frame)
#     return scatter,

# # Create animation
# ani = FuncAnimation(fig, update, frames=range(0, 360, 2), interval=50, blit=True)

# plt.tight_layout()
# plt.show()

# # If you want to save the animation (requires ffmpeg or imagemagick)
# # ani.save('car_point_cloud_rotation.gif', writer='pillow', fps=20, dpi=100)
