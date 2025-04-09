import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# Set random seed for reproducibility
np.random.seed(42)

def create_realistic_car_point_cloud(num_points=20000):
    # Car dimensions (sedan-like proportions)
    car_length = 4.8     # length in meters
    car_width = 1.8      # width in meters
    car_height = 1.4     # height in meters
    wheel_radius = 0.35  # wheel radius
    
    # Function to generate points on a parametric surface 
    def parametric_surface(u_range, v_range, u_points, v_points, function):
        points = []
        for i in range(u_points):
            u = u_range[0] + (u_range[1] - u_range[0]) * i / (u_points - 1)
            for j in range(v_points):
                v = v_range[0] + (v_range[1] - v_range[0]) * j / (v_points - 1)
                point = function(u, v)
                points.append(point)
        return np.array(points)
    
    # Function to add random jitter to points (for point cloud effect)
    def add_jitter(points, scale=0.02):
        jitter = np.random.normal(0, scale, points.shape)
        return points + jitter
    
    # === BODY CREATION ===
    # Base body shape using a modified superellipsoid function
    def body_function(u, v):
        # u controls length (0 = front, 1 = back)
        # v controls the angle around the central axis (0 = bottom, 0.5 = top, 1 = bottom)
        
        # Different cross-section profiles for different parts of the car
        if u < 0.2:  # Front (nose)
            x = u * car_length
            width_factor = 0.7 + u * 1.5  # Gradually widens
            height_factor = 0.6 + u * 2.0  # Gradually gets taller
            z_offset = 0.2  # Lower front profile
        elif u < 0.3:  # Hood transition
            x = u * car_length
            width_factor = 1.0
            height_factor = 1.0
            z_offset = 0.3 + (u - 0.2) * 3.0  # Rising to cabin height
        elif u < 0.7:  # Cabin
            x = u * car_length
            width_factor = 1.0
            height_factor = 1.0
            z_offset = 0.6
        elif u < 0.85:  # Trunk transition
            x = u * car_length
            width_factor = 1.0
            height_factor = 1.0 - (u - 0.7) * 0.6  # Gradually lowers
            z_offset = 0.6 - (u - 0.7) * 0.3  # Gradually lowers
        else:  # Rear
            x = u * car_length
            width_factor = 1.0 - (u - 0.85) * 1.5  # Gradually narrows
            height_factor = 0.7 - (u - 0.85) * 1.0  # Gradually lowers
            z_offset = 0.45
        
        # Calculate z based on a parametric curve for the roof profile
        if v < 0.25:  # Bottom front quarter
            y = (v * 4) * car_width * width_factor / 2 - car_width * width_factor / 2
            z = (0.5 - abs(v * 4 - 0.5)) * car_height * 0.4 * height_factor + z_offset * car_height
        elif v < 0.5:  # Bottom rear quarter
            y = (0.5 - (v - 0.25) * 4) * car_width * width_factor / 2 - car_width * width_factor / 2
            z = (0.5 - abs((v - 0.25) * 4 - 0.5)) * car_height * 0.4 * height_factor + z_offset * car_height
        elif v < 0.75:  # Top rear quarter
            y = ((v - 0.5) * 4) * car_width * width_factor / 2
            
            # Roof profile adjustment
            if 0.3 < u < 0.7:  # Cabin section
                base_height = car_height * 0.95  # Highest part of roof
            else:
                # Smooth transition from front to cabin and cabin to rear
                if u <= 0.3:
                    transition = u / 0.3
                    base_height = car_height * (0.6 + transition * 0.35)
                else:  # u >= 0.7
                    transition = (u - 0.7) / 0.3
                    base_height = car_height * (0.95 - transition * 0.3)
            
            z = (0.5 - abs((v - 0.5) * 4 - 0.5)) * base_height * height_factor + z_offset * 0.5 * car_height
        else:  # Top front quarter
            y = (1.0 - (v - 0.75) * 4) * car_width * width_factor / 2
            
            # Same roof profile adjustment as above
            if 0.3 < u < 0.7:
                base_height = car_height * 0.95
            else:
                if u <= 0.3:
                    transition = u / 0.3
                    base_height = car_height * (0.6 + transition * 0.35)
                else:
                    transition = (u - 0.7) / 0.3
                    base_height = car_height * (0.95 - transition * 0.3)
                    
            z = (0.5 - abs((v - 0.75) * 4 - 0.5)) * base_height * height_factor + z_offset * 0.5 * car_height
        
        return [x, y, z]
    
    # Generate the main car body
    body_density = int(np.sqrt(num_points * 0.5))  # Adjust density based on total points
    body_points = parametric_surface(
        [0, 1], [0, 1], 
        body_density, body_density, 
        body_function
    )
    
    # Add some random points inside the body for solidity
    interior_body_points = []
    num_interior = int(num_points * 0.1)
    for _ in range(num_interior):
        u = np.random.uniform(0, 1)
        v = np.random.uniform(0, 1)
        scale = np.random.uniform(0.5, 0.95)  # Scale to keep points inside
        point = body_function(u, v)
        # Move point toward center
        centroid = [point[0], 0, point[2] * 0.8]
        interior_point = [
            point[0],
            point[1] * scale,
            point[2] * scale + (1-scale) * centroid[2]
        ]
        interior_body_points.append(interior_point)
    
    interior_body_points = np.array(interior_body_points)
    
    # === WINDOWS ===
    # Windshield
    def windshield_function(u, v):
        # u from 0 (bottom) to 1 (top)
        # v from 0 (left) to 1 (right)
        x = (0.25 + u * 0.1) * car_length
        y = (v - 0.5) * car_width * 0.85
        z = (0.6 + u * 0.25) * car_height
        return [x, y, z]
    
    windshield_points = parametric_surface(
        [0, 1], [0, 1],
        int(np.sqrt(num_points * 0.03)), int(np.sqrt(num_points * 0.03)),
        windshield_function
    )
    
    # Rear window
    def rear_window_function(u, v):
        # u from 0 (top) to 1 (bottom)
        # v from 0 (left) to 1 (right)
        x = (0.65 + u * 0.1) * car_length
        y = (v - 0.5) * car_width * 0.85
        z = (0.85 - u * 0.2) * car_height
        return [x, y, z]
    
    rear_window_points = parametric_surface(
        [0, 1], [0, 1],
        int(np.sqrt(num_points * 0.02)), int(np.sqrt(num_points * 0.02)),
        rear_window_function
    )
    
    # Side windows (both sides)
    def side_window_function(u, v, side):
        # u from 0 (front) to 1 (back)
        # v from 0 (bottom) to 1 (top)
        x = (0.35 + u * 0.3) * car_length
        y = (0.43 + v * 0.12) * car_width * side  # side = 1 for right, -1 for left
        z = (0.65 + v * 0.15) * car_height
        return [x, y, z]
    
    side_window_right = parametric_surface(
        [0, 1], [0, 1],
        int(np.sqrt(num_points * 0.015)), int(np.sqrt(num_points * 0.015)),
        lambda u, v: side_window_function(u, v, 1)
    )
    
    side_window_left = parametric_surface(
        [0, 1], [0, 1],
        int(np.sqrt(num_points * 0.015)), int(np.sqrt(num_points * 0.015)),
        lambda u, v: side_window_function(u, v, -1)
    )
    
    # === WHEELS ===
    def wheel_function(u, v, wheel_pos):
        # u controls rotation (0 to 2π)
        # v controls radius (0 to wheel_radius)
        theta = u * 2 * np.pi
        rad = v * wheel_radius
        
        # Calculate position on wheel
        wx = wheel_pos[0] + 0.12 * np.cos(theta)  # wheel width = 0.24m
        wy = wheel_pos[1] + rad * np.sin(theta)
        wz = wheel_pos[2] + rad * np.cos(theta)
        
        return [wx, wy, wz]
    
    wheel_positions = [
        [car_length * 0.2, car_width * 0.5, wheel_radius],        # front-right
        [car_length * 0.2, -car_width * 0.5, wheel_radius],       # front-left
        [car_length * 0.8, car_width * 0.5, wheel_radius],        # rear-right
        [car_length * 0.8, -car_width * 0.5, wheel_radius]        # rear-left
    ]
    
    wheel_density = int(np.sqrt(num_points * 0.03))
    wheels_points = []
    
    for pos in wheel_positions:
        wheel = parametric_surface(
            [0, 1], [0.7, 1],  # Only outer part of wheel
            wheel_density, int(wheel_density/3),
            lambda u, v: wheel_function(u, v, pos)
        )
        
        # Add wheel face
        for i in range(wheel_density):
            theta = i * 2 * np.pi / wheel_density
            for j in range(int(wheel_density/5)):
                r = j * wheel_radius / (wheel_density/5) + 0.1  # Start from inner hub
                x = pos[0] + (0.12 if i % 2 == 0 else -0.12)  # Alternate between front and back face
                y = pos[1] + r * np.sin(theta)
                z = pos[2] + r * np.cos(theta)
                wheels_points.append([x, y, z])
                
        wheels_points.extend(wheel)
    
    wheels_points = np.array(wheels_points)
    
    # === DETAILS ===
    # Headlights
    headlights_points = []
    
    # Left headlight
    for _ in range(int(num_points * 0.02)):
        x = np.random.uniform(0.02, 0.06) * car_length
        y = np.random.uniform(-0.45, -0.35) * car_width
        z = np.random.uniform(0.4, 0.5) * car_height
        headlights_points.append([x, y, z])
        
    # Right headlight
    for _ in range(int(num_points * 0.02)):
        x = np.random.uniform(0.02, 0.06) * car_length
        y = np.random.uniform(0.35, 0.45) * car_width
        z = np.random.uniform(0.4, 0.5) * car_height
        headlights_points.append([x, y, z])
    
    headlights_points = np.array(headlights_points)
    
    # Taillights
    taillights_points = []
    
    # Left taillight
    for _ in range(int(num_points * 0.02)):
        x = np.random.uniform(0.96, 0.99) * car_length
        y = np.random.uniform(-0.45, -0.35) * car_width
        z = np.random.uniform(0.4, 0.5) * car_height
        taillights_points.append([x, y, z])
        
    # Right taillight
    for _ in range(int(num_points * 0.02)):
        x = np.random.uniform(0.96, 0.99) * car_length
        y = np.random.uniform(0.35, 0.45) * car_width
        z = np.random.uniform(0.4, 0.5) * car_height
        taillights_points.append([x, y, z])
    
    taillights_points = np.array(taillights_points)
    
    # Grill (front)
    grill_points = []
    for _ in range(int(num_points * 0.01)):
        x = np.random.uniform(0.01, 0.03) * car_length  # Very front
        y = np.random.uniform(-0.3, 0.3) * car_width
        z = np.random.uniform(0.3, 0.4) * car_height
        grill_points.append([x, y, z])
    
    grill_points = np.array(grill_points)
    
    # Door handles
    handles_points = []
    
    # Left door handle
    for _ in range(int(num_points * 0.005)):
        x = np.random.uniform(0.4, 0.45) * car_length
        y = np.random.uniform(-0.51, -0.49) * car_width
        z = np.random.uniform(0.53, 0.57) * car_height
        handles_points.append([x, y, z])
        
    # Right door handle
    for _ in range(int(num_points * 0.005)):
        x = np.random.uniform(0.4, 0.45) * car_length
        y = np.random.uniform(0.49, 0.51) * car_width
        z = np.random.uniform(0.53, 0.57) * car_height
        handles_points.append([x, y, z])
    
    handles_points = np.array(handles_points)
    
    # Mirrors
    mirrors_points = []
    
    # Left mirror
    for _ in range(int(num_points * 0.007)):
        x = np.random.uniform(0.3, 0.35) * car_length
        y = np.random.uniform(-0.55, -0.51) * car_width
        z = np.random.uniform(0.57, 0.63) * car_height
        mirrors_points.append([x, y, z])
        
    # Right mirror
    for _ in range(int(num_points * 0.007)):
        x = np.random.uniform(0.3, 0.35) * car_length
        y = np.random.uniform(0.51, 0.55) * car_width
        z = np.random.uniform(0.57, 0.63) * car_height
        mirrors_points.append([x, y, z])
    
    mirrors_points = np.array(mirrors_points)
    
    # === COMBINE ALL PARTS ===
    # Add some random jitter to make it look more like a point cloud
    body_points = add_jitter(body_points, 0.01)
    interior_body_points = add_jitter(interior_body_points, 0.01)
    windshield_points = add_jitter(windshield_points, 0.005)
    rear_window_points = add_jitter(rear_window_points, 0.005)
    side_window_left = add_jitter(side_window_left, 0.005)
    side_window_right = add_jitter(side_window_right, 0.005)
    wheels_points = add_jitter(wheels_points, 0.005)
    headlights_points = add_jitter(headlights_points, 0.003)
    taillights_points = add_jitter(taillights_points, 0.003)
    grill_points = add_jitter(grill_points, 0.003)
    handles_points = add_jitter(handles_points, 0.002)
    mirrors_points = add_jitter(mirrors_points, 0.003)
    
    # Define colors and part names
    body_colors = ['rgba(30, 45, 110, 0.8)'] * len(body_points)  # Deep blue for body
    interior_colors = ['rgba(25, 35, 90, 0.7)'] * len(interior_body_points)  # Darker blue for interior
    windshield_colors = ['rgba(180, 230, 255, 0.7)'] * len(windshield_points)  # Light blue for glass
    rear_window_colors = ['rgba(180, 230, 255, 0.7)'] * len(rear_window_points)
    side_window_left_colors = ['rgba(180, 230, 255, 0.7)'] * len(side_window_left)
    side_window_right_colors = ['rgba(180, 230, 255, 0.7)'] * len(side_window_right)
    wheels_colors = ['rgba(30, 30, 30, 0.9)'] * len(wheels_points)  # Dark for wheels
    headlights_colors = ['rgba(255, 255, 220, 0.9)'] * len(headlights_points)  # Yellow-white for headlights
    taillights_colors = ['rgba(255, 50, 50, 0.9)'] * len(taillights_points)  # Red for taillights
    grill_colors = ['rgba(50, 50, 50, 0.9)'] * len(grill_points)  # Dark for grill
    handles_colors = ['rgba(200, 200, 200, 0.9)'] * len(handles_points)  # Silver for handles
    mirrors_colors = ['rgba(200, 200, 200, 0.9)'] * len(mirrors_points)  # Silver for mirrors
    
    body_part = ['Body'] * len(body_points)
    interior_part = ['Interior'] * len(interior_body_points)
    windshield_part = ['Windshield'] * len(windshield_points)
    rear_window_part = ['Rear Window'] * len(rear_window_points)
    side_window_left_part = ['Left Window'] * len(side_window_left)
    side_window_right_part = ['Right Window'] * len(side_window_right)
    wheels_part = ['Wheels'] * len(wheels_points)
    headlights_part = ['Headlights'] * len(headlights_points)
    taillights_part = ['Taillights'] * len(taillights_points)
    grill_part = ['Grill'] * len(grill_points)
    handles_part = ['Door Handles'] * len(handles_points)
    mirrors_part = ['Side Mirrors'] * len(mirrors_points)
    
    # Combine all parts
    all_points = np.vstack([
        body_points,
        interior_body_points,
        windshield_points,
        rear_window_points,
        side_window_left,
        side_window_right,
        wheels_points,
        headlights_points,
        taillights_points,
        grill_points,
        handles_points,
        mirrors_points
    ])
    
    all_colors = (
        body_colors +
        interior_colors +
        windshield_colors +
        rear_window_colors +
        side_window_left_colors +
        side_window_right_colors +
        wheels_colors +
        headlights_colors +
        taillights_colors +
        grill_colors +
        handles_colors +
        mirrors_colors
    )
    
    all_parts = (
        body_part +
        interior_part +
        windshield_part +
        rear_window_part +
        side_window_left_part +
        side_window_right_part +
        wheels_part +
        headlights_part +
        taillights_part +
        grill_part +
        handles_part +
        mirrors_part
    )
    
    # Optional: Add ground reflection for visual effect
    ground_reflection_points = []
    ground_reflection_colors = []
    ground_reflection_parts = []
    
    # Sample a subset of the car points for reflection
    reflection_indices = np.random.choice(len(all_points), size=int(len(all_points)*0.3), replace=False)
    
    for idx in reflection_indices:
        reflected_point = all_points[idx].copy()
        reflected_point[2] = -reflected_point[2] * 0.06  # Flatten and invert
        ground_reflection_points.append(reflected_point)
        
        # Make reflection more transparent and darker
        orig_color = all_colors[idx]
        if orig_color.startswith('rgba('):
            r, g, b, a = map(float, orig_color.strip('rgba()').split(','))
            refl_color = f'rgba({r*0.4:.0f},{g*0.4:.0f},{b*0.4:.0f},0.15)'
        else:
            refl_color = 'rgba(20,20,20,0.1)'
            
        ground_reflection_colors.append(refl_color)
        ground_reflection_parts.append('Ground Reflection')
    
    # Add ground plane under the car
    ground_size = max(car_length * 1.5, car_width * 2.5)
    ground_points = []
    
    for _ in range(int(num_points * 0.2)):
        gx = np.random.uniform(-car_length * 0.2, car_length * 1.2)
        gy = np.random.uniform(-car_width * 1.2, car_width * 1.2)
        gz = 0.001 * np.random.normal()  # Very slight variation for visual effect
        ground_points.append([gx, gy, gz])
    
    ground_points = np.array(ground_points)
    ground_colors = ['rgba(50,50,50,0.15)'] * len(ground_points)
    ground_part = ['Ground'] * len(ground_points)
    
    # Add reflections and ground to final points
    final_points = np.vstack([
        all_points,
        np.array(ground_reflection_points),
        ground_points
    ])
    
    final_colors = all_colors + ground_reflection_colors + ground_colors
    final_parts = all_parts + ground_reflection_parts + ground_part
    
    return final_points, final_colors, final_parts

# Create the realistic car point cloud
car_points, car_colors, car_parts = create_realistic_car_point_cloud(25000)

# Create the interactive 3D plot with Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=car_points[:, 0],
    y=car_points[:, 1],
    z=car_points[:, 2],
    mode='markers',
    marker=dict(
        size=2,
        color=car_colors,
        opacity=0.8
    ),
    hoverinfo='text',
    text=car_parts,
    name='Car Point Cloud'
)])

# Configure the layout for better visualization
fig.update_layout(
    title='Realistic 3D Car Point Cloud Visualization',
    scene=dict(
        xaxis_title='X (Length)',
        yaxis_title='Y (Width)',
        zaxis_title='Z (Height)',
        aspectmode='data',  # Maintains proportions
        camera=dict(
            eye=dict(x=2.2, y=2.2, z=1.0),  # Initial camera position
            up=dict(x=0, y=0, z=1)
        ),
        xaxis=dict(range=[-1, 6]),
        yaxis=dict(range=[-2, 2]),
        zaxis=dict(range=[-0.5, 2.5])
    ),
    width=900,
    height=700,
    margin=dict(l=0, r=0, b=0, t=30)
)

# Add camera position buttons
camera_positions = {
    'Front View': dict(eye=dict(x=-3, y=0, z=0.8)),
    'Rear View': dict(eye=dict(x=7, y=0, z=0.8)),
    'Side View': dict(eye=dict(x=0, y=-3.5, z=0.8)),
    'Top View': dict(eye=dict(x=0, y=0, z=5)),
    '45° View': dict(eye=dict(x=2.2, y=2.2, z=1.8)),
    'Low Angle': dict(eye=dict(x=3, y=3, z=0.3))
}

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
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# # Set random seed for reproducibility
# np.random.seed(42)

# # Function to create a basic car shape using point clouds
# def create_car_point_cloud(num_points=8000):
#     # Car dimensions
#     car_length = 4.5  # length in meters
#     car_width = 2.0   # width in meters
#     car_height = 1.5  # height in meters
    
#     # Main body (cuboid)
#     body_points = np.random.rand(int(num_points * 0.6), 3)
#     body_points[:, 0] *= car_length        # x-axis (length)
#     body_points[:, 1] *= car_width         # y-axis (width)
#     body_points[:, 2] *= car_height * 0.7  # z-axis (height)
#     body_points[:, 1] -= car_width / 2     # Center the car on y-axis
#     body_colors = ['rgba(30, 100, 200, 0.8)'] * len(body_points)  # blue
#     body_part = ['Body'] * len(body_points)
    
#     # Windshield (front)
#     windshield_points = np.random.rand(int(num_points * 0.05), 3)
#     windshield_points[:, 0] = car_length * 0.25 + windshield_points[:, 0] * car_length * 0.15
#     windshield_points[:, 1] = (windshield_points[:, 1] - 0.5) * car_width * 0.8
#     windshield_points[:, 2] = car_height * 0.6 + windshield_points[:, 2] * car_height * 0.3
#     windshield_colors = ['rgba(150, 200, 255, 0.7)'] * len(windshield_points)  # light blue transparent
#     windshield_part = ['Windshield'] * len(windshield_points)
    
#     # Rear window
#     rear_window_points = np.random.rand(int(num_points * 0.05), 3)
#     rear_window_points[:, 0] = car_length * 0.75 - rear_window_points[:, 0] * car_length * 0.15
#     rear_window_points[:, 1] = (rear_window_points[:, 1] - 0.5) * car_width * 0.8
#     rear_window_points[:, 2] = car_height * 0.6 + rear_window_points[:, 2] * car_height * 0.3
#     rear_window_colors = ['rgba(150, 200, 255, 0.7)'] * len(rear_window_points)
#     rear_window_part = ['Rear Window'] * len(rear_window_points)
    
#     # Roof (slightly narrower than body)
#     roof_points = np.random.rand(int(num_points * 0.15), 3)
#     roof_points[:, 0] *= car_length * 0.5      # shorter than body
#     roof_points[:, 0] += car_length * 0.25     # shifted to middle
#     roof_points[:, 1] *= car_width * 0.8       # narrower than body
#     roof_points[:, 1] -= car_width * 0.4       # Center the roof
#     roof_points[:, 2] *= car_height * 0.3      # roof height
#     roof_points[:, 2] += car_height * 0.7      # above the body
#     roof_colors = ['rgba(20, 70, 150, 0.9)'] * len(roof_points)  # darker blue
#     roof_part = ['Roof'] * len(roof_points)
    
#     # Hood
#     hood_points = np.random.rand(int(num_points * 0.05), 3)
#     hood_points[:, 0] *= car_length * 0.25     # front quarter of car
#     hood_points[:, 1] *= car_width * 0.9      
#     hood_points[:, 1] -= car_width * 0.45      
#     hood_points[:, 2] *= car_height * 0.05     # thin layer
#     hood_points[:, 2] += car_height * 0.55     # at height of hood
#     hood_colors = ['rgba(40, 110, 210, 0.85)'] * len(hood_points)  # slightly different blue
#     hood_part = ['Hood'] * len(hood_points)
    
#     # Trunk
#     trunk_points = np.random.rand(int(num_points * 0.05), 3)
#     trunk_points[:, 0] *= car_length * 0.25    # rear quarter of car
#     trunk_points[:, 0] += car_length * 0.75    # start from 3/4 mark
#     trunk_points[:, 1] *= car_width * 0.9
#     trunk_points[:, 1] -= car_width * 0.45
#     trunk_points[:, 2] *= car_height * 0.05    # thin layer
#     trunk_points[:, 2] += car_height * 0.55    # at height of trunk
#     trunk_colors = ['rgba(40, 110, 210, 0.85)'] * len(trunk_points)
#     trunk_part = ['Trunk'] * len(trunk_points)
    
#     # Headlights
#     headlights_points = []
#     headlights_colors = []
#     headlights_part = []
    
#     # Left headlight
#     left_headlight = np.random.rand(int(num_points * 0.02), 3)
#     left_headlight[:, 0] *= car_length * 0.05
#     left_headlight[:, 1] = (left_headlight[:, 1] - 0.5) * car_width * 0.25 - car_width * 0.25
#     left_headlight[:, 2] = car_height * 0.4 + left_headlight[:, 2] * car_height * 0.15
#     headlights_points.append(left_headlight)
#     headlights_colors.extend(['rgba(255, 255, 200, 0.9)'] * len(left_headlight))  # yellow
#     headlights_part.extend(['Headlight'] * len(left_headlight))
    
#     # Right headlight
#     right_headlight = np.random.rand(int(num_points * 0.02), 3)
#     right_headlight[:, 0] *= car_length * 0.05
#     right_headlight[:, 1] = (right_headlight[:, 1] - 0.5) * car_width * 0.25 + car_width * 0.25
#     right_headlight[:, 2] = car_height * 0.4 + right_headlight[:, 2] * car_height * 0.15
#     headlights_points.append(right_headlight)
#     headlights_colors.extend(['rgba(255, 255, 200, 0.9)'] * len(right_headlight))
#     headlights_part.extend(['Headlight'] * len(right_headlight))
    
#     headlights_points = np.vstack(headlights_points) if headlights_points else np.empty((0, 3))
    
#     # Wheels (4 clusters of points)
#     wheel_radius = 0.35
#     wheel_width = 0.25
#     wheel_points = []
#     wheel_colors = []
#     wheel_part = []
    
#     wheel_positions = [
#         [car_length * 0.2, car_width * 0.5, wheel_radius],       # front-right
#         [car_length * 0.2, -car_width * 0.5, wheel_radius],      # front-left
#         [car_length * 0.8, car_width * 0.5, wheel_radius],       # rear-right
#         [car_length * 0.8, -car_width * 0.5, wheel_radius]       # rear-left
#     ]
    
#     for i, wheel_pos in enumerate(wheel_positions):
#         # Create denser wheel points for better visualization
#         wheel_pts = np.random.rand(int(num_points * 0.04), 3)
#         # Convert to cylindrical coordinates for wheel shape
#         theta = 2 * np.pi * wheel_pts[:, 0]
#         radius = wheel_radius * np.sqrt(wheel_pts[:, 1])
        
#         wx = wheel_pos[0] + wheel_width * (wheel_pts[:, 2] - 0.5)
#         wy = wheel_pos[1] + radius * np.cos(theta)
#         wz = radius * np.sin(theta)
        
#         wheel = np.column_stack((wx, wy, wz))
#         wheel_points.append(wheel)
#         wheel_colors.extend(['rgba(40, 40, 40, 0.9)'] * len(wheel))  # black
        
#         position = ["Front-Right", "Front-Left", "Rear-Right", "Rear-Left"][i]
#         wheel_part.extend([f'{position} Wheel'] * len(wheel))
    
#     wheel_points = np.vstack(wheel_points)
    
#     # Combine all points, colors and part labels
#     car_points = np.vstack([
#         body_points, 
#         roof_points, 
#         windshield_points,
#         rear_window_points,
#         hood_points,
#         trunk_points,
#         headlights_points,
#         wheel_points
#     ])
    
#     car_colors = (
#         body_colors + 
#         roof_colors + 
#         windshield_colors +
#         rear_window_colors +
#         hood_colors +
#         trunk_colors +
#         headlights_colors +
#         wheel_colors
#     )
    
#     car_parts = (
#         body_part + 
#         roof_part + 
#         windshield_part +
#         rear_window_part +
#         hood_part +
#         trunk_part +
#         headlights_part +
#         wheel_part
#     )
    
#     return car_points, car_colors, car_parts

# # Create the car point cloud
# car_points, car_colors, car_parts = create_car_point_cloud(12000)

# # Create the interactive 3D plot with Plotly
# fig = go.Figure(data=[go.Scatter3d(
#     x=car_points[:, 0],
#     y=car_points[:, 1],
#     z=car_points[:, 2],
#     mode='markers',
#     marker=dict(
#         size=2.5,
#         color=car_colors,
#         opacity=0.8
#     ),
#     hoverinfo='text',
#     text=car_parts,
#     name='Car Point Cloud'
# )])

# # Configure the layout for better visualization
# fig.update_layout(
#     title='Interactive 3D Car Point Cloud Visualization',
#     scene=dict(
#         xaxis_title='X (Length)',
#         yaxis_title='Y (Width)',
#         zaxis_title='Z (Height)',
#         aspectmode='data',  # Maintains proportions
#         camera=dict(
#             eye=dict(x=1.5, y=1.5, z=0.8),  # Initial camera position
#             up=dict(x=0, y=0, z=1)
#         ),
#         xaxis=dict(range=[0, 5]),
#         yaxis=dict(range=[-1.5, 1.5]),
#         zaxis=dict(range=[0, 2])
#     ),
#     width=900,
#     height=700,
#     margin=dict(l=0, r=0, b=0, t=30)
# )

# # Add sliders for different views
# camera_positions = {
#     'Front View': dict(eye=dict(x=-2, y=0, z=0.5)),
#     'Top View': dict(eye=dict(x=0, y=0, z=3)),
#     'Side View': dict(eye=dict(x=0, y=-2.5, z=0.5)),
#     'Iso View': dict(eye=dict(x=1.5, y=1.5, z=0.8))
# }

# # Add buttons for camera control
# button_list = []
# for view_name, cam_pos in camera_positions.items():
#     button = dict(
#         label=view_name,
#         method='relayout',
#         args=['scene.camera', cam_pos]
#     )
#     button_list.append(button)

# fig.update_layout(
#     updatemenus=[
#         dict(
#             type='buttons',
#             showactive=False,
#             buttons=button_list,
#             x=0.05,
#             y=1.15,
#             xanchor='left',
#             yanchor='top'
#         )
#     ]
# )

# # Add annotations and instructions
# fig.add_annotation(
#     text="Drag to rotate | Scroll to zoom | Double-click to reset view",
#     xref="paper", yref="paper",
#     x=0.5, y=1.06,
#     showarrow=False,
#     font=dict(size=14)
# )

# # Show the figure
# fig.show()


# # import numpy as np
# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D
# # from matplotlib.animation import FuncAnimation

# # # Set random seed for reproducibility
# # np.random.seed(42)

# # # Function to create a basic car shape using point clouds
# # def create_car_point_cloud(num_points=5000):
# #     # Car dimensions
# #     car_length = 4.5  # length in meters
# #     car_width = 2.0   # width in meters
# #     car_height = 1.5  # height in meters
    
# #     # Main body (cuboid)
# #     body_points = np.random.rand(int(num_points * 0.7), 3)
# #     body_points[:, 0] *= car_length        # x-axis (length)
# #     body_points[:, 1] *= car_width         # y-axis (width)
# #     body_points[:, 2] *= car_height * 0.7  # z-axis (height)
# #     body_points[:, 1] -= car_width / 2     # Center the car on y-axis
    
# #     # Roof (slightly narrower than body)
# #     roof_points = np.random.rand(int(num_points * 0.15), 3)
# #     roof_points[:, 0] *= car_length * 0.6      # shorter than body
# #     roof_points[:, 0] += car_length * 0.15     # shifted backwards
# #     roof_points[:, 1] *= car_width * 0.8       # narrower than body
# #     roof_points[:, 1] -= car_width * 0.4       # Center the roof
# #     roof_points[:, 2] *= car_height * 0.3      # roof height
# #     roof_points[:, 2] += car_height * 0.7      # above the body
    
# #     # Wheels (4 clusters of points)
# #     wheel_radius = 0.35
# #     wheel_width = 0.2
# #     wheel_points = []
    
# #     wheel_positions = [
# #         [car_length * 0.2, car_width * 0.5, wheel_radius],       # front-right
# #         [car_length * 0.2, -car_width * 0.5, wheel_radius],      # front-left
# #         [car_length * 0.8, car_width * 0.5, wheel_radius],       # rear-right
# #         [car_length * 0.8, -car_width * 0.5, wheel_radius]       # rear-left
# #     ]
    
# #     for wheel_pos in wheel_positions:
# #         wheel_pts = np.random.rand(int(num_points * 0.0375), 3)
# #         # Convert to cylindrical coordinates for wheel shape
# #         theta = 2 * np.pi * wheel_pts[:, 0]
# #         radius = wheel_radius * np.sqrt(wheel_pts[:, 1])
        
# #         wx = wheel_pos[0] + wheel_width * (wheel_pts[:, 2] - 0.5)
# #         wy = wheel_pos[1] + radius * np.cos(theta)
# #         wz = radius * np.sin(theta)
        
# #         wheel = np.column_stack((wx, wy, wz))
# #         wheel_points.append(wheel)
    
# #     # Combine all points
# #     car_points = np.vstack([body_points, roof_points] + wheel_points)
    
# #     # Add color information (RGB)
# #     # Body is blue, roof is darker blue, wheels are black
# #     colors = np.ones((car_points.shape[0], 3)) * 0.2  # Initialize with dark gray
    
# #     # Body points are blue
# #     body_indices = range(body_points.shape[0])
# #     colors[body_indices] = np.array([0.1, 0.3, 0.8])  # Blue
    
# #     # Roof points are darker blue
# #     roof_indices = range(body_points.shape[0], body_points.shape[0] + roof_points.shape[0])
# #     colors[roof_indices] = np.array([0.05, 0.15, 0.5])  # Darker blue
    
# #     # Wheels are black
# #     wheel_indices = range(body_points.shape[0] + roof_points.shape[0], car_points.shape[0])
# #     colors[wheel_indices] = np.array([0.1, 0.1, 0.1])  # Black
    
# #     return car_points, colors

# # # Create the car point cloud
# # car_points, car_colors = create_car_point_cloud(8000)

# # # Create 3D plot
# # fig = plt.figure(figsize=(12, 8))
# # ax = fig.add_subplot(111, projection='3d')

# # # Plot the car point cloud
# # scatter = ax.scatter(
# #     car_points[:, 0], car_points[:, 1], car_points[:, 2],
# #     c=car_colors, s=5, alpha=0.8
# # )

# # # Set plot limits and labels
# # ax.set_xlim(0, 5)
# # ax.set_ylim(-1.5, 1.5)
# # ax.set_zlim(0, 2)
# # ax.set_xlabel('X (length)')
# # ax.set_ylabel('Y (width)')
# # ax.set_zlabel('Z (height)')
# # ax.set_title('3D Car Point Cloud Visualization')

# # # Set equal aspect ratio
# # ax.set_box_aspect([5/3, 3/3, 2/3])  # Based on the plot limits

# # # Function to update the plot for animation
# # def update(frame):
# #     ax.view_init(elev=20, azim=frame)
# #     return scatter,

# # # Create animation
# # ani = FuncAnimation(fig, update, frames=range(0, 360, 2), interval=50, blit=True)

# # plt.tight_layout()
# # plt.show()

# # # If you want to save the animation (requires ffmpeg or imagemagick)
# # # ani.save('car_point_cloud_rotation.gif', writer='pillow', fps=20, dpi=100)
