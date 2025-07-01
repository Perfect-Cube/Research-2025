import pyrealsense2 as rs
import numpy as np
import cv2

# --- User-Configurable Parameters ---
# This is the most important parameter to adjust.
# It defines the maximum distance (in meters) for the objects you want to capture.
# Objects closer than this distance will be in the "blue range" of the colormap.
MAX_DISTANCE_METERS = 0.75 

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Create an align object
# rs.align allows us to perform alignment of depth frames to other frames
align_to = rs.stream.color
align = rs.align(align_to)

# Create a pointcloud object
pc = rs.pointcloud()

# --- Initialize the Threshold Filter ---
# This filter will remove depth data outside a given range
threshold = rs.threshold_filter()
# Set the range of depth values to keep (in meters)
# We start from 0.1m to avoid noise from the camera lens
threshold.set_option(rs.option.min_distance, 0.1) 
threshold.set_option(rs.option.max_distance, MAX_DISTANCE_METERS)

print(f"Capturing points up to {MAX_DISTANCE_METERS} meters away (the 'blue' region).")
print("Press 's' to save the filtered point cloud to a .ply file.")
print("Press 'q' to quit.")

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        
        # Align the depth frame to the color frame
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        # --- Apply the Threshold Filter ---
        # This will zero out all depth pixels outside the specified range
        filtered_depth_frame = threshold.process(depth_frame)

        # Convert images to numpy arrays for visualization
        # We use the filtered depth frame to show exactly what will be saved
        depth_image = np.asanyarray(filtered_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on the filtered depth image (for visualization)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        # The background of the depth image will be black, representing discarded points
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense - Filtered Depth', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense - Filtered Depth', images)
        key = cv2.waitKey(1)

        # Press 's' to save the point cloud
        if key & 0xFF == ord('s'):
            print("Generating and saving filtered point cloud...")
            
            # Generate point cloud from the *filtered* depth frame
            pc.map_to(color_frame)
            points = pc.calculate(filtered_depth_frame)
            
            print("Saving to pointcloud-filtered.ply...")
            points.export_to_ply("pointcloud-filtered.ply", color_frame)
            print("Done.")
        
        # Press 'q' or Esc to close the image window
        elif key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    # Stop streaming
    pipeline.stop()