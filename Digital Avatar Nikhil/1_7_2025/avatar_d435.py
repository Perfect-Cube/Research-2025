import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import time
import copy

# --- 1. Configuration and Setup ---

class RealsenseCamera:
    """A helper class to manage the Realsense camera stream."""
    def __init__(self, width=640, height=480, fps=30):
        self.width, self.height, self.fps = width, height, fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        self.align = rs.align(rs.stream.color)

    def start(self):
        """Starts the camera pipeline and gets intrinsics."""
        profile = self.pipeline.start(self.config)
        
        # Get camera intrinsics for Open3D
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
        )
        print("Camera started. Intrinsics captured.")
        # Allow auto-exposure to settle
        time.sleep(2)

    def get_rgbd_frame(self):
        """Captures and aligns a single RGB-D frame."""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        # Convert to Open3D format
        depth_image = o3d.geometry.Image(np.asanyarray(depth_frame.get_data()))
        color_image = o3d.geometry.Image(np.asanyarray(color_frame.get_data()))
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image, depth_image, convert_rgb_to_intensity=False
        )
        return rgbd_image
        
    def stop(self):
        self.pipeline.stop()
        print("Camera stopped.")


def main():
    # --- 2. Live Reconstruction Loop ---
    
    # Initialize camera
    camera = RealsenseCamera()
    camera.start()

    # Voxel size for downsampling
    VOXEL_SIZE = 0.01  # Meters. Adjust this based on object size and desired detail.
    
    # Odometry and Integration parameters
    odometry_option = o3d.pipelines.odometry.OdometryOption()
    odometry_option.max_depth_diff = 0.03 # Default is 0.03

    # The final, integrated point cloud
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_SIZE,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    # Visualization setup
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Live Reconstruction", width=1280, height=720)
    
    is_capturing = False
    is_finished = False
    
    def toggle_capture(vis):
        nonlocal is_capturing
        is_capturing = not is_capturing
        print(f"Capture {'started' if is_capturing else 'paused'}.")

    def stop_integration(vis):
        nonlocal is_finished
        is_finished = True
        print("Integration finished. Closing window.")

    vis.register_key_callback(ord(" "), toggle_capture)  # Spacebar to start/pause
    vis.register_key_callback(ord("Q"), stop_integration) # Q to finish and process
    vis.register_key_callback(256, stop_integration)      # ESC to finish

    # Main state variables
    current_pose = np.identity(4)
    previous_rgbd = None
    geometry_added = False

    print("\n" + "="*50)
    print("INSTRUCTIONS:")
    print(" - Place your object in front of the camera.")
    print(" - Press [SPACE] to start/pause the reconstruction.")
    print(" - Slowly move the camera around the object to capture all angles.")
    print(" - Press [Q] or [ESC] when you are finished.")
    print("="*50 + "\n")

    while not is_finished:
        if is_capturing:
            # Capture the current RGB-D frame
            rgbd_image = camera.get_rgbd_frame()
            
            if rgbd_image is None:
                continue

            if previous_rgbd is None:
                # This is the first frame
                previous_rgbd = rgbd_image
                continue

            # --- Calculate Odometry (camera motion) ---
            # This aligns the current frame (source) to the previous frame (target)
            # using both color and depth information.
            success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
                previous_rgbd, 
                rgbd_image, 
                camera.pinhole_camera_intrinsic,
                np.identity(4), # Initial guess for the transform
                o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                odometry_option
            )

            if success:
                # Update the camera's global pose
                current_pose = np.dot(current_pose, trans)

                # --- Integrate the frame into the volume ---
                # This adds the current frame's data to our 3D model
                volume.integrate(
                    rgbd_image,
                    camera.pinhole_camera_intrinsic,
                    np.linalg.inv(current_pose) # We need the inverse pose for integration
                )

                # Update the previous frame for the next iteration
                previous_rgbd = rgbd_image
            else:
                print("Odometry failed. Try moving slower or check lighting.")
        
        # --- Visualization ---
        vis.poll_events()
        vis.update_renderer()
        # To avoid constant redraws, we only update the geometry periodically
        if is_capturing and is_finished == False:
            temp_pcd = volume.extract_point_cloud()
            # Flip for visualization
            temp_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            if not geometry_added:
                vis.add_geometry(temp_pcd)
                geometry_added = True
            else:
                vis.update_geometry(temp_pcd)
    
    # --- 3. Post-Processing and Meshing ---
    vis.destroy_window()
    camera.stop()

    print("Extracting final point cloud from volume...")
    pcd = volume.extract_point_cloud()
    
    # Optional: Clean up the point cloud
    print("Cleaning up the point cloud with statistical outlier removal...")
    pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_clean = pcd.select_by_index(ind)

    # Save the cleaned point cloud
    pcd_output_path = "final_point_cloud.ply"
    o3d.io.write_point_cloud(pcd_output_path, pcd_clean)
    print(f"Final point cloud saved to {pcd_output_path}")

    print("Creating a mesh from the point cloud using Poisson reconstruction...")
    # Poisson reconstruction requires normals
    pcd_clean.estimate_normals()
    
    # The 'depth' parameter is crucial. Higher value means more detail.
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_clean, depth=9
    )

    # Clean the mesh by removing low-density vertices
    print("Cleaning the mesh...")
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # Save the final mesh as an OBJ file
    mesh_output_path = "final_asset.obj"
    o3d.io.write_triangle_mesh(mesh_output_path, mesh)
    print(f"Final 3D asset saved to {mesh_output_path}")

    # Visualize the final result
    print("Displaying final mesh. Press 'Q' to close.")
    # Flip for visualization
    mesh.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([mesh], "Final 3D Asset")


if __name__ == "__main__":
    main()
