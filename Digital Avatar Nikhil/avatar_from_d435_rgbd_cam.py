'''
Avatar Creation Pipeline (Offline)

1. RGB-D Capture & Volumetric Fusion with Open3D
2. Mesh Cleanup & Simplification
3. UV Unwrapping & Texture Baking in Blender (via script)
4. USD Export
'''

# 1. Volumetric Fusion using Open3D
import open3d as o3d
import numpy as np
import pyrealsense2 as rs

# Parameters
VOXEL_LENGTH = 4e-3       # 4mm voxels
SDF_TRUNC    = 0.02       # truncation
DEPTH_SCALE  = 1000.0     # RealSense depth scale

# Initialize RealSense
pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Get intrinsics
profile = pipeline.get_active_profile()
depth_profile = profile.get_stream(rs.stream.depth)
intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
pinhole = o3d.camera.PinholeCameraIntrinsic(
    intrinsics.width, intrinsics.height,
    intrinsics.fx, intrinsics.fy,
    intrinsics.ppx, intrinsics.ppy)

# Create TSDF Volume
tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=VOXEL_LENGTH,
    sdf_trunc=SDF_TRUNC,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

print("Starting capture. Move around the object...")
try:
    for i in range(200):  # capture 200 frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        # Convert images
        depth = o3d.geometry.Image(np.asanyarray(depth_frame.get_data()))
        color = o3d.geometry.Image(np.asanyarray(color_frame.get_data()))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth,
            depth_scale=DEPTH_SCALE,
            depth_trunc=4.0,
            convert_rgb_to_intensity=False)
        # Integrate
        tsdf.integrate(rgbd, pinhole, np.linalg.inv(np.eye(4)))
finally:
    pipeline.stop()

# Extract mesh
mesh = tsdf.extract_triangle_mesh()
mesh.compute_vertex_normals()

# 2. Mesh Simplification & Smoothing
print("Simplifying mesh...")
mesh_smp = mesh.simplify_quadric_decimation(target_number_of_triangles=100000)
mesh_smp = mesh_smp.filter_smooth_simple(number_of_iterations=2)

# Save intermediate mesh
o3d.io.write_triangle_mesh("avatar_raw.ply", mesh_smp)
print("Saved simplified mesh: avatar_raw.ply")

# 3. Export for Blender UV & Rigging
# You can import 'avatar_raw.ply' into Blender and run the Blender script below.

##### Blender Script (run inside Blender Text Editor) #####
'''
import bpy

# Load mesh
bpy.ops.import_mesh.ply(filepath="/path/to/avatar_raw.ply")
obj = bpy.context.selected_objects[0]

# Smooth shading
bpy.ops.object.shade_smooth()

# Decimate if needed further
dec = obj.modifiers.new(name='Decimate', type='DECIMATE')
dec.ratio = 0.5  # keep 50%
bpy.ops.object.modifier_apply(modifier='Decimate')

# UV Unwrap
bpy.ops.object.mode_set(mode='EDIT')
for face in obj.data.polygons:
    face.select = True
bpy.ops.uv.smart_project(angle_limit=66.0)
bpy.ops.object.mode_set(mode='OBJECT')

# Bake texture
# Assume you have a material with an image node named 'BakeTex'
img = bpy.data.images.new("DiffuseBake", width=2048, height=2048)
mat = obj.active_material
nodes = mat.node_tree.nodes
bake_node = nodes.new('ShaderNodeTexImage')
bake_node.image = img
nodes.active = bake_node
bpy.ops.object.bake(type='DIFFUSE', use_clear=True, margin=16)
img.filepath_raw = "/path/to/avatar_diffuse.png"
img.file_format = 'PNG'
img.save()

# Export USD
bpy.ops.wm.usd_export(filepath="/path/to/avatar.usd", export_textures=True)
print("Exported USD avatar at /path/to/avatar.usd")
'''
