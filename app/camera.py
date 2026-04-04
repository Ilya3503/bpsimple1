import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from datetime import datetime
from pathlib import Path


def capture_pointcloud(output_dir: str = "data"):

    Path(output_dir).mkdir(exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()

    # стабильная конфигурация для D415
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)

    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

    fx = intr.fx
    fy = intr.fy
    cx = intr.ppx
    cy = intr.ppy

    frames = pipeline.wait_for_frames()
    frames = align.process(frames)

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth_frame = spatial.process(depth_frame)
    depth_frame = temporal.process(depth_frame)
    depth_frame = hole_filling.process(depth_frame)

    depth = np.asanyarray(depth_frame.get_data())
    color = np.asanyarray(color_frame.get_data())

    pipeline.stop()

    depth = depth * depth_scale

    h, w = depth.shape

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    colors = color.reshape(-1, 3) / 255.0

    mask = z.reshape(-1) > 0
    points = points[mask]
    colors = colors[mask]

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    filename = f"pointcloud_{timestamp}.ply"
    filepath = Path(output_dir) / filename

    o3d.io.write_point_cloud(str(filepath), pcd, write_ascii=True)

    return str(filepath)