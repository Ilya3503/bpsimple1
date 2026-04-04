import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import time
from pathlib import Path


def capture_pointcloud(output_dir: str = "data"):

    Path(output_dir).mkdir(exist_ok=True)

    # --- запуск камеры ---
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    # --- параметры камеры ---
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

    fx = intr.fx
    fy = intr.fy
    cx = intr.ppx
    cy = intr.ppy

    # --- получаем кадр ---
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth = np.asanyarray(depth_frame.get_data())
    color = np.asanyarray(color_frame.get_data())

    pipeline.stop()

    # --- depth → meters ---
    depth = depth * depth_scale

    h, w = depth.shape

    # --- сетка пикселей ---
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # --- xyz ---
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # --- цвета ---
    colors = color.reshape(-1, 3) / 255.0

    # --- фильтр нулевой глубины ---
    mask = z.reshape(-1) > 0
    points = points[mask]
    colors = colors[mask]

    # --- pointcloud ---
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # --- имя файла ---
    timestamp = int(time.time())
    filename = f"pointcloud_{timestamp}.ply"
    filepath = Path(output_dir) / filename

    # --- сохраняем ---
    o3d.io.write_point_cloud(str(filepath), pcd, write_ascii=True)

    return str(filepath)