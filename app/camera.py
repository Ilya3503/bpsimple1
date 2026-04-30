import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from datetime import datetime
from pathlib import Path


def capture_pointcloud(output_dir: str = "data"):

    Path(output_dir).mkdir(exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

    profile = pipeline.start(config)

    # --- align depth → color ---
    align = rs.align(rs.stream.color)

    # --- фильтры ---
    ###decimation = rs.decimation_filter()
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    # --- параметры камеры ---
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    intr = profile.get_stream(rs.stream.depth)\
        .as_video_stream_profile().get_intrinsics()

    fx = intr.fx
    fy = intr.fy
    cx = intr.ppx
    cy = intr.ppy

    # --- получаем кадр ---
    frames = pipeline.wait_for_frames()

    # align
    frames = align.process(frames)

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # --- применяем фильтры ---

    depth_frame = spatial.process(depth_frame)
    depth_frame = temporal.process(depth_frame)
    depth_frame = hole_filling.process(depth_frame)

    depth = np.asanyarray(depth_frame.get_data())
    color = np.asanyarray(color_frame.get_data())

    pipeline.stop()

# ====================== ROI CROP (БЕЗОПАСНЫЙ ВАРИАНТ) ======================
    h, w = depth.shape

    left   = 330 
    right  = 200   
    top    = 50
    bottom = 0

    if top > 0:
        depth[:top, :] = 0
        color[:top, :] = 0
    if bottom > 0:
        depth[-bottom:, :] = 0     
        color[-bottom:, :] = 0
    if left > 0:
        depth[:, :left] = 0
        color[:, :left] = 0
    if right > 0:
        depth[:, -right:] = 0       # <- важно: -right
        color[:, -right:] = 0

    print(f"DEBUG: После кропа depth shape = {depth.shape}, non-zero pixels = {np.count_nonzero(depth)}")
    # =====================================================================

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

    # --- timestamp ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    filename = f"pointcloud_{timestamp}.ply"
    filepath = Path(output_dir) / filename

    # --- сохраняем ---
    o3d.io.write_point_cloud(str(filepath), pcd)

    return str(filepath)