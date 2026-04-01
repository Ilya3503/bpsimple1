# main.py
from fastapi import FastAPI
from threading import Lock
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from datetime import datetime
import os
from app.processing import voxel_downsample, remove_outliers, remove_plane, cluster_dbscan
from app.utils import save_pcd

app = FastAPI(title="Jetson RealSense PointCloud Pipeline")

camera = None
camera_lock = Lock()
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(config)

    def stop(self):
        self.pipeline.stop()

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            return None, None, None
        depth_image = np.asanyarray(depth.get_data())
        color_image = np.asanyarray(color.get_data())
        return color_image, depth_image, depth

    def get_pointcloud(self):
        color, depth_image, depth_frame = self.get_frame()
        if color is None:
            return None, None
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        v = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        return v, color

# создаём камеру при старте
@app.on_event("startup")
def startup_event():
    global camera
    camera = RealSenseCamera()

# останавливаем камеру при shutdown
@app.on_event("shutdown")
def shutdown_event():
    global camera
    if camera:
        camera.stop()

# Эндпоинт для захвата pointcloud и сохранения в файл
@app.get("/capture")
def capture_pointcloud():
    global camera
    with camera_lock:
        points, color = camera.get_pointcloud()
    if points is None:
        return {"error": "Failed to capture pointcloud"}

    # создаём Open3D pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # сохраняем с таймстемпом
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(DATA_DIR, f"capture_{timestamp}.ply")
    o3d.io.write_point_cloud(filename, pcd)

    return {"ply_file": filename}

# Эндпоинт для обработки последнего pointcloud
@app.get("/process")
def process_last_pointcloud(
    voxel_size: float = 0.01,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    distance_threshold: float = 0.01,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    dbscan_eps: float = 0.02,
    dbscan_min_samples: int = 10
):
    # ищем последний файл по таймстемпу
    ply_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".ply")])
    if not ply_files:
        return {"error": "No captured pointclouds found"}
    latest_file = os.path.join(DATA_DIR, ply_files[-1])

    # загружаем pointcloud
    pcd = o3d.io.read_point_cloud(latest_file)

    # пайплайн обработки
    pcd = voxel_downsample(pcd, voxel_size=voxel_size)
    pcd = remove_outliers(pcd, nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd = remove_plane(pcd, distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
    labels = cluster_dbscan(pcd, eps=dbscan_eps, min_samples=dbscan_min_samples)

    # сохраняем обработанный файл
    processed_filename = save_pcd(pcd, prefix="processed")

    clusters_count = len(set(labels)) - (1 if -1 in labels else 0)
    return {
        "clusters_count": clusters_count,
        "processed_ply_file": processed_filename
    }