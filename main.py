# main.py
from fastapi import FastAPI
from threading import Lock
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from app.processing import voxel_downsample, remove_outliers, remove_plane, cluster_dbscan
from app.utils import save_pcd

app = FastAPI(title="Jetson RealSense PointCloud Pipeline")

camera = None
camera_lock = Lock()

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

@app.get("/process")
def process_pointcloud():
    global camera
    with camera_lock:
        points, color = camera.get_pointcloud()
    if points is None:
        return {"error": "Failed to capture pointcloud"}

    # создаём Open3D pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # пример обработки (можно добавить весь ваш pipeline)
    pcd = voxel_downsample(pcd, voxel_size=0.01)
    ply_file = save_pcd(pcd, "processed")

    return {"ply_file": ply_file}