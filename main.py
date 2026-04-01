from fastapi import FastAPI
import os
import datetime
import open3d as o3d

from app.camera import RealSenseCamera
from app.processing import voxel_downsample, remove_outliers, remove_plane, cluster_dbscan
from app.utils import save_pcd

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="Jetson RealSense PointCloud Pipeline")

camera = RealSenseCamera()

@app.on_event("shutdown")
def shutdown_event():
    camera.stop()

@app.get("/capture")
def capture_pointcloud():
    points, color = camera.get_pointcloud()
    if points is None:
        return {"error": "Failed to capture pointcloud"}

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = save_pcd(pcd, prefix=f"raw_{timestamp}", directory=DATA_DIR)
    return {"file": filename}

@app.get("/process")
def process_pointcloud(
    voxel_size: float = 0.01,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    distance_threshold: float = 0.01,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    dbscan_eps: float = 0.02,
    dbscan_min_samples: int = 10,
):
    ply_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".ply")])
    if not ply_files:
        return {"error": "No pointclouds to process"}

    latest_file = os.path.join(DATA_DIR, ply_files[-1])
    pcd = o3d.io.read_point_cloud(latest_file)

    pcd = voxel_downsample(pcd, voxel_size=voxel_size)
    pcd = remove_outliers(pcd, nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd = remove_plane(pcd, distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
    labels = cluster_dbscan(pcd, eps=dbscan_eps, min_samples=dbscan_min_samples)

    processed_filename = save_pcd(pcd, prefix="processed", directory=DATA_DIR)

    clusters_count = len(set(labels)) - (1 if -1 in labels else 0)

    return {"processed_file": processed_filename, "clusters_count": clusters_count}