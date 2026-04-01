from fastapi import FastAPI
import os, glob, time
import open3d as o3d
from app.camera import RealSenseCamera

app = FastAPI(title="RealSense Safe Pipeline")

camera = RealSenseCamera()  # синхронная инициализация

SAVE_DIR = "captures"
os.makedirs(SAVE_DIR, exist_ok=True)

def save_pcd(pcd, name_prefix):
    timestamp = int(time.time())
    filename = os.path.join(SAVE_DIR, f"{name_prefix}_{timestamp}.ply")
    o3d.io.write_point_cloud(filename, pcd)
    return filename

def get_last_ply():
    files = glob.glob(os.path.join(SAVE_DIR, "*.ply"))
    if not files:
        return None
    return max(files, key=os.path.getctime)

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
    ply_file = save_pcd(pcd, "raw")
    return {"ply_file": ply_file}

@app.get("/process")
def process_last_pointcloud():
    last_file = get_last_ply()
    if last_file is None:
        return {"error": "No PLY file found"}

    pcd = o3d.io.read_point_cloud(last_file)

    # Простейшая обработка: вокселизация + очистка шумов
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)

    processed_file = save_pcd(pcd, "processed")
    return {"processed_ply_file": processed_file}