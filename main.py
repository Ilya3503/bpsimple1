from fastapi import FastAPI, Query
from app.camera import RealSenseCamera
from app.processing import voxel_downsample, remove_outliers, remove_plane, cluster_dbscan
from app.utils import save_pcd
import open3d as o3d

app = FastAPI(title="Jetson RealSense PointCloud Pipeline")

camera = RealSenseCamera()

@app.get("/process")
def process_pointcloud(
    voxel_size: float = Query(0.01, description="Voxel size for downsampling"),
    nb_neighbors: int = Query(20, description="Number of neighbors for outlier removal"),
    std_ratio: float = Query(2.0, description="Std ratio for outlier removal"),
    distance_threshold: float = Query(0.01, description="Distance threshold for plane removal"),
    ransac_n: int = Query(3, description="RANSAC points"),
    num_iterations: int = Query(1000, description="RANSAC iterations"),
    dbscan_eps: float = Query(0.02, description="EPS for DBSCAN"),
    dbscan_min_samples: int = Query(10, description="Min samples for DBSCAN"),
    save_intermediate: bool = Query(True, description="Save intermediate .ply files")
):
    """
    Захват pointcloud с RealSense и обработка пайплайном:
    вокселизация -> удаление шумов -> RANSAC -> DBSCAN.
    Возвращает количество кластеров и список файлов .ply.
    """
    ply_files = []

    # 1. Захват pointcloud
    points, _ = camera.get_pointcloud()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if save_intermediate: ply_files.append(save_pcd(pcd, "raw"))

    # 2. Вокселизация
    pcd = voxel_downsample(pcd, voxel_size=voxel_size, save=save_intermediate)
    if save_intermediate: ply_files.append(save_pcd(pcd, "voxel"))

    # 3. Удаление шумов
    pcd = remove_outliers(pcd, nb_neighbors=nb_neighbors, std_ratio=std_ratio, save=save_intermediate)
    if save_intermediate: ply_files.append(save_pcd(pcd, "outliers"))

    # 4. Удаление опорной плоскости
    pcd = remove_plane(pcd, distance_threshold=distance_threshold, ransac_n=ransac_n,
                       num_iterations=num_iterations, save=save_intermediate)
    if save_intermediate: ply_files.append(save_pcd(pcd, "plane_removed"))

    # 5. Кластеризация DBSCAN
    labels = cluster_dbscan(pcd, eps=dbscan_eps, min_samples=dbscan_min_samples, save=save_intermediate)
    if save_intermediate: ply_files.append(save_pcd(pcd, "dbscan"))

    clusters_count = len(set(labels)) - (1 if -1 in labels else 0)

    return {
        "clusters_count": clusters_count,
        "ply_files": ply_files
    }