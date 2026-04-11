import open3d as o3d
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

cluster = o3d.io.read_point_cloud(str(BASE_DIR / "results/clusters/cluster_000.ply"))
cad = o3d.io.read_point_cloud(str(BASE_DIR / "cad_models/Cube_30х30х30.ply"))

pts_cad = np.asarray(cad.points).copy()
pts_cad -= pts_cad.mean(axis=0)

cluster_extent = np.asarray(cluster.get_axis_aligned_bounding_box().get_extent())
cad_extent = pts_cad.max(axis=0) - pts_cad.min(axis=0)
scale = np.mean(cluster_extent) / (np.mean(cad_extent) + 1e-9)

print(f"Кластер extent: {cluster_extent}")
print(f"CAD extent до масштаба: {cad_extent}")
print(f"Scale: {scale:.6f}")

pts_cad *= scale
print(f"CAD extent после масштаба: {pts_cad.max(axis=0) - pts_cad.min(axis=0)}")
print(f"CAD точек: {len(pts_cad)}")

cluster_center = np.asarray(cluster.points).mean(axis=0)
pts_cad += cluster_center
print(f"CAD center после сдвига: {pts_cad.mean(axis=0)}")
print(f"Кластер center: {cluster_center}")