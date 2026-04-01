import open3d as o3d
from datetime import datetime

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_pcd(pcd, prefix="pcd"):
    filename = f"{prefix}_{timestamp()}.ply"
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved: {filename}")
    return filename