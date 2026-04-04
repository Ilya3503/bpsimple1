import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import json
import matplotlib.pyplot as plt


# -------------------- Работа с файлами --------------------
def get_latest_file(folder: str, extension: str = ".ply") -> Path:
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = list(folder_path.glob(f"*{extension}"))
    if not files:
        raise FileNotFoundError(f"No {extension} files found in folder: {folder}")
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    return latest_file


def load_point_cloud(file_path: str) -> o3d.geometry.PointCloud:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".ply":
        return o3d.io.read_point_cloud(str(path))
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path))
        pts = arr[:, :3].astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd
    raise ValueError(f"Unsupported file extension: {path.suffix}")


def save_point_cloud(pcd: o3d.geometry.PointCloud, file_path: str) -> str:
    path = Path(file_path)
    if path.suffix.lower() == ".ply":
        o3d.io.write_point_cloud(str(path), pcd)
    elif path.suffix.lower() == ".npy":
        np.save(str(path), np.asarray(pcd.points))
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")
    return str(path)


# -------------------- Предобработка --------------------
def voxel_downsample(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.01) -> o3d.geometry.PointCloud:
    return pcd.voxel_down_sample(voxel_size=voxel_size)


def remove_noise(pcd: o3d.geometry.PointCloud, nb_neighbors: int = 20, std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
    cl, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return cl





def clean_point_cloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    mask = np.isfinite(pts).all(axis=1)
    clean_pcd = o3d.geometry.PointCloud()
    clean_pcd.points = o3d.utility.Vector3dVector(pts[mask])
    return clean_pcd





# -------------------- Кластеризация --------------------
def cluster_dbscan(pcd: o3d.geometry.PointCloud, eps: float = 0.03, min_points: int = 20, max_points: Optional[int] = None) -> List[o3d.geometry.PointCloud]:
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return []
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    unique_labels = np.unique(labels)
    clusters = []
    for lab in unique_labels:
        if lab == -1:
            continue
        idx = np.where(labels == lab)[0]
        if idx.size == 0:
            continue
        if max_points is not None and idx.size > max_points:
            continue
        cluster = pcd.select_by_index(idx.tolist())
        clusters.append(cluster)
    return clusters


def get_obb_for_cluster(cluster: o3d.geometry.PointCloud) -> Dict:
    obb = cluster.get_oriented_bounding_box()
    center = list(map(float, obb.center))
    extent = list(map(float, obb.extent))
    R = np.asarray(obb.R)
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return {"center": center, "extent": extent, "yaw": yaw}


def ensure_results_dir(results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    clusters_dir = results_dir / "clusters"
    clusters_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def save_cluster_files(clusters: List[o3d.geometry.PointCloud], clusters_dir: str) -> List[str]:
    clusters_dir = Path(clusters_dir)
    clusters_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for i, c in enumerate(clusters):
        fname = clusters_dir / f"cluster_{i:03d}.ply"
        o3d.io.write_point_cloud(str(fname), c)
        saved_paths.append(str(fname))
    return saved_paths


# -------------------- Аннотированное облако --------------------
def create_and_save_annotated_pointcloud(pcd: o3d.geometry.PointCloud, clusters: List[o3d.geometry.PointCloud], results_dir: str) -> str:
    results_path = Path(results_dir)
    out_path = results_path / "annotated_pointcloud.ply"
    if not clusters:
        o3d.io.write_point_cloud(str(out_path), pcd)
        return str(out_path)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))[:, :3]
    all_points, all_colors = [], []
    for i, cluster in enumerate(clusters):
        pts = np.asarray(cluster.points)
        clr = np.tile(colors[i % len(colors)], (pts.shape[0], 1))
        all_points.append(pts)
        all_colors.append(clr)
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    merged.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    o3d.io.write_point_cloud(str(out_path), merged)
    return str(out_path)


def save_position_json(result: dict, results_dir: str) -> str:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "position.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return str(out)


# -------------------- CAD / ICP (заглушка) --------------------
def load_cad_model(file_path: str) -> Optional[o3d.geometry.PointCloud]:
    if file_path is None:
        return None
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CAD file not found: {file_path}")
    return load_point_cloud(str(path))


def apply_icp_to_cluster(cluster: o3d.geometry.PointCloud, cad_model: Optional[o3d.geometry.PointCloud], threshold: float = 50.0):
    if cad_model is None:
        return None
    # TODO: Реализовать ICP после получения CAD
    return None


# -------------------- Главная функция обработки --------------------
def process_pointcloud(
    input_file: Optional[str],
    results_dir: str,
    use_latest: bool = True,
    folder: str = "data",
    voxel_size: float = 0.001,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    eps: float = 30,
    min_points: int = 20,
    max_points: Optional[int] = None,
    cad_file: Optional[str] = None
) -> dict:
    results_dir = Path(results_dir)
    results_dir = ensure_results_dir(results_dir)
    clusters_dir = results_dir / "clusters"

    if use_latest:
        input_file = str(get_latest_file(folder))
    elif input_file is None:
        raise ValueError("No input file specified and use_latest=False")

    print(f"[process] Загрузка облака точек: {input_file}")
    pcd = load_point_cloud(input_file)
    print(f"[process] Исходное облако: {len(pcd.points)} точек")

    # Даунсэмплинг
    pcd = voxel_downsample(pcd, voxel_size)
    print(f"[process] После даунсэмплинга (voxel_size={voxel_size}): {len(pcd.points)} точек")

    # Удаление шума
    pcd = remove_noise(pcd, nb_neighbors, std_ratio)
    print(f"[process] После фильтрации шума (nb_neighbors={nb_neighbors}, std_ratio={std_ratio}): {len(pcd.points)} точек")



    # Кроп


    # Проверка перед кластеризацией
    if len(pcd.points) == 0:
        print("[process] Внимание: после предобработки нет точек для кластеризации!")
        clusters = []
    else:
        clusters = cluster_dbscan(pcd, eps, min_points, max_points)
        print(f"[process] После DBSCAN (eps={eps}, min_points={min_points}): найдено {len(clusters)} кластеров")

    clusters_info = []
    for i, c in enumerate(clusters):
        info = get_obb_for_cluster(c)
        info["id"] = i
        info["points_count"] = int(len(c.points))
        clusters_info.append(info)

    save_cluster_files(clusters, str(clusters_dir))
    annotated_path = create_and_save_annotated_pointcloud(pcd, clusters, str(results_dir))

    # Заглушка ICP
    cad_model = load_cad_model(cad_file)
    for cluster in clusters:
        apply_icp_to_cluster(cluster, cad_model)

    result = {
        "status": "ok",
        "preprocessed_file": input_file,
        "results_dir": str(results_dir),
        "num_clusters": len(clusters),
        "clusters": clusters_info,
        "annotated_ply": annotated_path
    }

    save_position_json(result, str(results_dir))
    return result