import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import json
import matplotlib.pyplot as plt


# ==============================================================================
# ФАЙЛОВЫЕ ОПЕРАЦИИ
# ==============================================================================

def get_latest_file(folder: str, extension: str = ".ply") -> Path:
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Папка не найдена: {folder}")
    files = list(folder_path.glob(f"*{extension}"))
    if not files:
        raise FileNotFoundError(f"Файлы {extension} не найдены в папке: {folder}")
    return max(files, key=lambda f: f.stat().st_mtime)


def load_point_cloud(file_path: str) -> o3d.geometry.PointCloud:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    if path.suffix.lower() == ".ply":
        return o3d.io.read_point_cloud(str(path))
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path))
        pts = arr[:, :3].astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd
    raise ValueError(f"Неподдерживаемый формат файла: {path.suffix}")


def save_point_cloud(pcd: o3d.geometry.PointCloud, file_path: str) -> str:
    path = Path(file_path)
    if path.suffix.lower() == ".ply":
        o3d.io.write_point_cloud(str(path), pcd)
    elif path.suffix.lower() == ".npy":
        np.save(str(path), np.asarray(pcd.points))
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {path.suffix}")
    return str(path)


def ensure_dirs(results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "clusters").mkdir(parents=True, exist_ok=True)
    return results_dir


# ==============================================================================
# ПРЕДОБРАБОТКА
# ==============================================================================

def clean_point_cloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Удаляет точки с NaN и Inf координатами."""
    pts = np.asarray(pcd.points)
    mask = np.isfinite(pts).all(axis=1)
    clean = o3d.geometry.PointCloud()
    clean.points = o3d.utility.Vector3dVector(pts[mask])
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        clean.colors = o3d.utility.Vector3dVector(colors[mask])
    return clean


def crop_roi(
    pcd: o3d.geometry.PointCloud,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    z_min: float, z_max: float,
) -> o3d.geometry.PointCloud:
    """
    Обрезка по заданным границам (все координаты в метрах).
    Рекомендуемые значения для текущей сцены:
        x: -0.5 .. +0.5
        y: -0.25 .. +0.25
        z:  0.50 .. 0.75
    """
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return pcd
    mask = (
        (pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) &
        (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max) &
        (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    )
    cropped = o3d.geometry.PointCloud()
    cropped.points = o3d.utility.Vector3dVector(pts[mask])
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        cropped.colors = o3d.utility.Vector3dVector(colors[mask])
    return cropped


def voxel_downsample(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.01,
) -> o3d.geometry.PointCloud:
    """
    Вокселизация для pipeline.
    Рекомендуемый размер: 0.01 (1см).
    ВАЖНО: этот параметр НЕ связан с icp_voxel_size.
    """
    return pcd.voxel_down_sample(voxel_size=voxel_size)


def remove_noise(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
    """Statistical outlier removal."""
    if len(pcd.points) < nb_neighbors:
        return pcd
    cl, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    return cl


def remove_plane(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.01,
    ransac_n: int = 3,
    num_iterations: int = 1000,
) -> Tuple[o3d.geometry.PointCloud, Optional[List[float]]]:
    """
    Удаление доминирующей плоскости (стол) через RANSAC.
    Возвращает облако без плоскости и уравнение [a,b,c,d].
    distance_threshold=0.01 (1см) — допуск для плоского стола.
    """
    pts = np.asarray(pcd.points)
    if pts.shape[0] < ransac_n:
        print(f"[remove_plane] Недостаточно точек: {pts.shape[0]}")
        return pcd, None
    try:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        pcd_no_plane = pcd.select_by_index(inliers, invert=True)
        return pcd_no_plane, list(plane_model)
    except Exception as e:
        print(f"[remove_plane] Ошибка: {e}")
        return pcd, None


# ==============================================================================
# КЛАСТЕРИЗАЦИЯ
# ==============================================================================

def cluster_dbscan(
    pcd: o3d.geometry.PointCloud,
    eps: float = 0.03,
    min_points: int = 30,
    max_points: Optional[int] = None,
    min_extent: float = 0.02,
    max_extent: float = 0.30,
) -> List[o3d.geometry.PointCloud]:
    """
    DBSCAN кластеризация с фильтрацией по физическому размеру.
    eps=0.03 (3см), min_extent/max_extent отсекают мусор и фон.
    """
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return []
    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )
    clusters = []
    for lab in np.unique(labels):
        if lab == -1:
            continue
        idx = np.where(labels == lab)[0]
        if max_points is not None and idx.size > max_points:
            continue
        cluster = pcd.select_by_index(idx.tolist())
        extent = cluster.get_axis_aligned_bounding_box().get_extent()
        max_dim = float(np.max(extent))
        if max_dim < min_extent or max_dim > max_extent:
            continue
        clusters.append(cluster)
    return clusters


def get_cluster_info(cluster: o3d.geometry.PointCloud, cluster_id: int) -> Dict:
    """Геометрические характеристики кластера с защитой от плоских объектов."""
    pts = np.asarray(cluster.points)

    if len(pts) < 6:
        aabb = cluster.get_axis_aligned_bounding_box()
        center = list(map(float, aabb.get_center()))
        extent = list(map(float, aabb.get_extent()))
        R = np.eye(3)
        yaw = 0.0
        is_obb = False
    else:
        try:
            # Пытаемся сделать OBB
            obb = cluster.get_oriented_bounding_box()
            center = list(map(float, obb.center))
            extent = list(map(float, obb.extent))
            R = np.asarray(obb.R)
            yaw = float(np.arctan2(R[1, 0], R[0, 0]))
            is_obb = True
        except Exception as e:
            print(f"[WARNING] OBB failed for cluster {cluster_id}: {e}")
            # Fallback на AABB
            aabb = cluster.get_axis_aligned_bounding_box()
            center = list(map(float, aabb.get_center()))
            extent = list(map(float, aabb.get_extent()))
            R = np.eye(3)
            yaw = 0.0
            is_obb = False

    return {
        "id": cluster_id,
        "center": center,
        "extent": extent,
        "yaw": yaw,
        "rotation_matrix": R.tolist(),
        "points_count": int(len(pts)),
        "is_obb": is_obb
    }


# ==============================================================================
# ICP — ОЦЕНКА ПОЗЫ
# ==============================================================================

def rotation_matrix_to_quaternion(R: np.ndarray) -> List[float]:
    """Матрица вращения 3x3 → кватернион [qx, qy, qz, qw]. Формула Shepperd."""
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def transformation_to_pose(T: np.ndarray) -> Dict:
    """Матрица 4x4 → {position, orientation quaternion}."""
    position = T[:3, 3].tolist()
    R = T[:3, :3]
    orientation = rotation_matrix_to_quaternion(R)
    return {
        "position": [float(v) for v in position],
        "orientation": orientation,
    }


def load_cad_model(file_path: str) -> Optional[o3d.geometry.PointCloud]:
    """Загружает CAD PLY файл. Возвращает None если путь не задан."""
    if file_path is None:
        return None
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CAD файл не найден: {file_path}")
    return load_point_cloud(str(path))


def estimate_pose_from_obb(cluster: o3d.geometry.PointCloud) -> Dict:
    """
    Fallback: оценка позы по OBB без ICP.
    method='obb_fallback' сигнализирует об этом downstream.
    """
    obb = cluster.get_oriented_bounding_box()
    R = np.asarray(obb.R)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = obb.center
    pose = transformation_to_pose(T)
    return {
        "method": "obb_fallback",
        "fitness": None,
        "inlier_rmse": None,
        "transformation": T.tolist(),
        "position": pose["position"],
        "orientation": pose["orientation"],
        "extent": list(map(float, obb.extent)),
    }


def _icp_step(src: np.ndarray, tgt: np.ndarray, max_dist: float):
    """
    Улучшенный расчёт fitness:
    - Учитываем оба облака (CAD и реальный кластер)
    - Добавляем bidirectional check
    """
    from scipy.spatial import KDTree

    if len(src) == 0 or len(tgt) == 0:
        return np.eye(4), float('inf'), 0.0

    # Прямой поиск: CAD → Cluster
    tree_tgt = KDTree(tgt)
    dists_forward, _ = tree_tgt.query(src, k=1)

    # Обратный поиск: Cluster → CAD
    tree_src = KDTree(src)
    dists_backward, _ = tree_src.query(tgt, k=1)

    # Inliers в обе стороны
    inliers_fwd = dists_forward < max_dist
    inliers_bwd = dists_backward < max_dist

    num_inliers = min(inliers_fwd.sum(), inliers_bwd.sum())

    fitness = float(num_inliers) / max(len(src), len(tgt))   # ← изменили
    rmse = float(np.sqrt((dists_forward[inliers_fwd] ** 2).mean())) if inliers_fwd.any() else float('inf')

    # SVD для трансформации (остаётся как было)
    src_matched = src[inliers_fwd]
    tgt_matched = tgt[tree_tgt.query(src, k=1)[1][inliers_fwd]]

    if len(src_matched) < 6:
        return np.eye(4), float('inf'), 0.0

    src_center = src_matched.mean(axis=0)
    tgt_center = tgt_matched.mean(axis=0)
    src_c = src_matched - src_center
    tgt_c = tgt_matched - tgt_center

    H = src_c.T @ tgt_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = tgt_center - R @ src_center

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, :3] = R
    T[:3, 3] = t

    return T, rmse, fitness


def run_icp(
    cluster: o3d.geometry.PointCloud,
    cad_model: o3d.geometry.PointCloud,
    voxel_size: float = 0.003,
    max_correspondence_distance: float = 0.02,
) -> Dict:
    """
    ICP реализован через numpy+scipy — без Open3D registration.
    Open3D registration_icp крашит на Jetson с версией 0.18.0.
    """
    # --- 1. Копируем CAD через numpy ---
    pts_cad = np.asarray(cad_model.points).copy()

    # --- 2. Центрируем ---
    pts_cad -= pts_cad.mean(axis=0)

    # --- 3. Масштабируем под кластер ---
    cluster_extent = np.asarray(
        cluster.get_axis_aligned_bounding_box().get_extent()
    )
    cad_extent = pts_cad.max(axis=0) - pts_cad.min(axis=0)
    scale = np.mean(cluster_extent) / (np.mean(cad_extent) + 1e-9)
    pts_cad *= scale

    # --- 4. Начальное приближение: совмещение центров ---
    cluster_pts = np.asarray(cluster.points).copy()
    cluster_center = cluster_pts.mean(axis=0)
    pts_cad += cluster_center

    # --- 5. Даунсэмплинг через numpy ---
    def downsample(pts, vsize):
        if vsize <= 0 or len(pts) == 0:
            return pts
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd_ds = pcd.voxel_down_sample(vsize)
        return np.asarray(pcd_ds.points)

    src = downsample(pts_cad, voxel_size)
    tgt = downsample(cluster_pts, voxel_size)

    print(f"[ICP] src (CAD DS):     {len(src)} точек")
    print(f"[ICP] tgt (кластер DS): {len(tgt)} точек")

    if len(src) < 6 or len(tgt) < 6:
        print("[ICP] Слишком мало точек — fallback на OBB")
        return estimate_pose_from_obb(cluster)

    # --- 6. Итерации ICP ---
    T_total = np.eye(4)
    prev_rmse = float('inf')

    for iteration in range(50):
        T_step, rmse, fitness = _icp_step(src, tgt, max_correspondence_distance)

        # Применяем трансформацию к src
        src_h = np.hstack([src, np.ones((len(src), 1))])
        src = (T_step @ src_h.T).T[:, :3]

        # Накапливаем трансформацию
        T_total = T_step @ T_total

        # Проверяем сходимость
        if abs(prev_rmse - rmse) < 1e-6:
            print(f"[ICP] Сошёлся на итерации {iteration}")
            break
        prev_rmse = rmse

    print(f"[ICP] fitness={fitness:.3f}  rmse={rmse:.5f}")

    if fitness < 0.3:
        print(f"[ICP] Низкий fitness — fallback на OBB")
        result_obb = estimate_pose_from_obb(cluster)
        result_obb["icp_fitness"] = fitness
        return result_obb

    cluster_center = np.asarray(cluster.points).mean(axis=0)

    R_final = T_total[:3, :3]

    T_pose = np.eye(4)
    T_pose[:3, :3] = R_final
    T_pose[:3, 3] = cluster_center

    pose = transformation_to_pose(T_pose)

    return {
        "method": "icp",
        "fitness": fitness,
        "inlier_rmse": rmse,
        "transformation": T_pose.tolist(),
        "position": pose["position"],
        "orientation": pose["orientation"],
        "extent": list(map(float, cluster_extent)),
    }


def estimate_pose_for_cluster(
    cluster: o3d.geometry.PointCloud,
    cad_model: Optional[o3d.geometry.PointCloud],
    icp_voxel_size: float = 0.003,
) -> Dict:
    """
    Оценка позы для одного кластера.
    icp_voxel_size=0.003 — фиксированный, НЕ зависит от pipeline voxel_size.
    """
    if cad_model is None:
        return estimate_pose_from_obb(cluster)
    try:
        return run_icp(cluster, cad_model, voxel_size=icp_voxel_size)
    except Exception as e:
        print(f"[ICP] Ошибка — fallback на OBB: {e}")
        result = estimate_pose_from_obb(cluster)
        result["icp_error"] = str(e)
        return result


# ==============================================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ==============================================================================

def save_cluster_files(
    clusters: List[o3d.geometry.PointCloud],
    clusters_dir: str,
) -> List[str]:
    clusters_dir = Path(clusters_dir)
    clusters_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, c in enumerate(clusters):
        fname = clusters_dir / f"cluster_{i:03d}.ply"
        o3d.io.write_point_cloud(str(fname), c)
        saved.append(str(fname))
    return saved


def create_annotated_pointcloud(
    pcd: o3d.geometry.PointCloud,
    clusters: List[o3d.geometry.PointCloud],
    results_dir: str,
) -> str:
    out_path = Path(results_dir) / "annotated_pointcloud.ply"
    if not clusters:
        o3d.io.write_point_cloud(str(out_path), pcd)
        return str(out_path)
    cmap = plt.get_cmap("tab10")(np.linspace(0, 1, 10))[:, :3]
    all_points, all_colors = [], []
    for i, cluster in enumerate(clusters):
        pts = np.asarray(cluster.points)
        clr = np.tile(cmap[i % len(cmap)], (pts.shape[0], 1))
        all_points.append(pts)
        all_colors.append(clr)
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    merged.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    o3d.io.write_point_cloud(str(out_path), merged)
    return str(out_path)


def save_position_json(result: dict, results_dir: str) -> str:
    out = Path(results_dir) / "position.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return str(out)


# ==============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ==============================================================================

def process_pointcloud(
    input_file: Optional[str],
    results_dir: str,
    use_latest: bool = True,
    folder: str = "data",
    roi_x_min: float = -0.5,
    roi_x_max: float = 0.5,
    roi_y_min: float = -0.25,
    roi_y_max: float = 0.25,
    roi_z_min: float = 0.50,
    roi_z_max: float = 0.75,
    # Pipeline voxel — НЕ ICP voxel
    voxel_size: float = 0.01,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    remove_table: bool = True,
    plane_distance_threshold: float = 0.01,
    plane_ransac_n: int = 3,
    plane_num_iterations: int = 1000,
    eps: float = 0.03,
    min_points: int = 30,
    max_points: Optional[int] = None,
    min_extent: float = 0.02,
    max_extent: float = 0.30,
    # ICP voxel — отдельный параметр, мелкий
    icp_voxel_size: float = 0.003,
    cad_file: Optional[str] = None,
) -> dict:

    results_path = Path(results_dir)
    ensure_dirs(results_path)
    clusters_dir = results_path / "clusters"

    if use_latest:
        input_file = str(get_latest_file(folder))
    elif input_file is None:
        raise ValueError("input_file не задан и use_latest=False")

    print(f"\n{'='*50}")
    print(f"[pipeline] Файл: {input_file}")

    pcd = load_point_cloud(input_file)
    print(f"[1] Загружено:          {len(pcd.points):>8} точек")

    pcd = clean_point_cloud(pcd)
    print(f"[2] После очистки:      {len(pcd.points):>8} точек")

    pcd = crop_roi(pcd,
        x_min=roi_x_min, x_max=roi_x_max,
        y_min=roi_y_min, y_max=roi_y_max,
        z_min=roi_z_min, z_max=roi_z_max,
    )
    print(f"[3] После ROI:          {len(pcd.points):>8} точек")

    if len(pcd.points) == 0:
        print("[pipeline] После ROI — 0 точек. Проверь границы ROI.")
        return _empty_result(input_file, results_dir)

    pcd = voxel_downsample(pcd, voxel_size)
    print(f"[4] После voxel DS:     {len(pcd.points):>8} точек  (voxel={voxel_size})")

    pcd = remove_noise(pcd, nb_neighbors, std_ratio)
    print(f"[5] После noise filter: {len(pcd.points):>8} точек")

    if len(pcd.points) == 0:
        print("[pipeline] После фильтрации — 0 точек.")
        return _empty_result(input_file, results_dir)

    plane_model = None
    if remove_table:
        pcd, plane_model = remove_plane(pcd,
            distance_threshold=plane_distance_threshold,
            ransac_n=plane_ransac_n,
            num_iterations=plane_num_iterations,
        )
        print(f"[6] После RANSAC:       {len(pcd.points):>8} точек")
        if plane_model:
            print(f"    Плоскость: a={plane_model[0]:.3f} b={plane_model[1]:.3f} "
                  f"c={plane_model[2]:.3f} d={plane_model[3]:.3f}")

    if len(pcd.points) == 0:
        clusters = []
    else:
        clusters = cluster_dbscan(pcd,
            eps=eps, min_points=min_points,
            max_points=max_points,
            min_extent=min_extent, max_extent=max_extent,
        )
    print(f"[7] Кластеры:           {len(clusters)} шт  (eps={eps}, min_pts={min_points})")

    cad_model = load_cad_model(cad_file)
    if cad_model:
        print(f"[8] CAD: {cad_file} — {len(cad_model.points)} точек")
        print(f"    pipeline voxel={voxel_size}  icp voxel={icp_voxel_size}")
    else:
        print(f"[8] CAD не задан — OBB fallback")

    clusters_info = []
    for i, cluster in enumerate(clusters):
        info = get_cluster_info(cluster, i)
        pose = estimate_pose_for_cluster(cluster, cad_model, icp_voxel_size)
        info["pose"] = pose
        clusters_info.append(info)
        fitness_str = f"fitness={pose['fitness']:.3f}" if pose.get("fitness") else "fitness=N/A"
        print(f"    Кластер {i}: {info['points_count']} точек | "
              f"метод={pose['method']} | {fitness_str} | "
              f"pos=[{pose['position'][0]:.3f}, "
              f"{pose['position'][1]:.3f}, "
              f"{pose['position'][2]:.3f}]")

    save_cluster_files(clusters, str(clusters_dir))
    annotated_path = create_annotated_pointcloud(pcd, clusters, str(results_path))

    result = {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "input_file": input_file,
        "results_dir": str(results_path),
        "pipeline": {
            "roi": {
                "x": [roi_x_min, roi_x_max],
                "y": [roi_y_min, roi_y_max],
                "z": [roi_z_min, roi_z_max],
            },
            "voxel_size": voxel_size,
            "icp_voxel_size": icp_voxel_size,
            "plane_removed": remove_table,
            "plane_model": plane_model,
        },
        "num_clusters": len(clusters),
        "clusters": clusters_info,
        "annotated_ply": annotated_path,
    }

    save_position_json(result, str(results_path))
    print(f"[pipeline] Готово. Найдено объектов: {len(clusters)}")
    print(f"{'='*50}\n")
    return result


def _empty_result(input_file: str, results_dir: str) -> dict:
    return {
        "status": "empty",
        "timestamp": datetime.now().isoformat(),
        "input_file": input_file,
        "results_dir": results_dir,
        "num_clusters": 0,
        "clusters": [],
    }