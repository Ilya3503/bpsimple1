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

    Для текущей сцены рекомендуемые значения:
        x: -0.5  .. +0.5
        y: -0.25 .. +0.25
        z:  0.50 ..  0.75
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
    voxel_size: float = 0.02,
) -> o3d.geometry.PointCloud:
    """
    Вокселизация. Рекомендуемый размер вокселя: 0.005 (5мм).
    Баланс: достаточно точек для ICP, достаточно быстро для Jetson.
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

    Возвращает:
        pcd_no_plane  — облако без плоскости
        plane_model   — [a, b, c, d] уравнение плоскости (или None)

    Рекомендуемый distance_threshold: 0.01 (1см).
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
    DBSCAN кластеризация с фильтрацией по размеру кластера.

    Параметры для кубов на столе (все в метрах):
        eps=0.03        — 3см, расстояние между точками одного объекта
        min_points=30   — минимум точек в кластере
        min_extent=0.02 — отсекаем мусор меньше 2см
        max_extent=0.30 — отсекаем стены/фон больше 30см

    max_points — жёсткий лимит точек (опционально).
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

        # Фильтр по физическому размеру кластера
        extent = cluster.get_axis_aligned_bounding_box().get_extent()
        max_dim = float(np.max(extent))
        if max_dim < min_extent or max_dim > max_extent:
            continue

        clusters.append(cluster)

    return clusters


def get_cluster_info(cluster: o3d.geometry.PointCloud, cluster_id: int) -> Dict:
    """
    Вычисляет геометрические характеристики кластера.

    Возвращает словарь совместимый с форматом pose estimation:
        center      — центр OBB [x, y, z] в метрах
        extent      — размеры [dx, dy, dz] в метрах
        yaw         — угол поворота вокруг Y (вертикальная ось камеры)
        rotation_matrix — полная матрица вращения OBB (3x3)
        points_count
    """
    obb = cluster.get_oriented_bounding_box()
    center = list(map(float, obb.center))
    extent = list(map(float, obb.extent))
    R = np.asarray(obb.R)
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))

    return {
        "id": cluster_id,
        "center": center,
        "extent": extent,
        "yaw": yaw,
        "rotation_matrix": R.tolist(),
        "points_count": int(len(cluster.points)),
    }


# ==============================================================================
# ICP — ОЦЕНКА ПОЗЫ
# ==============================================================================

def rotation_matrix_to_quaternion(R: np.ndarray) -> List[float]:
    """
    Конвертация матрицы вращения 3x3 в кватернион [qx, qy, qz, qw].
    Используется формула Shepperd.
    """
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
    """
    Извлекает position и orientation (quaternion) из матрицы 4x4.
    """
    position = T[:3, 3].tolist()
    R = T[:3, :3]
    orientation = rotation_matrix_to_quaternion(R)
    return {
        "position": [float(v) for v in position],
        "orientation": orientation,  # [qx, qy, qz, qw]
    }


def load_cad_model(file_path: str) -> Optional[o3d.geometry.PointCloud]:
    """
    Загружает CAD-модель как облако точек.
    Поддерживает .ply и .npy.
    """
    if file_path is None:
        return None
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CAD файл не найден: {file_path}")
    return load_point_cloud(str(path))


def estimate_pose_from_obb(cluster: o3d.geometry.PointCloud) -> Dict:
    """
    Заглушка ICP: оценивает позу объекта по OBB кластера.

    Используется когда CAD-модель недоступна.
    Возвращает тот же формат что и полноценный ICP,
    чтобы robot_controller мог работать без изменений.

    Поле 'method' = 'obb_fallback' сигнализирует что это приближение.
    """
    obb = cluster.get_oriented_bounding_box()
    R = np.asarray(obb.R)

    # Строим матрицу трансформации 4x4
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = obb.center

    pose = transformation_to_pose(T)

    return {
        "method": "obb_fallback",       # заменится на "icp" когда появятся CAD
        "fitness": None,                # ICP fitness score (None для заглушки)
        "inlier_rmse": None,            # ICP RMSE (None для заглушки)
        "transformation": T.tolist(),   # 4x4 матрица
        "position": pose["position"],
        "orientation": pose["orientation"],
        "extent": list(map(float, obb.extent)),
    }


def run_icp(
    cluster: o3d.geometry.PointCloud,
    cad_model: o3d.geometry.PointCloud,
    voxel_size: float = 0.005,
    max_correspondence_distance: float = 0.02,
) -> Dict:
    """
    ICP выравнивание кластера с CAD-моделью.

    Стратегия:
        1. Нормализуем масштаб и центрируем CAD-модель
        2. Начальное приближение: совмещение центров масс
        3. Point-to-plane ICP для точного выравнивания

    Возвращает тот же формат что estimate_pose_from_obb,
    но method='icp' и заполнены fitness/inlier_rmse.

    TODO: раскомментировать и протестировать когда появятся CAD-модели.
    """
    # --- 1. Нормализация CAD модели ---
    cad = o3d.geometry.PointCloud(cad_model)  # копия
    cad_center = cad.get_center()
    cad.translate(-cad_center)  # центрируем в (0,0,0)

    # Масштабируем CAD под размер кластера
    cluster_extent = np.asarray(
        cluster.get_axis_aligned_bounding_box().get_extent()
    )
    cad_extent = np.asarray(
        cad.get_axis_aligned_bounding_box().get_extent()
    )
    scale = np.mean(cluster_extent) / (np.mean(cad_extent) + 1e-9)
    cad.scale(scale, center=np.zeros(3))

    # --- 2. Начальное приближение: совмещение центров масс ---
    cluster_center = cluster.get_center()
    init_T = np.eye(4)
    init_T[:3, 3] = cluster_center

    # --- 3. Нормали (нужны для point-to-plane ICP) ---
    cluster.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    cad.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )

    # --- 4. ICP ---
    result = o3d.pipelines.registration.registration_icp(
        source=cad,
        target=cluster,
        max_correspondence_distance=max_correspondence_distance,
        init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=100
        ),
    )

    T = np.asarray(result.transformation)
    pose = transformation_to_pose(T)

    return {
        "method": "icp",
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
        "transformation": T.tolist(),
        "position": pose["position"],
        "orientation": pose["orientation"],
        "extent": list(map(float, cluster_extent)),
    }


def estimate_pose_for_cluster(
    cluster: o3d.geometry.PointCloud,
    cad_model: Optional[o3d.geometry.PointCloud],
    icp_voxel_size: float = 0.02,
) -> Dict:
    """
    Главная функция оценки позы для одного кластера.
    Автоматически выбирает ICP или OBB-заглушку.
    """
    if cad_model is None:
        return estimate_pose_from_obb(cluster)
    try:
        return run_icp(cluster, cad_model, voxel_size=icp_voxel_size)
    except Exception as e:
        print(f"[ICP] Ошибка, fallback на OBB: {e}")
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

    # Источник данных
    use_latest: bool = True,
    folder: str = "data",

    # ROI (метры)
    roi_x_min: float = -0.5,
    roi_x_max: float = 0.5,
    roi_y_min: float = -0.25,
    roi_y_max: float = 0.25,
    roi_z_min: float = 0.50,
    roi_z_max: float = 0.75,

    # Предобработка
    voxel_size: float = 0.05,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,

    # RANSAC удаление плоскости
    remove_table: bool = True,
    plane_distance_threshold: float = 0.01,
    plane_ransac_n: int = 3,
    plane_num_iterations: int = 1000,

    # DBSCAN
    eps: float = 0.03,
    min_points: int = 30,
    max_points: Optional[int] = None,
    min_extent: float = 0.02,
    max_extent: float = 0.30,

    # ICP / pose estimation
    cad_file: Optional[str] = None,
) -> dict:

    results_path = Path(results_dir)
    ensure_dirs(results_path)
    clusters_dir = results_path / "clusters"

    # --- Загрузка ---
    if use_latest:
        input_file = str(get_latest_file(folder))
    elif input_file is None:
        raise ValueError("input_file не задан и use_latest=False")

    print(f"\n{'='*50}")
    print(f"[pipeline] Файл: {input_file}")

    pcd = load_point_cloud(input_file)
    print(f"[1] Загружено:          {len(pcd.points):>8} точек")

    # --- Очистка NaN/Inf ---
    pcd = clean_point_cloud(pcd)
    print(f"[2] После очистки:      {len(pcd.points):>8} точек")

    # --- ROI crop ---
    pcd = crop_roi(
        pcd,
        x_min=roi_x_min, x_max=roi_x_max,
        y_min=roi_y_min, y_max=roi_y_max,
        z_min=roi_z_min, z_max=roi_z_max,
    )
    print(f"[3] После ROI:          {len(pcd.points):>8} точек")

    if len(pcd.points) == 0:
        print("[pipeline] После ROI — 0 точек. Проверь границы ROI.")
        return _empty_result(input_file, results_dir)

    # --- Voxel downsample ---
    pcd = voxel_downsample(pcd, voxel_size)
    print(f"[4] После voxel DS:     {len(pcd.points):>8} точек  (voxel={voxel_size})")

    # --- Noise removal ---
    pcd = remove_noise(pcd, nb_neighbors, std_ratio)
    print(f"[5] После noise filter: {len(pcd.points):>8} точек")

    if len(pcd.points) == 0:
        print("[pipeline] После фильтрации — 0 точек.")
        return _empty_result(input_file, results_dir)

    # --- RANSAC plane removal ---
    plane_model = None
    if remove_table:
        pcd, plane_model = remove_plane(
            pcd,
            distance_threshold=plane_distance_threshold,
            ransac_n=plane_ransac_n,
            num_iterations=plane_num_iterations,
        )
        print(f"[6] После RANSAC:       {len(pcd.points):>8} точек")
        if plane_model:
            print(f"    Плоскость: a={plane_model[0]:.3f} b={plane_model[1]:.3f} "
                  f"c={plane_model[2]:.3f} d={plane_model[3]:.3f}")

    # --- DBSCAN ---
    if len(pcd.points) == 0:
        clusters = []
    else:
        clusters = cluster_dbscan(
            pcd,
            eps=eps,
            min_points=min_points,
            max_points=max_points,
            min_extent=min_extent,
            max_extent=max_extent,
        )
    print(f"[7] Кластеры:           {len(clusters)} шт  "
          f"(eps={eps}, min_pts={min_points})")

    # --- Pose estimation для каждого кластера ---
    cad_model = load_cad_model(cad_file)
    if cad_model:
        print(f"[8] CAD модель загружена: {cad_file}")
    else:
        print(f"[8] CAD модель не задана — используется OBB fallback")

    clusters_info = []
    for i, cluster in enumerate(clusters):
        info = get_cluster_info(cluster, i)
        pose = estimate_pose_for_cluster(cluster, cad_model, voxel_size)
        info["pose"] = pose
        clusters_info.append(info)
        print(f"    Кластер {i}: {info['points_count']} точек | "
              f"метод={pose['method']} | "
              f"pos=[{pose['position'][0]:.3f}, "
              f"{pose['position'][1]:.3f}, "
              f"{pose['position'][2]:.3f}]")

    # --- Сохранение ---
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
    """Возвращает пустой результат при отсутствии точек."""
    return {
        "status": "empty",
        "timestamp": datetime.now().isoformat(),
        "input_file": input_file,
        "results_dir": results_dir,
        "num_clusters": 0,
        "clusters": [],
    }