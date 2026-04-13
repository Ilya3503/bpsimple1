import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Optional, List
from datetime import datetime


# ==============================================================================
# ТРАНСФОРМАЦИЯ МЕЖДУ ПОЗИЦИЯМИ КАМЕРЫ
# ==============================================================================

CALIBRATION_FILE = Path(__file__).resolve().parent / "T_camB_to_camA.npy"

try:
    T_camB_to_camA = np.load(str(CALIBRATION_FILE))
    print(f"[calibration] Загружена трансформация: {CALIBRATION_FILE}")
    print(f"[calibration] Матрица:\n{np.round(T_camB_to_camA, 4)}")
    _transform_is_stub = np.allclose(T_camB_to_camA, np.eye(4))
except FileNotFoundError:
    print(f"[calibration] Файл не найден: {CALIBRATION_FILE}")
    print("[calibration] Используется заглушка (единичная матрица)")
    T_camB_to_camA = np.eye(4)
    _transform_is_stub = True

# ==============================================================================
# ЗАГРУЗКА ФАЙЛОВ
# ==============================================================================

def get_two_latest_files(folder: str, extension: str = ".ply") -> List[Path]:
    """
    Возвращает два последних файла из папки по времени изменения.
    Первый — самый новый, второй — предыдущий.
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Папка не найдена: {folder}")

    files = sorted(
        folder_path.glob(f"*{extension}"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    if len(files) < 2:
        raise FileNotFoundError(
            f"Нужно минимум 2 файла {extension} в папке {folder}, "
            f"найдено: {len(files)}"
        )

    return [files[0], files[1]]


def load_pcd(path: str) -> o3d.geometry.PointCloud:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    pcd = o3d.io.read_point_cloud(str(p))
    if len(pcd.points) == 0:
        raise ValueError(f"Файл пустой: {path}")
    return pcd


# ==============================================================================
# ОБЪЕДИНЕНИЕ ОБЛАКОВ
# ==============================================================================

def merge_two_clouds(pcd_a, pcd_b, T_b_to_a=None, voxel_size=0.0):
    T = T_b_to_a if T_b_to_a is not None else T_camB_to_camA

    print("[merge] Начало transform...")
    # Трансформируем через numpy — без копирования Open3D объекта
    pts_b = np.asarray(pcd_b.points).copy()
    ones = np.ones((pts_b.shape[0], 1))
    pts_b_h = np.hstack([pts_b, ones])       # homogeneous coordinates
    pts_b_transformed = (T @ pts_b_h.T).T[:, :3]
    print("[merge] Transform готов")

    print("[merge] Начало сложения облаков...")
    pts_a = np.asarray(pcd_a.points)
    pts_merged = np.vstack([pts_a, pts_b_transformed])

    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(pts_merged)
    print(f"[merge] Сложение готово: {len(merged.points)} точек")

    return merged


def merge_point_cloud_files(
    file_a: str,
    file_b: str,
    output_dir: str = "data",
    T_b_to_a: Optional[np.ndarray] = None,
    voxel_size: float = 0.005,
) -> str:
    """
    Загружает два файла, объединяет и сохраняет результат.
    Даунсэмплинг применяется ДО merge чтобы не вешать память.
    """
    pcd_a = load_pcd(file_a)
    pcd_b = load_pcd(file_b)

    print(f"[merge] Файл A: {file_a} — {len(pcd_a.points)} точек")
    print(f"[merge] Файл B: {file_b} — {len(pcd_b.points)} точек")

    # --- Даунсэмплинг ДО merge ---
    if voxel_size > 0:
        pcd_a = pcd_a.voxel_down_sample(voxel_size)
        pcd_b = pcd_b.voxel_down_sample(voxel_size)
        print(f"[merge] После DS: A={len(pcd_a.points)} B={len(pcd_b.points)} точек")

    T = T_b_to_a if T_b_to_a is not None else T_camB_to_camA
    is_stub = T_b_to_a is None or np.allclose(T_camB_to_camA, np.eye(4))
    if is_stub:
        print("[merge] ВНИМАНИЕ: используется единичная трансформация (заглушка)")

    merged = merge_two_clouds(pcd_a, pcd_b, T, voxel_size=0)  # DS уже сделан
    print(f"[merge] После merge: {len(merged.points)} точек")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = Path(output_dir) / f"merged_{timestamp}.ply"
    print("[merge] Начало записи файла...")
    o3d.io.write_point_cloud(str(out_path), merged)
    print("[merge] Файл записан")
    print(f"[merge] Сохранено: {out_path}")

    return str(out_path), is_stub