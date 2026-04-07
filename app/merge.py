import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Optional, List
from datetime import datetime


# ==============================================================================
# ТРАНСФОРМАЦИЯ МЕЖДУ ПОЗИЦИЯМИ КАМЕРЫ
# ==============================================================================

# Матрица трансформации из позиции B в позицию A (4x4).
#
# Описывает как повёрнута и смещена камера в позиции B
# относительно камеры в позиции A.
#
# Как заполнить:
#   Напарник измеряет физическое расположение двух креплений камеры.
#   Либо через калибровочную мишень (шахматная доска),
#   либо рулеткой + транспортиром приблизительно.
#
#   Формат матрицы 4x4:
#   [ R  | t ]     R — матрица вращения 3x3
#   [ 0  | 1 ]     t — вектор смещения [x, y, z] в метрах
#
# Пример: камера B смещена на 20см вправо и повёрнута на 30° вокруг Y:
#   import numpy as np
#   angle = np.radians(30)
#   T = np.eye(4)
#   T[0, 3] = 0.20   # смещение X
#   T[0, 0] =  np.cos(angle)
#   T[0, 2] =  np.sin(angle)
#   T[2, 0] = -np.sin(angle)
#   T[2, 2] =  np.cos(angle)
#
# Сейчас стоит единичная матрица (заглушка) —
# облака просто складываются без трансформации.
# Это даст задвоение объектов, но цепочка кода работает.

DEFAULT_T_B_TO_A = np.eye(4)  # ЗАГЛУШКА — заменить после калибровки


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

def merge_two_clouds(pcd_a, pcd_b, T_b_to_a=None, voxel_size=0.005):
    T = T_b_to_a if T_b_to_a is not None else DEFAULT_T_B_TO_A

    print("[merge] Начало transform...")
    pcd_b_transformed = o3d.geometry.PointCloud(pcd_b)
    pcd_b_transformed.transform(T)
    print("[merge] Transform готов")

    print("[merge] Начало сложения облаков...")
    merged = pcd_a + pcd_b_transformed
    print(f"[merge] Сложение готово: {len(merged.points)} точек")

    if voxel_size > 0:
        print("[merge] Начало voxel DS...")
        merged = merged.voxel_down_sample(voxel_size)
        print(f"[merge] DS готов: {len(merged.points)} точек")

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

    T = T_b_to_a if T_b_to_a is not None else DEFAULT_T_B_TO_A
    is_stub = T_b_to_a is None or np.allclose(T, np.eye(4))
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