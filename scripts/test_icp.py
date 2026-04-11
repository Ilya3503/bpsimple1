# scripts/test_icp.py

import open3d as o3d
import numpy as np
import sys
from pathlib import Path

# --- Базовая директория проекта (корень) ---
BASE_DIR = Path(__file__).resolve().parent.parent

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(BASE_DIR))

# Теперь импорт будет стабильным
from app.processing import run_icp, estimate_pose_from_obb


def main():
    # --- Пути к данным ---
    cluster_path = BASE_DIR / "results" / "clusters" / "cluster_000.ply"
    cad_path = BASE_DIR / "cad_models" / "Cube_30x30x30.ply"

    # --- Проверка существования файлов ---
    if not cluster_path.exists():
        raise FileNotFoundError(f"Не найден файл кластера: {cluster_path}")

    if not cad_path.exists():
        raise FileNotFoundError(f"Не найден CAD файл: {cad_path}")

    # --- Загружаем кластер ---
    cluster = o3d.io.read_point_cloud(str(cluster_path))
    print(f"Кластер: {len(cluster.points)} точек")
    print(f"Extent: {cluster.get_axis_aligned_bounding_box().get_extent()}")

    # --- Загружаем CAD ---
    cad = o3d.io.read_point_cloud(str(cad_path))
    print(f"CAD: {len(cad.points)} точек")

    # --- Запускаем ICP ---
    print("\n--- ICP ---")
    result = run_icp(cluster, cad, voxel_size=0.005)

    print(f"Метод: {result['method']}")
    print(f"Позиция: {result['position']}")
    print(f"Ориентация: {result['orientation']}")

    if result.get("fitness") is not None:
        print(f"Fitness: {result['fitness']:.3f}")
        print(f"RMSE: {result['inlier_rmse']:.4f}")

    # --- Сравнение с OBB ---
    print("\n--- OBB fallback для сравнения ---")
    obb = estimate_pose_from_obb(cluster)
    print(f"Позиция OBB: {obb['position']}")


if __name__ == "__main__":
    main()