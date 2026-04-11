import open3d as o3d
import numpy as np
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.processing import run_icp, estimate_pose_from_obb


def main():
    cluster_path = BASE_DIR / "results" / "clusters" / "cluster_000.ply"

    # Ищем первый PLY в cad_models автоматически — без ручного указания имени
    cad_dir = BASE_DIR / "cad_models"
    cad_files = sorted(cad_dir.glob("*.ply"))
    if not cad_files:
        raise FileNotFoundError(f"Нет PLY файлов в {cad_dir}")

    cad_path = cad_files[0]
    print(f"CAD модель: {cad_path.name}")

    if not cluster_path.exists():
        raise FileNotFoundError(f"Нет кластера: {cluster_path}")

    # --- Загружаем данные ---
    cluster = o3d.io.read_point_cloud(str(cluster_path))
    cad = o3d.io.read_point_cloud(str(cad_path))

    print(f"Кластер:  {len(cluster.points)} точек | "
          f"extent={cluster.get_axis_aligned_bounding_box().get_extent()}")
    print(f"CAD:      {len(cad.points)} точек | "
          f"extent={cad.get_axis_aligned_bounding_box().get_extent()}")

    if len(cluster.points) < 10:
        raise ValueError(f"Слишком мало точек в кластере: {len(cluster.points)}")

    # --- ICP ---
    print("\n" + "="*40)
    print("ICP alignment")
    print("="*40)

    result = run_icp(cluster, cad, voxel_size=0.003)

    print(f"\nРезультат ICP:")
    print(f"  Метод:      {result['method']}")
    print(f"  Позиция:    {[round(v, 4) for v in result['position']]}")
    print(f"  Ориентация: {[round(v, 4) for v in result['orientation']]}")
    if result.get("fitness") is not None:
        print(f"  Fitness:    {result['fitness']:.3f}  "
              f"(>0.5 хорошо, >0.8 отлично)")
        print(f"  RMSE:       {result['inlier_rmse']:.5f}  "
              f"(чем меньше тем точнее)")

    # --- OBB для сравнения ---
    print("\n" + "="*40)
    print("OBB fallback (сравнение)")
    print("="*40)
    obb = estimate_pose_from_obb(cluster)
    print(f"  Позиция: {[round(v, 4) for v in obb['position']]}")

    icp_pos = np.array(result['position'])
    obb_pos = np.array(obb['position'])
    diff = np.linalg.norm(icp_pos - obb_pos)
    print(f"\nРасстояние ICP vs OBB: {diff:.4f} м")
    if diff < 0.005:
        print("  → Позиции почти совпадают")
    else:
        print("  → ICP уточнил позицию относительно OBB")


if __name__ == "__main__":
    main()