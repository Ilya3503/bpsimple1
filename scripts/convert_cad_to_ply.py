# scripts/convert_cad.py
import open3d as o3d
import numpy as np
from pathlib import Path
import traceback


def mesh_to_pointcloud(input_path: str, output_path: str, num_points: int = 5000):
    print(f"\nОбработка: {input_path}")

    # Читаем меш
    mesh = o3d.io.read_triangle_mesh(input_path)
    print(f"  Вершин: {len(mesh.vertices)}, треугольников: {len(mesh.triangles)}")

    if len(mesh.vertices) == 0:
        print(f"  ПРОПУСК: меш пустой")
        return False

    if len(mesh.triangles) == 0:
        print(f"  ПРОПУСК: нет треугольников")
        return False

    mesh.compute_vertex_normals()

    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    print(f"  Точек сэмплировано: {len(pcd.points)}")

    center = pcd.get_center()
    pcd.translate(-center)

    pts = np.asarray(pcd.points)
    scale = np.max(np.abs(pts))
    if scale > 0:
        pcd.scale(1.0 / scale, center=np.zeros(3))

    o3d.io.write_point_cloud(output_path, pcd)
    print(f"  Сохранено: {output_path}")
    print(f"  Extent: {pcd.get_axis_aligned_bounding_box().get_extent()}")
    return True


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    cad_dir = BASE_DIR / "cad_models"

    print(f"Папка: {cad_dir}")
    print(f"Файлы:")

    all_files = list(cad_dir.glob("*.stl")) + list(cad_dir.glob("*.obj"))

    if not all_files:
        print("  Файлов не найдено")
    else:
        for f in all_files:
            print(f"  {f.name}")

    print("\n--- Конвертация ---")

    for mesh_file in all_files:
        try:
            out = cad_dir / (mesh_file.stem + ".ply")
            mesh_to_pointcloud(str(mesh_file), str(out))
        except Exception as e:
            print(f"  ОШИБКА: {e}")
            traceback.print_exc()

    print("\nГотово.")