# scripts/convert_cad.py
import open3d as o3d
import numpy as np
from pathlib import Path
import traceback


def mesh_to_pointcloud(input_path: str, output_path: str, num_points: int = 5000):
    print(f"\nОбработка: {input_path}")

    mesh = o3d.io.read_triangle_mesh(input_path)
    print(f"  Вершин: {len(mesh.vertices)}, треугольников: {len(mesh.triangles)}")

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        print(f"  ПРОПУСК: меш пустой")
        return False

    mesh.compute_vertex_normals()

    # Ограничиваем num_points до разумного максимума
    # для простых мешей с 12 треугольниками 5000 точек — это segfault
    max_safe = len(mesh.triangles) * 100
    actual_points = min(num_points, max_safe)
    print(f"  Сэмплируем {actual_points} точек (макс безопасное: {max_safe})")

    pcd = mesh.sample_points_uniformly(number_of_points=actual_points)
    print(f"  Точек получено: {len(pcd.points)}")

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

    all_files = (
        list(cad_dir.glob("*.stl")) +
        list(cad_dir.glob("*.obj"))
    )

    for mesh_file in all_files:
        try:
            out = cad_dir / (mesh_file.stem + ".ply")
            if out.exists():
                print(f"  Пропуск (уже есть): {out.name}")
                continue
            mesh_to_pointcloud(str(mesh_file), str(out))
        except Exception as e:
            print(f"  ОШИБКА {mesh_file.name}: {e}")
            traceback.print_exc()

    print("\nГотово.")