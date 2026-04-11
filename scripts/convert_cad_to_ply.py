# scripts/convert_cad.py
import open3d as o3d
import numpy as np
from pathlib import Path
import traceback


def mesh_to_pointcloud(input_path: str, output_path: str, num_points: int = 2000):
    print(f"\nОбработка: {input_path}")

    mesh = o3d.io.read_triangle_mesh(input_path)
    print(f"  Вершин: {len(mesh.vertices)}, треугольников: {len(mesh.triangles)}")

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        print(f"  ПРОПУСК: меш пустой")
        return False

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Сэмплируем точки на поверхности треугольников вручную
    # Без вызова Open3D sample_points_uniformly
    points = []

    # Сначала все вершины
    points.append(vertices)

    # Затем случайные точки внутри каждого треугольника
    pts_per_triangle = max(1, num_points // len(triangles))

    for tri in triangles:
        v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        # Случайные барицентрические координаты
        r = np.random.rand(pts_per_triangle, 2)
        # Приводим к треугольнику
        mask = r[:, 0] + r[:, 1] > 1
        r[mask] = 1 - r[mask]
        u, v = r[:, 0:1], r[:, 1:2]
        pts = v0 + u * (v1 - v0) + v * (v2 - v0)
        points.append(pts)

    all_points = np.vstack(points)

    # Если точек больше чем нужно — прореживаем случайно
    if len(all_points) > num_points:
        idx = np.random.choice(len(all_points), num_points, replace=False)
        all_points = all_points[idx]

    print(f"  Точек сгенерировано: {len(all_points)}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)

    # Центрируем в (0, 0, 0)
    center = np.mean(all_points, axis=0)
    all_points -= center
    pcd.points = o3d.utility.Vector3dVector(all_points)

    # Нормализуем масштаб до ~1м
    scale = np.max(np.abs(all_points))
    if scale > 0:
        all_points /= scale
        pcd.points = o3d.utility.Vector3dVector(all_points)

    o3d.io.write_point_cloud(output_path, pcd)
    print(f"  Сохранено: {output_path}")
    print(f"  Extent: {pcd.get_axis_aligned_bounding_box().get_extent()}")
    return True


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    cad_dir = BASE_DIR / "cad_models"

    all_files = list(cad_dir.glob("*.stl")) + list(cad_dir.glob("*.obj"))

    for mesh_file in all_files:
        try:
            out = cad_dir / (mesh_file.stem + ".ply")
            if out.exists():
                print(f"Пропуск (уже есть): {out.name}")
                continue
            mesh_to_pointcloud(str(mesh_file), str(out))
        except Exception as e:
            print(f"ОШИБКА {mesh_file.name}: {e}")
            traceback.print_exc()

    print("\nГотово.")