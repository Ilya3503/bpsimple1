# scripts/convert_cad.py
import open3d as o3d
import numpy as np
from pathlib import Path



def mesh_to_pointcloud(
        input_path: str,
        output_path: str,
        num_points: int = 5000,
):
    mesh = o3d.io.read_triangle_mesh(input_path)

    if len(mesh.vertices) == 0:
        raise ValueError(f"Меш пустой или не загрузился: {input_path}")

    mesh.compute_vertex_normals()

    # Сэмплируем точки равномерно по поверхности
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)

    # Центрируем модель в (0, 0, 0)
    center = pcd.get_center()
    pcd.translate(-center)

    # Нормализуем масштаб — приводим к размеру ~1м
    # В run_icp масштаб подгонится под реальный кластер
    pts = np.asarray(pcd.points)
    scale = np.max(np.abs(pts))
    if scale > 0:
        pcd.scale(1.0 / scale, center=np.zeros(3))

    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Конвертировано: {input_path}")
    print(f"  → {output_path}")
    print(f"  Точек: {len(pcd.points)}")
    print(f"  Extent: {pcd.get_axis_aligned_bounding_box().get_extent()}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    cad_dir = BASE_DIR / "cad_models"

    for stl_file in cad_dir.glob("*.stl"):
        out = cad_dir / (stl_file.stem + ".ply")
        mesh_to_pointcloud(str(stl_file), str(out))

    for obj_file in cad_dir.glob("*.obj"):
        out = cad_dir / (obj_file.stem + ".ply")
        mesh_to_pointcloud(str(obj_file), str(out))

    print("\nГотово. PLY файлы сохранены в cad_models/")