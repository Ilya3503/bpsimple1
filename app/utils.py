# def remove_plane(pcd: o3d.geometry.PointCloud,
#                  distance_threshold: float = 0.01,
#                  ransac_n: int = 3,
#                  num_iterations: int = 1000) -> o3d.geometry.PointCloud:
#     pts = np.asarray(pcd.points)
#     if pts.shape[0] < ransac_n:
#         # Недостаточно точек для сегментации плоскости
#         print(f"[remove_plane] Пропуск сегментации плоскости: точек {pts.shape[0]} < ransac_n {ransac_n}")
#         return pcd
#     try:
#         plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
#                                                  ransac_n=ransac_n,
#                                                  num_iterations=num_iterations)
#         return pcd.select_by_index(inliers, invert=True)
#     except Exception as e:
#         print(f"[remove_plane] Ошибка сегментации плоскости: {e}")
#         return pcd
#
#
#
#
#
#     distance_threshold: float = 70.0,
#     ransac_n: int = 3,
#     num_iterations: int = 1000,
#
#
#
#     distance_threshold: float = Query(70.0, description="Порог расстояния при удалении плоскости"),
#     ransac_n: int = Query(3, description="Количество точек для RANSAC"),
#     num_iterations: int = Query(1000, description="Количество итераций RANSAC"),
#
#
#
#
#     # Удаление плоскости
#     pcd = remove_plane(pcd, distance_threshold, ransac_n, num_iterations)
#     print(f"[process] После удаления плоскости (distance_threshold={distance_threshold}): {len(pcd.points)} точек")
#
#     distance_threshold = distance_threshold,
#     ransac_n = ransac_n,
#     num_iterations = num_iterations,
#
#
# min_bound_x: float = Query(-231, description="Мин. X для обрезки"),
# min_bound_y: float = Query(-190, description="Мин. Y для обрезки"),
# min_bound_z: float = Query(474, description="Мин. Z для обрезки"),
# max_bound_x: float = Query(264, description="Макс. X для обрезки"),
# max_bound_y: float = Query(190, description="Макс. Y для обрезки"),
# max_bound_z: float = Query(670, description="Макс. Z для обрезки"),
#
# min_bound: tuple = (-231, -190, 474),
# max_bound: tuple = (264, 190, 670),
#
#
#
#
# def crop_points_numpy(pcd: o3d.geometry.PointCloud,
#                       min_bound: Optional[tuple] = None,
#                       max_bound: Optional[tuple] = None) -> o3d.geometry.PointCloud:
#     if min_bound is None or max_bound is None:
#         return pcd
#     pcd = clean_point_cloud(pcd)
#     pts = np.asarray(pcd.points)
#     if pts.size == 0:
#         return o3d.geometry.PointCloud()
#     mask = (
#         (pts[:, 0] >= min_bound[0]) & (pts[:, 0] <= max_bound[0]) &
#         (pts[:, 1] >= min_bound[1]) & (pts[:, 1] <= max_bound[1]) &
#         (pts[:, 2] >= min_bound[2]) & (pts[:, 2] <= max_bound[2])
#     )
#     cropped = o3d.geometry.PointCloud()
#     cropped.points = o3d.utility.Vector3dVector(pts[mask])
#     return cropped
#
#
#
# #min_bound = (min_bound_x, min_bound_y, min_bound_z)
#         max_bound = (max_bound_x, max_bound_y, max_bound_z)
# #
# #
# #
#     pcd = crop_points_numpy(pcd, min_bound, max_bound)
#     print(f"[process] После обрезки по границам {min_bound} - {max_bound}: {len(pcd.points)} точек")