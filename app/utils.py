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