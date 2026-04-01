import open3d as o3d
from .utils import save_pcd

def voxel_downsample(pcd, voxel_size=0.01, save=False):
    pcd = pcd.voxel_down_sample(voxel_size)
    if save: save_pcd(pcd, "voxel")
    return pcd

def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0, save=False):
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    if save: save_pcd(pcd, "outliers")
    return pcd

def remove_plane(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000, save=False):
    plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
    pcd = pcd.select_by_index(inliers, invert=True)
    if save: save_pcd(pcd, "plane_removed")
    return pcd

def crop_roi(pcd, min_bound, max_bound, save=False):
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    pcd = pcd.crop(bbox)
    if save: save_pcd(pcd, "cropped")
    return pcd

def cluster_dbscan(pcd, eps=0.02, min_samples=10, save=False):
    import numpy as np
    from sklearn.cluster import DBSCAN
    points = np.asarray(pcd.points)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    if save: save_pcd(pcd, "dbscan")
    return labels

def apply_icp(source_pcd, target_pcd, threshold=0.02, save=False):
    reg = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    if save: save_pcd(source_pcd, "icp_aligned")
    return reg.transformation