import open3d as o3d
import numpy as np

def crop_points_o3d(points_xyz: np.ndarray, center: np.ndarray, half_side: float) -> np.ndarray:
    pc = o3d.t.geometry.PointCloud(o3d.core.Tensor(points_xyz, device="CUDA:0"))
    min_bound = o3d.core.Tensor(center - half_side, device="CUDA:0")
    max_bound = o3d.core.Tensor(center + half_side, device="CUDA:0")
    aabb = o3d.core.AxisAlignedBoundingBox(min_bound, max_bound)
    idx = pc.get_point_indices_within_bounding_box(aabb)
    return points_xyz[idx.cpu().numpy()]