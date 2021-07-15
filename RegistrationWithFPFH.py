from typing import KeysView
import open3d as o3d
import copy
import numpy as np
# from open3d.pipelines.registration import registration_ransac_based_on_feature_matching as RANSAC
# from open3d.pipelines.registration import registration_icp as ICP
# from open3d.pipelines.registration import compute_fpfh_feature as FPFH

def show(model, scene, model_to_scene_trans=np.identity(4)):
    model_t = copy.deepcopy(model)
    scene_t = copy.deepcopy(scene)

    model_t.paint_uniform_color([1, 0, 0])
    scene_t.paint_uniform_color([0, 0, 1])

    model_t.transform(model_to_scene_trans)

    o3d.visualization.draw_geometries([model_t, scene_t],width=1200,height=800)

src = o3d.io.read_point_cloud("./bunny/data/bun000.ply")
tgt = o3d.io.read_point_cloud("./bunny/data/bun045.ply")

size = np.abs((src.get_max_bound() - src.get_min_bound())).max() / 20
kdt_n = o3d.geometry.KDTreeSearchParamHybrid(radius=size, max_nn=50)
kdt_f = o3d.geometry.KDTreeSearchParamHybrid(radius=size * 50, max_nn=50)

src.estimate_normals(kdt_n)
tgt.estimate_normals(kdt_n)

src_d = src.voxel_down_sample(size)
tgt_d = tgt.voxel_down_sample(size)

src_d.estimate_normals(kdt_n)
tgt_d.estimate_normals(kdt_n)

src_f = o3d.pipelines.registration.compute_fpfh_feature(src_d,kdt_f)
tgt_f = o3d.pipelines.registration.compute_fpfh_feature(tgt_d,kdt_f)

checker = [
    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(size * 2)
]

eptp  = o3d.pipelines.registration.TransformationEstimationPointToPoint()
eptpl = o3d.pipelines.registration.TransformationEstimationPointToPlane()

criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(
    max_iteration=100000,
    confidence=9.990000e-01
)

result1 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    src_d, tgt_d,
    src_f, tgt_f,
    mutual_filter = False,
    max_correspondence_distance=size * 2,
    estimation_method=eptp,
    ransac_n=4,
    checkers=checker,
    criteria=criteria)

show(src_d, tgt_d, result1.transformation)

