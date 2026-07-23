#!/usr/bin/env python3
"""Tester-style low-overlap RANSAC scenario on a structured surface:
does correct mutual filtering rescue registration?"""
import sys, time
import numpy as np
import tiny3d as t3d

g, u, reg = t3d.geometry, t3d.utility, t3d.pipelines.registration
r = np.random.default_rng(21)

pts = np.load("/tmp/surf.npy")
n = len(pts)
R = g.get_rotation_matrix_from_xyz(np.array([0.3, -0.2, 0.4]))
# low overlap: source = left 60% of the surface, target = right 60%, transformed
src_pts = pts[pts[:, 0] < 2.4]
tgt_pts = pts[pts[:, 0] > 1.6] @ R.T + [0.2, 0.1, -0.3]
tgt_pts = tgt_pts[r.permutation(len(tgt_pts))]

src = g.PointCloud(); src.points = u.Vector3dVector(src_pts)
tgt = g.PointCloud(); tgt.points = u.Vector3dVector(tgt_pts)
for c in (src, tgt):
    c.estimate_normals(g.KDTreeSearchParamKNN(20))
fs = reg.compute_fpfh_feature(src, g.KDTreeSearchParamHybrid(radius=0.25, max_nn=60))
ft = reg.compute_fpfh_feature(tgt, g.KDTreeSearchParamHybrid(radius=0.25, max_nn=60))

fwd = np.asarray(reg.correspondences_from_features(fs, ft))
rev = np.asarray(reg.correspondences_from_features(ft, fs))
true_mut = fwd[rev[fwd[:, 1], 1] == fwd[:, 0]]
corr_mut = np.asarray(reg.correspondences_from_features(fs, ft, mutual_filter=True,
                                                        mutual_consistency_ratio=0.1))
print(f"src={len(src_pts)} tgt={len(tgt_pts)} | plain={len(fwd)} "
      f"true_mutual={len(true_mut)} package_mutual={len(corr_mut)}")

for label, mf in [("mutual=False", False), ("mutual=True", True)]:
    u.random.seed(3)
    t0 = time.perf_counter()
    res = reg.registration_ransac_based_on_feature_matching(
        src, tgt, fs, ft, mf, 0.05,
        reg.TransformationEstimationPointToPoint(False), 3,
        [reg.CorrespondenceCheckerBasedOnDistance(0.05),
         reg.CorrespondenceCheckerBasedOnEdgeLength(0.9)],
        reg.RANSACConvergenceCriteria(40000, 0.999))
    dt = time.perf_counter() - t0
    err_R = np.abs(res.transformation[:3, :3] - R).max()
    print(f"{label:13s} fitness={res.fitness:.3f} rmse={res.inlier_rmse:.5f} "
          f"rot_err={err_R:.2e} time={dt*1000:.0f}ms")
