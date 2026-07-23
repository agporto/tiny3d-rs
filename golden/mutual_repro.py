#!/usr/bin/env python3
"""Reproduce the mutual_filter bug: compare package output against a
numpy-computed ground-truth mutual set (Open3D semantics)."""
import numpy as np
import tiny3d as t3d

g, u, reg = t3d.geometry, t3d.utility, t3d.pipelines.registration
r = np.random.default_rng(9)

# tester-like baseline: target = transformed noisy SUBSET, shuffled order
n = 4000
src_pts = r.random((n, 3))
R = g.get_rotation_matrix_from_xyz(np.array([0.05, -0.04, 0.08]))
keep = r.permutation(n)[: int(n * 0.6)]          # 60% overlap, shuffled
tgt_pts = src_pts[keep] @ R.T + [0.05, 0.03, -0.02] + r.standard_normal((len(keep), 3)) * 0.002

src = g.PointCloud(); src.points = u.Vector3dVector(src_pts)
tgt = g.PointCloud(); tgt.points = u.Vector3dVector(tgt_pts)
for c in (src, tgt):
    c.estimate_normals(g.KDTreeSearchParamKNN(20))
fs = reg.compute_fpfh_feature(src, g.KDTreeSearchParamKNN(40))
ft = reg.compute_fpfh_feature(tgt, g.KDTreeSearchParamKNN(40))

plain = np.asarray(reg.correspondences_from_features(fs, ft))
mutual = np.asarray(reg.correspondences_from_features(fs, ft, mutual_filter=True,
                                                      mutual_consistency_ratio=0.1))

# ground truth mutual set (Open3D semantics) from the two one-way searches
rev = np.asarray(reg.correspondences_from_features(ft, fs))  # target->source
fwd_j = plain[:, 1]
back_i = rev[fwd_j, 1]
true_mutual = plain[back_i == plain[:, 0]]
expected = true_mutual if len(true_mutual) >= int(0.1 * n) else plain

print(f"plain (unfiltered):        {len(plain)}")
print(f"package mutual_filter=True: {len(mutual)}")
print(f"true mutual (numpy):       {len(true_mutual)}")
print(f"expected (with fallback):  {len(expected)}")
print("package == unfiltered:", len(mutual) == len(plain) and np.array_equal(mutual, plain))
print("package == expected:  ", mutual.shape == expected.shape and np.array_equal(mutual, expected))
