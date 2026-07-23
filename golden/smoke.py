#!/usr/bin/env python3
"""Cross-platform smoke test for the tiny3d package.

Functional checks only (no bit-exact comparisons — those are the golden
suite, which is platform-specific to Linux x86_64). Exercises every module:
geometry, kdtree, normals, registration, FPFH/RANSAC, voxel grid, mesh,
I/O round-trips, numpy view semantics.
"""
import os
import sys
import tempfile
import importlib

import numpy as np
import tiny3d as t3d

g, u, reg = t3d.geometry, t3d.utility, t3d.pipelines.registration
rng = np.random.default_rng(12345)
checks = 0


def ok(cond, what):
    global checks
    checks += 1
    if not cond:
        print(f"FAIL: {what}")
        sys.exit(1)
    print(f"  ok: {what}")

def raises(expected, what, fn):
    global checks
    checks += 1
    try:
        fn()
    except expected:
        print(f"  ok: {what}")
        return
    except BaseException as exc:
        print(f"FAIL: {what} raised {type(exc).__name__}, expected {expected.__name__}")
        sys.exit(1)
    print(f"FAIL: {what} did not raise {expected.__name__}")
    sys.exit(1)


print(f"tiny3d {t3d.__version__} on {sys.platform}, python {sys.version.split()[0]}")
for alias, module in [
        ("tiny3d.geometry", g),
        ("tiny3d.io", t3d.io),
        ("tiny3d.utility", u),
        ("tiny3d.pipelines", t3d.pipelines),
        ("tiny3d.pipelines.registration", reg)]:
    ok(importlib.import_module(alias) is module, f"{alias} module alias")
raises(ValueError, "Feature.resize rejects negative dimensions",
       lambda: reg.Feature().resize(-1, 1))
raises(ValueError, "zero quaternion is rejected",
       lambda: g.get_rotation_matrix_from_quaternion(np.zeros(4)))

# --- construction + numpy views ---
pts = rng.random((1000, 3))
pcd = g.PointCloud()
pcd.points = u.Vector3dVector(pts)
ok(len(pcd.points) == 1000, "PointCloud construction")
arr = np.asarray(pcd.points)
ok(arr.shape == (1000, 3) and np.array_equal(arr, pts), "asarray round-trip")
arr[0] = [9.0, 9.0, 9.0]
ok(np.allclose(np.asarray(pcd.points)[0], [9.0, 9.0, 9.0]), "view write-through")
pcd.points[1] = [7.0, 7.0, 7.0]
ok(np.asarray(pcd.points)[1][0] == 7.0, "vector __setitem__ write-through")

# --- bounding volumes / transform ---
aabb = pcd.get_axis_aligned_bounding_box()
ok(np.all(aabb.get_max_bound() >= aabb.get_min_bound()), "AABB")
T = np.eye(4)
T[:3, :3] = g.get_rotation_matrix_from_xyz(np.array([0.1, 0.2, 0.3]))
T[:3, 3] = [1.0, 2.0, 3.0]
c2 = g.PointCloud(pcd)
c2.transform(T)
ok(not np.allclose(np.asarray(c2.points)[5], np.asarray(pcd.points)[5]), "transform")

# --- voxel downsample ---
big = g.PointCloud()
big.points = u.Vector3dVector(rng.random((20000, 3)) * 5.0)
down = big.voxel_down_sample(0.5)
ok(0 < len(down.points) < 20000, "voxel_down_sample")
voxel_grid = g.VoxelGrid.create_from_point_cloud(big, 0.5)
voxel_origin = np.asarray(voxel_grid.origin).copy()
bad_voxel_transform = np.eye(4)
bad_voxel_transform[0, 0] = 2.0
raises(RuntimeError, "VoxelGrid rejects non-translation transforms",
       lambda: voxel_grid.transform(bad_voxel_transform))
ok(np.array_equal(np.asarray(voxel_grid.origin), voxel_origin),
   "rejected VoxelGrid transform is atomic")
raises(RuntimeError, "VoxelGrid rejects rotations",
       lambda: voxel_grid.rotate(
           g.get_rotation_matrix_from_xyz(np.array([0.0, 0.0, 0.1]))))

# --- kdtree ---
tree = g.KDTreeFlann(big)
k, idx, d2 = tree.search_knn_vector_3d(np.asarray(big.points)[0], 10)
ok(k == 10 and idx[0] == 0 and d2[0] == 0.0, "kdtree knn (self is nearest)")
k, idx, d2 = tree.search_radius_vector_3d(np.asarray(big.points)[0], 0.3)
ok(k >= 1 and all(x <= 0.3**2 + 1e-12 for x in d2), "kdtree radius")

# --- normals ---
nc = g.PointCloud()
nc.points = u.Vector3dVector(rng.random((2000, 3)))
nc.estimate_normals(g.KDTreeSearchParamKNN(20))
n = np.asarray(nc.normals)
ok(n.shape == (2000, 3) and np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-9),
   "estimate_normals unit vectors")

# --- registration (ICP p2p + p2l, evaluate) ---
ns = 3000
src = g.PointCloud()
src.points = u.Vector3dVector(rng.random((ns, 3)))
Rm = g.get_rotation_matrix_from_xyz(np.array([0.02, -0.015, 0.03]))
tgt = g.PointCloud()
tgt.points = u.Vector3dVector(np.asarray(src.points) @ Rm.T + [0.01, 0.02, -0.01])
ev = reg.evaluate_registration(src, tgt, 0.1)
ok(ev.fitness > 0.9, "evaluate_registration")
res = reg.registration_icp(src, tgt, 0.1, np.eye(4),
                           reg.TransformationEstimationPointToPoint(False),
                           reg.ICPConvergenceCriteria(1e-10, 1e-10, 30))
ok(res.inlier_rmse < 0.01 and res.fitness > 0.99, "ICP point-to-point converges")
bad_corres = u.Vector2iVector(np.array([[-1, 0], [-1, 1], [-1, 2]], np.int32))
estimator = reg.TransformationEstimationPointToPoint(False)
raises(ValueError, "transformation estimation rejects invalid indices",
       lambda: estimator.compute_rmse(src, tgt, bad_corres))
raises(ValueError, "compute_transformation rejects invalid indices",
       lambda: estimator.compute_transformation(src, tgt, bad_corres))
checker = reg.CorrespondenceCheckerBasedOnDistance(0.1)
raises(ValueError, "correspondence checker rejects invalid indices",
       lambda: checker.Check(src, tgt, bad_corres, np.eye(4)))
raises(ValueError, "RANSAC rejects invalid correspondence indices",
       lambda: reg.registration_ransac_based_on_correspondence(
           src, tgt, bad_corres, 0.1,
           estimator, 3, [],
           reg.RANSACConvergenceCriteria(1, 0.999)))
tgt.estimate_normals(g.KDTreeSearchParamKNN(20))
res = reg.registration_icp(src, tgt, 0.1, np.eye(4),
                           reg.TransformationEstimationPointToPlane(),
                           reg.ICPConvergenceCriteria(1e-10, 1e-10, 30))
ok(res.fitness > 0.99, "ICP point-to-plane converges")

# --- FPFH + RANSAC ---
u.random.seed(3)
rs = g.PointCloud()
rs.points = u.Vector3dVector(rng.random((500, 3)))
rs.estimate_normals(g.KDTreeSearchParamKNN(20))
rt = g.PointCloud()
rt.points = u.Vector3dVector(np.asarray(rs.points) @ Rm.T + [0.05, 0.0, 0.02])
rt.estimate_normals(g.KDTreeSearchParamKNN(20))
fs = reg.compute_fpfh_feature(rs, g.KDTreeSearchParamKNN(40))
ft = reg.compute_fpfh_feature(rt, g.KDTreeSearchParamKNN(40))
ok(np.asarray(fs.data).shape[0] == 33, "FPFH shape")
rr = reg.registration_ransac_based_on_feature_matching(
    rs, rt, fs, ft, False, 0.03,
    reg.TransformationEstimationPointToPoint(False), 3,
    [reg.CorrespondenceCheckerBasedOnDistance(0.03)],
    reg.RANSACConvergenceCriteria(1000, 0.999))
ok(rr.fitness > 0.5, "RANSAC feature matching")

# --- mutual correspondence filtering (Open3D semantics; DIVERGENCES.md #4) ---
fwd = np.asarray(reg.correspondences_from_features(fs, ft))
rev = np.asarray(reg.correspondences_from_features(ft, fs))
mut = np.asarray(reg.correspondences_from_features(fs, ft, mutual_filter=True,
                                                   mutual_consistency_ratio=0.1))
expected = fwd[rev[fwd[:, 1], 1] == fwd[:, 0]]
if len(expected) < int(np.float32(0.1) * np.float32(len(fwd))):
    expected = fwd  # documented ratio fallback
ok(np.array_equal(mut, expected), "mutual filter keeps exactly the mutual pairs")

# --- mesh ---
mesh = g.TriangleMesh()
mesh.vertices = u.Vector3dVector(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float))
mesh.triangles = u.Vector3iVector(np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int32))
mesh.compute_vertex_normals()
ok(np.asarray(mesh.vertex_normals).shape == (4, 3), "mesh vertex normals")

# --- I/O round-trips ---
tmp = tempfile.mkdtemp()
bad_cloud_path = os.path.join(tmp, "partial-cloud.ply")
with open(bad_cloud_path, "w", encoding="ascii") as f:
    f.write("""ply
format ascii 1.0
element vertex 2
property float x
property float y
property float z
end_header
0 0 0
""")
bad_cloud = t3d.io.read_point_cloud(bad_cloud_path)
ok(len(bad_cloud.points) == 0, "failed PLY read returns no partial point cloud")
bad_mesh_path = os.path.join(tmp, "partial.ply")
with open(bad_mesh_path, "w", encoding="ascii") as f:
    f.write("""ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 2
property list uchar int vertex_indices
end_header
0 0 0
1 0 0
0 1 0
2 0 0
3 0 1 2
4 0 0 0 0
""")
bad_mesh = t3d.io.read_triangle_mesh(bad_mesh_path)
ok(len(bad_mesh.vertices) == 0 and len(bad_mesh.triangles) == 0,
   "failed PLY triangulation returns no partial geometry")
io_cloud = g.PointCloud()
io_cloud.points = u.Vector3dVector(rng.random((500, 3)))
io_cloud.normals = u.Vector3dVector(rng.random((500, 3)))
io_cloud.colors = u.Vector3dVector(rng.random((500, 3)))
raises(ValueError, "unsupported byte format is rejected",
       lambda: t3d.io.write_point_cloud_to_bytes(io_cloud, format="auto"))
# binary PLY is lossless (doubles); ascii PLY (%g) and xyz (%.10f) are not
for name, kw, atol in [("b.ply", {}, 0.0),
                       ("a.ply", {"write_ascii": True}, 1e-5),
                       ("c.xyz", {}, 1e-9)]:
    path = os.path.join(tmp, name)
    ok(t3d.io.write_point_cloud(path, io_cloud, **kw), f"write {name}")
    back = t3d.io.read_point_cloud(path)
    ok(len(back.points) == 500 and
       np.allclose(np.asarray(back.points), np.asarray(io_cloud.points), atol=atol),
       f"read-back {name}")

print(f"SMOKE OK ({checks} checks)")
