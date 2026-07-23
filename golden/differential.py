#!/usr/bin/env python3
"""Randomized differential battery: run with both the C++ and Rust tiny3d on
PYTHONPATH and compare dumps. Usage: differential.py <seed> <outfile>"""
import sys, os, struct
import numpy as np
import tiny3d as t3d

g, u, reg = t3d.geometry, t3d.utility, t3d.pipelines.registration
seed = int(sys.argv[1])
out = open(sys.argv[2], "w")
r = np.random.default_rng(seed)

def dump(tag, arr):
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    out.write(f"{tag} {a.shape} {a.tobytes().hex()[:200000]}\n")

def dumpi(tag, arr):
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.int64))
    out.write(f"{tag} {a.shape} {a.tobytes().hex()[:200000]}\n")

def cloud(n, scale=1.0, colors=False, normals=False):
    p = g.PointCloud()
    p.points = u.Vector3dVector(r.random((n, 3)) * scale)
    if colors:
        p.colors = u.Vector3dVector(r.random((n, 3)))
    if normals:
        p.normals = u.Vector3dVector(r.standard_normal((n, 3)))
    return p

# 1. big voxel down (hash growth deep) — 20k points, small voxels
p = cloud(20000, scale=10.0, colors=True, normals=True)
q = p.voxel_down_sample(0.11)
dump("bigvoxel_pts", q.points)
dump("bigvoxel_col", q.colors)
dump("bigvoxel_nrm", q.normals)

# 2. big voxelgrid
vg = g.VoxelGrid.create_from_point_cloud(p, 0.13)
vox = vg.get_voxels()
gi = np.array([v.grid_index for v in vox])
dumpi("bigvg_idx", gi)
dump("bigvg_center", vg.get_center())

# 3. kdtree stress: clustered + duplicated points
pts = np.concatenate([
    r.random((3000, 3)),
    np.repeat(r.random((50, 3)), 10, axis=0),  # duplicates
    r.random((500, 3)) * 0.01 + 0.5,           # dense cluster
])
pc = g.PointCloud(); pc.points = u.Vector3dVector(pts)
tree = g.KDTreeFlann(pc)
for qi in range(30):
    qp = r.random(3)
    k, idx, dist = tree.search_knn_vector_3d(qp, int(r.integers(1, 40)))
    dumpi(f"kdt_knn{qi}_i", idx); dump(f"kdt_knn{qi}_d", dist)
    k, idx, dist = tree.search_radius_vector_3d(qp, float(r.random() * 0.3))
    dumpi(f"kdt_rad{qi}_i", idx); dump(f"kdt_rad{qi}_d", dist)
    k, idx, dist = tree.search_hybrid_vector_3d(qp, float(r.random() * 0.3), int(r.integers(1, 30)))
    dumpi(f"kdt_hyb{qi}_i", idx); dump(f"kdt_hyb{qi}_d", dist)

# 4. normals on structured surface (plane + noise: near-degenerate covariances)
n = 2000
xy = r.random((n, 2))
z = 0.001 * r.standard_normal(n)
sp = g.PointCloud()
sp.points = u.Vector3dVector(np.column_stack([xy, z]))
sp.normals = u.Vector3dVector(np.tile([0.0, 0.0, 1.0], (n, 1)))
sp.estimate_normals(g.KDTreeSearchParamHybrid(radius=0.08, max_nn=25))
dump("plane_normals", sp.normals)
sp2 = g.PointCloud(sp)
sp2.estimate_normals(g.KDTreeSearchParamKNN(12), fast_normal_computation=False)
dump("plane_normals_slow", sp2.normals)

# 5. FPFH + registration on plane-ish data
src = cloud(800)
R = g.get_rotation_matrix_from_axis_angle(r.standard_normal(3) * 0.1)
tgt_pts = np.asarray(src.points) @ R.T + r.standard_normal(3) * 0.05 + r.standard_normal((800, 3)) * 0.002
tgt = g.PointCloud(); tgt.points = u.Vector3dVector(tgt_pts)
for c in (src, tgt):
    c.normals = u.Vector3dVector(np.tile([0.0, 0.0, 1.0], (800, 1)))
    c.estimate_normals(g.KDTreeSearchParamKNN(20))
fs = reg.compute_fpfh_feature(src, g.KDTreeSearchParamHybrid(radius=0.2, max_nn=40))
ft = reg.compute_fpfh_feature(tgt, g.KDTreeSearchParamHybrid(radius=0.2, max_nn=40))
dump("fpfh_src", fs.data)
# one-way correspondences both directions (bit-exact in both builds).
# mutual_filter=True is intentionally divergent (DIVERGENCES.md #4: the C++
# mutual check is broken) and is covered by the golden suite instead.
corr = reg.correspondences_from_features(fs, ft)
dumpi("fm_corr", corr)
corr_rev = reg.correspondences_from_features(ft, fs)
dumpi("fm_corr_rev", corr_rev)
u.random.seed(seed * 7 + 1)
res = reg.registration_ransac_based_on_feature_matching(
    src, tgt, fs, ft, False, 0.1,
    reg.TransformationEstimationPointToPoint(False), 4,
    [reg.CorrespondenceCheckerBasedOnDistance(0.1),
     reg.CorrespondenceCheckerBasedOnEdgeLength(0.8),
     reg.CorrespondenceCheckerBasedOnNormal(0.6)],
    reg.RANSACConvergenceCriteria(2000, 0.99))
dump("ransac_T", res.transformation)
dump("ransac_f", [res.fitness, res.inlier_rmse])
dumpi("ransac_cs", res.correspondence_set)

# 6. ICP chains
res = reg.registration_icp(src, tgt, 0.3, np.eye(4),
                           reg.TransformationEstimationPointToPoint(True),
                           reg.ICPConvergenceCriteria(1e-9, 1e-9, 25))
dump("icp_scaled_T", res.transformation)
res = reg.registration_icp(src, tgt, 0.3, res.transformation,
                           reg.TransformationEstimationPointToPlane())
dump("icp_p2l_T", res.transformation)
info = reg.get_information_matrix_from_point_clouds(src, tgt, 0.3, res.transformation)
dump("infomat", info)

# 7. PLY round trips with extreme values
import tempfile
tmp = tempfile.mkdtemp()
ext = g.PointCloud()
vals = np.array([
    [1e-308, -1e-308, 5e-324],       # denormals
    [1e308, -1e308, 0.0],
    [-0.0, 0.0, 123456789.123456789],
    [1.5e-10, 2.5, 3.5],
    [np.pi, np.e, -np.sqrt(2)],
])
ext.points = u.Vector3dVector(np.vstack([vals, r.random((20, 3))]))
ext.colors = u.Vector3dVector(np.clip(r.random((25, 3)) * 1.5 - 0.2, -0.5, 1.5))
for tag, ascii_mode in [("a", True), ("b", False)]:
    path = os.path.join(tmp, f"ext_{tag}.ply")
    t3d.io.write_point_cloud(path, ext, write_ascii=ascii_mode)
    data = open(path, "rb").read()
    out.write(f"ply_{tag}_bytes {len(data)} {data.hex()[:100000]}\n")
    p2 = t3d.io.read_point_cloud(path)
    dump(f"ply_{tag}_rt", p2.points)
xyzp = os.path.join(tmp, "e.xyz")
t3d.io.write_point_cloud(xyzp, ext)
out.write("xyz_bytes " + open(xyzp, "rb").read().hex()[:100000] + "\n")
dump("xyz_rt", np.asarray(t3d.io.read_point_cloud(xyzp).points))

# 8. mesh (ulp-sensitive keys use rounded comparison)
m = g.TriangleMesh()
m.vertices = u.Vector3dVector(r.random((200, 3)))
m.triangles = u.Vector3iVector(r.integers(0, 200, (400, 3)))
m.compute_vertex_normals()
vn = np.asarray(m.vertex_normals)
# normalize is +-1ulp alignment-dependent in the C++; compare at reduced precision
out.write("mesh_vn_round " + np.round(vn, 12).tobytes().hex()[:200000] + "\n")
mp = os.path.join(tmp, "m.ply")
t3d.io.write_triangle_mesh(mp, m)
m2 = t3d.io.read_triangle_mesh(mp)
dump("mesh_rt_v", m2.vertices)
dumpi("mesh_rt_t", m2.triangles)

# 9. transforms and bounds on the big cloud
p9 = cloud(5000, scale=3.0, normals=True)
T = np.eye(4)
T[:3, :3] = g.get_rotation_matrix_from_quaternion(r.standard_normal(4)) * 1.3
T[:3, 3] = r.standard_normal(3)
p9.transform(T)
dump("big_transform", p9.points)
dump("big_tnormals", p9.normals)
bb = p9.get_axis_aligned_bounding_box()
dump("big_bb", np.concatenate([bb.get_min_bound(), bb.get_max_bound()]))

out.close()
print("done", sys.argv[2])
