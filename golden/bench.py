#!/usr/bin/env python3
"""Benchmark battery for tiny3d (run under either the C++ or Rust package).
Usage: bench.py <label> [--quick]"""
import sys, time, json, os, tempfile
import numpy as np
import tiny3d as t3d

g, u, reg = t3d.geometry, t3d.utility, t3d.pipelines.registration
label = sys.argv[1]
quick = "--quick" in sys.argv

results = {}

def bench(name, fn, repeat=3, number=1):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        dt = (time.perf_counter() - t0) / number
        best = min(best, dt)
    results[name] = best
    print(f"{label:12s} {name:34s} {best*1000:10.2f} ms", flush=True)

r = np.random.default_rng(0)

def make_cloud(n, scale=1.0, normals=False):
    p = g.PointCloud()
    p.points = u.Vector3dVector(r.random((n, 3)) * scale)
    if normals:
        p.normals = u.Vector3dVector(np.tile([0.0, 0.0, 1.0], (n, 1)))
    return p

N = 50000 if not quick else 5000

# --- point ops ---
big = make_cloud(200000, scale=5.0)
bigc = g.PointCloud(big)
bigc.colors = u.Vector3dVector(r.random((200000, 3)))
bench("voxel_down_200k", lambda: bigc.voxel_down_sample(0.05))

T = np.eye(4); T[:3, :3] = g.get_rotation_matrix_from_xyz(np.array([0.1, 0.2, 0.3]))
def do_transform():
    c = g.PointCloud(big)
    c.transform(T)
bench("transform_200k", do_transform)

# --- kdtree ---
kd_cloud = make_cloud(100000)
bench("kdtree_build_100k", lambda: g.KDTreeFlann(kd_cloud))
tree = g.KDTreeFlann(kd_cloud)
queries = r.random((2000, 3))
def knn_queries():
    for q in queries:
        tree.search_knn_vector_3d(q, 20)
bench("kdtree_knn20_x2000", knn_queries)
def radius_queries():
    for q in queries:
        tree.search_radius_vector_3d(q, 0.05)
bench("kdtree_radius_x2000", radius_queries)
def hybrid_queries():
    for q in queries:
        tree.search_hybrid_vector_3d(q, 0.05, 30)
bench("kdtree_hybrid_x2000", hybrid_queries)

# --- normals ---
nc = make_cloud(N, normals=True)
def en_knn():
    c = g.PointCloud(nc)
    c.estimate_normals(g.KDTreeSearchParamKNN(30))
bench(f"estimate_normals_knn30_{N//1000}k", en_knn)
def en_hybrid():
    c = g.PointCloud(nc)
    c.estimate_normals(g.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
bench(f"estimate_normals_hybrid_{N//1000}k", en_hybrid)
def en_slow():
    c = g.PointCloud(nc)
    c.estimate_normals(g.KDTreeSearchParamKNN(30), fast_normal_computation=False)
bench(f"estimate_normals_slow_{N//1000}k", en_slow)

# --- fpfh ---
fc = make_cloud(20000 if not quick else 3000, normals=True)
fc.estimate_normals(g.KDTreeSearchParamKNN(30))
bench("fpfh_20k_hybrid_r0.05", lambda: reg.compute_fpfh_feature(fc, g.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)))

# --- registration ---
ns = 20000 if not quick else 3000
src = make_cloud(ns)
Rm = g.get_rotation_matrix_from_xyz(np.array([0.02, -0.015, 0.03]))
tgt_pts = np.asarray(src.points) @ Rm.T + [0.01, 0.02, -0.01] + r.standard_normal((ns, 3)) * 0.001
tgt = g.PointCloud(); tgt.points = u.Vector3dVector(tgt_pts)
bench("evaluate_registration_20k", lambda: reg.evaluate_registration(src, tgt, 0.05))
bench("icp_p2p_20k", lambda: reg.registration_icp(src, tgt, 0.05, np.eye(4), reg.TransformationEstimationPointToPoint(False), reg.ICPConvergenceCriteria(1e-12, 1e-12, 20)))
tgt.normals = u.Vector3dVector(np.tile([0.0, 0.0, 1.0], (ns, 1)))
tgt.estimate_normals(g.KDTreeSearchParamKNN(30))
bench("icp_p2l_20k", lambda: reg.registration_icp(src, tgt, 0.05, np.eye(4), reg.TransformationEstimationPointToPlane(), reg.ICPConvergenceCriteria(1e-12, 1e-12, 20)))

# ransac (feature-based, moderate)
rs_n = 2000 if not quick else 500
rsrc = make_cloud(rs_n, normals=True); rsrc.estimate_normals(g.KDTreeSearchParamKNN(20))
rtgt_pts = np.asarray(rsrc.points) @ Rm.T + [0.05, 0.0, 0.02]
rtgt = g.PointCloud(); rtgt.points = u.Vector3dVector(rtgt_pts)
rtgt.normals = u.Vector3dVector(np.tile([0.0, 0.0, 1.0], (rs_n, 1))); rtgt.estimate_normals(g.KDTreeSearchParamKNN(20))
fs = reg.compute_fpfh_feature(rsrc, g.KDTreeSearchParamKNN(60))
ft = reg.compute_fpfh_feature(rtgt, g.KDTreeSearchParamKNN(60))
def ransac():
    u.random.seed(3)
    reg.registration_ransac_based_on_feature_matching(
        rsrc, rtgt, fs, ft, False, 0.03,
        reg.TransformationEstimationPointToPoint(False), 3,
        [reg.CorrespondenceCheckerBasedOnDistance(0.03)],
        reg.RANSACConvergenceCriteria(4000, 0.999))
bench("ransac_feature_2k", ransac)
bench("corres_from_features_mutual", lambda: reg.correspondences_from_features(fs, ft, True))

# --- io ---
tmp = tempfile.mkdtemp()
io_cloud = make_cloud(100000, normals=True)
io_cloud.colors = u.Vector3dVector(r.random((100000, 3)))
pb = os.path.join(tmp, "b.ply"); pa = os.path.join(tmp, "a.ply"); px = os.path.join(tmp, "x.xyz")
bench("ply_write_bin_100k", lambda: t3d.io.write_point_cloud(pb, io_cloud))
bench("ply_write_ascii_100k", lambda: t3d.io.write_point_cloud(pa, io_cloud, write_ascii=True))
bench("ply_read_bin_100k", lambda: t3d.io.read_point_cloud(pb))
bench("ply_read_ascii_100k", lambda: t3d.io.read_point_cloud(pa))
bench("xyz_write_100k", lambda: t3d.io.write_point_cloud(px, io_cloud))
bench("xyz_read_100k", lambda: t3d.io.read_point_cloud(px))

# --- bindings overhead ---
small = make_cloud(1000)
bench("points_getter_1k_x1000", lambda: [np.asarray(small.points) for _ in range(1000)])
arr = r.random((100000, 3))
bench("points_setter_100k_x10", lambda: [setattr(make_cloud(1), "points", u.Vector3dVector(arr)) for _ in range(10)])

json.dump(results, open(f"/tmp/bench_{label}.json", "w"), indent=1)
print("saved", f"/tmp/bench_{label}.json")
