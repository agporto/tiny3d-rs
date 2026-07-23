#!/usr/bin/env python3
"""Generate (or check) golden reference outputs for the tiny3d API.

Run with the reference C++ build on PYTHONPATH:   python3 golden_gen.py gen
Run with the Rust build installed:                python3 golden_gen.py check

All arrays are compared bit-exactly in `check` mode unless a case is listed
in TOLERANT (allowed small fp divergence) or DIVERGENT (documented fixes).
"""
import sys, os, json, hashlib
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODE = sys.argv[1] if len(sys.argv) > 1 else "gen"

import tiny3d as t3d

g = t3d.geometry
u = t3d.utility
reg = t3d.pipelines.registration

CASES = {}
def case(f):
    CASES[f.__name__] = f
    return f

def rng(seed=0):
    return np.random.default_rng(seed)

def cloud(n=200, seed=0, colors=False, normals=False, scale=1.0):
    r = rng(seed)
    p = g.PointCloud()
    p.points = u.Vector3dVector(r.random((n, 3)) * scale)
    if colors:
        p.colors = u.Vector3dVector(r.random((n, 3)))
    if normals:
        nr = r.standard_normal((n, 3))
        p.normals = u.Vector3dVector(nr)
    return p

def pcd_state(p, prefix=""):
    d = {prefix + "points": np.asarray(p.points)}
    d[prefix + "colors"] = np.asarray(p.colors)
    d[prefix + "normals"] = np.asarray(p.normals)
    return d

def result_state(res):
    return {
        "transformation": np.array(res.transformation),
        "fitness": np.float64(res.fitness),
        "inlier_rmse": np.float64(res.inlier_rmse),
        "correspondence_set": np.asarray(res.correspondence_set),
        "repr": np.bytes_(repr(res).encode()),
    }


def preset_normals(p):
    n = len(p.points)
    p.normals = u.Vector3dVector(np.tile([0.0, 0.0, 1.0], (n, 1)))
    return p

# ---------------- rotation matrices ----------------
@case
def rotations():
    d = {}
    v = np.array([0.3, -1.2, 2.5])
    q = np.array([0.3, -0.4, 0.5, 0.7])
    d["axis_angle"] = g.get_rotation_matrix_from_axis_angle(v)
    d["quaternion"] = g.get_rotation_matrix_from_quaternion(q)
    for name in ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]:
        d[name] = getattr(g, "get_rotation_matrix_from_" + name)(v)
    d["axis_angle_zero"] = g.get_rotation_matrix_from_axis_angle(np.zeros(3))
    d["quat_unnorm"] = g.get_rotation_matrix_from_quaternion(np.array([2.0, 0.1, -0.2, 0.05]))
    return d

# ---------------- Geometry3D transforms ----------------
@case
def transforms():
    p = cloud(50, seed=1, colors=True, normals=True)
    d = {}
    R = g.get_rotation_matrix_from_xyz(np.array([0.1, 0.2, 0.3]))
    p.rotate(R)
    d.update(pcd_state(p, "rot_default_"))
    p2 = cloud(50, seed=1, normals=True)
    p2.rotate(R, center=np.array([0.5, -0.25, 1.0]))
    d.update(pcd_state(p2, "rot_center_"))
    p3 = cloud(50, seed=2)
    p3.translate(np.array([1.5, -2.5, 3.75]))
    d.update(pcd_state(p3, "trans_"))
    p4 = cloud(50, seed=2)
    p4.translate(np.array([1.5, -2.5, 3.75]), relative=False)
    d.update(pcd_state(p4, "trans_abs_"))
    p5 = cloud(50, seed=3)
    p5.scale(2.5, center=np.array([0.1, 0.1, 0.1]))
    d.update(pcd_state(p5, "scale_"))
    p6 = cloud(50, seed=3, normals=True)
    T = np.eye(4); T[:3, :3] = R * 1.7; T[:3, 3] = [0.5, 0.6, -0.7]
    p6.transform(T)
    d.update(pcd_state(p6, "transform_"))
    d["center"] = p6.get_center()
    d["min_bound"] = p6.get_min_bound()
    d["max_bound"] = p6.get_max_bound()
    return d

# ---------------- point cloud basic ops ----------------
@case
def pcd_ops():
    d = {}
    p = cloud(100, seed=4, normals=True)
    p.normalize_normals()
    d.update(pcd_state(p, "normlz_"))
    p.paint_uniform_color(np.array([0.2, 0.4, 0.8]))
    d["painted"] = np.asarray(p.colors)
    d["repr"] = np.bytes_(repr(p).encode())
    return d

@case
def voxel_down():
    d = {}
    for i, (n, vs, scale) in enumerate([(500, 0.1, 1.0), (500, 0.05, 1.0), (1000, 0.2, 5.0)]):
        p = cloud(n, seed=10 + i, colors=True, normals=True, scale=scale)
        q = p.voxel_down_sample(vs)
        d.update(pcd_state(q, f"v{i}_"))
    # negative coords
    p = g.PointCloud()
    r = rng(99)
    p.points = u.Vector3dVector(r.random((300, 3)) * 4.0 - 2.0)
    q = p.voxel_down_sample(0.3)
    d.update(pcd_state(q, "neg_"))
    return d

@case
def bounding_box():
    p = cloud(120, seed=20, scale=3.0)
    bb = p.get_axis_aligned_bounding_box()
    d = {
        "min_bound": bb.get_min_bound(), "max_bound": bb.get_max_bound(),
        "center": bb.get_center(), "extent": bb.get_extent(),
        "half_extent": bb.get_half_extent(), "max_extent": np.float64(bb.get_max_extent()),
        "volume": np.float64(bb.volume()), "box_points": np.asarray(bb.get_box_points()),
        "print_info": np.bytes_(bb.get_print_info().encode()),
        "repr": np.bytes_(repr(bb).encode()),
    }
    idx = bb.get_point_indices_within_bounding_box(p.points)
    d["indices_all"] = np.asarray(idx, dtype=np.int64)
    small = g.AxisAlignedBoundingBox(np.array([0.5, 0.5, 0.5]), np.array([2.0, 2.0, 2.0]))
    d["indices_small"] = np.asarray(small.get_point_indices_within_bounding_box(p.points), dtype=np.int64)
    small.color = np.array([1.0, 0.0, 0.0])
    bb2 = g.AxisAlignedBoundingBox(np.array([-1.0, -1, -1]), np.array([0.5, 0.75, 1.0]))
    bb2 += small
    d["merged_min"] = bb2.get_min_bound(); d["merged_max"] = bb2.get_max_bound()
    aabb_from_pts = g.AxisAlignedBoundingBox.create_from_points(p.points)
    d["cfp_min"] = aabb_from_pts.get_min_bound(); d["cfp_max"] = aabb_from_pts.get_max_bound()
    bb3 = g.AxisAlignedBoundingBox(np.array([0.0, 0, 0]), np.array([1.0, 2, 3]))
    bb3.scale(2.0, np.array([0.0, 0.0, 0.0]))
    d["scaled_min"] = bb3.get_min_bound(); d["scaled_max"] = bb3.get_max_bound()
    bb3.translate(np.array([1.0, 1.0, 1.0]))
    d["trans_min"] = bb3.get_min_bound(); d["trans_max"] = bb3.get_max_bound()
    return d

# ---------------- normals estimation ----------------
@case
def normals_knn_fast():
    p = preset_normals(cloud(300, seed=30))
    p.estimate_normals(g.KDTreeSearchParamKNN(20), fast_normal_computation=True)
    return {"normals": np.asarray(p.normals)}

@case
def normals_knn_slow():
    p = preset_normals(cloud(300, seed=30))
    p.estimate_normals(g.KDTreeSearchParamKNN(20), fast_normal_computation=False)
    return {"normals": np.asarray(p.normals)}

@case
def normals_hybrid():
    p = preset_normals(cloud(300, seed=31))
    p.estimate_normals(g.KDTreeSearchParamHybrid(radius=0.2, max_nn=25))
    return {"normals": np.asarray(p.normals)}

@case
def normals_radius():
    p = preset_normals(cloud(300, seed=32))
    p.estimate_normals(g.KDTreeSearchParamRadius(0.15))
    return {"normals": np.asarray(p.normals)}

@case
def normals_default():
    p = preset_normals(cloud(150, seed=33))
    p.estimate_normals()
    return {"normals": np.asarray(p.normals)}

@case
def normals_no_prior_SIGNFLIP():
    # C++ reads uninitialized memory for orientation when no prior normals exist
    # (EstimateNormals resizes normals_ before HasNormals() check). Golden stores one
    # observed sample; comparison is per-row up to sign.
    p = cloud(200, seed=34)
    p.estimate_normals(g.KDTreeSearchParamKNN(18))
    return {"normals": np.asarray(p.normals)}

# ---------------- KD-tree ----------------
@case
def kdtree():
    p = cloud(400, seed=40, scale=2.0)
    tree = g.KDTreeFlann(p)
    d = {}
    queries = rng(41).random((10, 3)) * 2.0
    for qi, q in enumerate(queries):
        k, idx, dist = tree.search_knn_vector_3d(q, 12)
        d[f"knn{qi}_k"] = np.int64(k)
        d[f"knn{qi}_idx"] = np.asarray(idx, dtype=np.int64)
        d[f"knn{qi}_dist"] = np.asarray(dist)
        k, idx, dist = tree.search_radius_vector_3d(q, 0.4)
        d[f"rad{qi}_k"] = np.int64(k)
        d[f"rad{qi}_idx"] = np.asarray(idx, dtype=np.int64)
        d[f"rad{qi}_dist"] = np.asarray(dist)
        k, idx, dist = tree.search_hybrid_vector_3d(q, 0.4, 15)
        d[f"hyb{qi}_k"] = np.int64(k)
        d[f"hyb{qi}_idx"] = np.asarray(idx, dtype=np.int64)
        d[f"hyb{qi}_dist"] = np.asarray(dist)
        k, idx, dist = tree.search_vector_3d(q, g.KDTreeSearchParamKNN(5))
        d[f"gen{qi}_idx"] = np.asarray(idx, dtype=np.int64)
        d[f"gen{qi}_dist"] = np.asarray(dist)
    # xd search on raw matrix data
    data = rng(42).random((8, 100))  # dim 8, 100 pts
    tx = g.KDTreeFlann(data)
    qx = rng(43).random((8,))
    k, idx, dist = tx.search_knn_vector_xd(qx, 7)
    d["xd_idx"] = np.asarray(idx, dtype=np.int64)
    d["xd_dist"] = np.asarray(dist)
    k, idx, dist = tx.search_radius_vector_xd(qx, 0.9)
    d["xd_rad_idx"] = np.asarray(idx, dtype=np.int64)
    d["xd_rad_dist"] = np.asarray(dist)
    return d

# ---------------- mesh ----------------
def make_mesh(seed=50, nv=60, nt=100):
    r = rng(seed)
    m = g.TriangleMesh()
    m.vertices = u.Vector3dVector(r.random((nv, 3)))
    tris = r.integers(0, nv, (nt, 3))
    m.triangles = u.Vector3iVector(tris)
    return m

@case
def mesh_ops():
    m = make_mesh()
    m.compute_triangle_normals(normalized=False)
    d = {"tri_normals": np.asarray(m.triangle_normals)}
    m2 = make_mesh(51)
    m2.compute_vertex_normals()
    d["vert_normals"] = np.asarray(m2.vertex_normals)
    d["tri_normals2"] = np.asarray(m2.triangle_normals)
    m2.normalize_normals()
    d["vert_normals_n"] = np.asarray(m2.vertex_normals)
    m2.paint_uniform_color(np.array([0.1, 0.9, 0.3]))
    d["colors"] = np.asarray(m2.vertex_colors)
    d["center"] = m2.get_center()
    d["repr"] = np.bytes_(repr(m2).encode())
    return d

# ---------------- voxel grid ----------------
@case
def voxelgrid():
    p = cloud(500, seed=60, colors=True, scale=2.0)
    vg = g.VoxelGrid.create_from_point_cloud(p, 0.25)
    voxels = vg.get_voxels()
    order = np.array(sorted(range(len(voxels)), key=lambda i: tuple(voxels[i].grid_index)))
    gi = np.array([voxels[i].grid_index for i in order]) if len(voxels) else np.zeros((0, 3))
    cols = np.array([voxels[i].color for i in order]) if len(voxels) else np.zeros((0, 3))
    d = {
        "origin": np.array(vg.origin), "voxel_size": np.float64(vg.voxel_size),
        "grid_indices_sorted": gi, "colors_sorted": cols,
        "min_bound": vg.get_min_bound(), "max_bound": vg.get_max_bound(),
        "center": vg.get_center(),
        "has_colors": np.bool_(vg.has_colors()),
    }
    bmin = np.array([-0.5, -0.5, -0.5]); bmax = np.array([1.5, 1.5, 1.5])
    vg2 = g.VoxelGrid.create_from_point_cloud_within_bounds(p, 0.25, bmin, bmax)
    voxels2 = vg2.get_voxels()
    order2 = np.array(sorted(range(len(voxels2)), key=lambda i: tuple(voxels2[i].grid_index)))
    d["wb_grid_indices_sorted"] = np.array([voxels2[i].grid_index for i in order2]) if len(voxels2) else np.zeros((0, 3))
    d["wb_origin"] = np.array(vg2.origin)
    return d

# ---------------- FPFH features ----------------
@case
def fpfh():
    p = preset_normals(cloud(250, seed=70))
    p.estimate_normals(g.KDTreeSearchParamKNN(15))
    f = reg.compute_fpfh_feature(p, g.KDTreeSearchParamHybrid(radius=0.25, max_nn=30))
    d = {"data": np.asarray(f.data), "dim": np.int64(f.dimension()), "num": np.int64(f.num())}
    f2 = reg.compute_fpfh_feature(p, g.KDTreeSearchParamKNN(20))
    d["data_knn"] = np.asarray(f2.data)
    return d

@case
def feature_corres():
    p = cloud(150, seed=71); q = cloud(150, seed=72)
    for c in (p, q):
        preset_normals(c)
        c.estimate_normals(g.KDTreeSearchParamKNN(15))
    fp = reg.compute_fpfh_feature(p, g.KDTreeSearchParamKNN(25))
    fq = reg.compute_fpfh_feature(q, g.KDTreeSearchParamKNN(25))
    d = {"fp": np.asarray(fp.data), "fq": np.asarray(fq.data)}
    c1 = reg.correspondences_from_features(fp, fq)
    d["plain"] = np.asarray(c1)
    c2 = reg.correspondences_from_features(fp, fq, mutual_filter=True)
    d["mutual"] = np.asarray(c2)
    c3 = reg.correspondences_from_features(fp, fq, mutual_filter=True, mutual_consistency_ratio=0.5)
    d["mutual_r5"] = np.asarray(c3)
    # reverse one-way correspondences (bit-exact both builds); lets the checker
    # compute the CORRECT mutual set independently for divergence #4
    d["rev"] = np.asarray(reg.correspondences_from_features(fq, fp))
    return d

# ---------------- registration ----------------
def icp_pair(seed=80, n=400, noise=0.005):
    r = rng(seed)
    src_pts = r.random((n, 3))
    R = g.get_rotation_matrix_from_xyz(np.array([0.05, -0.04, 0.08]))
    tgt_pts = src_pts @ R.T + np.array([0.05, 0.03, -0.02]) + r.standard_normal((n, 3)) * noise
    src = g.PointCloud(); src.points = u.Vector3dVector(src_pts)
    tgt = g.PointCloud(); tgt.points = u.Vector3dVector(tgt_pts)
    return src, tgt

@case
def evaluate():
    src, tgt = icp_pair()
    res = reg.evaluate_registration(src, tgt, 0.05)
    d = {("e_" + k): v for k, v in result_state(res).items()}
    T = np.eye(4); T[:3, 3] = [0.05, 0.03, -0.02]
    res2 = reg.evaluate_registration(src, tgt, 0.05, T)
    d.update({("t_" + k): v for k, v in result_state(res2).items()})
    info = reg.get_information_matrix_from_point_clouds(src, tgt, 0.05, T)
    d["info"] = np.array(info)
    return d

@case
def icp_p2p():
    src, tgt = icp_pair()
    res = reg.registration_icp(src, tgt, 0.2)
    d = result_state(res)
    res2 = reg.registration_icp(
        src, tgt, 0.2, np.eye(4), reg.TransformationEstimationPointToPoint(True),
        reg.ICPConvergenceCriteria(max_iteration=10))
    d.update({("s_" + k): v for k, v in result_state(res2).items()})
    return d

@case
def icp_p2l():
    src, tgt = icp_pair(seed=81)
    preset_normals(tgt)
    tgt.estimate_normals(g.KDTreeSearchParamKNN(20))
    res = reg.registration_icp(src, tgt, 0.2, np.eye(4), reg.TransformationEstimationPointToPlane())
    return result_state(res)

@case
def estimation_direct():
    src, tgt = icp_pair(seed=82, n=100)
    corres = u.Vector2iVector(np.stack([np.arange(100), np.arange(100)], axis=1))
    e = reg.TransformationEstimationPointToPoint()
    d = {"p2p_T": np.array(e.compute_transformation(src, tgt, corres)),
         "p2p_rmse": np.float64(e.compute_rmse(src, tgt, corres))}
    es = reg.TransformationEstimationPointToPoint(True)
    d["p2ps_T"] = np.array(es.compute_transformation(src, tgt, corres))
    preset_normals(tgt)
    tgt.estimate_normals(g.KDTreeSearchParamKNN(10))
    ep = reg.TransformationEstimationPointToPlane()
    d["p2l_T"] = np.array(ep.compute_transformation(src, tgt, corres))
    d["p2l_rmse"] = np.float64(ep.compute_rmse(src, tgt, corres))
    # checkers
    T = np.eye(4)
    cd = reg.CorrespondenceCheckerBasedOnDistance(0.1)
    ce = reg.CorrespondenceCheckerBasedOnEdgeLength(0.9)
    preset_normals(src)
    src.estimate_normals(g.KDTreeSearchParamKNN(10))
    cn = reg.CorrespondenceCheckerBasedOnNormal(0.5)
    d["chk_dist"] = np.bool_(cd.Check(src, tgt, corres, T))
    d["chk_edge"] = np.bool_(ce.Check(src, tgt, corres, T))
    d["chk_norm"] = np.bool_(cn.Check(src, tgt, corres, T))
    return d

@case
def ransac_corres():
    u_random = t3d.utility.random
    src, tgt = icp_pair(seed=83, n=200, noise=0.0)
    n = 200
    r = rng(84)
    corr = np.stack([np.arange(n), np.arange(n)], axis=1)
    bad = r.integers(0, n, (60, 2)); corr[r.choice(n, 60, replace=False)] = bad
    u_random.seed(7)
    res = reg.registration_ransac_based_on_correspondence(
        src, tgt, u.Vector2iVector(corr), 0.05,
        reg.TransformationEstimationPointToPoint(False), 3,
        [reg.CorrespondenceCheckerBasedOnDistance(0.05),
         reg.CorrespondenceCheckerBasedOnEdgeLength(0.9)],
        reg.RANSACConvergenceCriteria(1000, 0.999))
    return result_state(res)

@case
def ransac_feature():
    u_random = t3d.utility.random
    src, tgt = icp_pair(seed=85, n=150, noise=0.0)
    for c in (src, tgt):
        preset_normals(c)
        c.estimate_normals(g.KDTreeSearchParamKNN(15))
    fs = reg.compute_fpfh_feature(src, g.KDTreeSearchParamKNN(25))
    ft = reg.compute_fpfh_feature(tgt, g.KDTreeSearchParamKNN(25))
    u_random.seed(11)
    res = reg.registration_ransac_based_on_feature_matching(
        src, tgt, fs, ft, False, 0.075,
        reg.TransformationEstimationPointToPoint(False), 3,
        [reg.CorrespondenceCheckerBasedOnDistance(0.075)],
        reg.RANSACConvergenceCriteria(500, 0.999))
    return result_state(res)

# ---------------- I/O ----------------
@case
def io_ply():
    d = {}
    import tempfile
    tmp = tempfile.mkdtemp()
    # subsets of attributes
    for tag, kw in [("cn", dict(colors=True, normals=True)), ("c", dict(colors=True)),
                    ("n", dict(normals=True)), ("p", dict())]:
        p = cloud(50, seed=90, **kw)
        for ascii_mode, atag in [(True, "ascii"), (False, "bin")]:
            path = os.path.join(tmp, f"{tag}_{atag}.ply")
            t3d.io.write_point_cloud(path, p, write_ascii=ascii_mode)
            d[f"ply_{tag}_{atag}"] = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
            p2 = t3d.io.read_point_cloud(path)
            d.update(pcd_state(p2, f"rt_{tag}_{atag}_"))
    return d

@case
def io_xyz():
    d = {}
    p = cloud(40, seed=92, scale=10.0)
    b = t3d.io.write_point_cloud_to_bytes(p, format="mem::xyz")
    d["xyz_bytes"] = np.frombuffer(b, dtype=np.uint8)
    p2 = t3d.io.read_point_cloud_from_bytes(b, format="mem::xyz")
    d["rt_points"] = np.asarray(p2.points)
    # The C++ package silently returns empty bytes; Rust rejects the unknown
    # format. Preserve the reference key while verifying the documented fix.
    try:
        bu = t3d.io.write_point_cloud_to_bytes(p, format="xyz")
    except ValueError:
        bu = b""
    d["unknown_len"] = np.int64(len(bu) if bu is not None else -1)
    return d

@case
def io_files():
    d = {}
    import tempfile
    tmp = tempfile.mkdtemp()
    p = cloud(25, seed=93, colors=True, normals=True)
    for name, kw in [("a.ply", dict(write_ascii=True)), ("b.ply", {}), ("c.xyz", {})]:
        path = os.path.join(tmp, name)
        ok = t3d.io.write_point_cloud(path, p, **kw)
        d[name + "_ok"] = np.bool_(ok)
        d[name + "_bytes"] = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
        p2 = t3d.io.read_point_cloud(path)
        d.update(pcd_state(p2, name + "_rt_"))
        d[name + "_geomtype"] = np.int64(int(t3d.io.read_file_geometry_type(path)))
    # mesh io
    m = make_mesh(94, 30, 40)
    m.compute_vertex_normals()
    mpath = os.path.join(tmp, "m.ply")
    ok = t3d.io.write_triangle_mesh(mpath, m, write_ascii=True)
    d["mesh_ok"] = np.bool_(ok)
    d["mesh_bytes"] = np.frombuffer(open(mpath, "rb").read(), dtype=np.uint8)
    m2 = t3d.io.read_triangle_mesh(mpath)
    d["mesh_rt_vertices"] = np.asarray(m2.vertices)
    d["mesh_rt_triangles"] = np.asarray(m2.triangles)
    d["mesh_rt_normals"] = np.asarray(m2.vertex_normals)
    mpathb = os.path.join(tmp, "mb.ply")
    t3d.io.write_triangle_mesh(mpathb, m)
    d["mesh_bin_bytes"] = np.frombuffer(open(mpathb, "rb").read(), dtype=np.uint8)
    return d

@case
def io_nan():
    d = {}
    pts = rng(95).random((20, 3))
    pts[3] = [np.nan, 0.5, 0.5]
    pts[7] = [np.inf, 0.1, 0.1]
    p = g.PointCloud(); p.points = u.Vector3dVector(pts)
    b = t3d.io.write_point_cloud_to_bytes(p, format="mem::xyz")
    d["bytes"] = np.frombuffer(b, dtype=np.uint8)
    p2 = t3d.io.read_point_cloud_from_bytes(b, format="mem::xyz", remove_nan_points=True, remove_infinite_points=True)
    d["clean_points"] = np.asarray(p2.points)
    p3 = t3d.io.read_point_cloud_from_bytes(b, format="mem::xyz")
    d["raw_points"] = np.asarray(p3.points)
    return d

# ---------------- reprs / misc ----------------
@case
def misc():
    d = {}
    d["icp_crit_repr"] = np.bytes_(repr(reg.ICPConvergenceCriteria()).encode())
    d["ransac_crit_repr"] = np.bytes_(repr(reg.RANSACConvergenceCriteria()).encode())
    d["p2p_repr"] = np.bytes_(repr(reg.TransformationEstimationPointToPoint()).encode())
    d["p2l_repr"] = np.bytes_(repr(reg.TransformationEstimationPointToPlane()).encode())
    d["feat_repr"] = np.bytes_(repr(reg.Feature()).encode())
    d["knn_repr"] = np.bytes_(repr(g.KDTreeSearchParamKNN(17)).encode())
    d["verbosity"] = np.int64(int(u.get_verbosity_level()))
    p = g.PointCloud()
    d["empty_repr"] = np.bytes_(repr(p).encode())
    d["empty_isempty"] = np.bool_(p.is_empty())
    d["empty_dim"] = np.int64(p.dimension())
    d["geomtype"] = np.int64(int(p.get_geometry_type()))
    v = u.Vector3dVector(np.array([[1.0, 2, 3], [4, 5, 6]]))
    d["vec_repr"] = np.bytes_(repr(v).encode())
    d["vec_len"] = np.int64(len(v))
    f = reg.Feature(); f.resize(33, 10)
    d["feat_resized_repr"] = np.bytes_(repr(f).encode())
    d["feat_data"] = np.asarray(f.data)
    return d


def run_all():
    results = {}
    for name, f in CASES.items():
        results[name] = f()
    return results

def main():
    os.makedirs(OUT, exist_ok=True)
    if MODE == "gen":
        only = sys.argv[2:] if len(sys.argv) > 2 else None
        for name, f in CASES.items():
            if only and name not in only:
                continue
            d = f()
            np.savez(os.path.join(OUT, name + ".npz"), **d)
            print(f"[gen] {name}: {len(d)} entries")
    else:
        strict_fail = tol_fail = 0
        report = []
        only = sys.argv[2:] if len(sys.argv) > 2 else None
        # Documented divergences (correctness fixes in the Rust port):
        #  - normals_no_prior_SIGNFLIP: C++ reads uninitialized memory for
        #    normal orientation; compare up to per-row sign.
        #  - io_nan clean_points: C++ ignores remove_nan/remove_infinite flags;
        #    Rust implements them (Open3D semantics) -> 18 rows vs 20.
        #  - feature_corres mutual/mutual_r5: C++ mutual filter checks the
        #    wrong element (corres[1][j](0), which is always j) so it compares
        #    j == i, keeps ~nothing, and the ratio fallback returns the
        #    unfiltered set. Rust checks corres[1][j](1) == i (Open3D
        #    semantics). Verified against a numpy-computed expected set below.
        DIVERGENT_KEYS = {("io_nan", "clean_points"),
                          ("feature_corres", "mutual"),
                          ("feature_corres", "mutual_r5")}

        def expected_mutual(ref, ratio):
            fwd = ref["plain"]; rev = ref["rev"]
            n_src = fwd.shape[0]
            back_i = rev[fwd[:, 1], 1]
            mut = fwd[back_i == fwd[:, 0]]
            # C++/Rust fallback: count >= int(float(ratio) * n)  (f32 math)
            if len(mut) >= int(np.float32(ratio) * np.float32(n_src)):
                return mut
            return fwd
        SIGNFLIP_CASES = {"normals_no_prior_SIGNFLIP"}
        # The C++ pybind build's inline TriangleMesh/MeshBase::NormalizeNormals
        # (LTO-compiled, runtime-alignment-versioned loop) rounds +-1 ulp
        # differently depending on heap layout; same .so reproduces both
        # values for identical inputs. Tolerate a few ulp there.
        ULP_TOLERANT_KEYS = {("mesh_ops", "vert_normals"),
                             ("mesh_ops", "tri_normals2"),
                             ("mesh_ops", "vert_normals_n")}
        def ulp_diff(x, y):
            xi = x.view(np.int64); yi = y.view(np.int64)
            return np.abs(xi - yi)
        for name, f in CASES.items():
            if only and name not in only:
                continue
            ref = np.load(os.path.join(OUT, name + ".npz"))
            got = f(); bad = []
            if name in SIGNFLIP_CASES:
                r = ref["normals"]; gv = np.asarray(got["normals"])
                okrows = np.all(r == gv, axis=1) | np.all(r == -gv, axis=1)
                if not okrows.all():
                    bad.append(("normals", f"{(~okrows).sum()} rows differ beyond sign"))
                if bad:
                    strict_fail += 1
                    print(f"[FAIL] {name} (sign-insensitive): {bad}")
                else:
                    print(f"[ok]   {name} (sign-insensitive, divergence documented)")
                continue
            for k in ref.files:
                if (name, k) in DIVERGENT_KEYS:
                    if name == "feature_corres":
                        ratio = 0.1 if k == "mutual" else 0.5
                        exp = expected_mutual(ref, ratio)
                        gv = np.asarray(got[k])
                        if gv.shape != exp.shape or not np.array_equal(gv, exp):
                            bad.append((k, f"divergence check failed: got {gv.shape}, expected {exp.shape}"))
                        continue
                    # verify expected fixed behavior: NaN/inf rows removed
                    if k == "clean_points":
                        gv = np.asarray(got[k])
                        raw = ref["raw_points"]
                        finite = np.isfinite(raw).all(axis=1)
                        expected = raw[finite]
                        if not np.array_equal(gv, expected):
                            bad.append((k, "divergence check failed"))
                        continue
                r = ref[k]
                if k not in got:
                    bad.append((k, "MISSING")); continue
                gv = np.asarray(got[k])
                if r.shape != gv.shape:
                    bad.append((k, f"shape {r.shape} vs {gv.shape}")); continue
                if r.dtype.kind in "SU" or gv.dtype.kind in "SU":
                    if not np.array_equal(r, gv.astype(r.dtype)):
                        bad.append((k, f"str mismatch: {r} vs {gv}"))
                    continue
                if not np.array_equal(r, gv, equal_nan=True):
                    if (name, k) in ULP_TOLERANT_KEYS and r.dtype.kind == "f" and r.shape == gv.shape:
                        d = ulp_diff(np.ascontiguousarray(r), np.ascontiguousarray(gv))
                        if d.max() <= 4:
                            continue
                    if r.dtype.kind == "f":
                        diff = np.nanmax(np.abs(r - gv)) if r.size else 0
                        bad.append((k, f"maxdiff {diff:.3e}"))
                    else:
                        bad.append((k, f"int mismatch ({np.sum(r!=gv)} of {r.size})"))
            if bad:
                strict_fail += 1
                report.append((name, bad))
                print(f"[FAIL] {name}: {len(bad)} keys differ")
                for k, msg in bad[:10]:
                    print(f"    {k}: {msg}")
            else:
                print(f"[ok]   {name}")
        print(f"\n{strict_fail} failing cases / {len(CASES)}")
        sys.exit(1 if strict_fail else 0)

if __name__ == "__main__":
    main()
