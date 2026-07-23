//! Feature (FPFH) port — Feature.cpp.

use crate::geometry::search_param::KdTreeSearchParam;
use crate::geometry::PointCloud;
use crate::kdtree::KdTreeFlann;
use crate::linalg::*;

use super::estimation::Correspondence;

/// Feature: dim x n column-major matrix (data[col * dim + row]).
#[derive(Clone, Default)]
pub struct Feature {
    pub dim: usize,
    pub num: usize,
    pub data: Vec<f64>,
}

impl Feature {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn resize(&mut self, dim: usize, n: usize) {
        // Eigen resize on a Matrix leaves values uninitialized; tiny3d's
        // Resize sets... Feature::Resize does data_.resize(dim, n) — Eigen
        // resize does NOT zero. But both FPFH paths fill/or += after a
        // setZero? SPFH uses feature->data_(...) += — relies on zero init.
        // Eigen MatrixXd::resize leaves garbage; however Feature::Resize in
        // tiny3d calls data_.resize(dim, n) followed by setZero() (see
        // Feature.h). We zero here.
        self.dim = dim;
        self.num = n;
        self.data = vec![0.0; dim * n];
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[col * self.dim + row]
    }
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, v: f64) {
        self.data[col * self.dim + row] = v;
    }
    #[inline]
    pub fn col(&self, col: usize) -> &[f64] {
        &self.data[col * self.dim..(col + 1) * self.dim]
    }

    pub fn dimension(&self) -> usize {
        self.dim
    }
    pub fn n(&self) -> usize {
        self.num
    }
}

/// ComputePairFeatures
fn compute_pair_features(p1: V3, n1: V3, p2: V3, n2: V3) -> [f64; 4] {
    let mut result = [0.0f64; 4];
    let mut dp2p1 = sub3(p2, p1);
    result[3] = norm3(dp2p1);
    if result[3] == 0.0 {
        return [0.0; 4];
    }
    let mut n1_copy = n1;
    let mut n2_copy = n2;
    let angle1 = (dot3(n1_copy, dp2p1) / result[3]).clamp(-1.0, 1.0);
    let angle2 = (dot3(n2_copy, dp2p1) / result[3]).clamp(-1.0, 1.0);
    if angle1.abs() < angle2.abs() {
        n1_copy = n2;
        n2_copy = n1;
        dp2p1 = scale3(dp2p1, -1.0);
        result[2] = -angle2;
    } else {
        result[2] = angle1;
    }
    let mut v = cross3(dp2p1, n1_copy);
    let v_norm = norm3(v);
    if v_norm == 0.0 {
        return [0.0; 4];
    }
    v = div3(v, v_norm);
    let w = cross3(n1_copy, v);
    result[1] = dot3(v, n2_copy);
    result[0] = dot3(w, n2_copy).atan2(dot3(n1_copy, n2_copy));
    result
}

fn compute_spfh(input: &PointCloud, neighbor_indices: &[Vec<i64>]) -> Feature {
    use rayon::prelude::*;
    let n = input.points.len();
    let mut feature = Feature::new();
    feature.resize(33, n);
    // Parallel per point: each iteration writes only column i (disjoint
    // 33-element chunks) -> identical to the serial loop.
    feature
        .data
        .par_chunks_mut(33)
        .enumerate()
        .for_each(|(i, col)| {
            spfh_column(input, neighbor_indices, i, col);
        });
    feature
}

fn spfh_column(input: &PointCloud, neighbor_indices: &[Vec<i64>], i: usize, col: &mut [f64]) {
    {
        let point = input.points[i];
        let normal = input.normals[i];
        let indices = &neighbor_indices[i];
        if indices.len() > 1 {
            let hist_incr = 100.0 / (indices.len() - 1) as f64;
            for &k in &indices[1..] {
                let pf = compute_pair_features(
                    point,
                    normal,
                    input.points[k as usize],
                    input.normals[k as usize],
                );
                let mut h = (11.0 * (pf[0] + std::f64::consts::PI) / (2.0 * std::f64::consts::PI))
                    .floor() as i64;
                h = h.clamp(0, 10);
                col[h as usize] += hist_incr;

                let mut h = (11.0 * (pf[1] + 1.0) * 0.5).floor() as i64;
                h = h.clamp(0, 10);
                col[h as usize + 11] += hist_incr;

                let mut h = (11.0 * (pf[2] + 1.0) * 0.5).floor() as i64;
                h = h.clamp(0, 10);
                col[h as usize + 22] += hist_incr;
            }
        }
    }
}

pub fn compute_fpfh_feature(
    input: &PointCloud,
    search_param: &KdTreeSearchParam,
) -> Result<Feature, String> {
    if !input.has_normals() {
        return Err("Failed because input point cloud has no normal.".to_string());
    }
    let n_points = input.points.len();
    let mut kdtree = KdTreeFlann::new();
    kdtree.set_points(&input.points);

    use rayon::prelude::*;
    let mut neighbor_indices: Vec<Vec<i64>> = vec![Vec::new(); n_points];
    let mut neighbor_dist2: Vec<Vec<f64>> = vec![Vec::new(); n_points];
    // Parallel per point; disjoint writes.
    let points = &input.points;
    neighbor_indices
        .par_iter_mut()
        .zip(neighbor_dist2.par_iter_mut())
        .enumerate()
        .for_each(|(i, (ni, nd))| {
            kdtree.search(&points[i], search_param, ni, nd);
        });

    let mut feature = Feature::new();
    feature.resize(33, n_points);
    let spfh = compute_spfh(input, &neighbor_indices);

    // Parallel per point: writes only column i.
    let spfh_ref = &spfh;
    feature
        .data
        .par_chunks_mut(33)
        .enumerate()
        .for_each(|(i, col)| {
            let indices = &neighbor_indices[i];
            let dist2 = &neighbor_dist2[i];
            if indices.len() > 1 {
                let mut sum = [0.0f64; 3];
                for k in 1..indices.len() {
                    let dist = dist2[k];
                    if dist == 0.0 {
                        continue;
                    }
                    let p_index_k = indices[k] as usize;
                    let spfh_col = spfh_ref.col(p_index_k);
                    for j in 0..33 {
                        let val = spfh_col[j] / dist;
                        sum[j / 11] += val;
                        col[j] += val;
                    }
                }
                for s in sum.iter_mut() {
                    if *s != 0.0 {
                        *s = 100.0 / *s;
                    }
                }
                for (j, c) in col.iter_mut().enumerate() {
                    let v = *c * sum[j / 11];
                    *c = v + spfh_ref.col(i)[j];
                }
            }
        });
    Ok(feature)
}

/// CorrespondencesFromFeatures
pub fn correspondences_from_features(
    source_features: &Feature,
    target_features: &Feature,
    mutual_filter: bool,
    mutual_consistent_ratio: f32,
) -> Vec<Correspondence> {
    if source_features.num == 0 || target_features.num == 0 {
        return Vec::new();
    }
    let num_searches = if mutual_filter { 2 } else { 1 };
    let features = [source_features, target_features];
    let num_pts = [source_features.num as i32, target_features.num as i32];
    let mut corres: Vec<Vec<Correspondence>> = vec![Vec::new(); num_searches];

    #[allow(clippy::needless_range_loop)]
    for k in 0..num_searches {
        let mut kdtree = KdTreeFlann::new();
        let feat_tree = features[1 - k];
        kdtree.set_matrix_data(feat_tree.dim, feat_tree.num, feat_tree.data.clone());
        use rayon::prelude::*;
        let n_k = num_pts[k];
        corres[k] = vec![[0, -1]; n_k as usize];
        let f = features[k];
        let kdtree_ref = &kdtree;
        corres[k].par_iter_mut().enumerate().for_each_init(
            || (Vec::new(), Vec::new()),
            |(idx, d2), (i, out)| {
                let nn = kdtree_ref.search_knn(f.col(i), 1, idx, d2);
                if nn > 0 {
                    *out = [i as i32, idx[0] as i32];
                } else {
                    *out = [i as i32, -1];
                }
            },
        );
    }

    let filter_valid = |input: &[Correspondence]| -> Vec<Correspondence> {
        input
            .iter()
            .filter(|c| c[1] >= 0 && c[1] < num_pts[1])
            .copied()
            .collect()
    };

    if !mutual_filter {
        return filter_valid(&corres[0]);
    }

    let mut corres_mutual: Vec<Correspondence> = Vec::new();
    let num_src_pts = num_pts[0];
    for i in 0..num_src_pts {
        let j = corres[0][i as usize][1];
        // Upstream Open3D checks `corres[1][j](1) == i` (the reverse search's
        // match points back at i). The tiny3D C++ checks element (0) — which
        // is always j itself — so its mutual filter compares j == i and keeps
        // nearly nothing, then the ratio fallback silently returns the
        // unfiltered set. Deliberate divergence: see DIVERGENCES.md #4.
        if j >= 0 && j < num_pts[1] && corres[1][j as usize][1] == i {
            corres_mutual.push([i, j]);
        }
    }
    if corres_mutual.len() as i32 >= (mutual_consistent_ratio * num_src_pts as f32) as i32 {
        return corres_mutual;
    }
    // LogWarning: too few correspondences after mutual filter
    filter_valid(&corres[0])
}
