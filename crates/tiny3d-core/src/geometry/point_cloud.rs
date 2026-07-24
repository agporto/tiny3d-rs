//! tiny3d::geometry::PointCloud (PointCloud.cpp port).

use crate::eigen_solvers::{compute_inverse3_with_check, self_adjoint_eigen3};
use crate::kdtree::KdTreeFlann;
use crate::linalg::*;
use crate::stdhash::StdUnorderedMap;

use super::search_param::KdTreeSearchParam;

#[derive(Clone, Default)]
pub struct PointCloud {
    pub points: Vec<V3>,
    pub normals: Vec<V3>,
    pub colors: Vec<V3>,
}

// ---- shared geometry helpers (Geometry3D / PointCloud.cpp anonymous helpers) ----

pub fn compute_min_bound(points: &[V3]) -> V3 {
    if points.is_empty() {
        return [f64::NAN; 3];
    }
    let mut acc = points[0];
    for p in &points[1..] {
        for i in 0..3 {
            acc[i] = acc[i].min(p[i]);
        }
    }
    acc
}

pub fn compute_max_bound(points: &[V3]) -> V3 {
    if points.is_empty() {
        return [f64::NAN; 3];
    }
    let mut acc = points[0];
    for p in &points[1..] {
        for i in 0..3 {
            acc[i] = acc[i].max(p[i]);
        }
    }
    acc
}

pub fn compute_center(points: &[V3]) -> V3 {
    let mut center = ZERO3;
    if points.is_empty() {
        return center;
    }
    for p in points {
        center = add3(center, *p);
    }
    div3_inplace(&mut center, points.len() as f64);
    center
}

#[inline]
fn div3_inplace(v: &mut V3, s: f64) {
    v[0] /= s;
    v[1] /= s;
    v[2] /= s;
}

pub fn transform_points(t: &M4, points: &mut [V3]) {
    for p in points.iter_mut() {
        let ph = [p[0], p[1], p[2], 1.0];
        let nh = m4v4(t, ph);
        if nh[3].abs() > 1e-9 {
            *p = [nh[0] / nh[3], nh[1] / nh[3], nh[2] / nh[3]];
        } else {
            *p = [f64::NAN; 3];
        }
    }
}

pub fn transform_normals(t: &M4, normals: &mut [V3]) {
    let linear = m4_block3x3(t);
    let (inv, invertible) = compute_inverse3_with_check(&linear);
    let normal_matrix = if invertible {
        m3_transpose(&inv)
    } else {
        m3_identity()
    };
    for n in normals.iter_mut() {
        *n = m3v3(&normal_matrix, *n);
        stable_normalize3(n);
        if n[0].is_nan() {
            *n = ZERO3;
        }
    }
}

pub fn translate_points(translation: V3, points: &mut [V3], relative: bool) {
    let tv = if relative {
        translation
    } else if !points.is_empty() {
        sub3(translation, compute_center(points))
    } else {
        ZERO3
    };
    for p in points.iter_mut() {
        *p = add3(*p, tv);
    }
}

pub fn scale_points(scale: f64, points: &mut [V3], center: V3) {
    for p in points.iter_mut() {
        // point = center + scale * (point - center)
        let d = sub3(*p, center);
        *p = add3(center, scale3(d, scale));
    }
}

pub fn rotate_points(r: &M3, points: &mut [V3], center: V3) {
    for p in points.iter_mut() {
        let d = sub3(*p, center);
        *p = add3(center, m3v3(r, d));
    }
}

pub fn rotate_normals(r: &M3, normals: &mut [V3]) {
    for n in normals.iter_mut() {
        *n = m3v3(r, *n);
    }
}

// ---- normal estimation helpers ----

fn compute_eigenvector0(a: &M3, eval0: f64) -> V3 {
    let row0: V3 = [a[0][0] - eval0, a[0][1], a[0][2]];
    let row1: V3 = [a[0][1], a[1][1] - eval0, a[1][2]];
    let row2: V3 = [a[0][2], a[1][2], a[2][2] - eval0];
    let r0xr1 = cross3(row0, row1);
    let r0xr2 = cross3(row0, row2);
    let r1xr2 = cross3(row1, row2);
    let d0 = dot3(r0xr1, r0xr1);
    let d1 = dot3(r0xr2, r0xr2);
    let d2 = dot3(r1xr2, r1xr2);

    let mut dmax = d0;
    let mut imax = 0;
    if d1 > dmax {
        dmax = d1;
        imax = 1;
    }
    if d2 > dmax {
        imax = 2;
    }
    if dmax <= 1e-16 {
        return ZERO3;
    }
    match imax {
        0 => div3(r0xr1, d0.sqrt()),
        1 => div3(r0xr2, d1.sqrt()),
        _ => div3(r1xr2, d2.sqrt()),
    }
}

pub fn fast_eigen_3x3(covariance: &M3) -> V3 {
    let mut a = *covariance;
    let mut max_coeff = 0.0f64;
    for row in a.iter() {
        for &x in row.iter() {
            let v = x.abs();
            if v > max_coeff {
                max_coeff = v;
            }
        }
    }
    if max_coeff == 0.0 {
        return ZERO3;
    }
    for row in a.iter_mut() {
        for x in row.iter_mut() {
            *x /= max_coeff;
        }
    }

    let norm = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
    if norm > 1e-16 {
        // trace()/3.0 — Eigen trace: strided diagonal -> scalar redux
        // unroller order c0 + (c1 + c2)
        let q = (a[0][0] + (a[1][1] + a[2][2])) / 3.0;
        let b00 = a[0][0] - q;
        let b11 = a[1][1] - q;
        let b22 = a[2][2] - q;
        let p = ((b00 * b00 + b11 * b11 + b22 * b22 + norm * 2.0) / 6.0).sqrt();
        let c00 = b11 * b22 - a[1][2] * a[1][2];
        let c01 = a[0][1] * b22 - a[1][2] * a[0][2];
        let c02 = a[0][1] * a[1][2] - b11 * a[0][2];
        let mut det = b00 * c00 - a[0][1] * c01 + a[0][2] * c02;
        if p < 1e-16 {
            return ZERO3;
        }
        det /= p * p * p;

        let mut half_det = det * 0.5;
        half_det = half_det.clamp(-1.0, 1.0);

        let angle = half_det.acos() / 3.0;
        let two_thirds_pi = 2.0 * std::f64::consts::PI / 3.0;
        let beta2 = angle.cos() * 2.0;
        let beta0 = (angle + two_thirds_pi).cos() * 2.0;
        let beta1 = -(beta0 + beta2);

        let eval = [q + p * beta0, q + p * beta1, q + p * beta2];
        let mut min_idx = 0;
        if eval[1] < eval[min_idx] {
            min_idx = 1;
        }
        if eval[2] < eval[min_idx] {
            min_idx = 2;
        }
        compute_eigenvector0(&a, eval[min_idx])
    } else {
        if a[0][0] <= a[1][1] && a[0][0] <= a[2][2] {
            [1.0, 0.0, 0.0]
        } else if a[1][1] <= a[0][0] && a[1][1] <= a[2][2] {
            [0.0, 1.0, 0.0]
        } else {
            [0.0, 0.0, 1.0]
        }
    }
}

pub fn compute_normal(covariance: &M3, fast_normal_computation: bool) -> V3 {
    if fast_normal_computation {
        fast_eigen_3x3(covariance)
    } else {
        let es = self_adjoint_eigen3(covariance);
        if !es.success {
            return ZERO3;
        }
        [
            es.eigenvectors[0][0],
            es.eigenvectors[1][0],
            es.eigenvectors[2][0],
        ]
    }
}

/// utility::ComputeCovariance(points, indices)
pub fn compute_covariance(points: &[V3], indices: &[i64]) -> M3 {
    if indices.is_empty() {
        return m3_identity();
    }
    let mut cum = [0.0f64; 9];
    for &idx in indices {
        let p = points[idx as usize];
        cum[0] += p[0];
        cum[1] += p[1];
        cum[2] += p[2];
        cum[3] += p[0] * p[0];
        cum[4] += p[0] * p[1];
        cum[5] += p[0] * p[2];
        cum[6] += p[1] * p[1];
        cum[7] += p[1] * p[2];
        cum[8] += p[2] * p[2];
    }
    let n = indices.len() as f64;
    for c in cum.iter_mut() {
        *c /= n;
    }
    let mut cov = [[0.0f64; 3]; 3];
    cov[0][0] = cum[3] - cum[0] * cum[0];
    cov[1][1] = cum[6] - cum[1] * cum[1];
    cov[2][2] = cum[8] - cum[2] * cum[2];
    cov[0][1] = cum[4] - cum[0] * cum[1];
    cov[1][0] = cov[0][1];
    cov[0][2] = cum[5] - cum[0] * cum[2];
    cov[2][0] = cov[0][2];
    cov[1][2] = cum[7] - cum[1] * cum[2];
    cov[2][1] = cov[1][2];
    cov
}

impl PointCloud {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.points.clear();
        self.normals.clear();
        self.colors.clear();
    }

    pub fn is_empty(&self) -> bool {
        !self.has_points()
    }
    pub fn has_points(&self) -> bool {
        !self.points.is_empty()
    }
    pub fn has_normals(&self) -> bool {
        !self.points.is_empty() && self.normals.len() == self.points.len()
    }
    pub fn has_colors(&self) -> bool {
        !self.points.is_empty() && self.colors.len() == self.points.len()
    }

    pub fn get_min_bound(&self) -> V3 {
        compute_min_bound(&self.points)
    }
    pub fn get_max_bound(&self) -> V3 {
        compute_max_bound(&self.points)
    }
    pub fn get_center(&self) -> V3 {
        compute_center(&self.points)
    }

    pub fn transform(&mut self, t: &M4) {
        transform_points(t, &mut self.points);
        if self.has_normals() {
            transform_normals(t, &mut self.normals);
        }
    }

    pub fn translate(&mut self, translation: V3, relative: bool) {
        translate_points(translation, &mut self.points, relative);
    }

    pub fn scale(&mut self, s: f64, center: V3) {
        scale_points(s, &mut self.points, center);
    }

    pub fn rotate(&mut self, r: &M3, center: V3) {
        rotate_points(r, &mut self.points, center);
        if self.has_normals() {
            rotate_normals(r, &mut self.normals);
        }
    }

    pub fn normalize_normals(&mut self) {
        for n in self.normals.iter_mut() {
            stable_normalize3(n);
            if n[0].is_nan() {
                *n = [0.0, 0.0, 1.0];
            }
        }
    }

    // Keep the reference max/min NaN behavior; f64::clamp is not equivalent.
    #[allow(clippy::manual_clamp)]
    pub fn paint_uniform_color(&mut self, color: V3) {
        let clipped = [
            color[0].max(0.0).min(1.0),
            color[1].max(0.0).min(1.0),
            color[2].max(0.0).min(1.0),
        ];
        self.colors = vec![clipped; self.points.len()];
    }

    /// VoxelDownSample — returns Err(msg) on LogError conditions.
    pub fn voxel_down_sample(&self, voxel_size: f64) -> Result<PointCloud, String> {
        let mut output = PointCloud::new();
        if voxel_size <= 0.0 {
            return Err("[VoxelDownSample] voxel_size must be positive.".to_string());
        }
        if !self.has_points() {
            // LogWarning in C++; returns empty cloud
            return Ok(output);
        }
        let voxel_min_bound = self.get_min_bound();
        let voxel_max_bound = self.get_max_bound();
        let extent_max = {
            let e = sub3(voxel_max_bound, voxel_min_bound);
            e[0].max(e[1]).max(e[2])
        };
        if voxel_size * (i32::MAX as f64) < extent_max + 1e-9 {
            return Err(
                "[VoxelDownSample] voxel_size is too small relative to the cloud extent."
                    .to_string(),
            );
        }

        #[derive(Clone)]
        struct AccPoint {
            num: i32,
            point: V3,
            normal: V3,
            color: V3,
            has_normals: bool,
            has_colors: bool,
        }
        impl Default for AccPoint {
            fn default() -> Self {
                AccPoint {
                    num: 0,
                    point: ZERO3,
                    normal: ZERO3,
                    color: ZERO3,
                    has_normals: false,
                    has_colors: false,
                }
            }
        }

        let mut map: StdUnorderedMap<AccPoint> = StdUnorderedMap::new();
        let origin = voxel_min_bound;
        let has_normals = self.has_normals();
        let has_colors = self.has_colors();
        for i in 0..self.points.len() {
            let p = self.points[i];
            let ref_coord = [
                (p[0] - origin[0]) / voxel_size,
                (p[1] - origin[1]) / voxel_size,
                (p[2] - origin[2]) / voxel_size,
            ];
            let vi = [
                ref_coord[0].floor() as i32,
                ref_coord[1].floor() as i32,
                ref_coord[2].floor() as i32,
            ];
            let acc = map.get_mut_or_insert_with(&vi, AccPoint::default);
            acc.point = add3(acc.point, p);
            if has_normals {
                let n = self.normals[i];
                if !n[0].is_nan() && !n[1].is_nan() && !n[2].is_nan() {
                    acc.normal = add3(acc.normal, n);
                    acc.has_normals = true;
                }
            }
            if has_colors {
                acc.color = add3(acc.color, self.colors[i]);
                acc.has_colors = true;
            }
            acc.num += 1;
        }

        for (_k, acc) in map.iter() {
            let n = acc.num as f64;
            output.points.push(if acc.num > 0 {
                div3(acc.point, n)
            } else {
                ZERO3
            });
            if has_normals {
                if acc.has_normals && acc.num > 0 {
                    output.normals.push(div3(acc.normal, n));
                } else {
                    output.normals.push(ZERO3);
                }
            }
            if has_colors {
                if acc.has_colors && acc.num > 0 {
                    output.colors.push(div3(acc.color, n));
                } else {
                    output.colors.push([0.5, 0.5, 0.5]);
                }
            }
        }
        if output.has_normals() {
            output.normalize_normals();
        }
        Ok(output)
    }

    /// EstimateNormals.
    ///
    /// DIVERGENCE (documented correctness fix): the C++ resizes `normals_`
    /// BEFORE checking HasNormals(), so when no prior normals exist the
    /// orientation step reads uninitialized memory and the resulting normal
    /// signs are nondeterministic. Here (matching upstream Open3D semantics)
    /// prior-normal orientation only happens if the cloud had normals before
    /// the call; otherwise the raw eigenvector direction is kept.
    pub fn estimate_normals(
        &mut self,
        search_param: &KdTreeSearchParam,
        fast_normal_computation: bool,
    ) {
        if !self.has_points() {
            return;
        }
        let has_original_normals = self.has_normals();
        let original_normals = if has_original_normals {
            self.normals.clone()
        } else {
            Vec::new()
        };
        if self.normals.len() != self.points.len() {
            self.normals = vec![ZERO3; self.points.len()];
        }

        let mut kdtree = KdTreeFlann::new();
        kdtree.set_points(&self.points);

        // Parallel over points: each normal depends only on immutable inputs
        // and is written to a disjoint index, so the result is identical to
        // the serial loop for any thread count.
        use rayon::prelude::*;
        let points = &self.points;
        let orig = &original_normals;
        self.normals.par_iter_mut().enumerate().for_each_init(
            || (Vec::new(), Vec::new()),
            |(nn_indices, nn_dists), (i, out)| {
                let cnt = kdtree.search(&points[i], search_param, nn_indices, nn_dists);
                if cnt < 3 {
                    *out = [0.0, 0.0, 1.0];
                    return;
                }
                let covariance = compute_covariance(points, nn_indices);
                let mut normal = compute_normal(&covariance, fast_normal_computation);
                if normal.iter().any(|x| x.is_nan()) || norm3(normal) < 1e-9 {
                    normal = [0.0, 0.0, 1.0];
                }
                if has_original_normals && dot3(normal, orig[i]) < 0.0 {
                    normal = [-normal[0], -normal[1], -normal[2]];
                }
                *out = normal;
            },
        );
    }

    /// Open3D `PointCloud::OrientNormalsToAlignWithDirection`.
    ///
    /// Flips every normal whose dot product with `orientation_reference` is
    /// negative. A zero-length normal is replaced by the reference vector
    /// itself, un-normalized (upstream Open3D behavior).
    ///
    /// Errors when the cloud has no normals, mirroring upstream
    /// `utility::LogError` (surfaced as a Python `RuntimeError` by the
    /// bindings). Not present in the C++ tiny3d subset; semantics follow
    /// upstream Open3D `PointCloud.cpp`.
    pub fn orient_normals_to_align_with_direction(
        &mut self,
        orientation_reference: V3,
    ) -> Result<(), String> {
        if !self.has_normals() {
            return Err(
                "No normals in the PointCloud. Call EstimateNormals() first.".to_string(),
            );
        }
        for normal in self.normals.iter_mut() {
            if norm3(*normal) == 0.0 {
                *normal = orientation_reference;
            } else if dot3(*normal, orientation_reference) < 0.0 {
                *normal = [-normal[0], -normal[1], -normal[2]];
            }
        }
        Ok(())
    }

    /// Open3D `PointCloud::OrientNormalsTowardsCameraLocation`.
    ///
    /// Flips every normal so it points from its point towards
    /// `camera_location`. A zero-length normal becomes the normalized
    /// point→camera direction, or `[0, 0, 1]` when the point coincides with
    /// the camera (upstream Open3D behavior).
    ///
    /// Errors when the cloud has no normals, mirroring upstream
    /// `utility::LogError`. Not present in the C++ tiny3d subset; semantics
    /// follow upstream Open3D `PointCloud.cpp`.
    pub fn orient_normals_towards_camera_location(
        &mut self,
        camera_location: V3,
    ) -> Result<(), String> {
        if !self.has_normals() {
            return Err(
                "No normals in the PointCloud. Call EstimateNormals() first.".to_string(),
            );
        }
        for (point, normal) in self.points.iter().zip(self.normals.iter_mut()) {
            let orientation_reference = [
                camera_location[0] - point[0],
                camera_location[1] - point[1],
                camera_location[2] - point[2],
            ];
            if norm3(*normal) == 0.0 {
                if norm3(orientation_reference) == 0.0 {
                    *normal = [0.0, 0.0, 1.0];
                } else {
                    // Eigen `normal.normalize()` divides by the norm.
                    *normal = normalized3(orientation_reference);
                }
            } else if dot3(*normal, orientation_reference) < 0.0 {
                *normal = [-normal[0], -normal[1], -normal[2]];
            }
        }
        Ok(())
    }

    /// Open3D `PointCloud::OrientNormalsConsistentTangentPlane`.
    ///
    /// Hoppe '92 consistent orientation propagated over a Riemannian-graph
    /// MST. See [`super::orient_tangent_plane`] for the algorithm and its
    /// documented divergences from upstream (Delaunay-free Euclidean MST,
    /// no IQR outlier exclusion for non-default `lambda`).
    pub fn orient_normals_consistent_tangent_plane(
        &mut self,
        k: usize,
        lambda: f64,
        cos_alpha_tol: f64,
    ) -> Result<(), String> {
        super::orient_tangent_plane::orient_normals_consistent_tangent_plane(
            self,
            k,
            lambda,
            cos_alpha_tol,
        )
    }
}

#[cfg(test)]
mod orient_tests {
    use super::*;

    fn cloud(points: Vec<V3>, normals: Vec<V3>) -> PointCloud {
        PointCloud {
            points,
            normals,
            colors: Vec::new(),
        }
    }

    #[test]
    fn align_flips_negative_dot_only() {
        let mut pc = cloud(
            vec![[0.0; 3]; 3],
            vec![[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
        );
        pc.orient_normals_to_align_with_direction([0.0, 0.0, 1.0])
            .unwrap();
        assert_eq!(pc.normals[0], [0.0, 0.0, 1.0]);
        assert_eq!(pc.normals[1], [0.0, 0.0, 1.0]);
        // dot == 0 is NOT flipped (strict `< 0` comparison, Open3D-faithful).
        assert_eq!(pc.normals[2], [1.0, 0.0, 0.0]);
    }

    #[test]
    fn align_zero_normal_becomes_unnormalized_reference() {
        let mut pc = cloud(vec![[0.0; 3]], vec![[0.0; 3]]);
        pc.orient_normals_to_align_with_direction([0.0, 0.0, 2.0])
            .unwrap();
        // Upstream assigns the reference verbatim — no normalization.
        assert_eq!(pc.normals[0], [0.0, 0.0, 2.0]);
    }

    #[test]
    fn align_errors_without_normals() {
        let mut pc = cloud(vec![[0.0; 3]], Vec::new());
        assert!(pc
            .orient_normals_to_align_with_direction([0.0, 0.0, 1.0])
            .is_err());
    }

    #[test]
    fn camera_flips_towards_camera() {
        // Point below origin-camera, normal pointing away (down): must flip up.
        let mut pc = cloud(
            vec![[0.0, 0.0, -2.0], [1.0, 0.0, 0.0]],
            vec![[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
        );
        pc.orient_normals_towards_camera_location([0.0, 0.0, 0.0])
            .unwrap();
        assert_eq!(pc.normals[0], [0.0, 0.0, 1.0]);
        // Normal pointing away from camera along +x flips to -x.
        assert_eq!(pc.normals[1], [-1.0, 0.0, 0.0]);
    }

    #[test]
    fn camera_zero_normal_cases() {
        let mut pc = cloud(
            vec![[0.0, 0.0, -2.0], [0.0, 0.0, 0.0]],
            vec![[0.0; 3], [0.0; 3]],
        );
        pc.orient_normals_towards_camera_location([0.0, 0.0, 0.0])
            .unwrap();
        // Zero normal away from camera: normalized point→camera direction.
        assert_eq!(pc.normals[0], [0.0, 0.0, 1.0]);
        // Zero normal AT the camera: upstream fallback [0, 0, 1].
        assert_eq!(pc.normals[1], [0.0, 0.0, 1.0]);
    }
}
