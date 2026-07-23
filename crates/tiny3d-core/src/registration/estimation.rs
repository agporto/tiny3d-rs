//! TransformationEstimation (point-to-point / point-to-plane) and the
//! linear-system helpers from utility::Eigen.

use crate::eigen_solvers::{jacobi_svd3, ldlt6};
use crate::geometry::PointCloud;
use crate::linalg::*;

pub type Correspondence = [i32; 2];

pub fn validate_correspondences(
    source: &PointCloud,
    target: &PointCloud,
    corres: &[Correspondence],
) -> Result<(), String> {
    for (position, correspondence) in corres.iter().enumerate() {
        let source_index = usize::try_from(correspondence[0]).map_err(|_| {
            format!(
                "correspondence {position} has negative source index {}",
                correspondence[0]
            )
        })?;
        let target_index = usize::try_from(correspondence[1]).map_err(|_| {
            format!(
                "correspondence {position} has negative target index {}",
                correspondence[1]
            )
        })?;
        if source_index >= source.points.len() {
            return Err(format!(
                "correspondence {position} source index {source_index} is out of bounds for {} points",
                source.points.len()
            ));
        }
        if target_index >= target.points.len() {
            return Err(format!(
                "correspondence {position} target index {target_index} is out of bounds for {} points",
                target.points.len()
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub enum TransformationEstimation {
    PointToPoint { with_scaling: bool },
    PointToPlane,
}

impl TransformationEstimation {
    pub fn compute_rmse(
        &self,
        source: &PointCloud,
        target: &PointCloud,
        corres: &[Correspondence],
    ) -> Result<f64, String> {
        validate_correspondences(source, target, corres)?;
        Ok(self.compute_rmse_unchecked(source, target, corres))
    }

    pub(crate) fn compute_rmse_unchecked(
        &self,
        source: &PointCloud,
        target: &PointCloud,
        corres: &[Correspondence],
    ) -> f64 {
        match self {
            TransformationEstimation::PointToPoint { .. } => {
                if corres.is_empty() {
                    return 0.0;
                }
                let mut err = 0.0f64;
                for c in corres {
                    let d = sub3(source.points[c[0] as usize], target.points[c[1] as usize]);
                    err += squared_norm3(d);
                }
                (err / corres.len() as f64).sqrt()
            }
            TransformationEstimation::PointToPlane => {
                if corres.is_empty() || !target.has_normals() {
                    return 0.0;
                }
                let mut err = 0.0f64;
                for c in corres {
                    let d = sub3(source.points[c[0] as usize], target.points[c[1] as usize]);
                    let r = dot3(d, target.normals[c[1] as usize]);
                    err += r * r;
                }
                (err / corres.len() as f64).sqrt()
            }
        }
    }

    pub fn compute_transformation(
        &self,
        source: &PointCloud,
        target: &PointCloud,
        corres: &[Correspondence],
    ) -> Result<M4, String> {
        validate_correspondences(source, target, corres)?;
        Ok(self.compute_transformation_unchecked(source, target, corres))
    }

    pub(crate) fn compute_transformation_unchecked(
        &self,
        source: &PointCloud,
        target: &PointCloud,
        corres: &[Correspondence],
    ) -> M4 {
        match self {
            TransformationEstimation::PointToPoint { with_scaling } => {
                compute_transformation_point_to_point_unchecked(
                    source,
                    target,
                    corres,
                    *with_scaling,
                )
            }
            TransformationEstimation::PointToPlane => {
                compute_transformation_point_to_plane_unchecked(source, target, corres)
            }
        }
    }
}

pub fn compute_transformation_point_to_point(
    source: &PointCloud,
    target: &PointCloud,
    corres: &[Correspondence],
    with_scaling: bool,
) -> Result<M4, String> {
    validate_correspondences(source, target, corres)?;
    Ok(compute_transformation_point_to_point_unchecked(
        source,
        target,
        corres,
        with_scaling,
    ))
}

pub(crate) fn compute_transformation_point_to_point_unchecked(
    source: &PointCloud,
    target: &PointCloud,
    corres: &[Correspondence],
    with_scaling: bool,
) -> M4 {
    if corres.is_empty() {
        return m4_identity();
    }
    let inv_n = 1.0 / corres.len() as f64;
    let mut mean_s = ZERO3;
    let mut mean_t = ZERO3;
    for c in corres {
        mean_s = add3(mean_s, source.points[c[0] as usize]);
        mean_t = add3(mean_t, target.points[c[1] as usize]);
    }
    mean_s = scale3(mean_s, inv_n);
    mean_t = scale3(mean_t, inv_n);

    let mut cov = [[0.0f64; 3]; 3];
    let mut var_s = 0.0f64;
    for c in corres {
        let ds = sub3(source.points[c[0] as usize], mean_s);
        let dt = sub3(target.points[c[1] as usize], mean_t);
        // cov += dt * ds^T
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += dt[i] * ds[j];
            }
        }
        var_s += squared_norm3(ds);
    }
    for row in cov.iter_mut() {
        for x in row.iter_mut() {
            *x *= inv_n;
        }
    }
    var_s *= inv_n;

    finish_umeyama(&cov, var_s, mean_s, mean_t, with_scaling)
}

/// Shared tail: SVD of cov, handedness fix, optional scaling, assemble T.
pub fn finish_umeyama(cov: &M3, var_s: f64, mean_s: V3, mean_t: V3, with_scaling: bool) -> M4 {
    let svd = jacobi_svd3(cov);
    let u = svd.u;
    let v = svd.v;
    if !m3_all_finite(&u) || !m3_all_finite(&v) {
        return m4_identity();
    }
    let mut diag: V3 = [1.0, 1.0, 1.0];
    // (U * V^T).determinant()
    let uvt = m3m3(&u, &m3_transpose(&v));
    if det3(&uvt) < 0.0 {
        diag[2] = -1.0;
    }
    // R = U * diag.asDiagonal() * V^T — evaluates (U * diag) then * V^T
    let mut ud = u;
    for row in ud.iter_mut() {
        for (j, x) in row.iter_mut().enumerate() {
            *x *= diag[j];
        }
    }
    let r = m3m3(&ud, &m3_transpose(&v));

    let mut scale = 1.0f64;
    if with_scaling && var_s > 0.0 {
        let sigma = svd.singular_values;
        scale = dot3(sigma, diag) / var_s;
    }

    let mut t = m4_identity();
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = scale * r[i][j];
        }
    }
    // mean_t - scale * R * mean_s ; Eigen: scale*R evaluated (scalar*matrix),
    // then (scale*R) * mean_s (3x3 * vec), then subtraction.
    let sr = m3_scale(&r, scale);
    let srm = m3v3(&sr, mean_s);
    let trans = sub3(mean_t, srm);
    for i in 0..3 {
        t[i][3] = trans[i];
    }
    t
}

/// Eigen 3x3 determinant (cofactor expansion as in Eigen's determinant_impl<3>).
pub fn det3(m: &M3) -> f64 {
    // Eigen: bruteforce_det3_helper terms: m(0,a)*m(1,b)*m(2,c) etc.
    // det = (m01*m12 - m02*m11? ...) Eigen uses:
    //   bruteforce_det3_helper(m,0,1,2) - bruteforce_det3_helper(m,1,0,2) + bruteforce_det3_helper(m,2,0,1)
    // where helper(m,a,b,c) = m(0,a) * (m(1,b)*m(2,c) - m(1,c)*m(2,b))
    let h = |a: usize, b: usize, c: usize| m[0][a] * (m[1][b] * m[2][c] - m[1][c] * m[2][b]);
    h(0, 1, 2) - h(1, 0, 2) + h(2, 0, 1)
}

pub fn compute_transformation_point_to_plane(
    source: &PointCloud,
    target: &PointCloud,
    corres: &[Correspondence],
) -> Result<M4, String> {
    validate_correspondences(source, target, corres)?;
    Ok(compute_transformation_point_to_plane_unchecked(
        source, target, corres,
    ))
}

pub(crate) fn compute_transformation_point_to_plane_unchecked(
    source: &PointCloud,
    target: &PointCloud,
    corres: &[Correspondence],
) -> M4 {
    if corres.is_empty() || !target.has_normals() {
        return m4_identity();
    }
    // Parallel Jacobian evaluation, serial in-order accumulation.
    use rayon::prelude::*;
    let jr: Vec<(V6, f64)> = corres
        .par_iter()
        .map(|c| {
            let vs = source.points[c[0] as usize];
            let vt = target.points[c[1] as usize];
            let nt = target.normals[c[1] as usize];
            let r = dot3(sub3(vs, vt), nt);
            let cr = cross3(vs, nt);
            let j_r: V6 = [cr[0], cr[1], cr[2], nt[0], nt[1], nt[2]];
            (j_r, r)
        })
        .collect();
    let mut jtj = m6_zero();
    let mut jtr = v6_zero();
    for (j_r, r) in jr.iter() {
        m6_add_scaled_outer(&mut jtj, j_r, 1.0);
        v6_add_scaled(&mut jtr, j_r, 1.0, *r);
    }
    let (ok, extrinsic) = solve_jacobian_system(&jtj, &jtr);
    if ok {
        extrinsic
    } else {
        m4_identity()
    }
}

/// utility::SolveJacobianSystemAndObtainExtrinsicMatrix
pub fn solve_jacobian_system(jtj: &M6, jtr: &V6) -> (bool, M4) {
    // SolveLinearSystemPSD(JTJ, -JTr): plain ldlt solve (no checks enabled)
    let neg: V6 = [-jtr[0], -jtr[1], -jtr[2], -jtr[3], -jtr[4], -jtr[5]];
    let x = ldlt6(jtj).solve(&neg);
    (true, transform_vector6d_to_matrix4d(&x))
}

/// utility::TransformVector6dToMatrix4d:
/// R = AngleAxis(z,UnitZ) * AngleAxis(y,UnitY) * AngleAxis(x,UnitX)
/// (computed via quaternion products, as Eigen does).
pub fn transform_vector6d_to_matrix4d(input: &V6) -> M4 {
    let (ax, ay, az) = (input[0], input[1], input[2]);
    // Quaternion from AngleAxis around a unit axis
    let qz = [(az * 0.5).cos(), 0.0, 0.0, (az * 0.5).sin()]; // (w,x,y,z)
    let qy = [(ay * 0.5).cos(), 0.0, (ay * 0.5).sin(), 0.0];
    let qx = [(ax * 0.5).cos(), (ax * 0.5).sin(), 0.0, 0.0];
    let q = quat_mul(quat_mul(qz, qy), qx);
    let r = quat_to_matrix(q);
    let mut out = m4_identity();
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = r[i][j];
        }
        out[i][3] = input[3 + i];
    }
    out
}

fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    // Eigen quat_product (scalar): (w,x,y,z)
    let (aw, ax, ay, az) = (a[0], a[1], a[2], a[3]);
    let (bw, bx, by, bz) = (b[0], b[1], b[2], b[3]);
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by + ay * bw + az * bx - ax * bz,
        aw * bz + az * bw + ax * by - ay * bx,
    ]
}

fn quat_to_matrix(q: [f64; 4]) -> M3 {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let tx = 2.0 * x;
    let ty = 2.0 * y;
    let tz = 2.0 * z;
    let twx = tx * w;
    let twy = ty * w;
    let twz = tz * w;
    let txx = tx * x;
    let txy = ty * x;
    let txz = tz * x;
    let tyy = ty * y;
    let tyz = tz * y;
    let tzz = tz * z;
    [
        [1.0 - (tyy + tzz), txy - twz, txz + twy],
        [txy + twz, 1.0 - (txx + tzz), tyz - twx],
        [txz - twy, tyz + twx, 1.0 - (txx + tyy)],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cloud_with_points(count: usize) -> PointCloud {
        let mut cloud = PointCloud::new();
        cloud.points = vec![[0.0; 3]; count];
        cloud
    }

    #[test]
    fn correspondence_validation_rejects_negative_and_large_indices() {
        let source = cloud_with_points(2);
        let target = cloud_with_points(2);

        assert!(validate_correspondences(&source, &target, &[[-1, 0]])
            .unwrap_err()
            .contains("negative source index"));
        assert!(validate_correspondences(&source, &target, &[[0, 2]])
            .unwrap_err()
            .contains("target index 2 is out of bounds"));
    }

    #[test]
    fn correspondence_validation_accepts_valid_indices() {
        let source = cloud_with_points(2);
        let target = cloud_with_points(3);
        assert!(validate_correspondences(&source, &target, &[[0, 2], [1, 1]]).is_ok());
    }

    #[test]
    fn standalone_estimators_reject_invalid_correspondences() {
        let source = cloud_with_points(1);
        let target = cloud_with_points(1);
        assert!(compute_transformation_point_to_point(&source, &target, &[[1, 0]], false).is_err());
        assert!(compute_transformation_point_to_plane(&source, &target, &[[0, -1]]).is_err());
    }
}
