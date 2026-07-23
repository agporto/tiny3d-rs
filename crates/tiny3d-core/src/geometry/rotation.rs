//! Rotation matrix constructors (Geometry3D statics + utility::RotationMatrix*).
use crate::linalg::{m3m3, norm3, M3, V3};

pub fn rotation_matrix_x(t: f64) -> M3 {
    [
        [1.0, 0.0, 0.0],
        [0.0, t.cos(), -t.sin()],
        [0.0, t.sin(), t.cos()],
    ]
}

pub fn rotation_matrix_y(t: f64) -> M3 {
    [
        [t.cos(), 0.0, t.sin()],
        [0.0, 1.0, 0.0],
        [-t.sin(), 0.0, t.cos()],
    ]
}

pub fn rotation_matrix_z(t: f64) -> M3 {
    [
        [t.cos(), -t.sin(), 0.0],
        [t.sin(), t.cos(), 0.0],
        [0.0, 0.0, 1.0],
    ]
}

pub fn from_xyz(r: V3) -> M3 {
    m3m3(
        &m3m3(&rotation_matrix_x(r[0]), &rotation_matrix_y(r[1])),
        &rotation_matrix_z(r[2]),
    )
}
pub fn from_yzx(r: V3) -> M3 {
    m3m3(
        &m3m3(&rotation_matrix_y(r[0]), &rotation_matrix_z(r[1])),
        &rotation_matrix_x(r[2]),
    )
}
pub fn from_zxy(r: V3) -> M3 {
    m3m3(
        &m3m3(&rotation_matrix_z(r[0]), &rotation_matrix_x(r[1])),
        &rotation_matrix_y(r[2]),
    )
}
pub fn from_xzy(r: V3) -> M3 {
    m3m3(
        &m3m3(&rotation_matrix_x(r[0]), &rotation_matrix_z(r[1])),
        &rotation_matrix_y(r[2]),
    )
}
pub fn from_zyx(r: V3) -> M3 {
    m3m3(
        &m3m3(&rotation_matrix_z(r[0]), &rotation_matrix_y(r[1])),
        &rotation_matrix_x(r[2]),
    )
}
pub fn from_yxz(r: V3) -> M3 {
    m3m3(
        &m3m3(&rotation_matrix_y(r[0]), &rotation_matrix_x(r[1])),
        &rotation_matrix_z(r[2]),
    )
}

/// Eigen AngleAxis::toRotationMatrix (used when |rotation| > 1e-12).
pub fn from_axis_angle(rotation: V3) -> M3 {
    let angle = norm3(rotation);
    if angle > 1e-12 {
        let axis = [
            rotation[0] / angle,
            rotation[1] / angle,
            rotation[2] / angle,
        ];
        angle_axis_to_matrix(angle, axis)
    } else {
        crate::linalg::m3_identity()
    }
}

fn angle_axis_to_matrix(angle: f64, axis: V3) -> M3 {
    let sin_a = angle.sin();
    let cos_a = angle.cos();
    let sin_axis = [sin_a * axis[0], sin_a * axis[1], sin_a * axis[2]];
    let c = 1.0 - cos_a;
    let cos1_axis = [c * axis[0], c * axis[1], c * axis[2]];
    let mut res = [[0.0f64; 3]; 3];
    let mut tmp;
    tmp = cos1_axis[0] * axis[1];
    res[0][1] = tmp - sin_axis[2];
    res[1][0] = tmp + sin_axis[2];
    tmp = cos1_axis[0] * axis[2];
    res[0][2] = tmp + sin_axis[1];
    res[2][0] = tmp - sin_axis[1];
    tmp = cos1_axis[1] * axis[2];
    res[1][2] = tmp - sin_axis[0];
    res[2][1] = tmp + sin_axis[0];
    res[0][0] = cos1_axis[0] * axis[0] + cos_a;
    res[1][1] = cos1_axis[1] * axis[1] + cos_a;
    res[2][2] = cos1_axis[2] * axis[2] + cos_a;
    res
}

/// GetRotationMatrixFromQuaternion: q = (w, x, y, z); normalize; toRotationMatrix.
pub fn from_quaternion(rotation: [f64; 4]) -> Result<M3, String> {
    let (w, x, y, z) = (rotation[0], rotation[1], rotation[2], rotation[3]);
    // Quaterniond::normalize(): coeffs (x,y,z,w) as Vector4d .normalize():
    // divide by norm computed with fixed-size redux — for Vector4d SSE2:
    // squaredNorm = redux over packets: ((x^2+z^2? ...)). Eigen Vector4d
    // squaredNorm with SSE2: packets [x,y],[z,w]: predux(p0*p0 + p1*p1) =
    // (x^2+z^2) + (y^2+w^2)... verified against the quat probe below through
    // the full pipeline.
    let sq = (x * x + z * z) + (y * y + w * w);
    if sq <= 0.0 || !sq.is_finite() {
        return Err("quaternion must be finite and have non-zero norm".to_string());
    }
    let n = sq.sqrt();
    let (x, y, z, w) = (x / n, y / n, z / n, w / n);
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
    Ok([
        [1.0 - (tyy + tzz), txy - twz, txz + twy],
        [txy + twz, 1.0 - (txx + tzz), tyz - twx],
        [txz - twy, tyz + twx, 1.0 - (txx + tyy)],
    ])
}
