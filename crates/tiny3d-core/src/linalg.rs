//! Minimal fixed-size linear algebra with operation orderings that bit-match
//! Eigen 3.4 compiled for x86-64 SSE2 (no FMA), as used by the C++ tiny3D build.
//!
//! Ordering notes (empirically verified against the reference build):
//! - dot/squaredNorm of Vector3d: (p0 + p1) + p2
//! - Matrix3d * Vector3d: rows 0,1 = (p0 + p1) + p2 ; row 2 (SSE tail) = p0 + (p1 + p2)
//! - Matrix3d * Matrix3d: same per column (rows 0,1 packet, row 2 tail)
//! - Matrix4d * Vector4d: ((p0 + p1) + p2) + p3 for every row
//! - stableNormalize: w = maxabs; z = (v/w).squaredNorm(); v /= sqrt(z)*w

pub type V3 = [f64; 3];
pub type V4 = [f64; 4];
pub type V6 = [f64; 6];
/// Row-major 3x3.
pub type M3 = [[f64; 3]; 3];
/// Row-major 4x4.
pub type M4 = [[f64; 4]; 4];
/// Row-major 6x6.
pub type M6 = [[f64; 6]; 6];

pub const ZERO3: V3 = [0.0; 3];

#[inline]
pub fn add3(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
pub fn sub3(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
pub fn scale3(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
pub fn div3(a: V3, s: f64) -> V3 {
    [a[0] / s, a[1] / s, a[2] / s]
}
#[inline]
pub fn dot3(a: V3, b: V3) -> f64 {
    (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]
}
#[inline]
pub fn squared_norm3(a: V3) -> f64 {
    dot3(a, a)
}
#[inline]
pub fn norm3(a: V3) -> f64 {
    squared_norm3(a).sqrt()
}
#[inline]
pub fn cross3(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
/// Eigen `v.normalized()` / `v.normalize()`: divides by norm (not *inv).
#[inline]
pub fn normalized3(a: V3) -> V3 {
    let n = norm3(a);
    [a[0] / n, a[1] / n, a[2] / n]
}
/// Eigen `v.stableNormalize()`.
#[inline]
pub fn stable_normalize3(a: &mut V3) {
    let w = a[0].abs().max(a[1].abs()).max(a[2].abs());
    let y = [a[0] / w, a[1] / w, a[2] / w];
    let z = squared_norm3(y);
    if z > 0.0 {
        let d = z.sqrt() * w;
        a[0] /= d;
        a[1] /= d;
        a[2] /= d;
    }
}

/// Eigen Matrix3d * Vector3d under SSE2: rows 0,1 sequential; row 2 tail order.
#[inline]
pub fn m3v3(m: &M3, v: V3) -> V3 {
    let r0 = (m[0][0] * v[0] + m[0][1] * v[1]) + m[0][2] * v[2];
    let r1 = (m[1][0] * v[0] + m[1][1] * v[1]) + m[1][2] * v[2];
    let r2 = m[2][0] * v[0] + (m[2][1] * v[1] + m[2][2] * v[2]);
    [r0, r1, r2]
}

/// Eigen Matrix3d * Matrix3d under SSE2 (per column: rows 0,1 packet, row 2 tail).
#[inline]
pub fn m3m3(a: &M3, b: &M3) -> M3 {
    let mut c = [[0.0; 3]; 3];
    for j in 0..3 {
        c[0][j] = (a[0][0] * b[0][j] + a[0][1] * b[1][j]) + a[0][2] * b[2][j];
        c[1][j] = (a[1][0] * b[0][j] + a[1][1] * b[1][j]) + a[1][2] * b[2][j];
        c[2][j] = a[2][0] * b[0][j] + (a[2][1] * b[1][j] + a[2][2] * b[2][j]);
    }
    c
}

#[inline]
pub fn m3_transpose(a: &M3) -> M3 {
    let mut t = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = a[j][i];
        }
    }
    t
}

#[inline]
pub fn m3_scale(a: &M3, s: f64) -> M3 {
    let mut r = *a;
    for row in r.iter_mut() {
        for x in row.iter_mut() {
            *x *= s;
        }
    }
    r
}

#[inline]
pub fn m3_identity() -> M3 {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

#[inline]
pub fn m4_identity() -> M4 {
    let mut m = [[0.0; 4]; 4];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    m
}

/// Eigen Matrix4d * Vector4d: fully sequential row dot.
#[inline]
pub fn m4v4(m: &M4, v: V4) -> V4 {
    let mut out = [0.0; 4];
    for i in 0..4 {
        out[i] = ((m[i][0] * v[0] + m[i][1] * v[1]) + m[i][2] * v[2]) + m[i][3] * v[3];
    }
    out
}

/// Eigen Matrix4d * Matrix4d: per column, 2 row-packets, sequential col-accumulate.
#[inline]
pub fn m4m4(a: &M4, b: &M4) -> M4 {
    let mut c = [[0.0; 4]; 4];
    for j in 0..4 {
        for i in 0..4 {
            c[i][j] =
                ((a[i][0] * b[0][j] + a[i][1] * b[1][j]) + a[i][2] * b[2][j]) + a[i][3] * b[3][j];
        }
    }
    c
}

#[inline]
pub fn m4_block3x3(t: &M4) -> M3 {
    [
        [t[0][0], t[0][1], t[0][2]],
        [t[1][0], t[1][1], t[1][2]],
        [t[2][0], t[2][1], t[2][2]],
    ]
}

#[inline]
pub fn m4_translation(t: &M4) -> V3 {
    [t[0][3], t[1][3], t[2][3]]
}

#[inline]
pub fn m4_all_finite(t: &M4) -> bool {
    t.iter().all(|r| r.iter().all(|x| x.is_finite()))
}

#[inline]
pub fn m3_all_finite(t: &M3) -> bool {
    t.iter().all(|r| r.iter().all(|x| x.is_finite()))
}

/// Eigen `.isIdentity(prec)` with default precision 1e-12: per-entry
/// |a_ij - delta_ij| <= prec * max(1, |a_ij|)? Eigen: isApprox on other identity —
/// isIdentity checks every coeff: for diagonal |c-1| <= prec, off-diag |c| <= prec.
#[inline]
pub fn m4_is_identity(t: &M4) -> bool {
    let prec = 1e-12;
    for (i, row) in t.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            let target = if i == j { 1.0 } else { 0.0 };
            if (*value - target).abs() > prec {
                return false;
            }
        }
    }
    true
}

#[inline]
pub fn m3_is_identity(t: &M3) -> bool {
    let prec = 1e-12;
    if !m3_all_finite(t) {
        return false;
    }
    for (i, row) in t.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            let target = if i == j { 1.0 } else { 0.0 };
            if (*value - target).abs() > prec {
                return false;
            }
        }
    }
    true
}

#[inline]
pub fn m4_is_pure_translation(t: &M4) -> bool {
    let prec = 1e-12;
    if !m4_all_finite(t) {
        return false;
    }
    for (i, row) in t.iter().take(3).enumerate() {
        for (j, value) in row.iter().take(3).enumerate() {
            let target = if i == j { 1.0 } else { 0.0 };
            if (*value - target).abs() > prec {
                return false;
            }
        }
    }
    t[3][0].abs() <= prec
        && t[3][1].abs() <= prec
        && t[3][2].abs() <= prec
        && (t[3][3] - 1.0).abs() <= prec
}

// ---- 6-dim helpers (registration) ----

#[inline]
pub fn v6_zero() -> V6 {
    [0.0; 6]
}
#[inline]
pub fn m6_zero() -> M6 {
    [[0.0; 6]; 6]
}

/// JTJ += (J*w) * J^T  — Eigen evaluates (J_r * w) first, then outer product.
#[inline]
pub fn m6_add_scaled_outer(jtj: &mut M6, j: &V6, w: f64) {
    let mut jw = [0.0; 6];
    for k in 0..6 {
        jw[k] = j[k] * w;
    }
    for r in 0..6 {
        for c in 0..6 {
            jtj[r][c] += jw[r] * j[c];
        }
    }
}

/// JTr += (J*w) * r
#[inline]
pub fn v6_add_scaled(jtr: &mut V6, j: &V6, w: f64, r: f64) {
    for k in 0..6 {
        jtr[k] += (j[k] * w) * r;
    }
}

/// C's `%g` (printf default precision 6) formatting, matching glibc.
pub fn format_g(value: f64) -> String {
    format_g_prec(value, 6)
}

/// Allocation-free `%g` (precision 6): appends to `out`, using `scratch` as a
/// reusable buffer. Byte-identical output to `format_g`.
pub fn format_g_into(out: &mut String, scratch: &mut String, value: f64) {
    use std::fmt::Write;
    if value.is_nan() {
        out.push_str(if value.is_sign_negative() {
            "-nan"
        } else {
            "nan"
        });
        return;
    }
    if value.is_infinite() {
        out.push_str(if value < 0.0 { "-inf" } else { "inf" });
        return;
    }
    if value == 0.0 {
        out.push_str(if value.is_sign_negative() { "-0" } else { "0" });
        return;
    }
    let p = 6usize;
    scratch.clear();
    let _ = write!(scratch, "{:.*e}", p - 1, value);
    let epos = scratch.rfind('e').unwrap();
    let x: i32 = scratch[epos + 1..].parse().unwrap();
    if x < -4 || x >= p as i32 {
        // %e style: mantissa with trailing zeros stripped
        let mant = &scratch[..epos];
        let mut end = mant.len();
        if mant.contains('.') {
            while mant.as_bytes()[end - 1] == b'0' {
                end -= 1;
            }
            if mant.as_bytes()[end - 1] == b'.' {
                end -= 1;
            }
        }
        out.push_str(&mant[..end]);
        out.push('e');
        out.push(if x < 0 { '-' } else { '+' });
        let ax = x.abs();
        if ax < 10 {
            out.push('0');
        }
        let _ = write!(out, "{}", ax);
    } else {
        // %f style with p-1-x fractional digits, trailing zeros stripped
        let frac_digits = (p as i32 - 1 - x).max(0) as usize;
        let start = out.len();
        let _ = write!(out, "{:.*}", frac_digits, value);
        if out[start..].contains('.') {
            while out.as_bytes()[out.len() - 1] == b'0' {
                out.truncate(out.len() - 1);
            }
            if out.as_bytes()[out.len() - 1] == b'.' {
                out.truncate(out.len() - 1);
            }
        }
    }
}

/// C's `%.*g` formatting for finite and non-finite doubles, matching glibc.
pub fn format_g_prec(value: f64, precision: usize) -> String {
    if value.is_nan() {
        return (if value.is_sign_negative() {
            "-nan"
        } else {
            "nan"
        })
        .to_string();
    }
    if value.is_infinite() {
        return (if value < 0.0 { "-inf" } else { "inf" }).to_string();
    }
    let p = if precision == 0 { 1 } else { precision };
    // %g: use %e if exponent < -4 or >= precision, else %f; strip trailing zeros.
    // Determine decimal exponent X of the value as rounded to p significant digits.
    if value == 0.0 {
        return (if value.is_sign_negative() { "-0" } else { "0" }).to_string();
    }
    // Format with %e at p-1 digits to obtain the exponent after rounding.
    let e_str = format!("{:.*e}", p - 1, value);
    // Rust formats like "1.234e5" / "1.234e-5"; parse exponent.
    let (mant, exp_part) = e_str.split_once('e').unwrap();
    let x: i32 = exp_part.parse().unwrap();
    if x < -4 || x >= p as i32 {
        // %e style with trailing zeros stripped from mantissa.
        let mut m = mant.to_string();
        if m.contains('.') {
            while m.ends_with('0') {
                m.pop();
            }
            if m.ends_with('.') {
                m.pop();
            }
        }
        // glibc prints exponent with at least 2 digits and explicit sign.
        let sign = if x < 0 { '-' } else { '+' };
        format!("{}e{}{:02}", m, sign, x.abs())
    } else {
        // %f style with p - 1 - x fractional digits, trailing zeros stripped.
        let frac_digits = (p as i32 - 1 - x).max(0) as usize;
        let mut s = format!("{:.*}", frac_digits, value);
        if s.contains('.') {
            while s.ends_with('0') {
                s.pop();
            }
            if s.ends_with('.') {
                s.pop();
            }
        }
        s
    }
}
