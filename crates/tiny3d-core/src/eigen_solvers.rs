// Indexed loops and explicit arithmetic forms intentionally mirror Eigen's
// evaluation order; iterator and algebraic rewrites can change final bits.
#![allow(
    clippy::assign_op_pattern,
    clippy::manual_swap,
    clippy::needless_range_loop,
    clippy::neg_multiply
)]

//! Independent implementations of the Eigen 3.4 decomposition behavior used
//! by tiny3D, matched to the pinned revision (da79095923) compiled for SSE2:
//! - LDLT (dynamic path, as used on 6x6 in SolveLinearSystemPSD)
//! - JacobiSVD 3x3 (full U/V, no QR preconditioning since square)
//! - SelfAdjointEigenSolver 3x3 (tridiagonalization + implicit QL)
//! - computeInverseWithCheck 3x3 (cofactor based)

use crate::linalg::{M3, M6, V3, V6};

const DBL_MIN: f64 = f64::MIN_POSITIVE; // std::numeric_limits<double>::min()
const DBL_EPS: f64 = f64::EPSILON;

// ---------------------------------------------------------------- LDLT (6x6)

/// Eigen dynamic-size redux (sum) over a non-direct-access expression with
/// SSE2 double packets (packetSize = 2, alignedStart = 0):
/// k<2 or k==2/3: sequential; k==4: (a0+a2)+(a1+a3); k==5: that + a4; etc.
#[inline]
fn eigen_redux_sum_dyn(vals: &[f64]) -> f64 {
    let size = vals.len();
    let packet = 2usize;
    let aligned_size = (size / packet) * packet;
    if aligned_size == 0 {
        // scalar path
        if size == 0 {
            return 0.0;
        }
        let mut res = vals[0];
        for &v in &vals[1..] {
            res += v;
        }
        return res;
    }
    // packet lanes
    let mut p0 = [vals[0], vals[1]];
    if aligned_size > packet {
        let aligned_size2 = (size / (2 * packet)) * (2 * packet);
        let mut p1 = [vals[2], vals[3]];
        let mut idx = 2 * packet;
        while idx < aligned_size2 {
            p0[0] += vals[idx];
            p0[1] += vals[idx + 1];
            p1[0] += vals[idx + 2];
            p1[1] += vals[idx + 3];
            idx += 2 * packet;
        }
        p0[0] += p1[0];
        p0[1] += p1[1];
        if aligned_size > aligned_size2 {
            p0[0] += vals[aligned_size2];
            p0[1] += vals[aligned_size2 + 1];
        }
    }
    let mut res = p0[0] + p0[1];
    for &v in &vals[aligned_size..] {
        res += v;
    }
    res
}

/// Inner product of two length-k slices in Eigen dynamic redux order.
#[inline]
fn eigen_dot_dyn(a: &[f64], b: &[f64]) -> f64 {
    let prods: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
    eigen_redux_sum_dyn(&prods)
}

pub struct Ldlt6 {
    /// Packed factor: strict lower = L, diagonal = D (Eigen's m_matrix).
    mat: M6,
    transpositions: [usize; 6],
}

/// Eigen's dynamic-size `.dot`/inner products on non-contiguous rows run the
/// scalar path (sequential); contiguous GEMV accumulates column-by-column.
/// All loops below mirror the scalar evaluation order of the C++ build
/// (verified against bit-level probes).
pub fn ldlt6(a: &M6) -> Ldlt6 {
    let mut mat = *a;
    let mut transpositions = [0usize; 6];
    let size = 6usize;

    for k in 0..size {
        // Find largest diagonal element in tail
        let mut index_of_biggest = k;
        let mut biggest = mat[k][k].abs();
        for i in (k + 1)..size {
            let v = mat[i][i].abs();
            if v > biggest {
                biggest = v;
                index_of_biggest = i;
            }
        }
        transpositions[k] = index_of_biggest;
        if k != index_of_biggest {
            let ib = index_of_biggest;
            let s = size - ib - 1;
            // swap row k head(k) with row ib head(k)
            for j in 0..k {
                let tmp = mat[k][j];
                mat[k][j] = mat[ib][j];
                mat[ib][j] = tmp;
            }
            // swap col k tail(s) with col ib tail(s)
            for i in (size - s)..size {
                let tmp = mat[i][k];
                mat[i][k] = mat[i][ib];
                mat[i][ib] = tmp;
            }
            // swap diagonal entries
            let tmp = mat[k][k];
            mat[k][k] = mat[ib][ib];
            mat[ib][ib] = tmp;
            // swap the "in-between" part (transpose swap)
            for i in (k + 1)..ib {
                let tmp = mat[i][k];
                mat[i][k] = mat[ib][i];
                mat[ib][i] = tmp;
            }
        }

        let rs = size - k - 1;
        if k > 0 {
            // temp.head(k) = D.head(k) * A10^T ; A10 = row k cols [0,k)
            let mut temp = [0.0f64; 6];
            for (i, t) in temp.iter_mut().enumerate().take(k) {
                *t = mat[i][i] * mat[k][i];
            }
            // mat(k,k) -= (A10 * temp.head(k)).value() — inner product of a
            // STRIDED row with a contiguous vector: not packet-accessible in
            // Eigen, so scalar sequential redux.
            let mut acc = 0.0f64;
            for (i, t) in temp.iter().enumerate().take(k) {
                acc += mat[k][i] * t;
            }
            mat[k][k] -= acc;
            if rs > 0 {
                // A21.noalias() -= A20 * temp.head(k): Eigen GEMV kernel —
                // per row: acc = sum_j A(i,j)*t[j] (sequential), then
                // res[i] = acc*alpha + res[i] with alpha = -1.
                for i in (k + 1)..size {
                    let mut acc = 0.0f64;
                    for (j, t) in temp.iter().enumerate().take(k) {
                        acc += mat[i][j] * t;
                    }
                    mat[i][k] = acc * (-1.0) + mat[i][k];
                }
            }
        }

        let real_akk = mat[k][k];
        let pivot_is_valid = real_akk.abs() > 0.0;
        if k == 0 && !pivot_is_valid {
            for (j, t) in transpositions.iter_mut().enumerate() {
                *t = j;
            }
            break;
        }
        if rs > 0 && pivot_is_valid {
            for i in (k + 1)..size {
                mat[i][k] /= real_akk;
            }
        }
    }

    Ldlt6 {
        mat,
        transpositions,
    }
}

impl Ldlt6 {
    /// Test hook: expose the packed factor and transpositions.
    pub fn debug_internals(&self) -> (M6, [usize; 6]) {
        (self.mat, self.transpositions)
    }

    /// LDLT::solve for a single RHS vector (Eigen `_solve_impl`).
    pub fn solve(&self, b: &V6) -> V6 {
        let size = 6usize;
        let mut dst = *b;
        // dst = P b : apply transpositions in order
        for (k, &t) in self.transpositions.iter().enumerate() {
            dst.swap(k, t);
        }
        // dst = L^-1 dst : unit lower triangular, col-major solver
        // Eigen col-major forward substitution: for each col j:
        //   (unit diag: no divide) dst[i] -= L(i,j)*dst[j] for i>j
        for j in 0..size {
            let xj = dst[j];
            for i in (j + 1)..size {
                dst[i] -= self.mat[i][j] * xj;
            }
        }
        // dst = D^-1 dst with pseudo-inverse tolerance = DBL_MIN
        for (i, d) in dst.iter_mut().enumerate() {
            let dd = self.mat[i][i];
            if dd.abs() > DBL_MIN {
                *d /= dd;
            } else {
                *d = 0.0;
            }
        }
        // dst = L^-T dst : L^T is unit upper triangular; Eigen solves the
        // transposed expression with a row-major-style backward substitution.
        // Col-major "OnTheLeft, Upper, RowMajor" solver iterates columns of L:
        // for j from size-1 down: dst[j] -= dot(L.col(j).tail, dst.tail)
        for j in (0..size).rev() {
            let k = size - j - 1;
            if k > 0 {
                let col: Vec<f64> = ((j + 1)..size).map(|i| self.mat[i][j]).collect();
                let acc = eigen_dot_dyn(&col, &dst[(j + 1)..size]);
                dst[j] -= acc;
            }
        }
        // dst = P^T dst : reverse transpositions
        for k in (0..size).rev() {
            dst.swap(k, self.transpositions[k]);
        }
        dst
    }
}

// ------------------------------------------------------------ JacobiSVD 3x3

#[derive(Clone, Copy)]
struct JRot {
    c: f64,
    s: f64,
}

impl JRot {
    fn transpose(self) -> JRot {
        JRot {
            c: self.c,
            s: -self.s,
        }
    }
    fn mul(self, o: JRot) -> JRot {
        JRot {
            c: self.c * o.c - self.s * o.s,
            s: self.c * o.s + self.s * o.c,
        }
    }
    /// makeJacobi(x, y, z) for real scalars.
    fn make_jacobi(x: f64, y: f64, z: f64) -> (JRot, bool) {
        let deno = 2.0 * y.abs();
        if deno < DBL_MIN {
            (JRot { c: 1.0, s: 0.0 }, false)
        } else {
            let tau = (x - z) / deno;
            let w = (tau * tau + 1.0).sqrt();
            let t = if tau > 0.0 {
                1.0 / (tau + w)
            } else {
                1.0 / (tau - w)
            };
            let sign_t = if t > 0.0 { 1.0 } else { -1.0 };
            let n = 1.0 / (t * t + 1.0).sqrt();
            let s = -sign_t * (y / y.abs()) * t.abs() * n;
            (JRot { c: n, s }, true)
        }
    }
    /// makeGivens(p, q) for real scalars (no r output needed).
    fn make_givens(p: f64, q: f64) -> JRot {
        if q == 0.0 {
            JRot {
                c: if p < 0.0 { -1.0 } else { 1.0 },
                s: 0.0,
            }
        } else if p == 0.0 {
            JRot {
                c: 0.0,
                s: if q < 0.0 { 1.0 } else { -1.0 },
            }
        } else if p.abs() > q.abs() {
            let t = q / p;
            let mut u = (1.0 + t * t).sqrt();
            if p < 0.0 {
                u = -u;
            }
            let c = 1.0 / u;
            JRot { c, s: -t * c }
        } else {
            let t = p / q;
            let mut u = (1.0 + t * t).sqrt();
            if q < 0.0 {
                u = -u;
            }
            let s = -1.0 / u;
            JRot { c: -t * s, s }
        }
    }
}

/// apply rotation on rows p,q of a 3x3 (B = J*B): x = c*x + s*y ; y = -s*x + c*y
fn apply_on_the_left(m: &mut M3, p: usize, q: usize, j: JRot) {
    for i in 0..3 {
        let xi = m[p][i];
        let yi = m[q][i];
        m[p][i] = j.c * xi + j.s * yi;
        m[q][i] = -j.s * xi + j.c * yi;
    }
}

/// apply rotation on cols p,q of a 3x3 (B = B*J): uses j.transpose() internally.
fn apply_on_the_right(m: &mut M3, p: usize, q: usize, j: JRot) {
    let jt = j.transpose();
    for i in 0..3 {
        let xi = m[i][p];
        let yi = m[i][q];
        m[i][p] = jt.c * xi + jt.s * yi;
        m[i][q] = -jt.s * xi + jt.c * yi;
    }
}

/// real_2x2_jacobi_svd on block (p,q) of `matrix`.
fn real_2x2_jacobi_svd(matrix: &M3, p: usize, q: usize) -> (JRot, JRot) {
    let mut m = [[matrix[p][p], matrix[p][q]], [matrix[q][p], matrix[q][q]]];
    let t = m[0][0] + m[1][1];
    let d = m[1][0] - m[0][1];
    let rot1 = if d.abs() < DBL_MIN {
        JRot { s: 0.0, c: 1.0 }
    } else {
        let u = t / d;
        let tmp = (1.0 + u * u).sqrt();
        JRot {
            s: 1.0 / tmp,
            c: u / tmp,
        }
    };
    // m.applyOnTheLeft(0,1,rot1)
    for i in 0..2 {
        let xi = m[0][i];
        let yi = m[1][i];
        m[0][i] = rot1.c * xi + rot1.s * yi;
        m[1][i] = -rot1.s * xi + rot1.c * yi;
    }
    let (j_right, _) = JRot::make_jacobi(m[0][0], m[0][1], m[1][1]);
    let j_left = rot1.mul(j_right.transpose());
    (j_left, j_right)
}

pub struct Svd3 {
    pub u: M3,
    pub v: M3,
    pub singular_values: V3,
}

/// `JacobiSVD<Matrix3d>` (m, ComputeFullU | ComputeFullV)
pub fn jacobi_svd3(matrix: &M3) -> Svd3 {
    let precision = 2.0 * DBL_EPS;
    let consider_as_zero = DBL_MIN;

    // max abs coeff (PropagateNaN — NaN wins; with finite input plain max)
    let mut scale = 0.0f64;
    let mut any_nan = false;
    for row in matrix.iter() {
        for &x in row.iter() {
            if x.is_nan() {
                any_nan = true;
            }
            let a = x.abs();
            if a > scale {
                scale = a;
            }
        }
    }
    if any_nan || !scale.is_finite() {
        // InvalidInput: Eigen leaves U/V unset (identity-sized garbage).
        // tiny3d checks allFinite on U/V; return NaN matrices to trigger that.
        let nanm = [[f64::NAN; 3]; 3];
        return Svd3 {
            u: nanm,
            v: nanm,
            singular_values: [f64::NAN; 3],
        };
    }
    if scale == 0.0 {
        scale = 1.0;
    }

    let mut work: M3 = *matrix;
    for row in work.iter_mut() {
        for x in row.iter_mut() {
            *x /= scale;
        }
    }
    let mut u = crate::linalg::m3_identity();
    let mut v = crate::linalg::m3_identity();

    let mut max_diag_entry = work[0][0].abs().max(work[1][1].abs()).max(work[2][2].abs());

    let mut finished = false;
    while !finished {
        finished = true;
        for p in 1..3usize {
            for q in 0..p {
                let threshold = consider_as_zero.max(precision * max_diag_entry);
                if work[p][q].abs() > threshold || work[q][p].abs() > threshold {
                    finished = false;
                    let (j_left, j_right) = real_2x2_jacobi_svd(&work, p, q);
                    apply_on_the_left(&mut work, p, q, j_left);
                    apply_on_the_right(&mut u, p, q, j_left.transpose());
                    apply_on_the_right(&mut work, p, q, j_right);
                    apply_on_the_right(&mut v, p, q, j_right);
                    max_diag_entry = max_diag_entry.max(work[p][p].abs().max(work[q][q].abs()));
                }
            }
        }
    }

    let mut sv = [0.0f64; 3];
    for i in 0..3 {
        let a = work[i][i];
        sv[i] = a.abs();
        if a < 0.0 {
            for row in u.iter_mut() {
                row[i] = -row[i];
            }
        }
    }
    for s in sv.iter_mut() {
        *s *= scale;
    }

    // Sort in decreasing order (selection with swaps)
    for i in 0..3usize {
        // maxCoeff over tail, first occurrence wins (Eigen visits sequentially,
        // strictly-greater updates)
        let mut pos = 0usize;
        let mut maxv = sv[i];
        for (jj, &val) in sv.iter().enumerate().skip(i + 1) {
            if val > maxv {
                maxv = val;
                pos = jj - i;
            }
        }
        if maxv == 0.0 {
            break;
        }
        if pos != 0 {
            let posi = pos + i;
            sv.swap(i, posi);
            for row in u.iter_mut() {
                row.swap(posi, i);
            }
            for row in v.iter_mut() {
                row.swap(posi, i);
            }
        }
    }

    Svd3 {
        u,
        v,
        singular_values: sv,
    }
}

// ------------------------------------------- SelfAdjointEigenSolver (3x3)

pub struct Saes3 {
    pub eigenvalues: V3,
    pub eigenvectors: M3,
    pub success: bool,
}

/// Test hook: run scaling + 3x3 tridiagonalization, return (diag, subdiag, Q).
pub fn debug_tridiag3(matrix: &M3) -> (V3, [f64; 2], M3) {
    let (diag, subdiag, q, _scale) = tridiag3_scaled(matrix);
    (diag, subdiag, q)
}

fn tridiag3_scaled(matrix: &M3) -> (V3, [f64; 2], M3, f64) {
    let mut mat = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..=i {
            mat[i][j] = matrix[i][j];
        }
    }
    let mut scale = 0.0f64;
    for row in mat.iter() {
        for &x in row.iter() {
            let a = x.abs();
            if a > scale {
                scale = a;
            }
        }
    }
    if scale == 0.0 {
        scale = 1.0;
    }
    for i in 0..3 {
        for j in 0..=i {
            mat[i][j] /= scale;
        }
    }
    let mut diag = [0.0f64; 3];
    let mut subdiag = [0.0f64; 2];
    let tol = DBL_MIN;
    diag[0] = mat[0][0];
    let v1norm2 = mat[2][0] * mat[2][0];
    let q: M3;
    if v1norm2 <= tol {
        diag[1] = mat[1][1];
        diag[2] = mat[2][2];
        subdiag[0] = mat[1][0];
        subdiag[1] = mat[2][1];
        q = crate::linalg::m3_identity();
    } else {
        let beta = (mat[1][0] * mat[1][0] + v1norm2).sqrt();
        let inv_beta = 1.0 / beta;
        let m01 = mat[1][0] * inv_beta;
        let m02 = mat[2][0] * inv_beta;
        let qq = 2.0 * m01 * mat[2][1] + m02 * (mat[2][2] - mat[1][1]);
        diag[1] = mat[1][1] + m02 * qq;
        diag[2] = mat[2][2] - m02 * qq;
        subdiag[0] = beta;
        subdiag[1] = mat[2][1] - m01 * qq;
        q = [[1.0, 0.0, 0.0], [0.0, m01, m02], [0.0, m02, -m01]];
    }
    (diag, subdiag, q, scale)
}

/// `SelfAdjointEigenSolver<Matrix3d>::compute` (m, ComputeEigenvectors)
/// (the iterative path, not computeDirect).
pub fn self_adjoint_eigen3(matrix: &M3) -> Saes3 {
    let (mut diag, mut subdiag, mut q, scale) = tridiag3_scaled(matrix);

    // computeFromTridiagonal_impl (n = 3, maxIterations = 30)
    let n = 3usize;
    let max_iterations = 30usize;
    let consider_as_zero = DBL_MIN;
    let precision_inv = 1.0 / DBL_EPS;
    let mut end = n - 1;
    let mut start = 0usize;
    let mut iter = 0usize;
    let success;
    loop {
        if end == 0 {
            success = true;
            break;
        }
        for i in start..end {
            if subdiag[i].abs() < consider_as_zero {
                subdiag[i] = 0.0;
            } else {
                let scaled_subdiag = precision_inv * subdiag[i];
                if scaled_subdiag * scaled_subdiag <= diag[i].abs() + diag[i + 1].abs() {
                    subdiag[i] = 0.0;
                }
            }
        }
        while end > 0 && subdiag[end - 1] == 0.0 {
            end -= 1;
        }
        if end == 0 {
            success = true;
            break;
        }
        iter += 1;
        if iter > max_iterations * n {
            success = false;
            break;
        }
        start = end - 1;
        while start > 0 && subdiag[start - 1] != 0.0 {
            start -= 1;
        }
        tridiagonal_qr_step(&mut diag, &mut subdiag, start, end, &mut q);
    }

    if success {
        // selection sort ascending by eigenvalue
        for i in 0..(n - 1) {
            // minCoeff over segment(i, n-i): first minimum
            let mut k = 0usize;
            let mut minv = diag[i];
            for (jj, &val) in diag.iter().enumerate().skip(i + 1) {
                if val < minv {
                    minv = val;
                    k = jj - i;
                }
            }
            if k > 0 {
                diag.swap(i, k + i);
                for row in q.iter_mut() {
                    row.swap(i, k + i);
                }
            }
        }
    }

    // scale back
    for d in diag.iter_mut() {
        *d *= scale;
    }

    Saes3 {
        eigenvalues: diag,
        eigenvectors: q,
        success,
    }
}

/// Test hook.
pub fn debug_qr_step(
    diag: &mut [f64; 3],
    subdiag: &mut [f64; 2],
    start: usize,
    end: usize,
    q: &mut M3,
) {
    tridiagonal_qr_step(diag, subdiag, start, end, q)
}

/// Eigen numext::hypot (positive_real_hypot): p = max(|x|,|y|);
/// qp = min(|x|,|y|)/p; p * sqrt(1 + qp*qp).
fn eigen_hypot(x: f64, y: f64) -> f64 {
    let (ax, ay) = (x.abs(), y.abs());
    if ax.is_infinite() || ay.is_infinite() {
        return f64::INFINITY;
    }
    if ax.is_nan() || ay.is_nan() {
        return f64::NAN;
    }
    let p = ax.max(ay);
    if p == 0.0 {
        return 0.0;
    }
    let qp = ax.min(ay) / p;
    p * (1.0 + qp * qp).sqrt()
}

fn tridiagonal_qr_step(
    diag: &mut [f64; 3],
    subdiag: &mut [f64; 2],
    start: usize,
    end: usize,
    q: &mut M3,
) {
    let td = (diag[end - 1] - diag[end]) * 0.5;
    let e = subdiag[end - 1];
    let mut mu = diag[end];
    if td == 0.0 {
        mu -= e.abs();
    } else if e != 0.0 {
        let e2 = e * e;
        let h = eigen_hypot(td, e);
        if e2 == 0.0 {
            mu -= e / ((td + if td > 0.0 { h } else { -h }) / e);
        } else {
            mu -= e2 / (td + if td > 0.0 { h } else { -h });
        }
    }

    let mut x = diag[start] - mu;
    let mut z = subdiag[start];
    let mut k = start;
    while k < end && z != 0.0 {
        let rot = JRot::make_givens(x, z);
        // T = G' T G
        let sdk = rot.s * diag[k] + rot.c * subdiag[k];
        let dkp1 = rot.s * subdiag[k] + rot.c * diag[k + 1];
        diag[k] = rot.c * (rot.c * diag[k] - rot.s * subdiag[k])
            - rot.s * (rot.c * subdiag[k] - rot.s * diag[k + 1]);
        diag[k + 1] = rot.s * sdk + rot.c * dkp1;
        subdiag[k] = rot.c * sdk - rot.s * dkp1;

        if k > start {
            subdiag[k - 1] = rot.c * subdiag[k - 1] - rot.s * z;
        }

        x = subdiag[k];
        if k < end - 1 {
            z = -rot.s * subdiag[k + 1];
            subdiag[k + 1] = rot.c * subdiag[k + 1];
        }

        // Q = Q * G
        apply_on_the_right_generic(q, k, k + 1, rot);
        k += 1;
    }
}

fn apply_on_the_right_generic(m: &mut M3, p: usize, q: usize, j: JRot) {
    let jt = j.transpose();
    for i in 0..3 {
        let xi = m[i][p];
        let yi = m[i][q];
        m[i][p] = jt.c * xi + jt.s * yi;
        m[i][q] = -jt.s * xi + jt.c * yi;
    }
}

// ------------------------------------------------- computeInverseWithCheck 3x3

fn cofactor_3x3(m: &M3, i: usize, j: usize) -> f64 {
    let i1 = (i + 1) % 3;
    let i2 = (i + 2) % 3;
    let j1 = (j + 1) % 3;
    let j2 = (j + 2) % 3;
    m[i1][j1] * m[i2][j2] - m[i1][j2] * m[i2][j1]
}

/// Returns (inverse, invertible) with Eigen's default threshold
/// (dummy_precision = 1e-12) — as used by computeInverseWithCheck.
pub fn compute_inverse3_with_check(m: &M3) -> (M3, bool) {
    let c0 = cofactor_3x3(m, 0, 0);
    let c1 = cofactor_3x3(m, 1, 0);
    let c2 = cofactor_3x3(m, 2, 0);
    // determinant = cwiseProduct with col 0, sum via fixed-size redux: (p0+p1)+p2
    let det = (c0 * m[0][0] + c1 * m[1][0]) + c2 * m[2][0];
    let invertible = det.abs() > 1e-12;
    let mut inv = [[0.0f64; 3]; 3];
    if !invertible {
        return (inv, false);
    }
    let invdet = 1.0 / det;
    let c01 = cofactor_3x3(m, 0, 1) * invdet;
    let c11 = cofactor_3x3(m, 1, 1) * invdet;
    let c02 = cofactor_3x3(m, 0, 2) * invdet;
    inv[1][2] = cofactor_3x3(m, 2, 1) * invdet;
    inv[2][1] = cofactor_3x3(m, 1, 2) * invdet;
    inv[2][2] = cofactor_3x3(m, 2, 2) * invdet;
    inv[1][0] = c01;
    inv[1][1] = c11;
    inv[2][0] = c02;
    inv[0][0] = c0 * invdet;
    inv[0][1] = c1 * invdet;
    inv[0][2] = c2 * invdet;
    (inv, true)
}
