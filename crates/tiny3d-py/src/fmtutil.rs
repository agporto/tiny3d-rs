//! C/fmt-compatible float formatting for reprs.

/// C `%e` with given precision (e.g. 6 -> "5.375000e-01").
pub fn c_format_e(v: f64, precision: usize) -> String {
    if v.is_nan() {
        return "nan".to_string();
    }
    if v.is_infinite() {
        return (if v < 0.0 { "-inf" } else { "inf" }).to_string();
    }
    let s = format!("{:.*e}", precision, v);
    // Rust: "5.375e-1" -> want 2-digit exponent with sign
    let (mant, exp) = s.split_once('e').unwrap();
    let x: i32 = exp.parse().unwrap();
    let sign = if x < 0 { '-' } else { '+' };
    format!("{}e{}{:02}", mant, sign, x.abs())
}

/// C `%f` with precision.
pub fn c_format_f(v: f64, precision: usize) -> String {
    format!("{:.*}", precision, v)
}

/// C++ ostream default double output (equivalent to %g with precision 6).
pub fn ostream_double(v: f64) -> String {
    tiny3d_core::linalg::format_g(v)
}

/// fmt::format("{}", v) — shortest round-trip representation.
pub fn shortest(v: f64) -> String {
    let s = format!("{}", v);
    s
}
