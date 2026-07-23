//! XYZ format I/O (FileXYZ.cpp port).

use crate::geometry::PointCloud;

/// sscanf("%lf %lf %lf") equivalent: parse three doubles separated by
/// whitespace; returns None unless all three parse.
fn parse_xyz_line(line: &str) -> Option<[f64; 3]> {
    let mut it = line.split_ascii_whitespace();
    let mut out = [0.0f64; 3];
    for o in out.iter_mut() {
        let tok = it.next()?;
        *o = parse_c_double(tok)?;
    }
    Some(out)
}

/// strtod-like: parse a leading double from the token (C sscanf %lf accepts
/// "nan", "inf", "infinity", hex floats; trailing junk in the token is fine
/// for sscanf as long as a valid prefix exists — but whitespace-split tokens
/// from tiny3d writers are clean, so parse the full token, with a prefix
/// fallback).
fn parse_c_double(tok: &str) -> Option<f64> {
    if let Ok(v) = tok.parse::<f64>() {
        return Some(v);
    }
    let lower = tok.to_ascii_lowercase();
    for (pref, val) in [
        ("nan", f64::NAN),
        ("-nan", f64::NAN),
        ("inf", f64::INFINITY),
        ("-inf", f64::NEG_INFINITY),
    ] {
        if lower.starts_with(pref) {
            return Some(if pref.starts_with('-') {
                -val.abs()
            } else {
                val
            });
        }
    }
    // longest valid numeric prefix
    let bytes = tok.as_bytes();
    for end in (1..bytes.len()).rev() {
        if let Ok(v) = tok[..end].parse::<f64>() {
            return Some(v);
        }
    }
    None
}

fn read_from_lines<'a>(lines: impl Iterator<Item = &'a str>, pcd: &mut PointCloud) {
    pcd.clear();
    for line in lines {
        if let Some(p) = parse_xyz_line(line) {
            pcd.points.push(p);
        }
    }
}

pub fn read_point_cloud_from_xyz_file(filename: &str, pcd: &mut PointCloud) -> bool {
    match std::fs::read_to_string(filename) {
        Ok(content) => {
            read_from_lines(content.lines(), pcd);
            true
        }
        Err(_) => false,
    }
}

pub fn read_point_cloud_from_xyz_bytes(bytes: &[u8], pcd: &mut PointCloud) -> bool {
    let content = String::from_utf8_lossy(bytes);
    read_from_lines(content.lines(), pcd);
    true
}

/// printf("%.10f") formatting, glibc-compatible (including nan/inf).
pub fn format_f64_fixed10(v: f64) -> String {
    if v.is_nan() {
        return (if v.is_sign_negative() { "-nan" } else { "nan" }).to_string();
    }
    if v.is_infinite() {
        return (if v < 0.0 { "-inf" } else { "inf" }).to_string();
    }
    format!("{:.10}", v)
}

fn push_fixed10(out: &mut String, v: f64) {
    use std::fmt::Write;
    if v.is_nan() {
        out.push_str(if v.is_sign_negative() { "-nan" } else { "nan" });
    } else if v.is_infinite() {
        out.push_str(if v < 0.0 { "-inf" } else { "inf" });
    } else {
        let _ = write!(out, "{:.10}", v);
    }
}

fn write_to_string(pcd: &PointCloud) -> String {
    // ~26 bytes per coordinate incl. separators, generous preallocation
    let mut out = String::with_capacity(pcd.points.len() * 60 + 16);
    for p in pcd.points.iter() {
        push_fixed10(&mut out, p[0]);
        out.push(' ');
        push_fixed10(&mut out, p[1]);
        out.push(' ');
        push_fixed10(&mut out, p[2]);
        out.push('\n');
    }
    out
}

pub fn write_point_cloud_to_xyz_file(filename: &str, pcd: &PointCloud) -> bool {
    std::fs::write(filename, write_to_string(pcd)).is_ok()
}

pub fn write_point_cloud_to_xyz_bytes(pcd: &PointCloud) -> Option<Vec<u8>> {
    Some(write_to_string(pcd).into_bytes())
}
