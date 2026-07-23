pub mod ply;
pub mod xyz;

use crate::geometry::{PointCloud, TriangleMesh};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileGeometry {
    ContentsUnknown = 0,
    ContainsPoints = 1,
    ContainsLines = 2,
    ContainsTriangles = 4,
}

pub fn format_from_filename(filename: &str, format: &str) -> String {
    if format == "auto" {
        match filename.rsplit_once('.') {
            Some((_, ext)) => ext.to_lowercase(),
            None => String::new(),
        }
    } else {
        format.to_string()
    }
}

/// io::ReadPointCloud dispatch. Returns Ok(cloud); unknown format or failed
/// read logs a warning in C++ and returns an empty cloud with success=false —
/// the pybind layer still returns the (empty) cloud object.
pub struct ReadPointCloudOptions {
    pub remove_nan_points: bool,
    pub remove_infinite_points: bool,
}

pub fn read_point_cloud(filename: &str, format: &str, opts: &ReadPointCloudOptions) -> PointCloud {
    let fmt = format_from_filename(filename, format);
    let mut pcd = PointCloud::new();
    let ok = match fmt.as_str() {
        "xyz" => xyz::read_point_cloud_from_xyz_file(filename, &mut pcd),
        "ply" => ply::read_point_cloud_from_ply_file(filename, &mut pcd),
        _ => false,
    };
    if ok {
        postprocess_cloud(&mut pcd, opts);
    }
    pcd
}

pub fn read_point_cloud_from_bytes(
    bytes: &[u8],
    format: &str,
    opts: &ReadPointCloudOptions,
) -> PointCloud {
    let mut pcd = PointCloud::new();
    // Only "mem::xyz" is registered in the C++.
    let ok = match format {
        "mem::xyz" => xyz::read_point_cloud_from_xyz_bytes(bytes, &mut pcd),
        _ => false,
    };
    if ok {
        postprocess_cloud(&mut pcd, opts);
    }
    pcd
}

/// RemoveNonFinitePoints (from PointCloudIO read postprocessing).
fn postprocess_cloud(pcd: &mut PointCloud, opts: &ReadPointCloudOptions) {
    if !opts.remove_nan_points && !opts.remove_infinite_points {
        return;
    }
    let has_normal = pcd.has_normals();
    let has_color = pcd.has_colors();
    let old_len = pcd.points.len();
    let mut k = 0usize;
    for i in 0..old_len {
        let p = pcd.points[i];
        let is_nan = p[0].is_nan() || p[1].is_nan() || p[2].is_nan();
        let is_infinite = p[0].is_infinite() || p[1].is_infinite() || p[2].is_infinite();
        if (!opts.remove_nan_points || !is_nan) && (!opts.remove_infinite_points || !is_infinite) {
            pcd.points[k] = pcd.points[i];
            if has_normal {
                pcd.normals[k] = pcd.normals[i];
            }
            if has_color {
                pcd.colors[k] = pcd.colors[i];
            }
            k += 1;
        }
    }
    pcd.points.truncate(k);
    if has_normal {
        pcd.normals.truncate(k);
    }
    if has_color {
        pcd.colors.truncate(k);
    }
}

pub fn write_point_cloud(
    filename: &str,
    pcd: &PointCloud,
    format: &str,
    write_ascii: bool,
) -> bool {
    let fmt = format_from_filename(filename, format);
    match fmt.as_str() {
        "xyz" => xyz::write_point_cloud_to_xyz_file(filename, pcd),
        "ply" => ply::write_point_cloud_to_ply_file(filename, pcd, write_ascii),
        _ => false,
    }
}

pub fn write_point_cloud_to_bytes(
    pcd: &PointCloud,
    format: &str,
    _write_ascii: bool,
) -> Option<Vec<u8>> {
    match format {
        "mem::xyz" => xyz::write_point_cloud_to_xyz_bytes(pcd),
        _ => None,
    }
}

pub fn read_triangle_mesh(filename: &str) -> TriangleMesh {
    let fmt = format_from_filename(filename, "auto");
    let mut mesh = TriangleMesh::new();
    let _ok = match fmt.as_str() {
        "ply" => ply::read_triangle_mesh_from_ply_file(filename, &mut mesh),
        _ => false,
    };
    mesh
}

#[allow(clippy::too_many_arguments)]
pub fn write_triangle_mesh(
    filename: &str,
    mesh: &TriangleMesh,
    write_ascii: bool,
    write_vertex_normals: bool,
    write_vertex_colors: bool,
) -> bool {
    let fmt = format_from_filename(filename, "auto");
    match fmt.as_str() {
        "ply" => ply::write_triangle_mesh_to_ply_file(
            filename,
            mesh,
            write_ascii,
            write_vertex_normals,
            write_vertex_colors,
        ),
        _ => false,
    }
}

pub fn read_file_geometry_type(path: &str) -> FileGeometry {
    let fmt = format_from_filename(path, "auto");
    match fmt.as_str() {
        "xyz" => FileGeometry::ContainsPoints,
        "ply" => ply::read_file_geometry_type_ply(path),
        _ => FileGeometry::ContentsUnknown,
    }
}
