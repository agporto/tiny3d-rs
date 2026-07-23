//! PLY I/O — independently implemented rply-compatible reader/writer.

use crate::geometry::{PointCloud, TriangleMesh};

use super::FileGeometry;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PlyType {
    Int8,
    Uint8,
    Int16,
    Uint16,
    Int32,
    Uint32,
    Float32,
    Float64,
}

impl PlyType {
    fn parse(s: &str) -> Option<PlyType> {
        Some(match s {
            "int8" | "char" => PlyType::Int8,
            "uint8" | "uchar" => PlyType::Uint8,
            "int16" | "short" => PlyType::Int16,
            "uint16" | "ushort" => PlyType::Uint16,
            "int32" | "int" => PlyType::Int32,
            "uint32" | "uint" => PlyType::Uint32,
            "float32" | "float" => PlyType::Float32,
            "float64" | "double" => PlyType::Float64,
            _ => return None,
        })
    }

    fn size(&self) -> usize {
        match self {
            PlyType::Int8 | PlyType::Uint8 => 1,
            PlyType::Int16 | PlyType::Uint16 => 2,
            PlyType::Int32 | PlyType::Uint32 | PlyType::Float32 => 4,
            PlyType::Float64 => 8,
        }
    }

    fn read_binary(&self, data: &[u8], le: bool) -> f64 {
        macro_rules! rd {
            ($t:ty, $n:expr) => {{
                let mut b = [0u8; $n];
                b.copy_from_slice(&data[..$n]);
                if le {
                    <$t>::from_le_bytes(b) as f64
                } else {
                    <$t>::from_be_bytes(b) as f64
                }
            }};
        }
        match self {
            PlyType::Int8 => data[0] as i8 as f64,
            PlyType::Uint8 => data[0] as f64,
            PlyType::Int16 => rd!(i16, 2),
            PlyType::Uint16 => rd!(u16, 2),
            PlyType::Int32 => rd!(i32, 4),
            PlyType::Uint32 => rd!(u32, 4),
            PlyType::Float32 => rd!(f32, 4),
            PlyType::Float64 => rd!(f64, 8),
        }
    }
}

#[derive(Clone, Debug)]
struct PlyProperty {
    name: String,
    ptype: PlyType,
    is_list: bool,
    length_type: PlyType,
}

#[derive(Clone, Debug)]
struct PlyElement {
    name: String,
    count: usize,
    properties: Vec<PlyProperty>,
}

struct PlyHeader {
    ascii: bool,
    little_endian: bool,
    elements: Vec<PlyElement>,
    data_offset: usize,
}

fn parse_header(data: &[u8]) -> Option<PlyHeader> {
    // find end_header line
    let mut ascii = true;
    let mut little_endian = true;
    let mut elements: Vec<PlyElement> = Vec::new();
    let mut pos = 0usize;
    let mut line_no = 0;
    let mut seen_ply = false;
    let mut seen_format = false;
    loop {
        let end = data[pos..].iter().position(|&b| b == b'\n')? + pos;
        let line = std::str::from_utf8(&data[pos..end]).ok()?;
        let line = line.trim_end_matches('\r');
        pos = end + 1;
        line_no += 1;
        if line_no == 1 {
            if line.trim() != "ply" {
                return None;
            }
            seen_ply = true;
            continue;
        }
        let mut it = line.split_ascii_whitespace();
        match it.next() {
            Some("format") => {
                let fmt = it.next()?;
                match fmt {
                    "ascii" => {
                        ascii = true;
                    }
                    "binary_little_endian" => {
                        ascii = false;
                        little_endian = true;
                    }
                    "binary_big_endian" => {
                        ascii = false;
                        little_endian = false;
                    }
                    _ => return None,
                }
                seen_format = true;
            }
            Some("comment") | Some("obj_info") => {}
            Some("element") => {
                let name = it.next()?.to_string();
                let count: usize = it.next()?.parse().ok()?;
                elements.push(PlyElement {
                    name,
                    count,
                    properties: Vec::new(),
                });
            }
            Some("property") => {
                let el = elements.last_mut()?;
                let t1 = it.next()?;
                if t1 == "list" {
                    let lt = PlyType::parse(it.next()?)?;
                    let vt = PlyType::parse(it.next()?)?;
                    let name = it.next()?.to_string();
                    el.properties.push(PlyProperty {
                        name,
                        ptype: vt,
                        is_list: true,
                        length_type: lt,
                    });
                } else {
                    let vt = PlyType::parse(t1)?;
                    let name = it.next()?.to_string();
                    el.properties.push(PlyProperty {
                        name,
                        ptype: vt,
                        is_list: false,
                        length_type: PlyType::Uint8,
                    });
                }
            }
            Some("end_header") => break,
            _ => {}
        }
    }
    if !seen_ply || !seen_format {
        return None;
    }
    Some(PlyHeader {
        ascii,
        little_endian,
        elements,
        data_offset: pos,
    })
}

/// Streaming value visitor: (element_index, instance, property_index,
/// value_index_within_list, list_length, value)
type Visitor<'a> = dyn FnMut(usize, usize, usize, usize, usize, f64) + 'a;

fn read_ply_data(data: &[u8], header: &PlyHeader, visit: &mut Visitor) -> bool {
    if header.ascii {
        let text = match std::str::from_utf8(&data[header.data_offset..]) {
            Ok(t) => t,
            Err(_) => return false,
        };
        let mut tokens = text.split_ascii_whitespace();
        for (ei, el) in header.elements.iter().enumerate() {
            for inst in 0..el.count {
                for (pi, prop) in el.properties.iter().enumerate() {
                    if prop.is_list {
                        let len_tok = match tokens.next() {
                            Some(t) => t,
                            None => return false,
                        };
                        let len = match len_tok.parse::<f64>() {
                            Ok(v) => v as usize,
                            Err(_) => return false,
                        };
                        visit(ei, inst, pi, 0, len, len as f64);
                        for k in 0..len {
                            let tok = match tokens.next() {
                                Some(t) => t,
                                None => return false,
                            };
                            let v = match tok.parse::<f64>() {
                                Ok(v) => v,
                                Err(_) => return false,
                            };
                            visit(ei, inst, pi, k + 1, len, v);
                        }
                    } else {
                        let tok = match tokens.next() {
                            Some(t) => t,
                            None => return false,
                        };
                        let v = match tok.parse::<f64>() {
                            Ok(v) => v,
                            Err(_) => return false,
                        };
                        visit(ei, inst, pi, 0, 0, v);
                    }
                }
            }
        }
        true
    } else {
        let le = header.little_endian;
        let mut off = header.data_offset;
        for (ei, el) in header.elements.iter().enumerate() {
            for inst in 0..el.count {
                for (pi, prop) in el.properties.iter().enumerate() {
                    if prop.is_list {
                        let lsz = prop.length_type.size();
                        if off + lsz > data.len() {
                            return false;
                        }
                        let len = prop.length_type.read_binary(&data[off..], le) as usize;
                        off += lsz;
                        visit(ei, inst, pi, 0, len, len as f64);
                        let vsz = prop.ptype.size();
                        for k in 0..len {
                            if off + vsz > data.len() {
                                return false;
                            }
                            let v = prop.ptype.read_binary(&data[off..], le);
                            off += vsz;
                            visit(ei, inst, pi, k + 1, len, v);
                        }
                    } else {
                        let vsz = prop.ptype.size();
                        if off + vsz > data.len() {
                            return false;
                        }
                        let v = prop.ptype.read_binary(&data[off..], le);
                        off += vsz;
                        visit(ei, inst, pi, 0, 0, v);
                    }
                }
            }
        }
        true
    }
}

fn find_prop(el: &PlyElement, name: &str) -> Option<usize> {
    el.properties.iter().position(|p| p.name == name)
}

pub fn read_point_cloud_from_ply_file(filename: &str, pcd: &mut PointCloud) -> bool {
    let data = match std::fs::read(filename) {
        Ok(d) => d,
        Err(_) => return false,
    };
    let header = match parse_header(&data) {
        Some(h) => h,
        None => return false,
    };
    let vei = match header.elements.iter().position(|e| e.name == "vertex") {
        Some(i) => i,
        None => return false,
    };
    let vel = &header.elements[vei];
    let vertex_num = if find_prop(vel, "x").is_some() {
        vel.count
    } else {
        0
    };
    if vertex_num == 0 {
        return false;
    }
    let normal_num = if find_prop(vel, "nx").is_some() {
        vel.count
    } else {
        0
    };
    let color_num = if find_prop(vel, "red").is_some() {
        vel.count
    } else {
        0
    };

    let mut parsed = PointCloud::new();
    parsed.points = vec![[0.0; 3]; vertex_num];
    parsed.normals = vec![[0.0; 3]; normal_num];
    parsed.colors = vec![[0.0; 3]; color_num];

    // map property index -> (target, coordinate)
    #[derive(Clone, Copy)]
    enum Target {
        Point(usize),
        Normal(usize),
        Color(usize),
        None,
    }
    let vel_props: Vec<Target> = vel
        .properties
        .iter()
        .map(|p| match p.name.as_str() {
            "x" => Target::Point(0),
            "y" => Target::Point(1),
            "z" => Target::Point(2),
            "nx" => Target::Normal(0),
            "ny" => Target::Normal(1),
            "nz" => Target::Normal(2),
            "red" => Target::Color(0),
            "green" => Target::Color(1),
            "blue" => Target::Color(2),
            _ => Target::None,
        })
        .collect();

    let success = read_ply_data(&data, &header, &mut |ei, inst, pi, _vk, _len, v| {
        if ei != vei {
            return;
        }
        match vel_props[pi] {
            Target::Point(c) => parsed.points[inst][c] = v,
            Target::Normal(c) => {
                if inst < normal_num {
                    parsed.normals[inst][c] = v;
                }
            }
            Target::Color(c) => {
                if inst < color_num {
                    parsed.colors[inst][c] = v / 255.0;
                }
            }
            Target::None => {}
        }
    });
    if success {
        *pcd = parsed;
    }
    success
}

pub fn read_triangle_mesh_from_ply_file(filename: &str, mesh: &mut TriangleMesh) -> bool {
    let data = match std::fs::read(filename) {
        Ok(d) => d,
        Err(_) => return false,
    };
    let header = match parse_header(&data) {
        Some(h) => h,
        None => return false,
    };
    let vei = match header.elements.iter().position(|e| e.name == "vertex") {
        Some(i) => i,
        None => return false,
    };
    let vel = &header.elements[vei];
    if find_prop(vel, "x").is_none() || vel.count == 0 {
        return false;
    }
    let vertex_num = vel.count;
    let normal_num = if find_prop(vel, "nx").is_some() {
        vel.count
    } else {
        0
    };
    let color_num = if find_prop(vel, "red").is_some() {
        vel.count
    } else {
        0
    };

    let fei = header.elements.iter().position(|e| e.name == "face");
    let mut face_prop = None;
    if let Some(fi) = fei {
        let fe = &header.elements[fi];
        face_prop = find_prop(fe, "vertex_indices")
            .or_else(|| find_prop(fe, "vertex_index"))
            .map(|pi| (fi, pi));
    }

    let mut vertices = vec![[0.0f64; 3]; vertex_num];
    let mut vertex_normals = vec![[0.0f64; 3]; normal_num];
    let mut vertex_colors = vec![[0.0f64; 3]; color_num];
    let mut faces: Vec<Vec<u32>> = Vec::new();

    #[derive(Clone, Copy)]
    enum Target {
        Vertex(usize),
        Normal(usize),
        Color(usize),
        None,
    }
    let vel_props: Vec<Target> = vel
        .properties
        .iter()
        .map(|p| match p.name.as_str() {
            "x" => Target::Vertex(0),
            "y" => Target::Vertex(1),
            "z" => Target::Vertex(2),
            "nx" => Target::Normal(0),
            "ny" => Target::Normal(1),
            "nz" => Target::Normal(2),
            "red" => Target::Color(0),
            "green" => Target::Color(1),
            "blue" => Target::Color(2),
            _ => Target::None,
        })
        .collect();

    let mut cur_face: Vec<u32> = Vec::new();
    let mut invalid_face_index = false;
    let ok = read_ply_data(&data, &header, &mut |ei, inst, pi, vk, len, v| {
        if ei == vei {
            match vel_props[pi] {
                Target::Vertex(c) => vertices[inst][c] = v,
                Target::Normal(c) => {
                    if inst < normal_num {
                        vertex_normals[inst][c] = v;
                    }
                }
                Target::Color(c) => {
                    if inst < color_num {
                        vertex_colors[inst][c] = v / 255.0;
                    }
                }
                Target::None => {}
            }
        } else if let Some((fi, fpi)) = face_prop {
            if ei == fi && pi == fpi {
                if vk == 0 {
                    cur_face.clear();
                } else {
                    if !v.is_finite() || v < 0.0 || v.fract() != 0.0 || v > u32::MAX as f64 {
                        invalid_face_index = true;
                    } else {
                        cur_face.push(v as u32);
                    }
                    if vk == len {
                        faces.push(cur_face.clone());
                    }
                }
            }
        }
    });
    if !ok || invalid_face_index {
        return false;
    }

    let mut parsed_mesh = TriangleMesh::new();
    parsed_mesh.vertices = vertices;
    parsed_mesh.vertex_normals = vertex_normals;
    parsed_mesh.vertex_colors = vertex_colors;
    for mut face in faces {
        if !add_triangles_by_ear_clipping(&mut parsed_mesh, &mut face) {
            // LogWarning: polygon could not be decomposed
            return false;
        }
    }
    *mesh = parsed_mesh;
    true
}

/// TriangleMeshIO.cpp IsPointInsidePolygon
fn is_point_inside_polygon(polygon: &[[f64; 2]], x: f64, y: f64) -> bool {
    let mut inside = false;
    let n = polygon.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let (vx0, vy0) = (polygon[i][0], polygon[i][1]);
        let (vx1, vy1) = (polygon[j][0], polygon[j][1]);
        if ((vy0 <= y) && (vy1 > y)) || ((vy1 <= y) && (vy0 > y)) {
            let cross = (vx1 - vx0) * (y - vy0) / (vy1 - vy0) + vx0;
            if cross < x {
                inside = !inside;
            }
        }
    }
    inside
}

/// TriangleMeshIO.cpp AddTrianglesByEarClipping
fn add_triangles_by_ear_clipping(mesh: &mut TriangleMesh, indices: &mut Vec<u32>) -> bool {
    use crate::linalg::{add3, cross3, dot3, sub3};
    let mut n = indices.len();
    if n < 3
        || indices
            .iter()
            .any(|&index| index as usize >= mesh.vertices.len())
    {
        return false;
    }
    let mut face_normal = [0.0f64; 3];
    if n > 3 {
        for i in 0..n {
            let v1 = sub3(
                mesh.vertices[indices[(i + 1) % n] as usize],
                mesh.vertices[indices[i % n] as usize],
            );
            let v2 = sub3(
                mesh.vertices[indices[(i + 2) % n] as usize],
                mesh.vertices[indices[(i + 1) % n] as usize],
            );
            face_normal = add3(face_normal, cross3(v1, v2));
        }
        let l = dot3(face_normal, face_normal).sqrt();
        if l == 0.0 || !l.is_finite() {
            return false;
        }
        face_normal = [
            face_normal[0] * (1.0 / l),
            face_normal[1] * (1.0 / l),
            face_normal[2] * (1.0 / l),
        ];
    }

    let mut found_ear = true;
    while n > 3 {
        if !found_ear {
            return false;
        }
        found_ear = false;
        for i in 1..(n - 1) {
            let v1 = sub3(
                mesh.vertices[indices[i] as usize],
                mesh.vertices[indices[i - 1] as usize],
            );
            let v2 = sub3(
                mesh.vertices[indices[i + 1] as usize],
                mesh.vertices[indices[i] as usize],
            );
            let is_convex = dot3(face_normal, cross3(v1, v2)) > 0.0;
            if is_convex {
                let mut is_ear = true;
                let mut polygon = [[0.0f64; 2]; 3];
                for (j, poly) in polygon.iter_mut().enumerate() {
                    let vv = mesh.vertices[indices[i + j - 1] as usize];
                    poly[0] = vv[0];
                    poly[1] = vv[1];
                }
                for (j, &index) in indices.iter().take(n).enumerate() {
                    if j == i - 1 || j == i || j == i + 1 {
                        continue;
                    }
                    let v = mesh.vertices[index as usize];
                    if is_point_inside_polygon(&polygon, v[0], v[1]) {
                        is_ear = false;
                        break;
                    }
                }
                if is_ear {
                    found_ear = true;
                    mesh.triangles.push([
                        indices[i - 1] as i32,
                        indices[i] as i32,
                        indices[i + 1] as i32,
                    ]);
                    indices.remove(i);
                    n = indices.len();
                    break;
                }
            }
        }
    }
    mesh.triangles
        .push([indices[0] as i32, indices[1] as i32, indices[2] as i32]);
    true
}

pub fn read_file_geometry_type_ply(path: &str) -> FileGeometry {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(_) => return FileGeometry::ContentsUnknown,
    };
    let header = match parse_header(&data) {
        Some(h) => h,
        None => return FileGeometry::ContentsUnknown,
    };
    let n_vertices = header
        .elements
        .iter()
        .find(|e| e.name == "vertex")
        .map(|e| e.count)
        .unwrap_or(0);
    let n_faces = header
        .elements
        .iter()
        .find(|e| e.name == "face")
        .filter(|e| {
            e.properties
                .iter()
                .any(|p| p.name == "vertex_indices" || p.name == "vertex_index")
        })
        .map(|e| e.count)
        .unwrap_or(0);
    if n_faces > 0 {
        FileGeometry::ContainsTriangles
    } else if n_vertices > 0 {
        FileGeometry::ContainsPoints
    } else {
        FileGeometry::ContentsUnknown
    }
}

// -------------------------------------------------------------- writers

fn color_to_uint8(c: f64) -> u8 {
    (c.clamp(0.0, 1.0) * 255.0).round() as u8
}

struct PlyWriter {
    ascii: bool,
    buf: Vec<u8>,
    fmt_out: String,
    fmt_scratch: String,
}

impl PlyWriter {
    fn write_double(&mut self, v: f64, last_in_line: bool) {
        if self.ascii {
            self.fmt_out.clear();
            crate::linalg::format_g_into(&mut self.fmt_out, &mut self.fmt_scratch, v);
            self.buf.extend_from_slice(self.fmt_out.as_bytes());
            self.push_sep(last_in_line);
        } else {
            self.buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    fn write_uchar(&mut self, v: u8, last_in_line: bool) {
        if self.ascii {
            use std::fmt::Write;
            self.fmt_out.clear();
            let _ = write!(self.fmt_out, "{}", v);
            self.buf.extend_from_slice(self.fmt_out.as_bytes());
            self.push_sep(last_in_line);
        } else {
            self.buf.push(v);
        }
    }
    fn write_uint(&mut self, v: u32, last_in_line: bool) {
        if self.ascii {
            use std::fmt::Write;
            self.fmt_out.clear();
            let _ = write!(self.fmt_out, "{}", v);
            self.buf.extend_from_slice(self.fmt_out.as_bytes());
            self.push_sep(last_in_line);
        } else {
            self.buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    fn push_sep(&mut self, last_in_line: bool) {
        if last_in_line {
            self.buf.push(b'\n');
        } else {
            self.buf.push(b' ');
        }
    }
}

pub fn write_point_cloud_to_ply_file(filename: &str, pcd: &PointCloud, write_ascii: bool) -> bool {
    if pcd.is_empty() {
        return false;
    }
    let has_normals = pcd.has_normals();
    let has_colors = pcd.has_colors();

    let mut header = String::new();
    header.push_str("ply\n");
    header.push_str(if write_ascii {
        "format ascii 1.0\n"
    } else {
        "format binary_little_endian 1.0\n"
    });
    header.push_str("comment Created by tiny3d\n");
    header.push_str(&format!("element vertex {}\n", pcd.points.len()));
    header.push_str("property double x\nproperty double y\nproperty double z\n");
    if has_normals {
        header.push_str("property double nx\nproperty double ny\nproperty double nz\n");
    }
    if has_colors {
        header.push_str("property uchar red\nproperty uchar green\nproperty uchar blue\n");
    }
    header.push_str("end_header\n");

    let mut buf = header.into_bytes();
    buf.reserve(pcd.points.len() * if write_ascii { 80 } else { 27 } + 64);
    let mut w = PlyWriter {
        ascii: write_ascii,
        buf,
        fmt_out: String::new(),
        fmt_scratch: String::new(),
    };
    for i in 0..pcd.points.len() {
        let p = pcd.points[i];
        let n_line_vals = 3 + if has_normals { 3 } else { 0 } + if has_colors { 3 } else { 0 };
        let mut vi = 0;
        let last = |vi: &mut i32| {
            *vi += 1;
            *vi == n_line_vals
        };
        w.write_double(p[0], last(&mut vi));
        w.write_double(p[1], last(&mut vi));
        w.write_double(p[2], last(&mut vi));
        if has_normals {
            let n = pcd.normals[i];
            w.write_double(n[0], last(&mut vi));
            w.write_double(n[1], last(&mut vi));
            w.write_double(n[2], last(&mut vi));
        }
        if has_colors {
            let c = pcd.colors[i];
            w.write_uchar(color_to_uint8(c[0]), last(&mut vi));
            w.write_uchar(color_to_uint8(c[1]), last(&mut vi));
            w.write_uchar(color_to_uint8(c[2]), last(&mut vi));
        }
    }
    std::fs::write(filename, &w.buf).is_ok()
}

pub fn write_triangle_mesh_to_ply_file(
    filename: &str,
    mesh: &TriangleMesh,
    write_ascii: bool,
    write_vertex_normals: bool,
    write_vertex_colors: bool,
) -> bool {
    if mesh.is_empty() {
        return false;
    }
    let write_vertex_normals = write_vertex_normals && mesh.has_vertex_normals();
    let write_vertex_colors = write_vertex_colors && mesh.has_vertex_colors();

    let mut header = String::new();
    header.push_str("ply\n");
    header.push_str(if write_ascii {
        "format ascii 1.0\n"
    } else {
        "format binary_little_endian 1.0\n"
    });
    header.push_str("comment Created by tiny3d\n");
    header.push_str(&format!("element vertex {}\n", mesh.vertices.len()));
    header.push_str("property double x\nproperty double y\nproperty double z\n");
    if write_vertex_normals {
        header.push_str("property double nx\nproperty double ny\nproperty double nz\n");
    }
    if write_vertex_colors {
        header.push_str("property uchar red\nproperty uchar green\nproperty uchar blue\n");
    }
    header.push_str(&format!("element face {}\n", mesh.triangles.len()));
    header.push_str("property list uchar uint vertex_indices\n");
    header.push_str("end_header\n");

    let mut buf = header.into_bytes();
    buf.reserve(
        mesh.vertices.len() * if write_ascii { 80 } else { 27 } + mesh.triangles.len() * 16 + 64,
    );
    let mut w = PlyWriter {
        ascii: write_ascii,
        buf,
        fmt_out: String::new(),
        fmt_scratch: String::new(),
    };
    let n_line_vals =
        3 + if write_vertex_normals { 3 } else { 0 } + if write_vertex_colors { 3 } else { 0 };
    for i in 0..mesh.vertices.len() {
        let mut vi = 0;
        let last = |vi: &mut i32| {
            *vi += 1;
            *vi == n_line_vals
        };
        let v = mesh.vertices[i];
        w.write_double(v[0], last(&mut vi));
        w.write_double(v[1], last(&mut vi));
        w.write_double(v[2], last(&mut vi));
        if write_vertex_normals {
            let n = mesh.vertex_normals[i];
            w.write_double(n[0], last(&mut vi));
            w.write_double(n[1], last(&mut vi));
            w.write_double(n[2], last(&mut vi));
        }
        if write_vertex_colors {
            let c = mesh.vertex_colors[i];
            w.write_uchar(color_to_uint8(c[0]), last(&mut vi));
            w.write_uchar(color_to_uint8(c[1]), last(&mut vi));
            w.write_uchar(color_to_uint8(c[2]), last(&mut vi));
        }
    }
    for t in mesh.triangles.iter() {
        w.write_uchar(3, false);
        w.write_uint(t[0] as u32, false);
        w.write_uint(t[1] as u32, false);
        w.write_uint(t[2] as u32, true);
    }
    std::fs::write(filename, &w.buf).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_temp_ply(name: &str, contents: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("tiny3d-{name}-{}.ply", std::process::id()));
        std::fs::write(&path, contents).unwrap();
        path
    }

    fn sentinel_mesh() -> TriangleMesh {
        let mut mesh = TriangleMesh::new();
        mesh.vertices = vec![[9.0, 9.0, 9.0]];
        mesh
    }

    #[test]
    fn truncated_point_cloud_does_not_mutate_output() {
        let path = write_temp_ply(
            "truncated-point-cloud",
            "ply\n\
             format ascii 1.0\n\
             element vertex 2\n\
             property float x\n\
             property float y\n\
             property float z\n\
             end_header\n\
             0 0 0\n",
        );
        let mut point_cloud = PointCloud::new();
        point_cloud.points = vec![[9.0, 9.0, 9.0]];

        assert!(!read_point_cloud_from_ply_file(
            path.to_str().unwrap(),
            &mut point_cloud
        ));
        assert_eq!(point_cloud.points, vec![[9.0, 9.0, 9.0]]);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn failed_polygon_triangulation_does_not_mutate_output() {
        let path = write_temp_ply(
            "degenerate-face",
            "ply\n\
             format ascii 1.0\n\
             element vertex 4\n\
             property float x\n\
             property float y\n\
             property float z\n\
             element face 2\n\
             property list uchar int vertex_indices\n\
             end_header\n\
             0 0 0\n\
             1 0 0\n\
             0 1 0\n\
             2 0 0\n\
             3 0 1 2\n\
             4 0 0 0 0\n",
        );
        let mut mesh = sentinel_mesh();

        assert!(!read_triangle_mesh_from_ply_file(
            path.to_str().unwrap(),
            &mut mesh
        ));
        assert_eq!(mesh.vertices, vec![[9.0, 9.0, 9.0]]);
        assert!(mesh.triangles.is_empty());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn out_of_range_face_index_is_rejected_without_panicking() {
        let path = write_temp_ply(
            "large-face-index",
            "ply\n\
             format ascii 1.0\n\
             element vertex 3\n\
             property float x\n\
             property float y\n\
             property float z\n\
             element face 1\n\
             property list uchar int vertex_indices\n\
             end_header\n\
             0 0 0\n\
             1 0 0\n\
             0 1 0\n\
             3 0 1 99\n",
        );
        let mut mesh = sentinel_mesh();

        assert!(!read_triangle_mesh_from_ply_file(
            path.to_str().unwrap(),
            &mut mesh
        ));
        assert_eq!(mesh.vertices, vec![[9.0, 9.0, 9.0]]);
        let _ = std::fs::remove_file(path);
    }
}
