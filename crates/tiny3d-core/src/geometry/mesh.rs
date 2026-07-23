//! tiny3d::geometry::MeshBase / TriangleMesh

use crate::linalg::*;

use super::point_cloud::{
    compute_center, compute_max_bound, compute_min_bound, rotate_normals, rotate_points,
    scale_points, transform_normals, transform_points, translate_points,
};

#[derive(Clone, Default)]
pub struct TriangleMesh {
    pub vertices: Vec<V3>,
    pub vertex_normals: Vec<V3>,
    pub vertex_colors: Vec<V3>,
    pub triangles: Vec<[i32; 3]>,
    pub triangle_normals: Vec<V3>,
}

impl TriangleMesh {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.vertex_normals.clear();
        self.vertex_colors.clear();
        self.triangles.clear();
        self.triangle_normals.clear();
    }

    pub fn has_vertices(&self) -> bool {
        !self.vertices.is_empty()
    }
    pub fn has_vertex_normals(&self) -> bool {
        self.has_vertices() && self.vertex_normals.len() == self.vertices.len()
    }
    pub fn has_vertex_colors(&self) -> bool {
        self.has_vertices() && self.vertex_colors.len() == self.vertices.len()
    }
    pub fn has_triangles(&self) -> bool {
        !self.triangles.is_empty()
    }
    pub fn has_triangle_normals(&self) -> bool {
        self.has_triangles() && self.triangle_normals.len() == self.triangles.len()
    }
    pub fn is_empty(&self) -> bool {
        !self.has_vertices()
    }

    pub fn get_min_bound(&self) -> V3 {
        compute_min_bound(&self.vertices)
    }
    pub fn get_max_bound(&self) -> V3 {
        compute_max_bound(&self.vertices)
    }
    pub fn get_center(&self) -> V3 {
        compute_center(&self.vertices)
    }

    pub fn transform(&mut self, t: &M4) {
        transform_points(t, &mut self.vertices);
        if self.has_vertex_normals() {
            transform_normals(t, &mut self.vertex_normals);
        }
        if self.has_triangle_normals() {
            transform_normals(t, &mut self.triangle_normals);
        }
    }

    pub fn translate(&mut self, translation: V3, relative: bool) {
        translate_points(translation, &mut self.vertices, relative);
    }

    pub fn scale(&mut self, s: f64, center: V3) {
        scale_points(s, &mut self.vertices, center);
    }

    pub fn rotate(&mut self, r: &M3, center: V3) {
        rotate_points(r, &mut self.vertices, center);
        if self.has_vertex_normals() {
            rotate_normals(r, &mut self.vertex_normals);
        }
        if self.has_triangle_normals() {
            rotate_normals(r, &mut self.triangle_normals);
        }
    }

    pub fn normalize_normals(&mut self) {
        for n in self.vertex_normals.iter_mut() {
            stable_normalize3(n);
            if n[0].is_nan() {
                *n = [0.0, 0.0, 1.0];
            }
        }
        for n in self.triangle_normals.iter_mut() {
            stable_normalize3(n);
            if n[0].is_nan() {
                *n = [0.0, 0.0, 1.0];
            }
        }
    }

    /// MeshBase::PaintUniformColor — note: unlike PointCloud, no clipping.
    pub fn paint_uniform_color(&mut self, color: V3) {
        self.vertex_colors = vec![color; self.vertices.len()];
    }

    pub fn compute_triangle_normals(&mut self, normalized: bool) {
        if !self.has_vertices() || !self.has_triangles() {
            return;
        }
        self.triangle_normals = vec![ZERO3; self.triangles.len()];
        for i in 0..self.triangles.len() {
            let t = self.triangles[i];
            let nv = self.vertices.len();
            if t.iter().any(|&x| x < 0 || x as usize >= nv) {
                self.triangle_normals[i] = ZERO3;
                continue;
            }
            let v0 = self.vertices[t[0] as usize];
            let v1 = self.vertices[t[1] as usize];
            let v2 = self.vertices[t[2] as usize];
            let v01 = sub3(v1, v0);
            let v02 = sub3(v2, v0);
            self.triangle_normals[i] = cross3(v01, v02);
        }
        if normalized {
            self.normalize_normals();
        }
    }

    pub fn compute_vertex_normals(&mut self, normalized: bool) {
        if !self.has_vertices() || !self.has_triangles() {
            return;
        }
        if !self.has_triangle_normals() {
            self.compute_triangle_normals(false);
        }
        self.vertex_normals = vec![ZERO3; self.vertices.len()];
        let nv = self.vertices.len();
        for i in 0..self.triangles.len() {
            let t = self.triangles[i];
            if t.iter().all(|&x| x >= 0 && (x as usize) < nv) {
                for &vi in t.iter() {
                    self.vertex_normals[vi as usize] =
                        add3(self.vertex_normals[vi as usize], self.triangle_normals[i]);
                }
            }
        }
        if normalized {
            self.normalize_normals();
        }
    }
}
