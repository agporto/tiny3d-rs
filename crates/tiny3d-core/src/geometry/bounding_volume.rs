//! tiny3d::geometry::AxisAlignedBoundingBox

use crate::linalg::*;

use super::point_cloud::{compute_max_bound, compute_min_bound};

#[derive(Clone, Debug)]
pub struct AxisAlignedBoundingBox {
    pub min_bound: V3,
    pub max_bound: V3,
    pub color: V3,
}

impl Default for AxisAlignedBoundingBox {
    fn default() -> Self {
        AxisAlignedBoundingBox {
            min_bound: ZERO3,
            max_bound: ZERO3,
            color: [1.0, 1.0, 1.0],
        }
    }
}

impl AxisAlignedBoundingBox {
    pub fn new(min_bound: V3, max_bound: V3) -> Self {
        let mut b = AxisAlignedBoundingBox {
            min_bound,
            max_bound,
            color: [1.0, 1.0, 1.0],
        };
        if (0..3).any(|i| b.max_bound[i] < b.min_bound[i]) {
            // LogWarning + correct bounds
            let cmin = [
                b.min_bound[0].min(b.max_bound[0]),
                b.min_bound[1].min(b.max_bound[1]),
                b.min_bound[2].min(b.max_bound[2]),
            ];
            let cmax = [
                b.min_bound[0].max(b.max_bound[0]),
                b.min_bound[1].max(b.max_bound[1]),
                b.min_bound[2].max(b.max_bound[2]),
            ];
            b.min_bound = cmin;
            b.max_bound = cmax;
        }
        b
    }

    pub fn clear(&mut self) {
        self.min_bound = ZERO3;
        self.max_bound = ZERO3;
        self.color = [1.0, 1.0, 1.0];
    }

    pub fn volume(&self) -> f64 {
        if (0..3).any(|i| self.max_bound[i] < self.min_bound[i]) {
            return 0.0;
        }
        let e = self.get_extent();
        // .prod() — Eigen redux product: (p0*p1)*p2
        (e[0] * e[1]) * e[2]
    }

    pub fn is_empty(&self) -> bool {
        self.volume() <= 1e-12 || (0..3).any(|i| self.max_bound[i] < self.min_bound[i])
    }

    pub fn get_min_bound(&self) -> V3 {
        self.min_bound
    }
    pub fn get_max_bound(&self) -> V3 {
        self.max_bound
    }

    pub fn get_center(&self) -> V3 {
        if self.is_empty() {
            return ZERO3;
        }
        let s = add3(self.min_bound, self.max_bound);
        scale3(s, 0.5)
    }

    pub fn get_extent(&self) -> V3 {
        sub3(self.max_bound, self.min_bound)
    }

    pub fn get_half_extent(&self) -> V3 {
        scale3(self.get_extent(), 0.5)
    }

    pub fn get_max_extent(&self) -> f64 {
        let e = self.get_extent();
        e[0].max(e[1]).max(e[2])
    }

    pub fn translate(&mut self, translation: V3, relative: bool) {
        if relative {
            self.min_bound = add3(self.min_bound, translation);
            self.max_bound = add3(self.max_bound, translation);
        } else {
            let center = self.get_center();
            let shift = sub3(translation, center);
            self.min_bound = add3(self.min_bound, shift);
            self.max_bound = add3(self.max_bound, shift);
        }
    }

    pub fn scale(&mut self, scale: f64, center: V3) {
        self.min_bound = add3(center, scale3(sub3(self.min_bound, center), scale));
        self.max_bound = add3(center, scale3(sub3(self.max_bound, center), scale));
        if scale < 0.0 {
            std::mem::swap(&mut self.min_bound, &mut self.max_bound);
        }
    }

    /// operator+=
    pub fn merge(&mut self, other: &AxisAlignedBoundingBox) {
        if self.is_empty() {
            *self = other.clone();
        } else if !other.is_empty() {
            for i in 0..3 {
                self.min_bound[i] = self.min_bound[i].min(other.min_bound[i]);
                self.max_bound[i] = self.max_bound[i].max(other.max_bound[i]);
            }
        }
    }

    pub fn create_from_points(points: &[V3]) -> Self {
        if points.is_empty() {
            // LogWarning
            return AxisAlignedBoundingBox::new(ZERO3, ZERO3);
        }
        AxisAlignedBoundingBox::new(compute_min_bound(points), compute_max_bound(points))
    }

    pub fn get_box_points(&self) -> Vec<V3> {
        let extent = self.get_extent();
        let min_c = extent[0].min(extent[1]).min(extent[2]);
        if min_c < 0.0 {
            return vec![self.min_bound; 8];
        }
        let mb = self.min_bound;
        vec![
            mb,
            add3(mb, [extent[0], 0.0, 0.0]),
            add3(mb, [0.0, extent[1], 0.0]),
            add3(mb, [0.0, 0.0, extent[2]]),
            add3(mb, [extent[0], extent[1], 0.0]),
            add3(mb, [0.0, extent[1], extent[2]]),
            add3(mb, [extent[0], 0.0, extent[2]]),
            self.max_bound,
        ]
    }

    pub fn get_point_indices_within_bounding_box(&self, points: &[V3]) -> Vec<usize> {
        let eps = 1e-9;
        let mut indices = Vec::new();
        for (idx, p) in points.iter().enumerate() {
            if (0..3).all(|i| p[i] >= self.min_bound[i] - eps)
                && (0..3).all(|i| p[i] <= self.max_bound[i] + eps)
            {
                indices.push(idx);
            }
        }
        indices
    }

    pub fn get_print_info(&self) -> String {
        format!(
            "AxisAlignedBoundingBox: min: ({:.4}, {:.4}, {:.4}), max: ({:.4}, {:.4}, {:.4})",
            self.min_bound[0],
            self.min_bound[1],
            self.min_bound[2],
            self.max_bound[0],
            self.max_bound[1],
            self.max_bound[2]
        )
    }
}
