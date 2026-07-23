//! tiny3d::geometry::VoxelGrid

use crate::linalg::*;
use crate::stdhash::StdUnorderedMap;

use super::bounding_volume::AxisAlignedBoundingBox;
use super::point_cloud::PointCloud;

#[derive(Clone, Debug, Default)]
pub struct Voxel {
    pub grid_index: [i32; 3],
    pub color: V3,
}

#[derive(Default)]
pub struct VoxelGrid {
    pub voxel_size: f64,
    pub origin: V3,
    /// unordered_map<Vector3i, Voxel> with libstdc++-faithful iteration order.
    pub voxels: StdUnorderedMap<Voxel>,
}

impl VoxelGrid {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.voxel_size = 0.0;
        self.origin = ZERO3;
        self.voxels = StdUnorderedMap::new();
    }

    pub fn has_voxels(&self) -> bool {
        !self.voxels.is_empty()
    }
    pub fn is_empty(&self) -> bool {
        !self.has_voxels()
    }
    pub fn has_colors(&self) -> bool {
        true
    }

    pub fn get_min_bound(&self) -> V3 {
        if !self.has_voxels() {
            self.origin
        } else {
            let mut min_gi = *self.voxels.iter().next().unwrap().0;
            for (gi, _) in self.voxels.iter() {
                for i in 0..3 {
                    min_gi[i] = min_gi[i].min(gi[i]);
                }
            }
            [
                self.origin[0] + min_gi[0] as f64 * self.voxel_size,
                self.origin[1] + min_gi[1] as f64 * self.voxel_size,
                self.origin[2] + min_gi[2] as f64 * self.voxel_size,
            ]
        }
    }

    pub fn get_max_bound(&self) -> V3 {
        if !self.has_voxels() {
            self.origin
        } else {
            let mut max_gi = *self.voxels.iter().next().unwrap().0;
            for (gi, _) in self.voxels.iter() {
                for i in 0..3 {
                    max_gi[i] = max_gi[i].max(gi[i]);
                }
            }
            [
                self.origin[0] + (max_gi[0] as f64 + 1.0) * self.voxel_size,
                self.origin[1] + (max_gi[1] as f64 + 1.0) * self.voxel_size,
                self.origin[2] + (max_gi[2] as f64 + 1.0) * self.voxel_size,
            ]
        }
    }

    pub fn get_center(&self) -> V3 {
        if !self.has_voxels() {
            return ZERO3;
        }
        let mut center_sum = ZERO3;
        let half = 0.5 * self.voxel_size;
        for (gi, _) in self.voxels.iter() {
            // origin + gi*voxel_size + half_voxel (Eigen elementwise, l-to-r)
            let v = [
                self.origin[0] + gi[0] as f64 * self.voxel_size + half,
                self.origin[1] + gi[1] as f64 * self.voxel_size + half,
                self.origin[2] + gi[2] as f64 * self.voxel_size + half,
            ];
            center_sum = add3(center_sum, v);
        }
        div3(center_sum, self.voxels.len() as f64)
    }

    pub fn get_axis_aligned_bounding_box(&self) -> AxisAlignedBoundingBox {
        AxisAlignedBoundingBox::new(self.get_min_bound(), self.get_max_bound())
    }

    /// Applies a transform that preserves the axis-aligned voxel lattice.
    ///
    /// Only identity and pure translations are supported. Other transforms
    /// are rejected before the grid is mutated.
    pub fn transform(&mut self, t: &M4) -> Result<(), String> {
        if !m4_is_pure_translation(t) {
            return Err(
                "VoxelGrid only supports identity or pure translation transforms".to_string(),
            );
        }
        let oh = [self.origin[0], self.origin[1], self.origin[2], 1.0];
        let nh = m4v4(t, oh);
        self.origin = [nh[0], nh[1], nh[2]];
        Ok(())
    }

    pub fn translate(&mut self, translation: V3, relative: bool) {
        if relative {
            self.origin = add3(self.origin, translation);
        } else {
            self.origin = translation;
        }
    }

    pub fn scale(&mut self, s: f64, center: V3) {
        self.origin = add3(center, scale3(sub3(self.origin, center), s));
        self.voxel_size *= s;
    }

    /// Accepts identity only because rotating an axis-aligned voxel lattice
    /// would require revoxelization.
    pub fn rotate(&mut self, r: &M3, _center: V3) -> Result<(), String> {
        if !m3_is_identity(r) {
            return Err(
                "VoxelGrid rotation is unsupported; revoxelize the geometry instead".to_string(),
            );
        }
        Ok(())
    }

    pub fn get_voxels(&self) -> Vec<Voxel> {
        self.voxels.iter().map(|(_, v)| v.clone()).collect()
    }

    pub fn create_from_point_cloud_within_bounds(
        input: &PointCloud,
        voxel_size: f64,
        min_bound: V3,
        max_bound: V3,
    ) -> Result<VoxelGrid, String> {
        let mut output = VoxelGrid::new();
        if voxel_size <= 0.0 {
            return Err("voxel_size must be positive.".to_string());
        }
        let ext = sub3(max_bound, min_bound);
        let max_extent = ext[0].max(ext[1]).max(ext[2]);
        if max_extent / voxel_size > i32::MAX as f64 {
            return Err(
                "voxel_size is potentially too small for the given bounds, may lead to integer overflow in indices."
                    .to_string(),
            );
        }
        output.voxel_size = voxel_size;
        output.origin = min_bound;
        let mut occupied: StdUnorderedMap<()> = StdUnorderedMap::new();
        for p in input.points.iter() {
            if (0..3).any(|i| p[i] < min_bound[i]) || (0..3).any(|i| p[i] >= max_bound[i]) {
                continue;
            }
            let rc = [
                (p[0] - min_bound[0]) / voxel_size,
                (p[1] - min_bound[1]) / voxel_size,
                (p[2] - min_bound[2]) / voxel_size,
            ];
            let vi = [
                rc[0].floor() as i32,
                rc[1].floor() as i32,
                rc[2].floor() as i32,
            ];
            occupied.insert(vi, ());
        }
        output.voxels.reserve(occupied.len());
        for (gi, _) in occupied.iter() {
            output.voxels.insert(
                *gi,
                Voxel {
                    grid_index: *gi,
                    color: [0.5, 0.5, 0.5],
                },
            );
        }
        Ok(output)
    }

    pub fn create_from_point_cloud(
        input: &PointCloud,
        voxel_size: f64,
    ) -> Result<VoxelGrid, String> {
        if input.is_empty() {
            // LogWarning; empty grid
            return Ok(VoxelGrid::new());
        }
        if voxel_size <= 0.0 {
            return Err("voxel_size must be positive for VoxelGrid creation.".to_string());
        }
        let mut min_bound = input.get_min_bound();
        let mut max_bound = input.get_max_bound();
        let half = 0.5 * voxel_size;
        for i in 0..3 {
            min_bound[i] -= half;
            max_bound[i] += half;
        }
        Self::create_from_point_cloud_within_bounds(input, voxel_size, min_bound, max_bound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_accepts_translation_and_rejects_other_transforms_atomically() {
        let mut grid = VoxelGrid::new();
        grid.origin = [1.0, 2.0, 3.0];

        let mut translation = m4_identity();
        translation[0][3] = 4.0;
        translation[1][3] = -1.0;
        grid.transform(&translation).unwrap();
        assert_eq!(grid.origin, [5.0, 1.0, 3.0]);

        let origin_before = grid.origin;
        let mut scaling = m4_identity();
        scaling[0][0] = 2.0;
        assert!(grid.transform(&scaling).is_err());
        assert_eq!(grid.origin, origin_before);
    }

    #[test]
    fn rotate_accepts_identity_and_rejects_rotation_atomically() {
        let mut grid = VoxelGrid::new();
        grid.origin = [1.0, 2.0, 3.0];
        assert!(grid.rotate(&m3_identity(), ZERO3).is_ok());

        let origin_before = grid.origin;
        let rotation = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(grid.rotate(&rotation, ZERO3).is_err());
        assert_eq!(grid.origin, origin_before);
    }
}
