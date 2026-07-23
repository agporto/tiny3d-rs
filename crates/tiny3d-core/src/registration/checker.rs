//! CorrespondenceChecker port.

use crate::geometry::PointCloud;
use crate::linalg::*;

use super::estimation::{validate_correspondences, Correspondence};

#[derive(Clone, Debug)]
pub enum CorrespondenceChecker {
    EdgeLength { similarity_threshold: f64 },
    Distance { distance_threshold: f64 },
    Normal { normal_angle_threshold: f64 },
}

impl CorrespondenceChecker {
    pub fn require_pointcloud_alignment(&self) -> bool {
        matches!(self, CorrespondenceChecker::Distance { .. })
    }

    pub fn check(
        &self,
        source: &PointCloud,
        target: &PointCloud,
        corres: &[Correspondence],
        transformation: &M4,
    ) -> Result<bool, String> {
        validate_correspondences(source, target, corres)?;
        Ok(self.check_unchecked(source, target, corres, transformation))
    }

    pub(crate) fn check_unchecked(
        &self,
        source: &PointCloud,
        target: &PointCloud,
        corres: &[Correspondence],
        transformation: &M4,
    ) -> bool {
        match self {
            CorrespondenceChecker::EdgeLength {
                similarity_threshold,
            } => {
                let st2 = similarity_threshold * similarity_threshold;
                for i in 0..corres.len() {
                    for j in (i + 1)..corres.len() {
                        let ds2 = squared_norm3(sub3(
                            source.points[corres[i][0] as usize],
                            source.points[corres[j][0] as usize],
                        ));
                        let dt2 = squared_norm3(sub3(
                            target.points[corres[i][1] as usize],
                            target.points[corres[j][1] as usize],
                        ));
                        if ds2 < dt2 * st2 || dt2 < ds2 * st2 {
                            return false;
                        }
                    }
                }
                true
            }
            CorrespondenceChecker::Distance { distance_threshold } => {
                let r = m4_block3x3(transformation);
                let t = m4_translation(transformation);
                let dt2 = distance_threshold * distance_threshold;
                for c in corres {
                    let pt = add3(m3v3(&r, source.points[c[0] as usize]), t);
                    if squared_norm3(sub3(target.points[c[1] as usize], pt)) > dt2 {
                        return false;
                    }
                }
                true
            }
            CorrespondenceChecker::Normal {
                normal_angle_threshold,
            } => {
                if !source.has_normals() || !target.has_normals() {
                    return true;
                }
                let r = m4_block3x3(transformation);
                let cos_th = normal_angle_threshold.cos();
                for c in corres {
                    let nt = m3v3(&r, source.normals[c[0] as usize]);
                    if dot3(target.normals[c[1] as usize], nt) < cos_th {
                        return false;
                    }
                }
                true
            }
        }
    }
}
