//! tiny3d::geometry::KDTreeSearchParam

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KdTreeSearchParam {
    Knn { knn: i32 },
    Radius { radius: f64 },
    Hybrid { radius: f64, max_nn: i32 },
}

impl KdTreeSearchParam {
    pub fn search_type(&self) -> i32 {
        match self {
            KdTreeSearchParam::Knn { .. } => 0,
            KdTreeSearchParam::Radius { .. } => 1,
            KdTreeSearchParam::Hybrid { .. } => 2,
        }
    }
}

impl Default for KdTreeSearchParam {
    fn default() -> Self {
        KdTreeSearchParam::Knn { knn: 30 }
    }
}
