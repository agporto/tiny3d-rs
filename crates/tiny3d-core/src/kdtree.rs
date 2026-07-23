// Indexed traversal mirrors nanoflann's dimension and node visitation order.
#![allow(clippy::needless_range_loop)]

//! Independent implementation compatible with nanoflann 1.5.0's observable
//! KDTreeSingleIndexAdaptor behavior (metric_L2, leaf_max_size = 15) as used
//! by tiny3D's KDTreeFlann, plus the KDTreeFlann search wrappers.
//!
//! Data layout matches the C++: a dims x n column-major matrix; "point i,
//! component d" = data[i * dims + d].

#[derive(Clone, Copy)]
struct Interval {
    low: f64,
    high: f64,
}

enum Node {
    Leaf {
        left: usize,
        right: usize,
    },
    Split {
        divfeat: usize,
        divlow: f64,
        divhigh: f64,
        child1: usize,
        child2: usize,
    },
}

pub struct KdTree {
    dims: usize,
    n: usize,
    /// Column-major data: point i component d at [i*dims + d].
    data: Vec<f64>,
    vacc: Vec<u32>,
    nodes: Vec<Node>,
    root: usize,
    root_bbox: Vec<Interval>,
    leaf_max_size: usize,
}

pub const SEARCH_FAILED: i32 = -1;

/// Per-thread reusable search scratch (rayon-friendly, zero steady-state
/// allocation). Purely a performance measure: values are fully overwritten
/// before use on every query.
struct Scratch {
    knn_indices: Vec<u32>,
    knn_dists: Vec<f64>,
    dim_dists: Vec<f64>,
    radius_out: Vec<(i64, f64)>,
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = const { std::cell::RefCell::new(Scratch {
        knn_indices: Vec::new(),
        knn_dists: Vec::new(),
        dim_dists: Vec::new(),
        radius_out: Vec::new(),
    }) };
}

impl KdTree {
    /// Build from column-major (dims x n) data, leaf size 15 (tiny3d default).
    pub fn build(dims: usize, n: usize, data: Vec<f64>) -> Option<KdTree> {
        if n == 0 || dims == 0 {
            return None;
        }
        assert_eq!(data.len(), dims * n);
        let mut t = KdTree {
            dims,
            n,
            data,
            vacc: (0..n as u32).collect(),
            nodes: Vec::new(),
            root: 0,
            root_bbox: vec![
                Interval {
                    low: 0.0,
                    high: 0.0
                };
                dims
            ],
            leaf_max_size: 15,
        };
        let mut bbox = t.compute_bounding_box();
        t.root = t.divide_tree(0, n, &mut bbox);
        t.root_bbox = bbox;
        Some(t)
    }

    #[inline(always)]
    fn get(&self, idx: u32, dim: usize) -> f64 {
        // SAFETY: idx < n and dim < dims by construction (vacc holds valid
        // indices; dims fixed at build).
        unsafe { *self.data.get_unchecked(idx as usize * self.dims + dim) }
    }

    pub fn num_points(&self) -> usize {
        self.n
    }
    pub fn dims(&self) -> usize {
        self.dims
    }

    fn compute_bounding_box(&self) -> Vec<Interval> {
        let mut bbox = vec![
            Interval {
                low: 0.0,
                high: 0.0
            };
            self.dims
        ];
        for (i, bi) in bbox.iter_mut().enumerate() {
            let v = self.get(self.vacc[0], i);
            bi.low = v;
            bi.high = v;
        }
        for k in 1..self.n {
            for i in 0..self.dims {
                let val = self.get(self.vacc[k], i);
                if val < bbox[i].low {
                    bbox[i].low = val;
                }
                if val > bbox[i].high {
                    bbox[i].high = val;
                }
            }
        }
        bbox
    }

    fn compute_min_max(&self, ind: usize, count: usize, element: usize) -> (f64, f64) {
        let mut min_elem = self.get(self.vacc[ind], element);
        let mut max_elem = min_elem;
        for i in 1..count {
            let val = self.get(self.vacc[ind + i], element);
            if val < min_elem {
                min_elem = val;
            }
            if val > max_elem {
                max_elem = val;
            }
        }
        (min_elem, max_elem)
    }

    fn divide_tree(&mut self, left: usize, right: usize, bbox: &mut [Interval]) -> usize {
        let node_idx = self.nodes.len();
        self.nodes.push(Node::Leaf { left: 0, right: 0 }); // placeholder
        let dims = self.dims;

        if right - left <= self.leaf_max_size {
            self.nodes[node_idx] = Node::Leaf { left, right };
            for i in 0..dims {
                let v = self.get(self.vacc[left], i);
                bbox[i].low = v;
                bbox[i].high = v;
            }
            for k in (left + 1)..right {
                for i in 0..dims {
                    let val = self.get(self.vacc[k], i);
                    if bbox[i].low > val {
                        bbox[i].low = val;
                    }
                    if bbox[i].high < val {
                        bbox[i].high = val;
                    }
                }
            }
        } else {
            let (idx, cutfeat, cutval) = self.middle_split(left, right - left, bbox);

            let mut left_bbox = bbox.to_vec();
            left_bbox[cutfeat].high = cutval;
            let child1 = self.divide_tree(left, left + idx, &mut left_bbox);

            let mut right_bbox = bbox.to_vec();
            right_bbox[cutfeat].low = cutval;
            let child2 = self.divide_tree(left + idx, right, &mut right_bbox);

            let divlow = left_bbox[cutfeat].high;
            let divhigh = right_bbox[cutfeat].low;

            for i in 0..dims {
                bbox[i].low = left_bbox[i].low.min(right_bbox[i].low);
                bbox[i].high = left_bbox[i].high.max(right_bbox[i].high);
            }

            self.nodes[node_idx] = Node::Split {
                divfeat: cutfeat,
                divlow,
                divhigh,
                child1,
                child2,
            };
        }
        node_idx
    }

    fn middle_split(&mut self, ind: usize, count: usize, bbox: &[Interval]) -> (usize, usize, f64) {
        const EPS: f64 = 0.00001;
        let dims = self.dims;
        let mut max_span = bbox[0].high - bbox[0].low;
        for bi in bbox.iter().take(dims).skip(1) {
            let span = bi.high - bi.low;
            if span > max_span {
                max_span = span;
            }
        }
        let mut max_spread = -1.0f64;
        let mut cutfeat = 0usize;
        for (i, bi) in bbox.iter().enumerate().take(dims) {
            let span = bi.high - bi.low;
            if span > (1.0 - EPS) * max_span {
                let (min_elem, max_elem) = self.compute_min_max(ind, count, i);
                let spread = max_elem - min_elem;
                if spread > max_spread {
                    cutfeat = i;
                    max_spread = spread;
                }
            }
        }
        let split_val = (bbox[cutfeat].low + bbox[cutfeat].high) / 2.0;
        let (min_elem, max_elem) = self.compute_min_max(ind, count, cutfeat);
        let cutval = if split_val < min_elem {
            min_elem
        } else if split_val > max_elem {
            max_elem
        } else {
            split_val
        };

        let (lim1, lim2) = self.plane_split(ind, count, cutfeat, cutval);

        let index = if lim1 > count / 2 {
            lim1
        } else if lim2 < count / 2 {
            lim2
        } else {
            count / 2
        };
        (index, cutfeat, cutval)
    }

    fn plane_split(
        &mut self,
        ind: usize,
        count: usize,
        cutfeat: usize,
        cutval: f64,
    ) -> (usize, usize) {
        let mut left = 0usize;
        let mut right = count - 1;
        loop {
            while left <= right && self.get(self.vacc[ind + left], cutfeat) < cutval {
                left += 1;
            }
            while right > 0 && left <= right && self.get(self.vacc[ind + right], cutfeat) >= cutval
            {
                right -= 1;
            }
            if left > right || right == 0 {
                break;
            }
            self.vacc.swap(ind + left, ind + right);
            left += 1;
            right -= 1;
        }
        let lim1 = left;
        right = count - 1;
        loop {
            while left <= right && self.get(self.vacc[ind + left], cutfeat) <= cutval {
                left += 1;
            }
            while right > 0 && left <= right && self.get(self.vacc[ind + right], cutfeat) > cutval {
                right -= 1;
            }
            if left > right || right == 0 {
                break;
            }
            self.vacc.swap(ind + left, ind + right);
            left += 1;
            right -= 1;
        }
        (lim1, left)
    }

    /// nanoflann L2_Adaptor::evalMetric (4-way unrolled accumulation, no early exit).
    #[inline(always)]
    fn eval_metric3(&self, query: &[f64; 3], b_idx: u32) -> f64 {
        // dims == 3 tail-only path of evalMetric: sequential accumulation.
        let base = b_idx as usize * 3;
        // SAFETY: base+2 < data.len() by construction.
        unsafe {
            let d0 = query[0] - *self.data.get_unchecked(base);
            let d1 = query[1] - *self.data.get_unchecked(base + 1);
            let d2 = query[2] - *self.data.get_unchecked(base + 2);
            ((d0 * d0) + d1 * d1) + d2 * d2
        }
    }

    /// nanoflann L2_Adaptor::evalMetric (4-way unrolled accumulation, no early exit).
    fn eval_metric(&self, query: &[f64], b_idx: u32) -> f64 {
        if self.dims == 3 {
            let q3: &[f64; 3] = query[..3].try_into().unwrap();
            return self.eval_metric3(q3, b_idx);
        }
        let size = self.dims;
        let mut result = 0.0f64;
        let mut a = 0usize;
        let mut d = 0usize;
        // groups of 4 while a + 4 <= size - ... (a < lastgroup where lastgroup = size-3)
        while a + 3 < size {
            let diff0 = query[a] - self.get(b_idx, d);
            let diff1 = query[a + 1] - self.get(b_idx, d + 1);
            let diff2 = query[a + 2] - self.get(b_idx, d + 2);
            let diff3 = query[a + 3] - self.get(b_idx, d + 3);
            d += 4;
            result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
            a += 4;
        }
        while a < size {
            let diff0 = query[a] - self.get(b_idx, d);
            a += 1;
            d += 1;
            result += diff0 * diff0;
        }
        result
    }

    #[inline]
    fn accum_dist(a: f64, b: f64) -> f64 {
        (a - b) * (a - b)
    }

    fn compute_initial_distances(&self, vec: &[f64], dists: &mut [f64]) -> f64 {
        let mut dist = 0.0f64;
        for i in 0..self.dims {
            if vec[i] < self.root_bbox[i].low {
                dists[i] = Self::accum_dist(vec[i], self.root_bbox[i].low);
                dist += dists[i];
            }
            if vec[i] > self.root_bbox[i].high {
                dists[i] = Self::accum_dist(vec[i], self.root_bbox[i].high);
                dist += dists[i];
            }
        }
        dist
    }

    /// KNN search. Returns number found; fills indices/dists (squared).
    pub fn knn_search(
        &self,
        query: &[f64],
        num_closest: usize,
        out_indices: &mut Vec<i64>,
        out_dists: &mut Vec<f64>,
    ) -> usize {
        SCRATCH.with(|sc| {
            let sc = &mut *sc.borrow_mut();
            sc.knn_indices.clear();
            sc.knn_indices.resize(num_closest, 0);
            sc.knn_dists.clear();
            sc.knn_dists.resize(num_closest, 0.0);
            if self.n != 0 {
                let eps_error = 1.0f64; // SearchParameters default eps = 0
                sc.dim_dists.clear();
                sc.dim_dists.resize(self.dims, 0.0);
                // reborrow scratch pieces disjointly
                let (dim_dists, rs_idx, rs_d) =
                    (&mut sc.dim_dists, &mut sc.knn_indices, &mut sc.knn_dists);
                let mut rs = KnnResultSet {
                    indices: rs_idx,
                    dists: rs_d,
                    capacity: num_closest,
                    count: 0,
                };
                if num_closest > 0 {
                    rs.dists[num_closest - 1] = f64::MAX;
                }
                let dist = self.compute_initial_distances(query, dim_dists);
                self.search_level_knn(&mut rs, query, self.root, dist, dim_dists, eps_error);
                let count = rs.count;
                out_indices.clear();
                out_dists.clear();
                out_indices.extend(sc.knn_indices.iter().take(count).map(|&i| i as i64));
                out_dists.extend(sc.knn_dists.iter().take(count));
                return count;
            }
            out_indices.clear();
            out_dists.clear();
            0
        })
    }

    fn search_level_knn(
        &self,
        result_set: &mut KnnResultSet<'_>,
        vec: &[f64],
        node: usize,
        mut mindist: f64,
        dists: &mut [f64],
        eps_error: f64,
    ) -> bool {
        match &self.nodes[node] {
            Node::Leaf { left, right } => {
                let worst_dist = result_set.worst_dist();
                for i in *left..*right {
                    let accessor = self.vacc[i];
                    let dist = self.eval_metric(vec, accessor);
                    if dist < worst_dist && !result_set.add_point(dist, accessor) {
                        return false;
                    }
                }
                true
            }
            Node::Split {
                divfeat,
                divlow,
                divhigh,
                child1,
                child2,
            } => {
                let idx = *divfeat;
                let val = vec[idx];
                let diff1 = val - divlow;
                let diff2 = val - divhigh;
                let (best_child, other_child, cut_dist) = if diff1 + diff2 < 0.0 {
                    (*child1, *child2, Self::accum_dist(val, *divhigh))
                } else {
                    (*child2, *child1, Self::accum_dist(val, *divlow))
                };
                if !self.search_level_knn(result_set, vec, best_child, mindist, dists, eps_error) {
                    return false;
                }
                let dst = dists[idx];
                mindist = mindist + cut_dist - dst;
                dists[idx] = cut_dist;
                if mindist * eps_error <= result_set.worst_dist()
                    && !self.search_level_knn(
                        result_set,
                        vec,
                        other_child,
                        mindist,
                        dists,
                        eps_error,
                    )
                {
                    return false;
                }
                dists[idx] = dst;
                true
            }
        }
    }

    /// Radius search with squared radius; results sorted by distance
    /// (std::sort with < on distance — order of exact ties is unspecified in
    /// C++ too; we use a stable sort by (dist) which matches in practice).
    pub fn radius_search(&self, query: &[f64], radius_sq: f64, out: &mut Vec<(i64, f64)>) -> usize {
        out.clear();
        if self.n == 0 {
            return 0;
        }
        SCRATCH.with(|sc| {
            let dists = &mut sc.borrow_mut().dim_dists;
            dists.clear();
            dists.resize(self.dims, 0.0);
            let dist = self.compute_initial_distances(query, dists);
            self.search_level_radius(query, self.root, dist, dists, radius_sq, out);
        });
        // std::sort with IndexDist_Sorter — libstdc++ introsort, so exact
        // tie order matches the C++ build.
        crate::stdsort::std_sort(out, |a, b| a.1 < b.1);
        out.len()
    }

    fn search_level_radius(
        &self,
        vec: &[f64],
        node: usize,
        mut mindist: f64,
        dists: &mut [f64],
        radius: f64,
        out: &mut Vec<(i64, f64)>,
    ) {
        match &self.nodes[node] {
            Node::Leaf { left, right } => {
                for i in *left..*right {
                    let accessor = self.vacc[i];
                    let dist = self.eval_metric(vec, accessor);
                    if dist < radius {
                        out.push((accessor as i64, dist));
                    }
                }
            }
            Node::Split {
                divfeat,
                divlow,
                divhigh,
                child1,
                child2,
            } => {
                let idx = *divfeat;
                let val = vec[idx];
                let diff1 = val - divlow;
                let diff2 = val - divhigh;
                let (best_child, other_child, cut_dist) = if diff1 + diff2 < 0.0 {
                    (*child1, *child2, Self::accum_dist(val, *divhigh))
                } else {
                    (*child2, *child1, Self::accum_dist(val, *divlow))
                };
                self.search_level_radius(vec, best_child, mindist, dists, radius, out);
                let dst = dists[idx];
                mindist = mindist + cut_dist - dst;
                dists[idx] = cut_dist;
                if mindist <= radius {
                    self.search_level_radius(vec, other_child, mindist, dists, radius, out);
                }
                dists[idx] = dst;
            }
        }
    }
}

struct KnnResultSet<'a> {
    indices: &'a mut [u32],
    dists: &'a mut [f64],
    capacity: usize,
    count: usize,
}

impl KnnResultSet<'_> {
    #[inline(always)]
    fn worst_dist(&self) -> f64 {
        // SAFETY: capacity > 0 in all search paths.
        unsafe { *self.dists.get_unchecked(self.capacity - 1) }
    }

    fn add_point(&mut self, dist: f64, index: u32) -> bool {
        let mut i = self.count;
        while i > 0 {
            if self.dists[i - 1] > dist {
                if i < self.capacity {
                    self.dists[i] = self.dists[i - 1];
                    self.indices[i] = self.indices[i - 1];
                }
            } else {
                break;
            }
            i -= 1;
        }
        if i < self.capacity {
            self.dists[i] = dist;
            self.indices[i] = index;
        }
        if self.count < self.capacity {
            self.count += 1;
        }
        true
    }
}

// ---------------- KDTreeFlann wrapper (tiny3d::geometry::KDTreeFlann) ----------------

use crate::geometry::search_param::KdTreeSearchParam;

pub struct KdTreeFlann {
    tree: Option<KdTree>,
}

impl Default for KdTreeFlann {
    fn default() -> Self {
        Self::new()
    }
}

impl KdTreeFlann {
    pub fn new() -> Self {
        KdTreeFlann { tree: None }
    }

    /// SetRawData: column-major dims x n. Returns false on empty.
    pub fn set_matrix_data(&mut self, dims: usize, n: usize, data: Vec<f64>) -> bool {
        if dims * n == 0 {
            self.tree = None;
            return false;
        }
        self.tree = KdTree::build(dims, n, data);
        self.tree.is_some()
    }

    pub fn set_points(&mut self, points: &[[f64; 3]]) -> bool {
        let mut data = Vec::with_capacity(points.len() * 3);
        for p in points {
            data.extend_from_slice(p);
        }
        self.set_matrix_data(3, points.len(), data)
    }

    pub fn dims(&self) -> usize {
        self.tree.as_ref().map_or(0, |t| t.dims())
    }

    pub fn is_empty(&self) -> bool {
        self.tree.is_none()
    }

    pub fn search(
        &self,
        query: &[f64],
        param: &KdTreeSearchParam,
        indices: &mut Vec<i64>,
        distance2: &mut Vec<f64>,
    ) -> i32 {
        match param {
            KdTreeSearchParam::Knn { knn } => self.search_knn(query, *knn, indices, distance2),
            KdTreeSearchParam::Radius { radius } => {
                self.search_radius(query, *radius, indices, distance2)
            }
            KdTreeSearchParam::Hybrid { radius, max_nn } => {
                self.search_hybrid(query, *radius, *max_nn, indices, distance2)
            }
        }
    }

    pub fn search_knn(
        &self,
        query: &[f64],
        knn: i32,
        indices: &mut Vec<i64>,
        distance2: &mut Vec<f64>,
    ) -> i32 {
        let tree = match &self.tree {
            Some(t) => t,
            None => return SEARCH_FAILED,
        };
        if query.len() != tree.dims() || knn < 0 {
            return SEARCH_FAILED;
        }
        let k = tree.knn_search(query, knn as usize, indices, distance2);
        k as i32
    }

    pub fn search_radius(
        &self,
        query: &[f64],
        radius: f64,
        indices: &mut Vec<i64>,
        distance2: &mut Vec<f64>,
    ) -> i32 {
        let tree = match &self.tree {
            Some(t) => t,
            None => return SEARCH_FAILED,
        };
        if query.len() != tree.dims() {
            return SEARCH_FAILED;
        }
        // Take the reusable buffer out of the thread-local (dropping the
        // RefCell guard) so the tree's internal scratch borrow can't overlap.
        let mut taken = SCRATCH.with(|sc| std::mem::take(&mut sc.borrow_mut().radius_out));
        let k = tree.radius_search(query, radius * radius, &mut taken);
        indices.clear();
        distance2.clear();
        for &(i, d) in taken.iter() {
            indices.push(i);
            distance2.push(d);
        }
        SCRATCH.with(|sc| sc.borrow_mut().radius_out = taken);
        k as i32
    }

    pub fn search_hybrid(
        &self,
        query: &[f64],
        radius: f64,
        max_nn: i32,
        indices: &mut Vec<i64>,
        distance2: &mut Vec<f64>,
    ) -> i32 {
        let tree = match &self.tree {
            Some(t) => t,
            None => return SEARCH_FAILED,
        };
        if query.len() != tree.dims() || max_nn < 0 {
            return SEARCH_FAILED;
        }
        if max_nn == 1 {
            // KDTreeFlann::SearchHybrid max_nn==1 fast path
            let k = tree.knn_search(query, 1, indices, distance2);
            if k > 0 && distance2[0] < radius * radius {
                indices.truncate(1);
                distance2.truncate(1);
                return 1;
            }
            indices.clear();
            distance2.clear();
            return 0;
        }
        let k = tree.knn_search(query, max_nn as usize, indices, distance2) as i32;
        // lower_bound(distance2.begin(), begin+k, radius*radius)
        let rr = radius * radius;
        let mut lo = 0usize;
        let mut hi = k as usize;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if distance2[mid] < rr {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        indices.truncate(lo);
        distance2.truncate(lo);
        lo as i32
    }
}
