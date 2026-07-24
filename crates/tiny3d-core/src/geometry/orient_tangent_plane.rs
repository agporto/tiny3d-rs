//! Open3D `PointCloud::OrientNormalsConsistentTangentPlane`
//! (EstimateNormals.cpp port): Hoppe et al. '92 consistent normal
//! orientation, propagated over a Riemannian graph.
//!
//! Upstream pipeline: build the Euclidean minimum spanning tree, reweight
//! its edges by the normal-alignment cost `1 - |n0 . n1|`, add kNN edges
//! with the same cost, extract a second MST (Kruskal) from that Riemannian
//! graph, then BFS from the minimum-z vertex — whose normal is first
//! oriented towards `(0, 0, -1)` — flipping every visited normal whose dot
//! product with its already-oriented parent is negative.
//!
//! DIVERGENCE (documented approximation): upstream obtains the Euclidean
//! MST from a Qhull Delaunay tetrahedralization (the EMST is a subgraph of
//! the Delaunay graph). A Delaunay/Qhull dependency is out of scope for
//! this pure-Rust subset, so the EMST is approximated by a Kruskal MST over
//! the symmetric kNN graph, plus deterministic nearest-pair bridging of any
//! components the kNN graph leaves disconnected. The bridging preserves the
//! guarantee upstream gets from Delaunay connectivity: the propagation
//! reaches every normal. On well-sampled surfaces the resulting spanning
//! tree matches upstream's up to weight ties; individual flip decisions can
//! differ near tie boundaries, global sign consistency does not.
//!
//! DIVERGENCE (unimplemented, non-default): when `lambda != 0.0`, upstream
//! additionally excludes kNN-phase neighbors whose distance to the tangent
//! plane is an outlier (`> q3 + 1.5 * iqr`). This subset applies the lambda
//! penalization to the Euclidean edge weights but skips the IQR exclusion.
//! At the default `lambda = 0.0` behavior is unaffected.

use super::point_cloud::PointCloud;
use crate::kdtree::KdTreeFlann;
use crate::linalg::*;
use crate::stdsort::std_sort;

use std::collections::VecDeque;

#[derive(Clone, Copy)]
struct WeightedEdge {
    v0: usize,
    v1: usize,
    weight: f64,
}

/// Union-find with path compression and union by size (upstream
/// `DisjointSet`).
struct DisjointSet {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        DisjointSet {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    /// Returns true when the two sets were distinct and are now merged.
    fn union(&mut self, a: usize, b: usize) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        let (big, small) = if self.size[ra] >= self.size[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small] = big;
        self.size[big] += self.size[small];
        true
    }
}

/// Kruskal MST (upstream helper). Upstream sorts with `std::sort` and a
/// weight-only comparator; the libstdc++-faithful `std_sort` keeps the
/// order of tied weights deterministic for a given input order.
fn kruskal(mut edges: Vec<WeightedEdge>, n_vertices: usize) -> Vec<WeightedEdge> {
    std_sort(&mut edges, |a, b| a.weight < b.weight);
    let mut ds = DisjointSet::new(n_vertices);
    let mut mst = Vec::new();
    for e in edges {
        if ds.union(e.v0, e.v1) {
            mst.push(e);
        }
    }
    mst
}

pub fn orient_normals_consistent_tangent_plane(
    pc: &mut PointCloud,
    k: usize,
    lambda: f64,
    cos_alpha_tol: f64,
) -> Result<(), String> {
    if !pc.has_normals() {
        return Err("No normals in the PointCloud. Call EstimateNormals() first.".to_string());
    }
    let n = pc.points.len();
    let points = &pc.points;

    let mut kdtree = KdTreeFlann::new();
    kdtree.set_points(points);

    // One kNN pass serves both graph phases. Upstream's kNN phase searches
    // `k` neighbors *including* the query point itself (`SearchKNN(p, k)` on
    // a member point) and skips the self match; the Euclidean phase (our
    // Delaunay substitute) uses the full list, with a floor of one true
    // neighbor so the graph is never trivially edgeless.
    let knn = k.max(2) as i32;
    // Parallel over points: each neighbor list depends only on immutable
    // inputs and is written to a disjoint index, so the result is identical
    // to the serial loop for any thread count.
    use rayon::prelude::*;
    let neighbor_lists: Vec<Vec<usize>> = (0..n)
        .into_par_iter()
        .map_init(
            || (Vec::new(), Vec::new()),
            |(idx_buf, d2_buf), i| {
                let cnt = kdtree.search_knn(&points[i], knn, idx_buf, d2_buf);
                (0..cnt.max(0) as usize)
                    .map(|t| idx_buf[t] as usize)
                    .collect()
            },
        )
        .collect();

    // Excludes edges nearly parallel to the source normal (upstream test;
    // inactive at the default `cos_alpha_tol = 1.0` for unit normals).
    // NaN from coincident points compares false and keeps the edge,
    // matching the C++ comparison semantics.
    let normals_ro = &pc.normals;
    let edge_excluded = |v0: usize, v1: usize| -> bool {
        let diff = sub3(points[v1], points[v0]);
        let proj = dot3(diff, normals_ro[v0]).abs();
        let cos_alpha = proj / norm3(diff);
        cos_alpha > cos_alpha_tol
    };

    // ---- Phase 1: Euclidean MST substitute over the kNN graph.
    // Upstream Delaunay-edge weight: squared distance plus the lambda
    // penalization `lambda * |diff . n0|`.
    let mut euclid_edges = Vec::new();
    for (v0, nbrs) in neighbor_lists.iter().enumerate() {
        for &v1 in nbrs {
            if v0 == v1 {
                continue;
            }
            if edge_excluded(v0, v1) {
                continue;
            }
            let diff = sub3(points[v1], points[v0]);
            let dist2 = squared_norm3(diff);
            let penalization = lambda * dot3(diff, normals_ro[v0]).abs();
            euclid_edges.push(WeightedEdge {
                v0,
                v1,
                weight: dist2 + penalization,
            });
        }
    }
    let mut mst = kruskal(euclid_edges, n);

    // ---- Bridging (divergence, see module docs): connect any components
    // the kNN graph missed, cheapest crossing pair first, deterministic
    // tie-break on vertex indices.
    let mut ds = DisjointSet::new(n);
    for e in &mst {
        ds.union(e.v0, e.v1);
    }
    let mut idx_buf = Vec::new();
    let mut d2_buf = Vec::new();
    loop {
        let anchor = ds.find(0);
        let Some(first_out) = (0..n).find(|&v| ds.find(v) != anchor) else {
            break;
        };
        let comp = ds.find(first_out);
        let members: Vec<usize> = (0..n).filter(|&v| ds.find(v) == comp).collect();
        // Nearest vertex outside this component over all members, found by
        // growing kNN probes (neighbors come back sorted by distance, so the
        // first crossing hit is the nearest for that member).
        let mut best: Option<(f64, usize, usize)> = None;
        for &v in &members {
            let mut m = (k.max(2) + 1).min(n);
            loop {
                let cnt = kdtree.search_knn(&points[v], m as i32, &mut idx_buf, &mut d2_buf);
                let mut found = false;
                for t in 0..cnt.max(0) as usize {
                    let u = idx_buf[t] as usize;
                    if ds.find(u) != comp {
                        let cand = (d2_buf[t], v, u);
                        let better = match best {
                            None => true,
                            Some(b) => {
                                cand.0 < b.0 || (cand.0 == b.0 && (cand.1, cand.2) < (b.1, b.2))
                            }
                        };
                        if better {
                            best = Some(cand);
                        }
                        found = true;
                        break;
                    }
                }
                if found || m >= n {
                    break;
                }
                m = (m * 2).min(n);
            }
        }
        match best {
            Some((d2, a, b)) => {
                mst.push(WeightedEdge {
                    v0: a,
                    v1: b,
                    weight: d2,
                });
                ds.union(a, b);
            }
            None => break, // unreachable: n > |members| implies a crossing pair
        }
    }

    // ---- Reweight the Euclidean tree to the Riemannian normal cost and add
    // the kNN edges with the same cost (upstream `NormalWeight`).
    let normal_weight =
        |v0: usize, v1: usize| -> f64 { 1.0 - dot3(normals_ro[v0], normals_ro[v1]).abs() };
    let mut riemannian_edges = mst;
    for e in riemannian_edges.iter_mut() {
        e.weight = normal_weight(e.v0, e.v1);
    }
    for (v0, nbrs) in neighbor_lists.iter().enumerate() {
        // Upstream searches `k` including self; our lists may hold up to
        // `max(k, 2)` entries, so truncate back to upstream's count.
        for &v1 in nbrs.iter().take(k) {
            if v0 == v1 {
                continue;
            }
            if edge_excluded(v0, v1) {
                continue;
            }
            riemannian_edges.push(WeightedEdge {
                v0,
                v1,
                weight: normal_weight(v0, v1),
            });
        }
    }
    let riemannian_mst = kruskal(riemannian_edges, n);

    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    for e in &riemannian_mst {
        adjacency[e.v0].push(e.v1);
        adjacency[e.v1].push(e.v0);
    }

    // ---- Seed: minimum-z vertex (`std::min_element` semantics: strict
    // less-than, first minimum wins), normal oriented towards (0, 0, -1).
    let mut v0 = 0usize;
    for i in 1..n {
        if points[i][2] < points[v0][2] {
            v0 = i;
        }
    }

    let normals = &mut pc.normals;
    if dot3([0.0, 0.0, -1.0], normals[v0]) < 0.0 {
        normals[v0] = [-normals[v0][0], -normals[v0][1], -normals[v0][2]];
    }

    // ---- BFS propagation (upstream `std::queue`): flip a child whose dot
    // product with its already-oriented parent is negative.
    let mut visited = vec![false; n];
    let mut queue = VecDeque::new();
    visited[v0] = true;
    queue.push_back(v0);
    while let Some(u) = queue.pop_front() {
        for &v in &adjacency[u] {
            if !visited[v] {
                visited[v] = true;
                if dot3(normals[u], normals[v]) < 0.0 {
                    normals[v] = [-normals[v][0], -normals[v][1], -normals[v][2]];
                }
                queue.push_back(v);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic quasi-uniform sphere sampling (Fibonacci lattice).
    fn fibonacci_sphere(n: usize, center: V3, radius: f64) -> Vec<V3> {
        let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        (0..n)
            .map(|i| {
                let y = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
                let r = (1.0 - y * y).sqrt();
                let th = golden * i as f64;
                [
                    center[0] + radius * r * th.cos(),
                    center[1] + radius * y,
                    center[2] + radius * r * th.sin(),
                ]
            })
            .collect()
    }

    /// Radial (outward) unit normals with a deterministic sign scramble.
    fn scrambled_radial_normals(points: &[V3], center: V3, period: usize) -> Vec<V3> {
        points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mut nrm = normalized3(sub3(*p, center));
                if i % period == 0 {
                    nrm = [-nrm[0], -nrm[1], -nrm[2]];
                }
                nrm
            })
            .collect()
    }

    fn outward_consistency(pc: &PointCloud, center: V3) -> (usize, usize) {
        let mut outward = 0;
        let mut inward = 0;
        for (p, nrm) in pc.points.iter().zip(pc.normals.iter()) {
            if dot3(*nrm, sub3(*p, center)) > 0.0 {
                outward += 1;
            } else {
                inward += 1;
            }
        }
        (outward, inward)
    }

    #[test]
    fn sphere_becomes_globally_consistent_outward() {
        let center = [0.0, 0.0, 0.0];
        let points = fibonacci_sphere(500, center, 1.0);
        let normals = scrambled_radial_normals(&points, center, 2);
        let mut pc = PointCloud {
            points,
            normals,
            colors: Vec::new(),
        };
        pc.orient_normals_consistent_tangent_plane(10, 0.0, 1.0)
            .unwrap();
        // Min-z seed towards (0,0,-1) is the outward direction at the sphere
        // bottom, so the whole sphere must come out outward.
        let (outward, inward) = outward_consistency(&pc, center);
        assert_eq!(inward, 0, "expected all outward, got {inward} inward");
        assert_eq!(outward, 500);
    }

    #[test]
    fn result_is_invariant_to_input_signs() {
        let center = [0.0, 0.0, 0.0];
        let points = fibonacci_sphere(400, center, 1.0);
        let mut pc_a = PointCloud {
            points: points.clone(),
            normals: scrambled_radial_normals(&points, center, 2),
            colors: Vec::new(),
        };
        let mut pc_b = PointCloud {
            points,
            normals: scrambled_radial_normals(&pc_a.points, center, 3),
            colors: Vec::new(),
        };
        pc_a.orient_normals_consistent_tangent_plane(10, 0.0, 1.0)
            .unwrap();
        pc_b.orient_normals_consistent_tangent_plane(10, 0.0, 1.0)
            .unwrap();
        // Weights use |dot| and the seed sign is forced, so the output must
        // not depend on the input sign pattern — bitwise.
        assert_eq!(pc_a.normals, pc_b.normals);
    }

    #[test]
    fn heightfield_oriented_downward_from_min_z_seed() {
        // z = 0.3 sin(3x) cos(3y) over a deterministic grid; normals as
        // analytic gradients with scrambled signs.
        let mut points = Vec::new();
        let mut normals = Vec::new();
        let w = 20;
        for i in 0..w {
            for j in 0..w {
                let x = -1.0 + 2.0 * i as f64 / (w - 1) as f64;
                let y = -1.0 + 2.0 * j as f64 / (w - 1) as f64;
                points.push([x, y, 0.3 * (3.0 * x).sin() * (3.0 * y).cos()]);
                let up = normalized3([
                    -0.9 * (3.0 * x).cos() * (3.0 * y).cos(),
                    0.9 * (3.0 * x).sin() * (3.0 * y).sin(),
                    1.0,
                ]);
                normals.push(if (i * w + j) % 2 == 0 {
                    up
                } else {
                    [-up[0], -up[1], -up[2]]
                });
            }
        }
        let mut pc = PointCloud {
            points,
            normals,
            colors: Vec::new(),
        };
        pc.orient_normals_consistent_tangent_plane(10, 0.0, 1.0)
            .unwrap();
        // Upstream seeds at the minimum-z vertex towards (0,0,-1), so an
        // open height field comes out consistently downward.
        assert!(pc.normals.iter().all(|nrm| nrm[2] < 0.0));
    }

    #[test]
    fn disconnected_components_are_bridged_and_oriented() {
        // Two spheres far apart: the kNN graph is disconnected, the bridge
        // must carry the orientation across.
        let c0 = [0.0, 0.0, 0.0];
        let c1 = [100.0, 0.0, 0.0];
        let mut points = fibonacci_sphere(200, c0, 1.0);
        points.extend(fibonacci_sphere(200, c1, 1.0));
        let normals = scrambled_radial_normals(&points, c0, 2)
            .into_iter()
            .zip(points.iter())
            .enumerate()
            .map(|(i, (_, p))| {
                let c = if i < 200 { c0 } else { c1 };
                let mut nrm = normalized3(sub3(*p, c));
                if i % 2 == 0 {
                    nrm = [-nrm[0], -nrm[1], -nrm[2]];
                }
                nrm
            })
            .collect();
        let mut pc = PointCloud {
            points,
            normals,
            colors: Vec::new(),
        };
        pc.orient_normals_consistent_tangent_plane(10, 0.0, 1.0)
            .unwrap();
        // Every point must have been visited: each sphere internally
        // consistent (all outward or all inward w.r.t. its own center).
        for (offset, c) in [(0usize, c0), (200usize, c1)] {
            let signs: Vec<bool> = (offset..offset + 200)
                .map(|i| dot3(pc.normals[i], sub3(pc.points[i], c)) > 0.0)
                .collect();
            assert!(
                signs.iter().all(|&s| s == signs[0]),
                "component at {c:?} not internally consistent"
            );
        }
    }

    #[test]
    fn errors_without_normals() {
        let mut pc = PointCloud {
            points: fibonacci_sphere(10, [0.0; 3], 1.0),
            normals: Vec::new(),
            colors: Vec::new(),
        };
        assert!(pc
            .orient_normals_consistent_tangent_plane(10, 0.0, 1.0)
            .is_err());
    }
}
