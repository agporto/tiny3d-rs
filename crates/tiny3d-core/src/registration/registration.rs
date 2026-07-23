//! Registration.cpp port: evaluate, ICP, RANSAC, information matrix.
//! Serial execution — matches the C++ built with OMP_NUM_THREADS=1.

use crate::geometry::PointCloud;
use crate::kdtree::KdTreeFlann;
use crate::linalg::*;
use crate::random::UniformIntGenerator;

use super::checker::CorrespondenceChecker;
use super::estimation::{
    solve_jacobian_system, validate_correspondences, Correspondence, TransformationEstimation,
};
use super::feature::{correspondences_from_features, Feature};

#[derive(Clone, Debug)]
pub struct IcpConvergenceCriteria {
    pub relative_fitness: f64,
    pub relative_rmse: f64,
    pub max_iteration: i32,
}

impl Default for IcpConvergenceCriteria {
    fn default() -> Self {
        IcpConvergenceCriteria {
            relative_fitness: 1e-6,
            relative_rmse: 1e-6,
            max_iteration: 30,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RansacConvergenceCriteria {
    pub max_iteration: i32,
    pub confidence: f64,
}

impl Default for RansacConvergenceCriteria {
    fn default() -> Self {
        RansacConvergenceCriteria {
            max_iteration: 100000,
            confidence: 0.999,
        }
    }
}

#[derive(Clone)]
pub struct RegistrationResult {
    pub transformation: M4,
    pub correspondence_set: Vec<Correspondence>,
    pub inlier_rmse: f64,
    pub fitness: f64,
}

impl Default for RegistrationResult {
    fn default() -> Self {
        Self::new(m4_identity())
    }
}

impl RegistrationResult {
    pub fn new(transformation: M4) -> Self {
        RegistrationResult {
            transformation,
            correspondence_set: Vec::new(),
            inlier_rmse: 0.0,
            fitness: 0.0,
        }
    }

    pub fn is_better_ransac_than(&self, other: &RegistrationResult) -> bool {
        self.fitness > other.fitness
            || (self.fitness == other.fitness && self.inlier_rmse < other.inlier_rmse)
    }
}

fn get_registration_result_and_correspondences(
    source: &PointCloud,
    target_kdtree: &KdTreeFlann,
    max_correspondence_distance: f64,
    transformation: &M4,
    with_correspondence_set: bool,
) -> RegistrationResult {
    let mut result = RegistrationResult::new(*transformation);
    if max_correspondence_distance <= 0.0 {
        return result;
    }
    let r = m4_block3x3(transformation);
    let t = m4_translation(transformation);

    // Parallel per-point nearest-neighbor queries (each independent), then a
    // SERIAL in-index-order fold for error2/count/correspondences — identical
    // to the serial loop bit-for-bit, for any thread count.
    use rayon::prelude::*;
    let hits: Vec<Option<(i64, f64)>> = source
        .points
        .par_iter()
        .map_init(
            || (Vec::new(), Vec::new()),
            |(indices, dists), p| {
                let point = add3(m3v3(&r, *p), t);
                if target_kdtree.search_hybrid(
                    &point,
                    max_correspondence_distance,
                    1,
                    indices,
                    dists,
                ) > 0
                {
                    Some((indices[0], dists[0]))
                } else {
                    None
                }
            },
        )
        .collect();
    let mut error2 = 0.0f64;
    let mut correspondence_count = 0usize;
    for (i, h) in hits.iter().enumerate() {
        if let Some((idx0, d0)) = h {
            error2 += d0;
            correspondence_count += 1;
            if with_correspondence_set {
                result.correspondence_set.push([i as i32, *idx0 as i32]);
            }
        }
    }

    if correspondence_count == 0 {
        result.fitness = 0.0;
        result.inlier_rmse = 0.0;
    } else {
        result.fitness = if source.points.is_empty() {
            0.0
        } else {
            correspondence_count as f64 / source.points.len() as f64
        };
        result.inlier_rmse = (error2 / correspondence_count as f64).sqrt();
    }
    result
}

fn compute_transformation_point_to_point_transformed_source(
    source: &PointCloud,
    target: &PointCloud,
    corres: &[Correspondence],
    transformation: &M4,
    with_scaling: bool,
) -> M4 {
    if corres.is_empty() {
        return m4_identity();
    }
    let linear = m4_block3x3(transformation);
    let t = m4_translation(transformation);
    let inv_n = 1.0 / corres.len() as f64;

    let mut mean_s = ZERO3;
    let mut mean_t = ZERO3;
    for c in corres {
        mean_s = add3(mean_s, add3(m3v3(&linear, source.points[c[0] as usize]), t));
        mean_t = add3(mean_t, target.points[c[1] as usize]);
    }
    // mean *= inv_n
    mean_s = scale3(mean_s, inv_n);
    mean_t = scale3(mean_t, inv_n);

    let mut cov = [[0.0f64; 3]; 3];
    let mut var_s = 0.0f64;
    for c in corres {
        let ds = sub3(add3(m3v3(&linear, source.points[c[0] as usize]), t), mean_s);
        let dt = sub3(target.points[c[1] as usize], mean_t);
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += dt[i] * ds[j];
            }
        }
        var_s += squared_norm3(ds);
    }
    for row in cov.iter_mut() {
        for x in row.iter_mut() {
            *x *= inv_n;
        }
    }
    var_s *= inv_n;

    super::estimation::finish_umeyama(&cov, var_s, mean_s, mean_t, with_scaling)
}

fn compute_transformation_point_to_plane_transformed_source(
    source: &PointCloud,
    target: &PointCloud,
    corres: &[Correspondence],
    transformation: &M4,
) -> M4 {
    if corres.is_empty() || !target.has_normals() {
        return m4_identity();
    }
    let linear = m4_block3x3(transformation);
    let t = m4_translation(transformation);
    // Parallel per-correspondence Jacobian evaluation (elementwise, exact),
    // then serial in-order accumulation (order-sensitive sums).
    use rayon::prelude::*;
    let jr: Vec<(V6, f64)> = corres
        .par_iter()
        .map(|c| {
            let vs = add3(m3v3(&linear, source.points[c[0] as usize]), t);
            let vt = target.points[c[1] as usize];
            let nt = target.normals[c[1] as usize];
            let r = dot3(sub3(vs, vt), nt);
            let cr = cross3(vs, nt);
            ([cr[0], cr[1], cr[2], nt[0], nt[1], nt[2]], r)
        })
        .collect();
    let mut jtj = m6_zero();
    let mut jtr = v6_zero();
    for (j_r, r) in jr.iter() {
        for a in 0..6 {
            for b in 0..6 {
                jtj[a][b] += j_r[a] * j_r[b];
            }
        }
        for a in 0..6 {
            jtr[a] += j_r[a] * r;
        }
    }
    let (ok, extrinsic) = solve_jacobian_system(&jtj, &jtr);
    if ok {
        extrinsic
    } else {
        m4_identity()
    }
}

fn compute_transformation_transformed_source(
    source: &PointCloud,
    target: &PointCloud,
    corres: &[Correspondence],
    transformation: &M4,
    estimation: &TransformationEstimation,
) -> M4 {
    match estimation {
        TransformationEstimation::PointToPoint { with_scaling } => {
            compute_transformation_point_to_point_transformed_source(
                source,
                target,
                corres,
                transformation,
                *with_scaling,
            )
        }
        TransformationEstimation::PointToPlane => {
            compute_transformation_point_to_plane_transformed_source(
                source,
                target,
                corres,
                transformation,
            )
        }
    }
}

fn get_registration_result_sampled(
    source: &PointCloud,
    target_kdtree: &KdTreeFlann,
    max_correspondence_distance: f64,
    transformation: &M4,
    mut max_samples: i32,
) -> RegistrationResult {
    let mut result = RegistrationResult::new(*transformation);
    if max_correspondence_distance <= 0.0 || source.points.is_empty() {
        return result;
    }
    if max_samples <= 0 {
        max_samples = 1;
    }
    let n_source = source.points.len() as i32;
    let stride = std::cmp::max(1, n_source / max_samples) as usize;
    let r = m4_block3x3(transformation);
    let t = m4_translation(transformation);

    // Parallel queries over the strided sample; serial in-order fold.
    use rayon::prelude::*;
    let sample_idx: Vec<usize> = (0..n_source as usize).step_by(stride).collect();
    let hits: Vec<Option<f64>> = sample_idx
        .par_iter()
        .map_init(
            || (Vec::new(), Vec::new()),
            |(indices, dists), &i| {
                let point = add3(m3v3(&r, source.points[i]), t);
                if target_kdtree.search_hybrid(
                    &point,
                    max_correspondence_distance,
                    1,
                    indices,
                    dists,
                ) > 0
                {
                    Some(dists[0])
                } else {
                    None
                }
            },
        )
        .collect();
    let mut error2 = 0.0f64;
    let mut inlier_count = 0i32;
    let mut sampled_count = 0i32;
    for h in hits.iter() {
        sampled_count += 1;
        if let Some(d) = h {
            error2 += d;
            inlier_count += 1;
        }
    }

    if inlier_count > 0 {
        result.fitness = if sampled_count > 0 {
            inlier_count as f64 / sampled_count as f64
        } else {
            0.0
        };
        result.inlier_rmse = (error2 / inlier_count as f64).sqrt();
    }
    result
}

fn evaluate_inlier_correspondence_ratio(
    source: &PointCloud,
    target: &PointCloud,
    corres: &[Correspondence],
    max_correspondence_distance: f64,
    transformation: &M4,
) -> f64 {
    if corres.is_empty() {
        return 0.0;
    }
    let mut inlier = 0i32;
    let mut sampled = 0i32;
    let max_samples = 5000i32;
    let stride = std::cmp::max(1, corres.len() as i32 / max_samples) as usize;
    let r = m4_block3x3(transformation);
    let t = m4_translation(transformation);
    let max_dis2 = max_correspondence_distance * max_correspondence_distance;
    let mut i = 0usize;
    while i < corres.len() {
        let c = corres[i];
        sampled += 1;
        let transformed = add3(m3v3(&r, source.points[c[0] as usize]), t);
        let dis2 = squared_norm3(sub3(transformed, target.points[c[1] as usize]));
        if dis2 < max_dis2 {
            inlier += 1;
        }
        i += stride;
    }
    if sampled > 0 {
        inlier as f64 / sampled as f64
    } else {
        0.0
    }
}

pub fn evaluate_registration(
    source: &PointCloud,
    target: &PointCloud,
    max_correspondence_distance: f64,
    transformation: &M4,
) -> RegistrationResult {
    let mut kdtree = KdTreeFlann::new();
    kdtree.set_points(&target.points);
    get_registration_result_and_correspondences(
        source,
        &kdtree,
        max_correspondence_distance,
        transformation,
        true,
    )
}

pub fn registration_icp(
    source: &PointCloud,
    target: &PointCloud,
    max_correspondence_distance: f64,
    init: &M4,
    estimation: &TransformationEstimation,
    criteria: &IcpConvergenceCriteria,
) -> Result<RegistrationResult, String> {
    if max_correspondence_distance <= 0.0 {
        return Err("Invalid max_correspondence_distance.".to_string());
    }
    if source.is_empty() || target.is_empty() {
        // LogWarning: skipped on empty point cloud
        return Ok(RegistrationResult::new(*init));
    }
    if matches!(estimation, TransformationEstimation::PointToPlane) && !target.has_normals() {
        return Err("PointToPlaneICP requires target pointcloud to have normals.".to_string());
    }

    let mut transformation = *init;
    let mut kdtree = KdTreeFlann::new();
    kdtree.set_points(&target.points);
    let mut result = get_registration_result_and_correspondences(
        source,
        &kdtree,
        max_correspondence_distance,
        &transformation,
        true,
    );
    for _i in 0..criteria.max_iteration {
        let update = compute_transformation_transformed_source(
            source,
            target,
            &result.correspondence_set,
            &transformation,
            estimation,
        );
        if !m4_all_finite(&update) {
            break;
        }
        transformation = m4m4(&update, &transformation);
        let backup = result.clone();
        result = get_registration_result_and_correspondences(
            source,
            &kdtree,
            max_correspondence_distance,
            &transformation,
            true,
        );
        if (backup.fitness - result.fitness).abs() < criteria.relative_fitness
            && (backup.inlier_rmse - result.inlier_rmse).abs() < criteria.relative_rmse
        {
            break;
        }
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
pub fn registration_ransac_based_on_correspondence(
    source: &PointCloud,
    target: &PointCloud,
    corres: &[Correspondence],
    max_correspondence_distance: f64,
    estimation: &TransformationEstimation,
    ransac_n: i32,
    checkers: &[CorrespondenceChecker],
    criteria: &RansacConvergenceCriteria,
) -> Result<RegistrationResult, String> {
    validate_correspondences(source, target, corres)?;
    if ransac_n < 3 || (corres.len() as i32) < ransac_n || max_correspondence_distance <= 0.0 {
        return Ok(RegistrationResult::default());
    }
    if source.is_empty() || target.is_empty() {
        return Ok(RegistrationResult::default());
    }

    let mut best_result = RegistrationResult::default();
    let mut kdtree = KdTreeFlann::new();
    kdtree.set_points(&target.points);

    // Deterministic batch-parallel RANSAC, bit-exact with the serial
    // (OMP_NUM_THREADS=1) C++ loop:
    //   1. pre-draw each batch's samples SERIALLY (the sampling loop is the
    //      only RNG consumer; a per-iteration engine snapshot allows exact
    //      rewind when the early-exit lands mid-batch, because iterations at
    //      or past est_k consume no RNG in the serial loop);
    //   2. evaluate iterations of the batch in PARALLEL (transform estimate,
    //      checkers, strided sampled validation, full validation — all pure).
    //      The best-so-far prune uses the batch-start best, which is <= the
    //      exact running best, so it only skips work the serial fold would
    //      skip too (outcomes are decided in the fold, never here);
    //   3. fold results SERIALLY in iteration order, replicating best-result
    //      selection and the est_k early-exit update exactly.
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    let mut est_k_global = criteria.max_iteration;
    let mut est_k_local = criteria.max_iteration;
    let mut best_result_local = RegistrationResult::default();
    let rand_gen = UniformIntGenerator::new(0, corres.len() as i64 - 1);
    let max_attempts = std::cmp::max(32 * ransac_n, 128);
    let n_threads = rayon::current_num_threads().max(1) as i32;

    // Single-thread: run the plain serial loop (identical results; the
    // batch machinery below only pays off with parallelism).
    if n_threads == 1 {
        let mut ransac_corres: Vec<Correspondence> = vec![[0, 0]; ransac_n as usize];
        for itr in 0..criteria.max_iteration {
            if itr < est_k_global {
                let mut sampled = 0i32;
                let mut attempts = 0i32;
                while sampled < ransac_n {
                    let candidate = corres[rand_gen.next() as usize];
                    let mut duplicated = false;
                    for j in 0..sampled {
                        if ransac_corres[j as usize] == candidate {
                            duplicated = true;
                            break;
                        }
                    }
                    if !duplicated {
                        ransac_corres[sampled as usize] = candidate;
                        sampled += 1;
                    }
                    attempts += 1;
                    if attempts > max_attempts {
                        break;
                    }
                }
                if sampled < ransac_n {
                    continue;
                }
                let transformation =
                    estimation.compute_transformation_unchecked(source, target, &ransac_corres);
                if !m4_all_finite(&transformation) {
                    continue;
                }
                let mut check = true;
                for checker in checkers {
                    if !checker.check_unchecked(source, target, &ransac_corres, &transformation) {
                        check = false;
                        break;
                    }
                }
                if !check {
                    continue;
                }
                let sampled_result = get_registration_result_sampled(
                    source,
                    &kdtree,
                    max_correspondence_distance,
                    &transformation,
                    2048,
                );
                if best_result_local.fitness > 0.0
                    && sampled_result.fitness + 0.02 < best_result_local.fitness
                {
                    continue;
                }
                if sampled_result.fitness <= 0.0 {
                    continue;
                }
                let result = get_registration_result_and_correspondences(
                    source,
                    &kdtree,
                    max_correspondence_distance,
                    &transformation,
                    false,
                );
                let result = if result.is_better_ransac_than(&best_result_local) {
                    get_registration_result_and_correspondences(
                        source,
                        &kdtree,
                        max_correspondence_distance,
                        &transformation,
                        true,
                    )
                } else {
                    continue;
                };
                if result.is_better_ransac_than(&best_result_local) {
                    best_result_local = result;
                    let corres_inlier_ratio = evaluate_inlier_correspondence_ratio(
                        source,
                        target,
                        corres,
                        max_correspondence_distance,
                        &transformation,
                    );
                    let mut est_k_local_d = est_k_local as f64;
                    let inlier_prob = corres_inlier_ratio.powi(ransac_n);
                    if inlier_prob >= 1.0 {
                        est_k_local_d = 1.0;
                    } else if inlier_prob > 0.0 {
                        est_k_local_d = (1.0 - criteria.confidence).ln() / (1.0 - inlier_prob).ln();
                        if est_k_local_d < 0.0 || !est_k_local_d.is_finite() {
                            est_k_local_d = est_k_local as f64;
                        }
                    }
                    est_k_local = if est_k_local_d < est_k_global as f64 {
                        est_k_local_d.ceil() as i32
                    } else {
                        est_k_local
                    };
                }
                if est_k_local < est_k_global {
                    est_k_global = est_k_local;
                }
            }
        }
        if best_result_local.is_better_ransac_than(&best_result) {
            best_result = best_result_local;
        }
        return Ok(best_result);
    }
    let batch_max = std::cmp::max(64, 16 * n_threads);
    let mut batch_size = std::cmp::max(4, 2 * n_threads); // ramp up

    struct Eval {
        transformation: M4,
        sampled_fitness: f64,
        // None => full validation pruned against the fitness watermark; the
        // fold recomputes it serially in the (rare) case the exact running
        // best would not have pruned it.
        full: Option<RegistrationResult>,
    }

    // Monotonic max of the fitness of ANY completed candidate this batch;
    // used only to SKIP speculative full validations (never to decide
    // outcomes — the serial fold does that with the exact running best).
    let watermark = AtomicU64::new(0f64.to_bits());
    let raise = |w: &AtomicU64, f: f64| {
        let mut cur = w.load(Ordering::Relaxed);
        while f > f64::from_bits(cur) {
            match w.compare_exchange_weak(cur, f.to_bits(), Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(c) => cur = c,
            }
        }
    };

    let mut itr = 0i32;
    'outer: while itr < criteria.max_iteration && itr < est_k_global {
        let batch_end = std::cmp::min(itr + batch_size, criteria.max_iteration);
        batch_size = std::cmp::min(batch_size * 2, batch_max);

        // --- phase 1: serial pre-draw (exact serial RNG consumption) ---
        // One engine snapshot per batch; a mid-batch early exit rewinds by
        // restoring it and replaying the (deterministic) draws.
        let batch_snap = crate::random::snapshot_engine();
        let n_batch = (batch_end - itr) as usize;
        let mut draws: Vec<Option<Vec<Correspondence>>> = Vec::with_capacity(n_batch);
        let mut ransac_corres: Vec<Correspondence> = vec![[0, 0]; ransac_n as usize];
        let mut draw_batch = |draws: &mut Vec<Option<Vec<Correspondence>>>, upto: usize| {
            for _ in draws.len()..upto {
                let mut sampled = 0i32;
                let mut attempts = 0i32;
                while sampled < ransac_n {
                    let candidate = corres[rand_gen.next() as usize];
                    let mut duplicated = false;
                    for j in 0..sampled {
                        if ransac_corres[j as usize] == candidate {
                            duplicated = true;
                            break;
                        }
                    }
                    if !duplicated {
                        ransac_corres[sampled as usize] = candidate;
                        sampled += 1;
                    }
                    attempts += 1;
                    if attempts > max_attempts {
                        break;
                    }
                }
                draws.push(if sampled < ransac_n {
                    None
                } else {
                    Some(ransac_corres.clone())
                });
            }
        };
        draw_batch(&mut draws, n_batch);

        // --- phase 2: parallel evaluation (pure per iteration) ---
        raise(&watermark, best_result_local.fitness);
        let evals: Vec<Option<Eval>> = draws
            .par_iter()
            .map(|samples| {
                let samples = samples.as_ref()?;
                let transformation =
                    estimation.compute_transformation_unchecked(source, target, samples);
                if !m4_all_finite(&transformation) {
                    return None;
                }
                for checker in checkers {
                    if !checker.check_unchecked(source, target, samples, &transformation) {
                        return None;
                    }
                }
                let sampled_result = get_registration_result_sampled(
                    source,
                    &kdtree,
                    max_correspondence_distance,
                    &transformation,
                    2048,
                );
                if sampled_result.fitness <= 0.0 {
                    return None;
                }
                let wm = f64::from_bits(watermark.load(Ordering::Relaxed));
                if wm > 0.0 && sampled_result.fitness + 0.02 < wm {
                    return Some(Eval {
                        transformation,
                        sampled_fitness: sampled_result.fitness,
                        full: None,
                    });
                }
                let full = get_registration_result_and_correspondences(
                    source,
                    &kdtree,
                    max_correspondence_distance,
                    &transformation,
                    false,
                );
                raise(&watermark, full.fitness);
                Some(Eval {
                    transformation,
                    sampled_fitness: sampled_result.fitness,
                    full: Some(full),
                })
            })
            .collect();

        // --- phase 3: serial fold in iteration order (exact semantics) ---
        for (k, ev) in evals.into_iter().enumerate() {
            let i_global = itr + k as i32;
            if i_global >= est_k_global {
                // Serial loop would skip this iteration WITHOUT drawing from
                // the RNG — rewind: restore the batch-start engine state and
                // replay the draws of the iterations that did execute.
                crate::random::restore_engine(batch_snap.clone());
                let mut replay: Vec<Option<Vec<Correspondence>>> = Vec::with_capacity(k);
                draw_batch(&mut replay, k);
                break 'outer;
            }
            let Some(ev) = ev else { continue };
            if best_result_local.fitness > 0.0
                && ev.sampled_fitness + 0.02 < best_result_local.fitness
            {
                continue;
            }
            let result = match ev.full {
                Some(r) => r,
                // Watermark-pruned, but the exact running best would NOT have
                // pruned it (watermark may include later iterations) —
                // recompute the full validation now, serially.
                None => get_registration_result_and_correspondences(
                    source,
                    &kdtree,
                    max_correspondence_distance,
                    &ev.transformation,
                    false,
                ),
            };
            let result = if result.is_better_ransac_than(&best_result_local) {
                get_registration_result_and_correspondences(
                    source,
                    &kdtree,
                    max_correspondence_distance,
                    &ev.transformation,
                    true,
                )
            } else {
                continue;
            };

            if result.is_better_ransac_than(&best_result_local) {
                best_result_local = result;
                let corres_inlier_ratio = evaluate_inlier_correspondence_ratio(
                    source,
                    target,
                    corres,
                    max_correspondence_distance,
                    &ev.transformation,
                );
                let mut est_k_local_d = est_k_local as f64;
                let inlier_prob = corres_inlier_ratio.powi(ransac_n);
                if inlier_prob >= 1.0 {
                    est_k_local_d = 1.0;
                } else if inlier_prob > 0.0 {
                    est_k_local_d = (1.0 - criteria.confidence).ln() / (1.0 - inlier_prob).ln();
                    if est_k_local_d < 0.0 || !est_k_local_d.is_finite() {
                        est_k_local_d = est_k_local as f64;
                    }
                }
                est_k_local = if est_k_local_d < est_k_global as f64 {
                    est_k_local_d.ceil() as i32
                } else {
                    est_k_local
                };
            }
            if est_k_local < est_k_global {
                est_k_global = est_k_local;
            }
        }

        itr = batch_end;
    }

    if best_result_local.is_better_ransac_than(&best_result) {
        best_result = best_result_local;
    }
    Ok(best_result)
}

#[allow(clippy::too_many_arguments)]
pub fn registration_ransac_based_on_feature_matching(
    source: &PointCloud,
    target: &PointCloud,
    source_features: &Feature,
    target_features: &Feature,
    mutual_filter: bool,
    max_correspondence_distance: f64,
    estimation: &TransformationEstimation,
    ransac_n: i32,
    checkers: &[CorrespondenceChecker],
    criteria: &RansacConvergenceCriteria,
) -> Result<RegistrationResult, String> {
    if ransac_n < 3 || max_correspondence_distance <= 0.0 {
        return Ok(RegistrationResult::default());
    }
    let corres =
        correspondences_from_features(source_features, target_features, mutual_filter, 0.1);
    registration_ransac_based_on_correspondence(
        source,
        target,
        &corres,
        max_correspondence_distance,
        estimation,
        ransac_n,
        checkers,
        criteria,
    )
}

pub fn get_information_matrix_from_point_clouds(
    source: &PointCloud,
    target: &PointCloud,
    max_correspondence_distance: f64,
    transformation: &M4,
) -> M6 {
    let mut kdtree = KdTreeFlann::new();
    kdtree.set_points(&target.points);
    let result = get_registration_result_and_correspondences(
        source,
        &kdtree,
        max_correspondence_distance,
        transformation,
        true,
    );

    let mut gtg = m6_zero();
    for c in result.correspondence_set.iter() {
        let ti = c[1] as usize;
        let x = target.points[ti][0];
        let y = target.points[ti][1];
        let z = target.points[ti][2];
        let mut g = [0.0f64; 6];
        g[1] = z;
        g[2] = -y;
        g[3] = 1.0;
        add_outer6(&mut gtg, &g);
        let mut g = [0.0f64; 6];
        g[0] = -z;
        g[2] = x;
        g[4] = 1.0;
        add_outer6(&mut gtg, &g);
        let mut g = [0.0f64; 6];
        g[0] = y;
        g[1] = -x;
        g[5] = 1.0;
        add_outer6(&mut gtg, &g);
    }
    gtg
}

#[inline]
fn add_outer6(m: &mut M6, g: &[f64; 6]) {
    for i in 0..6 {
        for j in 0..6 {
            m[i][j] += g[i] * g[j];
        }
    }
}
