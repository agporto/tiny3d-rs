//! tiny3d.cpu.pybind.pipelines.registration

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use tiny3d_core::registration as cr;

use crate::fmtutil::{c_format_e, c_format_f};
use crate::geometry::{m4_from_any, m4_to_numpy, KDTreeSearchParam, PointCloud};
use crate::vectors::Vector2iVector;

// ---------------------------------------------------------------- Feature

#[pyclass(name = "Feature", module = "tiny3d.cpu.pybind.pipelines.registration")]
#[derive(Clone, Default)]
pub struct Feature {
    pub inner: cr::Feature,
}

fn checked_feature_size(dim: usize, num: usize) -> PyResult<usize> {
    let elements = dim
        .checked_mul(num)
        .ok_or_else(|| PyValueError::new_err("feature dimensions are too large"))?;
    let bytes = elements
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| PyValueError::new_err("feature dimensions are too large"))?;
    if bytes > isize::MAX as usize {
        return Err(PyValueError::new_err("feature dimensions are too large"));
    }
    Ok(elements)
}

#[pymethods]
impl Feature {
    #[new]
    #[pyo3(signature = (other = None))]
    fn new(other: Option<PyRef<'_, Feature>>) -> Self {
        match other {
            Some(o) => o.clone(),
            None => Feature::default(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Feature class with dimension = {} and num = {}\nAccess its data via data member.",
            self.inner.dim, self.inner.num
        )
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }
    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    fn resize(&mut self, dim: i32, n: i32) -> PyResult<()> {
        let dim =
            usize::try_from(dim).map_err(|_| PyValueError::new_err("dim must be non-negative"))?;
        let n = usize::try_from(n).map_err(|_| PyValueError::new_err("n must be non-negative"))?;
        checked_feature_size(dim, n)?;
        self.inner.resize(dim, n);
        Ok(())
    }
    fn dimension(&self) -> i32 {
        self.inner.dim as i32
    }
    fn num(&self) -> i32 {
        self.inner.num as i32
    }

    #[getter(data)]
    fn data_getter(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // column-major dim x num -> numpy (dim, num)
        let dim = self.inner.dim;
        let num = self.inner.num;
        let mut row_major = Vec::with_capacity(dim * num);
        for r in 0..dim {
            for c in 0..num {
                row_major.push(self.inner.get(r, c));
            }
        }
        Ok(PyArray1::from_vec(py, row_major)
            .reshape([dim, num])?
            .into_any()
            .unbind())
    }
    #[setter(data)]
    fn data_setter(&mut self, value: PyReadonlyArray2<f64>) -> PyResult<()> {
        let view = value.as_array();
        let dim = view.shape()[0];
        let num = view.shape()[1];
        checked_feature_size(dim, num)?;
        self.inner.resize(dim, num);
        for r in 0..dim {
            for c in 0..num {
                self.inner.set(r, c, view[[r, c]]);
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------- criteria

#[pyclass(
    name = "ICPConvergenceCriteria",
    module = "tiny3d.cpu.pybind.pipelines.registration"
)]
#[derive(Clone)]
pub struct ICPConvergenceCriteria {
    pub inner: cr::IcpConvergenceCriteria,
}

#[pymethods]
impl ICPConvergenceCriteria {
    #[new]
    #[pyo3(signature = (relative_fitness = 1e-6, relative_rmse = 1e-6, max_iteration = 30))]
    fn new(relative_fitness: f64, relative_rmse: f64, max_iteration: i32) -> Self {
        ICPConvergenceCriteria {
            inner: cr::IcpConvergenceCriteria {
                relative_fitness,
                relative_rmse,
                max_iteration,
            },
        }
    }
    fn __repr__(&self) -> String {
        format!(
            "ICPConvergenceCriteria(relative_fitness={}, relative_rmse={}, max_iteration={})",
            c_format_e(self.inner.relative_fitness, 6),
            c_format_e(self.inner.relative_rmse, 6),
            self.inner.max_iteration
        )
    }
    fn __copy__(&self) -> Self {
        self.clone()
    }
    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }
    #[getter(relative_fitness)]
    fn rf(&self) -> f64 {
        self.inner.relative_fitness
    }
    #[setter(relative_fitness)]
    fn set_rf(&mut self, v: f64) {
        self.inner.relative_fitness = v;
    }
    #[getter(relative_rmse)]
    fn rr(&self) -> f64 {
        self.inner.relative_rmse
    }
    #[setter(relative_rmse)]
    fn set_rr(&mut self, v: f64) {
        self.inner.relative_rmse = v;
    }
    #[getter(max_iteration)]
    fn mi(&self) -> i32 {
        self.inner.max_iteration
    }
    #[setter(max_iteration)]
    fn set_mi(&mut self, v: i32) {
        self.inner.max_iteration = v;
    }
}

#[pyclass(
    name = "RANSACConvergenceCriteria",
    module = "tiny3d.cpu.pybind.pipelines.registration"
)]
#[derive(Clone)]
pub struct RANSACConvergenceCriteria {
    pub inner: cr::RansacConvergenceCriteria,
}

#[pymethods]
impl RANSACConvergenceCriteria {
    #[new]
    #[pyo3(signature = (max_iteration = 100000, confidence = 0.999))]
    fn new(max_iteration: i32, confidence: f64) -> Self {
        RANSACConvergenceCriteria {
            inner: cr::RansacConvergenceCriteria {
                max_iteration,
                confidence,
            },
        }
    }
    fn __repr__(&self) -> String {
        format!(
            "RANSACConvergenceCriteria(max_iteration={}, confidence={})",
            self.inner.max_iteration,
            c_format_e(self.inner.confidence, 6)
        )
    }
    fn __copy__(&self) -> Self {
        self.clone()
    }
    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }
    #[getter(max_iteration)]
    fn mi(&self) -> i32 {
        self.inner.max_iteration
    }
    #[setter(max_iteration)]
    fn set_mi(&mut self, v: i32) {
        self.inner.max_iteration = v;
    }
    #[getter(confidence)]
    fn conf(&self) -> f64 {
        self.inner.confidence
    }
    #[setter(confidence)]
    fn set_conf(&mut self, v: f64) {
        self.inner.confidence = v;
    }
}

// ---------------------------------------------------------------- estimation

#[pyclass(
    subclass,
    name = "TransformationEstimation",
    module = "tiny3d.cpu.pybind.pipelines.registration"
)]
#[derive(Clone)]
pub struct TransformationEstimation {
    pub inner: cr::TransformationEstimation,
}

#[pymethods]
impl TransformationEstimation {
    fn compute_rmse(
        &self,
        source: PyRef<'_, PointCloud>,
        target: PyRef<'_, PointCloud>,
        corres: PyRef<'_, Vector2iVector>,
    ) -> PyResult<f64> {
        self.inner
            .compute_rmse(&source.inner, &target.inner, &corres.data)
            .map_err(PyValueError::new_err)
    }

    fn compute_transformation(
        &self,
        py: Python<'_>,
        source: PyRef<'_, PointCloud>,
        target: PyRef<'_, PointCloud>,
        corres: PyRef<'_, Vector2iVector>,
    ) -> PyResult<Py<PyAny>> {
        let t = self
            .inner
            .compute_transformation(&source.inner, &target.inner, &corres.data)
            .map_err(PyValueError::new_err)?;
        m4_to_numpy(py, &t)
    }
}

#[pyclass(extends = TransformationEstimation, name = "TransformationEstimationPointToPoint", module = "tiny3d.cpu.pybind.pipelines.registration")]
pub struct TransformationEstimationPointToPoint;

#[pymethods]
impl TransformationEstimationPointToPoint {
    #[new]
    #[pyo3(signature = (with_scaling = false))]
    fn new(with_scaling: bool) -> (Self, TransformationEstimation) {
        (
            TransformationEstimationPointToPoint,
            TransformationEstimation {
                inner: cr::TransformationEstimation::PointToPoint { with_scaling },
            },
        )
    }
    fn __repr__(slf: PyRef<'_, Self>) -> String {
        if let cr::TransformationEstimation::PointToPoint { with_scaling } = slf.as_super().inner {
            format!(
                "TransformationEstimationPointToPoint(with_scaling={})",
                if with_scaling { "True" } else { "False" }
            )
        } else {
            unreachable!()
        }
    }
    fn __copy__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
    ) -> PyResult<Py<TransformationEstimationPointToPoint>> {
        let base: &TransformationEstimation = slf.as_super();
        Py::new(
            py,
            PyClassInitializer::from(base.clone())
                .add_subclass(TransformationEstimationPointToPoint),
        )
    }
    fn __deepcopy__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        _memo: &Bound<'_, PyAny>,
    ) -> PyResult<Py<TransformationEstimationPointToPoint>> {
        let base: &TransformationEstimation = slf.as_super();
        Py::new(
            py,
            PyClassInitializer::from(base.clone())
                .add_subclass(TransformationEstimationPointToPoint),
        )
    }
    #[getter(with_scaling)]
    fn ws(slf: PyRef<'_, Self>) -> bool {
        matches!(
            slf.as_super().inner,
            cr::TransformationEstimation::PointToPoint { with_scaling: true }
        )
    }
    #[setter(with_scaling)]
    fn set_ws(mut slf: PyRefMut<'_, Self>, v: bool) {
        slf.as_super().inner = cr::TransformationEstimation::PointToPoint { with_scaling: v };
    }
}

#[pyclass(extends = TransformationEstimation, name = "TransformationEstimationPointToPlane", module = "tiny3d.cpu.pybind.pipelines.registration")]
pub struct TransformationEstimationPointToPlane;

#[pymethods]
impl TransformationEstimationPointToPlane {
    #[new]
    fn new() -> (Self, TransformationEstimation) {
        (
            TransformationEstimationPointToPlane,
            TransformationEstimation {
                inner: cr::TransformationEstimation::PointToPlane,
            },
        )
    }
    fn __repr__(&self) -> &'static str {
        "TransformationEstimationPointToPlane"
    }
    fn __copy__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
    ) -> PyResult<Py<TransformationEstimationPointToPlane>> {
        let base: &TransformationEstimation = slf.as_super();
        Py::new(
            py,
            PyClassInitializer::from(base.clone())
                .add_subclass(TransformationEstimationPointToPlane),
        )
    }
    fn __deepcopy__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        _memo: &Bound<'_, PyAny>,
    ) -> PyResult<Py<TransformationEstimationPointToPlane>> {
        let base: &TransformationEstimation = slf.as_super();
        Py::new(
            py,
            PyClassInitializer::from(base.clone())
                .add_subclass(TransformationEstimationPointToPlane),
        )
    }
}

// ---------------------------------------------------------------- checkers

#[pyclass(
    subclass,
    name = "CorrespondenceChecker",
    module = "tiny3d.cpu.pybind.pipelines.registration"
)]
#[derive(Clone)]
pub struct CorrespondenceChecker {
    pub inner: cr::CorrespondenceChecker,
}

#[pymethods]
impl CorrespondenceChecker {
    #[allow(non_snake_case)]
    fn Check(
        &self,
        source: PyRef<'_, PointCloud>,
        target: PyRef<'_, PointCloud>,
        corres: PyRef<'_, Vector2iVector>,
        transformation: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let t = m4_from_any(transformation)?;
        self.inner
            .check(&source.inner, &target.inner, &corres.data, &t)
            .map_err(PyValueError::new_err)
    }

    #[getter]
    fn require_pointcloud_alignment_(&self) -> bool {
        self.inner.require_pointcloud_alignment()
    }
}

#[pyclass(extends = CorrespondenceChecker, name = "CorrespondenceCheckerBasedOnEdgeLength", module = "tiny3d.cpu.pybind.pipelines.registration")]
pub struct CorrespondenceCheckerBasedOnEdgeLength;

#[pymethods]
impl CorrespondenceCheckerBasedOnEdgeLength {
    #[new]
    #[pyo3(signature = (similarity_threshold = 0.9))]
    fn new(similarity_threshold: f64) -> (Self, CorrespondenceChecker) {
        (
            CorrespondenceCheckerBasedOnEdgeLength,
            CorrespondenceChecker {
                inner: cr::CorrespondenceChecker::EdgeLength {
                    similarity_threshold,
                },
            },
        )
    }
    fn __repr__(slf: PyRef<'_, Self>) -> String {
        if let cr::CorrespondenceChecker::EdgeLength {
            similarity_threshold,
        } = slf.as_super().inner
        {
            format!(
                "CorrespondenceCheckerBasedOnEdgeLength with similarity_threshold={}",
                c_format_f(similarity_threshold, 6)
            )
        } else {
            unreachable!()
        }
    }
    fn __copy__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
    ) -> PyResult<Py<CorrespondenceCheckerBasedOnEdgeLength>> {
        let base: &CorrespondenceChecker = slf.as_super();
        Py::new(
            py,
            PyClassInitializer::from(base.clone())
                .add_subclass(CorrespondenceCheckerBasedOnEdgeLength),
        )
    }
    fn __deepcopy__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        _m: &Bound<'_, PyAny>,
    ) -> PyResult<Py<CorrespondenceCheckerBasedOnEdgeLength>> {
        let base: &CorrespondenceChecker = slf.as_super();
        Py::new(
            py,
            PyClassInitializer::from(base.clone())
                .add_subclass(CorrespondenceCheckerBasedOnEdgeLength),
        )
    }
    #[getter]
    fn similarity_threshold(slf: PyRef<'_, Self>) -> f64 {
        if let cr::CorrespondenceChecker::EdgeLength {
            similarity_threshold,
        } = slf.as_super().inner
        {
            similarity_threshold
        } else {
            0.0
        }
    }
}

#[pyclass(extends = CorrespondenceChecker, name = "CorrespondenceCheckerBasedOnDistance", module = "tiny3d.cpu.pybind.pipelines.registration")]
pub struct CorrespondenceCheckerBasedOnDistance;

#[pymethods]
impl CorrespondenceCheckerBasedOnDistance {
    #[new]
    fn new(distance_threshold: f64) -> (Self, CorrespondenceChecker) {
        (
            CorrespondenceCheckerBasedOnDistance,
            CorrespondenceChecker {
                inner: cr::CorrespondenceChecker::Distance { distance_threshold },
            },
        )
    }
    fn __repr__(slf: PyRef<'_, Self>) -> String {
        if let cr::CorrespondenceChecker::Distance { distance_threshold } = slf.as_super().inner {
            format!(
                "CorrespondenceCheckerBasedOnDistance with distance_threshold={}",
                c_format_f(distance_threshold, 6)
            )
        } else {
            unreachable!()
        }
    }
    fn __copy__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
    ) -> PyResult<Py<CorrespondenceCheckerBasedOnDistance>> {
        let base: &CorrespondenceChecker = slf.as_super();
        Py::new(
            py,
            PyClassInitializer::from(base.clone())
                .add_subclass(CorrespondenceCheckerBasedOnDistance),
        )
    }
    fn __deepcopy__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        _m: &Bound<'_, PyAny>,
    ) -> PyResult<Py<CorrespondenceCheckerBasedOnDistance>> {
        let base: &CorrespondenceChecker = slf.as_super();
        Py::new(
            py,
            PyClassInitializer::from(base.clone())
                .add_subclass(CorrespondenceCheckerBasedOnDistance),
        )
    }
    #[getter]
    fn distance_threshold(slf: PyRef<'_, Self>) -> f64 {
        if let cr::CorrespondenceChecker::Distance { distance_threshold } = slf.as_super().inner {
            distance_threshold
        } else {
            0.0
        }
    }
}

#[pyclass(extends = CorrespondenceChecker, name = "CorrespondenceCheckerBasedOnNormal", module = "tiny3d.cpu.pybind.pipelines.registration")]
pub struct CorrespondenceCheckerBasedOnNormal;

#[pymethods]
impl CorrespondenceCheckerBasedOnNormal {
    #[new]
    fn new(normal_angle_threshold: f64) -> (Self, CorrespondenceChecker) {
        (
            CorrespondenceCheckerBasedOnNormal,
            CorrespondenceChecker {
                inner: cr::CorrespondenceChecker::Normal {
                    normal_angle_threshold,
                },
            },
        )
    }
    fn __repr__(slf: PyRef<'_, Self>) -> String {
        if let cr::CorrespondenceChecker::Normal {
            normal_angle_threshold,
        } = slf.as_super().inner
        {
            format!(
                "CorrespondenceCheckerBasedOnNormal with normal_threshold={}",
                c_format_f(normal_angle_threshold, 6)
            )
        } else {
            unreachable!()
        }
    }
    fn __copy__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
    ) -> PyResult<Py<CorrespondenceCheckerBasedOnNormal>> {
        let base: &CorrespondenceChecker = slf.as_super();
        Py::new(
            py,
            PyClassInitializer::from(base.clone()).add_subclass(CorrespondenceCheckerBasedOnNormal),
        )
    }
    fn __deepcopy__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        _m: &Bound<'_, PyAny>,
    ) -> PyResult<Py<CorrespondenceCheckerBasedOnNormal>> {
        let base: &CorrespondenceChecker = slf.as_super();
        Py::new(
            py,
            PyClassInitializer::from(base.clone()).add_subclass(CorrespondenceCheckerBasedOnNormal),
        )
    }
    #[getter]
    fn normal_angle_threshold(slf: PyRef<'_, Self>) -> f64 {
        if let cr::CorrespondenceChecker::Normal {
            normal_angle_threshold,
        } = slf.as_super().inner
        {
            normal_angle_threshold
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------- result

#[pyclass(
    name = "RegistrationResult",
    module = "tiny3d.cpu.pybind.pipelines.registration"
)]
#[derive(Clone, Default)]
pub struct RegistrationResult {
    pub inner: cr::RegistrationResult,
}

#[pymethods]
impl RegistrationResult {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn __repr__(&self) -> String {
        format!(
            "RegistrationResult with fitness={}, inlier_rmse={}, and correspondence_set size of {}\nAccess transformation to get result.",
            c_format_e(self.inner.fitness, 6),
            c_format_e(self.inner.inlier_rmse, 6),
            self.inner.correspondence_set.len()
        )
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }
    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    #[getter(transformation)]
    fn transformation_getter(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        m4_to_numpy(py, &self.inner.transformation)
    }
    #[setter(transformation)]
    fn transformation_setter(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.transformation = m4_from_any(v)?;
        Ok(())
    }
    #[getter(fitness)]
    fn fitness_getter(&self) -> f64 {
        self.inner.fitness
    }
    #[setter(fitness)]
    fn fitness_setter(&mut self, v: f64) {
        self.inner.fitness = v;
    }
    #[getter(inlier_rmse)]
    fn rmse_getter(&self) -> f64 {
        self.inner.inlier_rmse
    }
    #[setter(inlier_rmse)]
    fn rmse_setter(&mut self, v: f64) {
        self.inner.inlier_rmse = v;
    }
    #[getter(correspondence_set)]
    fn corres_getter(&self) -> Vector2iVector {
        Vector2iVector {
            data: self.inner.correspondence_set.clone(),
            owner: None,
        }
    }
    #[setter(correspondence_set)]
    fn corres_setter(&mut self, py: Python<'_>, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.correspondence_set = Vector2iVector::extract_from(py, v)?.data;
        Ok(())
    }
}

// ---------------------------------------------------------------- functions

fn default_estimation() -> cr::TransformationEstimation {
    cr::TransformationEstimation::PointToPoint {
        with_scaling: false,
    }
}

fn extract_checkers(
    checkers: Vec<PyRef<'_, CorrespondenceChecker>>,
) -> Vec<cr::CorrespondenceChecker> {
    checkers.into_iter().map(|c| c.inner.clone()).collect()
}

#[pyfunction]
fn compute_fpfh_feature(
    py: Python<'_>,
    input: PyRef<'_, PointCloud>,
    search_param: PyRef<'_, KDTreeSearchParam>,
) -> PyResult<Feature> {
    let inner = &input.inner;
    let param = search_param.param;
    let res = py.allow_threads(|| cr::compute_fpfh_feature(inner, &param));
    match res {
        Ok(f) => Ok(Feature { inner: f }),
        Err(e) => Err(PyRuntimeError::new_err(e)),
    }
}

#[pyfunction]
#[pyo3(signature = (source_features, target_features, mutual_filter = false, mutual_consistency_ratio = 0.1))]
fn correspondences_from_features(
    py: Python<'_>,
    source_features: PyRef<'_, Feature>,
    target_features: PyRef<'_, Feature>,
    mutual_filter: bool,
    mutual_consistency_ratio: f32,
) -> Vector2iVector {
    let sf = &source_features.inner;
    let tf = &target_features.inner;
    Vector2iVector {
        data: py.allow_threads(|| {
            cr::correspondences_from_features(sf, tf, mutual_filter, mutual_consistency_ratio)
        }),
        owner: None,
    }
}

#[pyfunction]
#[pyo3(signature = (source, target, max_correspondence_distance, transformation = None))]
fn evaluate_registration(
    source: PyRef<'_, PointCloud>,
    target: PyRef<'_, PointCloud>,
    max_correspondence_distance: f64,
    transformation: Option<&Bound<'_, PyAny>>,
) -> PyResult<RegistrationResult> {
    let t = match transformation {
        Some(t) => m4_from_any(t)?,
        None => tiny3d_core::linalg::m4_identity(),
    };
    let (s_in, t_in) = (&source.inner, &target.inner);
    let py = source.py();
    Ok(RegistrationResult {
        inner: py.allow_threads(|| {
            cr::evaluate_registration(s_in, t_in, max_correspondence_distance, &t)
        }),
    })
}

#[pyfunction]
#[pyo3(signature = (source, target, max_correspondence_distance, init = None, estimation_method = None, criteria = None))]
fn registration_icp(
    source: PyRef<'_, PointCloud>,
    target: PyRef<'_, PointCloud>,
    max_correspondence_distance: f64,
    init: Option<&Bound<'_, PyAny>>,
    estimation_method: Option<PyRef<'_, TransformationEstimation>>,
    criteria: Option<PyRef<'_, ICPConvergenceCriteria>>,
) -> PyResult<RegistrationResult> {
    let init_t = match init {
        Some(t) => m4_from_any(t)?,
        None => tiny3d_core::linalg::m4_identity(),
    };
    let est = estimation_method
        .map(|e| e.inner.clone())
        .unwrap_or_else(default_estimation);
    let crit = criteria.map(|c| c.inner.clone()).unwrap_or_default();
    let (s_in, t_in) = (&source.inner, &target.inner);
    let py = source.py();
    let res = py.allow_threads(|| {
        cr::registration_icp(
            s_in,
            t_in,
            max_correspondence_distance,
            &init_t,
            &est,
            &crit,
        )
    });
    match res {
        Ok(r) => Ok(RegistrationResult { inner: r }),
        Err(e) => Err(PyRuntimeError::new_err(e)),
    }
}

#[pyfunction]
#[pyo3(signature = (source, target, corres, max_correspondence_distance, estimation_method = None, ransac_n = 3, checkers = Vec::new(), criteria = None))]
#[allow(clippy::too_many_arguments)]
fn registration_ransac_based_on_correspondence(
    source: PyRef<'_, PointCloud>,
    target: PyRef<'_, PointCloud>,
    corres: PyRef<'_, Vector2iVector>,
    max_correspondence_distance: f64,
    estimation_method: Option<PyRef<'_, TransformationEstimation>>,
    ransac_n: i32,
    checkers: Vec<PyRef<'_, CorrespondenceChecker>>,
    criteria: Option<PyRef<'_, RANSACConvergenceCriteria>>,
) -> PyResult<RegistrationResult> {
    let est = estimation_method
        .map(|e| e.inner.clone())
        .unwrap_or_else(default_estimation);
    let crit = criteria.map(|c| c.inner.clone()).unwrap_or_default();
    let chk = extract_checkers(checkers);
    let (s_in, t_in, c_in) = (&source.inner, &target.inner, &corres.data);
    let py = source.py();
    let inner = py
        .allow_threads(|| {
            cr::registration_ransac_based_on_correspondence(
                s_in,
                t_in,
                c_in,
                max_correspondence_distance,
                &est,
                ransac_n,
                &chk,
                &crit,
            )
        })
        .map_err(PyValueError::new_err)?;
    Ok(RegistrationResult { inner })
}

#[pyfunction]
#[pyo3(signature = (source, target, source_feature, target_feature, mutual_filter, max_correspondence_distance, estimation_method = None, ransac_n = 3, checkers = Vec::new(), criteria = None))]
#[allow(clippy::too_many_arguments)]
fn registration_ransac_based_on_feature_matching(
    source: PyRef<'_, PointCloud>,
    target: PyRef<'_, PointCloud>,
    source_feature: PyRef<'_, Feature>,
    target_feature: PyRef<'_, Feature>,
    mutual_filter: bool,
    max_correspondence_distance: f64,
    estimation_method: Option<PyRef<'_, TransformationEstimation>>,
    ransac_n: i32,
    checkers: Vec<PyRef<'_, CorrespondenceChecker>>,
    criteria: Option<PyRef<'_, RANSACConvergenceCriteria>>,
) -> PyResult<RegistrationResult> {
    let est = estimation_method
        .map(|e| e.inner.clone())
        .unwrap_or_else(default_estimation);
    let crit = criteria.map(|c| c.inner.clone()).unwrap_or_default();
    let chk = extract_checkers(checkers);
    let (s_in, t_in) = (&source.inner, &target.inner);
    let (sf_in, tf_in) = (&source_feature.inner, &target_feature.inner);
    let py = source.py();
    let inner = py
        .allow_threads(|| {
            cr::registration_ransac_based_on_feature_matching(
                s_in,
                t_in,
                sf_in,
                tf_in,
                mutual_filter,
                max_correspondence_distance,
                &est,
                ransac_n,
                &chk,
                &crit,
            )
        })
        .map_err(PyValueError::new_err)?;
    Ok(RegistrationResult { inner })
}

#[pyfunction]
fn get_information_matrix_from_point_clouds(
    py: Python<'_>,
    source: PyRef<'_, PointCloud>,
    target: PyRef<'_, PointCloud>,
    max_correspondence_distance: f64,
    transformation: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let t = m4_from_any(transformation)?;
    let (s_in, t_in) = (&source.inner, &target.inner);
    let info = py.allow_threads(|| {
        cr::get_information_matrix_from_point_clouds(s_in, t_in, max_correspondence_distance, &t)
    });
    let flat: Vec<f64> = info.iter().flatten().copied().collect();
    Ok(PyArray1::from_vec(py, flat)
        .reshape([6, 6])?
        .into_any()
        .unbind())
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Feature>()?;
    m.add_class::<ICPConvergenceCriteria>()?;
    m.add_class::<RANSACConvergenceCriteria>()?;
    m.add_class::<TransformationEstimation>()?;
    m.add_class::<TransformationEstimationPointToPoint>()?;
    m.add_class::<TransformationEstimationPointToPlane>()?;
    m.add_class::<CorrespondenceChecker>()?;
    m.add_class::<CorrespondenceCheckerBasedOnEdgeLength>()?;
    m.add_class::<CorrespondenceCheckerBasedOnDistance>()?;
    m.add_class::<CorrespondenceCheckerBasedOnNormal>()?;
    m.add_class::<RegistrationResult>()?;
    m.add_function(wrap_pyfunction!(compute_fpfh_feature, m)?)?;
    m.add_function(wrap_pyfunction!(correspondences_from_features, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_registration, m)?)?;
    m.add_function(wrap_pyfunction!(registration_icp, m)?)?;
    m.add_function(wrap_pyfunction!(
        registration_ransac_based_on_correspondence,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        registration_ransac_based_on_feature_matching,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        get_information_matrix_from_point_clouds,
        m
    )?)?;
    Ok(())
}
