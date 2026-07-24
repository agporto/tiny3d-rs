//! tiny3d.cpu.pybind.geometry

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;

use tiny3d_core::geometry as cg;
use tiny3d_core::geometry::search_param::KdTreeSearchParam;
use tiny3d_core::kdtree::KdTreeFlann as CoreKdTree;
use tiny3d_core::linalg::{M3, M4, V3};

use crate::fmtutil::ostream_double;
use crate::vectors::{
    DoubleVector, IntVector, Vector3dVector, Vector3iVector, ViewAccess, VT_MESH_TRIANGLES,
    VT_MESH_TRIANGLE_NORMALS, VT_MESH_VERTEX_COLORS, VT_MESH_VERTEX_NORMALS, VT_MESH_VERTICES,
    VT_PCD_COLORS, VT_PCD_NORMALS, VT_PCD_POINTS,
};

// ---------------------------------------------------------------- helpers

pub fn v3_from_any(obj: &Bound<'_, PyAny>) -> PyResult<V3> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<f64>>() {
        let s = arr.as_slice()?;
        if s.len() == 3 {
            return Ok([s[0], s[1], s[2]]);
        }
    }
    let v: Vec<f64> = obj.extract()?;
    if v.len() != 3 {
        return Err(PyTypeError::new_err("expected 3-vector"));
    }
    Ok([v[0], v[1], v[2]])
}

pub fn v4_from_any(obj: &Bound<'_, PyAny>) -> PyResult<[f64; 4]> {
    let v: Vec<f64> = if let Ok(arr) = obj.extract::<PyReadonlyArray1<f64>>() {
        arr.as_slice()?.to_vec()
    } else {
        obj.extract()?
    };
    if v.len() != 4 {
        return Err(PyTypeError::new_err("expected 4-vector"));
    }
    Ok([v[0], v[1], v[2], v[3]])
}

pub fn m3_from_any(obj: &Bound<'_, PyAny>) -> PyResult<M3> {
    let arr: PyReadonlyArray2<f64> = obj.extract()?;
    let view = arr.as_array();
    if view.shape() != [3, 3] {
        return Err(PyTypeError::new_err("expected 3x3 matrix"));
    }
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = view[[i, j]];
        }
    }
    Ok(m)
}

pub fn m4_from_any(obj: &Bound<'_, PyAny>) -> PyResult<M4> {
    let arr: PyReadonlyArray2<f64> = obj.extract()?;
    let view = arr.as_array();
    if view.shape() != [4, 4] {
        return Err(PyTypeError::new_err("expected 4x4 matrix"));
    }
    let mut m = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = view[[i, j]];
        }
    }
    Ok(m)
}

pub fn v3_to_numpy(py: Python<'_>, v: V3) -> Py<PyArray1<f64>> {
    PyArray1::from_slice(py, &v).unbind()
}

pub fn m3_to_numpy(py: Python<'_>, m: &M3) -> PyResult<Py<PyAny>> {
    let flat: Vec<f64> = m.iter().flatten().copied().collect();
    Ok(PyArray1::from_vec(py, flat)
        .reshape([3, 3])?
        .into_any()
        .unbind())
}

pub fn m4_to_numpy(py: Python<'_>, m: &M4) -> PyResult<Py<PyAny>> {
    let flat: Vec<f64> = m.iter().flatten().copied().collect();
    Ok(PyArray1::from_vec(py, flat)
        .reshape([4, 4])?
        .into_any()
        .unbind())
}

// ------------------------------------------------- view dispatch (live vectors)

impl ViewAccess for [f64; 3] {
    fn with_view<R>(
        py: Python<'_>,
        owner: &Py<PyAny>,
        target: u8,
        f: impl FnOnce(&Vec<Self>) -> R,
    ) -> PyResult<R> {
        let b = owner.bind(py);
        match target {
            VT_PCD_POINTS | VT_PCD_NORMALS | VT_PCD_COLORS => {
                let cell = b
                    .downcast::<PointCloud>()
                    .map_err(|_| PyTypeError::new_err("view owner is not a PointCloud"))?
                    .borrow();
                let v = match target {
                    VT_PCD_POINTS => &cell.inner.points,
                    VT_PCD_NORMALS => &cell.inner.normals,
                    _ => &cell.inner.colors,
                };
                Ok(f(v))
            }
            _ => {
                let cell = b
                    .downcast::<TriangleMesh>()
                    .map_err(|_| PyTypeError::new_err("view owner is not a TriangleMesh"))?
                    .borrow();
                let v = match target {
                    VT_MESH_VERTICES => &cell.inner.vertices,
                    VT_MESH_VERTEX_NORMALS => &cell.inner.vertex_normals,
                    VT_MESH_VERTEX_COLORS => &cell.inner.vertex_colors,
                    VT_MESH_TRIANGLE_NORMALS => &cell.inner.triangle_normals,
                    _ => return Err(PyTypeError::new_err("invalid view target")),
                };
                Ok(f(v))
            }
        }
    }

    fn with_view_mut<R>(
        py: Python<'_>,
        owner: &Py<PyAny>,
        target: u8,
        f: impl FnOnce(&mut Vec<Self>) -> R,
    ) -> PyResult<R> {
        let b = owner.bind(py);
        match target {
            VT_PCD_POINTS | VT_PCD_NORMALS | VT_PCD_COLORS => {
                let mut cell = b
                    .downcast::<PointCloud>()
                    .map_err(|_| PyTypeError::new_err("view owner is not a PointCloud"))?
                    .borrow_mut();
                let v = match target {
                    VT_PCD_POINTS => &mut cell.inner.points,
                    VT_PCD_NORMALS => &mut cell.inner.normals,
                    _ => &mut cell.inner.colors,
                };
                Ok(f(v))
            }
            _ => {
                let mut cell = b
                    .downcast::<TriangleMesh>()
                    .map_err(|_| PyTypeError::new_err("view owner is not a TriangleMesh"))?
                    .borrow_mut();
                let v = match target {
                    VT_MESH_VERTICES => &mut cell.inner.vertices,
                    VT_MESH_VERTEX_NORMALS => &mut cell.inner.vertex_normals,
                    VT_MESH_VERTEX_COLORS => &mut cell.inner.vertex_colors,
                    VT_MESH_TRIANGLE_NORMALS => &mut cell.inner.triangle_normals,
                    _ => return Err(PyTypeError::new_err("invalid view target")),
                };
                Ok(f(v))
            }
        }
    }
}

impl ViewAccess for [i32; 3] {
    fn with_view<R>(
        py: Python<'_>,
        owner: &Py<PyAny>,
        target: u8,
        f: impl FnOnce(&Vec<Self>) -> R,
    ) -> PyResult<R> {
        if target != VT_MESH_TRIANGLES {
            return Err(PyTypeError::new_err("invalid view target"));
        }
        let b = owner.bind(py);
        let cell = b
            .downcast::<TriangleMesh>()
            .map_err(|_| PyTypeError::new_err("view owner is not a TriangleMesh"))?
            .borrow();
        Ok(f(&cell.inner.triangles))
    }

    fn with_view_mut<R>(
        py: Python<'_>,
        owner: &Py<PyAny>,
        target: u8,
        f: impl FnOnce(&mut Vec<Self>) -> R,
    ) -> PyResult<R> {
        if target != VT_MESH_TRIANGLES {
            return Err(PyTypeError::new_err("invalid view target"));
        }
        let b = owner.bind(py);
        let mut cell = b
            .downcast::<TriangleMesh>()
            .map_err(|_| PyTypeError::new_err("view owner is not a TriangleMesh"))?
            .borrow_mut();
        Ok(f(&mut cell.inner.triangles))
    }
}

// ---------------------------------------------------------------- GeometryType

#[pyclass(
    name = "GeometryType",
    module = "tiny3d.cpu.pybind.geometry",
    frozen,
    eq,
    hash
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct GeometryType {
    #[pyo3(get)]
    pub value: i32,
    pub name_str: &'static str,
}

#[pymethods]
impl GeometryType {
    #[getter]
    fn name(&self) -> &'static str {
        self.name_str
    }
    fn __int__(&self) -> i32 {
        self.value
    }
    fn __index__(&self) -> i32 {
        self.value
    }
    fn __repr__(&self) -> String {
        format!("<Type.{}: {}>", self.name_str, self.value)
    }
}

pub const GT_UNSPECIFIED: GeometryType = GeometryType {
    value: 0,
    name_str: "Unspecified",
};
pub const GT_POINTCLOUD: GeometryType = GeometryType {
    value: 1,
    name_str: "PointCloud",
};
pub const GT_VOXELGRID: GeometryType = GeometryType {
    value: 2,
    name_str: "VoxelGrid",
};
pub const GT_TRIANGLEMESH: GeometryType = GeometryType {
    value: 6,
    name_str: "TriangleMesh",
};

// ---------------------------------------------------------------- base classes

#[pyclass(subclass, name = "Geometry", module = "tiny3d.cpu.pybind.geometry")]
pub struct Geometry;

#[pymethods]
impl Geometry {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const Unspecified: GeometryType = GT_UNSPECIFIED;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const PointCloud: GeometryType = GT_POINTCLOUD;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const VoxelGrid: GeometryType = GT_VOXELGRID;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const TriangleMesh: GeometryType = GT_TRIANGLEMESH;
    #[classattr]
    // Preserve the pybind-compatible public attribute spelling.
    #[allow(non_snake_case, non_upper_case_globals)]
    fn Type(py: Python<'_>) -> Py<PyAny> {
        py.get_type::<GeometryType>().into_any().unbind()
    }
}

#[pyclass(subclass, extends = Geometry, name = "Geometry3D", module = "tiny3d.cpu.pybind.geometry")]
pub struct Geometry3D;

#[pymethods]
impl Geometry3D {
    #[staticmethod]
    fn get_rotation_matrix_from_xyz(
        py: Python<'_>,
        rotation: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        m3_to_numpy(py, &cg::rotation::from_xyz(v3_from_any(rotation)?))
    }
    #[staticmethod]
    fn get_rotation_matrix_from_yzx(
        py: Python<'_>,
        rotation: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        m3_to_numpy(py, &cg::rotation::from_yzx(v3_from_any(rotation)?))
    }
    #[staticmethod]
    fn get_rotation_matrix_from_zxy(
        py: Python<'_>,
        rotation: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        m3_to_numpy(py, &cg::rotation::from_zxy(v3_from_any(rotation)?))
    }
    #[staticmethod]
    fn get_rotation_matrix_from_xzy(
        py: Python<'_>,
        rotation: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        m3_to_numpy(py, &cg::rotation::from_xzy(v3_from_any(rotation)?))
    }
    #[staticmethod]
    fn get_rotation_matrix_from_zyx(
        py: Python<'_>,
        rotation: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        m3_to_numpy(py, &cg::rotation::from_zyx(v3_from_any(rotation)?))
    }
    #[staticmethod]
    fn get_rotation_matrix_from_yxz(
        py: Python<'_>,
        rotation: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        m3_to_numpy(py, &cg::rotation::from_yxz(v3_from_any(rotation)?))
    }
    #[staticmethod]
    fn get_rotation_matrix_from_axis_angle(
        py: Python<'_>,
        rotation: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        m3_to_numpy(py, &cg::rotation::from_axis_angle(v3_from_any(rotation)?))
    }
    #[staticmethod]
    fn get_rotation_matrix_from_quaternion(
        py: Python<'_>,
        rotation: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let matrix =
            cg::rotation::from_quaternion(v4_from_any(rotation)?).map_err(PyValueError::new_err)?;
        m3_to_numpy(py, &matrix)
    }
}

#[pyclass(subclass, extends = Geometry, name = "Geometry2D", module = "tiny3d.cpu.pybind.geometry")]
pub struct Geometry2D;

#[pymethods]
impl Geometry2D {}

// ---------------------------------------------------------------- PointCloud

#[pyclass(extends = Geometry3D, name = "PointCloud", module = "tiny3d.cpu.pybind.geometry", subclass)]
#[derive(Default)]
pub struct PointCloud {
    pub inner: cg::PointCloud,
}

pub fn pcd_init_pub(inner: cg::PointCloud) -> PyClassInitializer<PointCloud> {
    pcd_init(inner)
}

fn pcd_init(inner: cg::PointCloud) -> PyClassInitializer<PointCloud> {
    PyClassInitializer::from(Geometry)
        .add_subclass(Geometry3D)
        .add_subclass(PointCloud { inner })
}

#[pymethods]
impl PointCloud {
    #[new]
    #[pyo3(signature = (other = None))]
    fn new(other: Option<PyRef<'_, PointCloud>>) -> PyClassInitializer<PointCloud> {
        match other {
            Some(o) => pcd_init(o.inner.clone()),
            None => pcd_init(cg::PointCloud::new()),
        }
    }

    fn __repr__(&self) -> String {
        format!("PointCloud with {} points.", self.inner.points.len())
    }

    fn __copy__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PointCloud>> {
        Py::new(py, pcd_init(slf.inner.clone()))
    }
    fn __deepcopy__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        _memo: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PointCloud>> {
        Py::new(py, pcd_init(slf.inner.clone()))
    }

    #[getter]
    fn get_points(slf: PyRef<'_, Self>) -> Vector3dVector {
        let owner: Py<PointCloud> = slf.into();
        Vector3dVector {
            data: Vec::new(),
            owner: Some((owner.into_any(), VT_PCD_POINTS)),
        }
    }
    #[setter]
    fn set_points(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        Vector3dVector::assign_into(py, value, &mut self.inner.points)?;
        Ok(())
    }
    #[getter]
    fn get_normals(slf: PyRef<'_, Self>) -> Vector3dVector {
        let owner: Py<PointCloud> = slf.into();
        Vector3dVector {
            data: Vec::new(),
            owner: Some((owner.into_any(), VT_PCD_NORMALS)),
        }
    }
    #[setter]
    fn set_normals(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        Vector3dVector::assign_into(py, value, &mut self.inner.normals)?;
        Ok(())
    }
    #[getter]
    fn get_colors(slf: PyRef<'_, Self>) -> Vector3dVector {
        let owner: Py<PointCloud> = slf.into();
        Vector3dVector {
            data: Vec::new(),
            owner: Some((owner.into_any(), VT_PCD_COLORS)),
        }
    }
    #[setter]
    fn set_colors(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        Vector3dVector::assign_into(py, value, &mut self.inner.colors)?;
        Ok(())
    }

    fn clear(mut slf: PyRefMut<'_, Self>) -> Py<Self> {
        slf.inner.clear();
        slf.into()
    }
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    fn dimension(&self) -> i32 {
        3
    }
    fn get_geometry_type(&self) -> GeometryType {
        GT_POINTCLOUD
    }
    fn has_points(&self) -> bool {
        self.inner.has_points()
    }
    fn has_normals(&self) -> bool {
        self.inner.has_normals()
    }
    fn has_colors(&self) -> bool {
        self.inner.has_colors()
    }
    fn get_min_bound(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        v3_to_numpy(py, self.inner.get_min_bound())
    }
    fn get_max_bound(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        v3_to_numpy(py, self.inner.get_max_bound())
    }
    fn get_center(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        v3_to_numpy(py, self.inner.get_center())
    }
    fn get_axis_aligned_bounding_box(
        &self,
        py: Python<'_>,
    ) -> PyResult<Py<AxisAlignedBoundingBox>> {
        Py::new(
            py,
            aabb_init(cg::AxisAlignedBoundingBox::new(
                self.inner.get_min_bound(),
                self.inner.get_max_bound(),
            )),
        )
    }

    fn transform(
        mut slf: PyRefMut<'_, Self>,
        transformation: &Bound<'_, PyAny>,
    ) -> PyResult<Py<Self>> {
        let t = m4_from_any(transformation)?;
        slf.inner.transform(&t);
        Ok(slf.into())
    }
    #[pyo3(signature = (translation, relative = true))]
    fn translate(
        mut slf: PyRefMut<'_, Self>,
        translation: &Bound<'_, PyAny>,
        relative: bool,
    ) -> PyResult<Py<Self>> {
        let t = v3_from_any(translation)?;
        slf.inner.translate(t, relative);
        Ok(slf.into())
    }
    #[pyo3(signature = (scale, center))]
    fn scale(
        mut slf: PyRefMut<'_, Self>,
        scale: f64,
        center: &Bound<'_, PyAny>,
    ) -> PyResult<Py<Self>> {
        let c = v3_from_any(center)?;
        slf.inner.scale(scale, c);
        Ok(slf.into())
    }
    #[pyo3(signature = (r, center = None))]
    fn rotate(
        mut slf: PyRefMut<'_, Self>,
        r: &Bound<'_, PyAny>,
        center: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<Self>> {
        let rm = m3_from_any(r)?;
        let c = match center {
            Some(c) => v3_from_any(c)?,
            None => slf.inner.get_center(),
        };
        slf.inner.rotate(&rm, c);
        Ok(slf.into())
    }

    fn normalize_normals(mut slf: PyRefMut<'_, Self>) -> Py<Self> {
        slf.inner.normalize_normals();
        slf.into()
    }
    fn paint_uniform_color(
        mut slf: PyRefMut<'_, Self>,
        color: &Bound<'_, PyAny>,
    ) -> PyResult<Py<Self>> {
        let c = v3_from_any(color)?;
        slf.inner.paint_uniform_color(c);
        Ok(slf.into())
    }

    fn voxel_down_sample(&self, py: Python<'_>, voxel_size: f64) -> PyResult<Py<PointCloud>> {
        let inner = &self.inner;
        let res = py.allow_threads(|| inner.voxel_down_sample(voxel_size));
        match res {
            Ok(out) => Py::new(py, pcd_init(out)),
            Err(msg) => Err(PyRuntimeError::new_err(msg)),
        }
    }

    #[pyo3(signature = (search_param = None, fast_normal_computation = true))]
    fn estimate_normals(
        &mut self,
        py: Python<'_>,
        search_param: Option<PyRef<'_, KDTreeSearchParam>>,
        fast_normal_computation: bool,
    ) {
        let param = match search_param {
            Some(p) => p.param,
            None => KdTreeSearchParam::Knn { knn: 30 },
        };
        let inner = &mut self.inner;
        py.allow_threads(|| inner.estimate_normals(&param, fast_normal_computation));
    }

    /// Function to orient the normals of a point cloud.
    ///
    /// Args:
    ///     orientation_reference (numpy.ndarray[numpy.float64[3, 1]],
    ///         optional, default=array([0., 0., 1.])): Normals are oriented
    ///         with respect to orientation_reference.
    ///
    /// Returns:
    ///     None
    #[pyo3(signature = (orientation_reference = None))]
    fn orient_normals_to_align_with_direction(
        &mut self,
        py: Python<'_>,
        orientation_reference: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let reference = match orientation_reference {
            Some(v) => v3_from_any(v)?,
            None => [0.0, 0.0, 1.0],
        };
        let inner = &mut self.inner;
        py.allow_threads(|| inner.orient_normals_to_align_with_direction(reference))
            .map_err(PyRuntimeError::new_err)
    }

    /// Function to orient the normals of a point cloud.
    ///
    /// Args:
    ///     camera_location (numpy.ndarray[numpy.float64[3, 1]], optional,
    ///         default=array([0., 0., 0.])): Normals are oriented with
    ///         towards the camera_location.
    ///
    /// Returns:
    ///     None
    #[pyo3(signature = (camera_location = None))]
    fn orient_normals_towards_camera_location(
        &mut self,
        py: Python<'_>,
        camera_location: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let camera = match camera_location {
            Some(v) => v3_from_any(v)?,
            None => [0.0, 0.0, 0.0],
        };
        let inner = &mut self.inner;
        py.allow_threads(|| inner.orient_normals_towards_camera_location(camera))
            .map_err(PyRuntimeError::new_err)
    }

    /// Function to consistently orient the normals of a point cloud based
    /// on tangent planes.
    ///
    /// Args:
    ///     k (int): Number of k nearest neighbors used in constructing the
    ///         Riemannian graph used to propagate normal orientation.
    ///     lambda (float, optional, default=0.0): penalty constant on the
    ///         distance to the tangent plane.
    ///     cos_alpha_tol (float, optional, default=1.0): treshold that
    ///         defines the amplitude of the cone spanned by the reference
    ///         normal.
    ///
    /// Returns:
    ///     None
    #[pyo3(signature = (k, lambda = 0.0, cos_alpha_tol = 1.0))]
    fn orient_normals_consistent_tangent_plane(
        &mut self,
        py: Python<'_>,
        k: usize,
        lambda: f64,
        cos_alpha_tol: f64,
    ) -> PyResult<()> {
        let inner = &mut self.inner;
        py.allow_threads(|| {
            inner.orient_normals_consistent_tangent_plane(k, lambda, cos_alpha_tol)
        })
        .map_err(PyRuntimeError::new_err)
    }
}

// ---------------------------------------------------------------- AABB

#[pyclass(extends = Geometry3D, name = "AxisAlignedBoundingBox", module = "tiny3d.cpu.pybind.geometry")]
pub struct AxisAlignedBoundingBox {
    pub inner: cg::AxisAlignedBoundingBox,
}

pub fn aabb_init(inner: cg::AxisAlignedBoundingBox) -> PyClassInitializer<AxisAlignedBoundingBox> {
    PyClassInitializer::from(Geometry)
        .add_subclass(Geometry3D)
        .add_subclass(AxisAlignedBoundingBox { inner })
}

#[pymethods]
impl AxisAlignedBoundingBox {
    #[new]
    #[pyo3(signature = (min_bound = None, max_bound = None))]
    fn new(
        min_bound: Option<&Bound<'_, PyAny>>,
        max_bound: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<AxisAlignedBoundingBox>> {
        let inner = match (min_bound, max_bound) {
            (Some(mn), Some(mx)) => {
                // copy-constructor style: first arg may be another AABB
                if let Ok(other) = mn.extract::<PyRef<'_, AxisAlignedBoundingBox>>() {
                    let _ = mx;
                    other.inner.clone()
                } else {
                    cg::AxisAlignedBoundingBox::new(v3_from_any(mn)?, v3_from_any(mx)?)
                }
            }
            (Some(o), None) => {
                let other = o.extract::<PyRef<'_, AxisAlignedBoundingBox>>()?;
                other.inner.clone()
            }
            _ => cg::AxisAlignedBoundingBox::default(),
        };
        Ok(aabb_init(inner))
    }

    fn __repr__(&self) -> String {
        format!(
            "AxisAlignedBoundingBox: min: ({}, {}, {}), max: ({}, {}, {})",
            ostream_double(self.inner.min_bound[0]),
            ostream_double(self.inner.min_bound[1]),
            ostream_double(self.inner.min_bound[2]),
            ostream_double(self.inner.max_bound[0]),
            ostream_double(self.inner.max_bound[1]),
            ostream_double(self.inner.max_bound[2]),
        )
    }

    fn __copy__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<AxisAlignedBoundingBox>> {
        Py::new(py, aabb_init(slf.inner.clone()))
    }
    fn __deepcopy__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        _memo: &Bound<'_, PyAny>,
    ) -> PyResult<Py<AxisAlignedBoundingBox>> {
        Py::new(py, aabb_init(slf.inner.clone()))
    }

    fn __iadd__(&mut self, other: PyRef<'_, AxisAlignedBoundingBox>) {
        self.inner.merge(&other.inner);
    }

    #[getter(min_bound)]
    fn min_bound_getter(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        v3_to_numpy(py, self.inner.min_bound)
    }
    #[setter(min_bound)]
    fn min_bound_setter(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.min_bound = v3_from_any(v)?;
        Ok(())
    }
    #[getter(max_bound)]
    fn max_bound_getter(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        v3_to_numpy(py, self.inner.max_bound)
    }
    #[setter(max_bound)]
    fn max_bound_setter(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.max_bound = v3_from_any(v)?;
        Ok(())
    }
    #[getter(color)]
    fn color_getter(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        v3_to_numpy(py, self.inner.color)
    }
    #[setter(color)]
    fn color_setter(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.color = v3_from_any(v)?;
        Ok(())
    }

    fn clear(mut slf: PyRefMut<'_, Self>) -> Py<Self> {
        slf.inner.clear();
        slf.into()
    }
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    fn dimension(&self) -> i32 {
        3
    }
    fn get_geometry_type(&self) -> GeometryType {
        GeometryType {
            value: 12,
            name_str: "AxisAlignedBoundingBox",
        }
    }
    fn get_min_bound(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        v3_to_numpy(py, self.inner.get_min_bound())
    }
    fn get_max_bound(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        v3_to_numpy(py, self.inner.get_max_bound())
    }
    fn get_center(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        v3_to_numpy(py, self.inner.get_center())
    }
    fn get_extent(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        v3_to_numpy(py, self.inner.get_extent())
    }
    fn get_half_extent(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        v3_to_numpy(py, self.inner.get_half_extent())
    }
    fn get_max_extent(&self) -> f64 {
        self.inner.get_max_extent()
    }
    fn volume(&self) -> f64 {
        self.inner.volume()
    }
    fn get_box_points(&self) -> Vector3dVector {
        Vector3dVector {
            data: self.inner.get_box_points(),
            owner: None,
        }
    }
    fn get_point_indices_within_bounding_box(
        &self,
        py: Python<'_>,
        points: PyRef<'_, Vector3dVector>,
    ) -> PyResult<Vec<usize>> {
        points.read(py, |d| self.inner.get_point_indices_within_bounding_box(d))
    }
    fn get_print_info(&self) -> String {
        self.inner.get_print_info()
    }
    fn get_axis_aligned_bounding_box(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
    ) -> PyResult<Py<AxisAlignedBoundingBox>> {
        Py::new(py, aabb_init(slf.inner.clone()))
    }
    #[staticmethod]
    fn create_from_points(
        py: Python<'_>,
        points: PyRef<'_, Vector3dVector>,
    ) -> PyResult<Py<AxisAlignedBoundingBox>> {
        let bb = points.read(py, |d| cg::AxisAlignedBoundingBox::create_from_points(d))?;
        Py::new(py, aabb_init(bb))
    }
    #[pyo3(signature = (translation, relative = true))]
    fn translate(
        mut slf: PyRefMut<'_, Self>,
        translation: &Bound<'_, PyAny>,
        relative: bool,
    ) -> PyResult<Py<Self>> {
        let t = v3_from_any(translation)?;
        slf.inner.translate(t, relative);
        Ok(slf.into())
    }
    fn scale(
        mut slf: PyRefMut<'_, Self>,
        scale: f64,
        center: &Bound<'_, PyAny>,
    ) -> PyResult<Py<Self>> {
        let c = v3_from_any(center)?;
        slf.inner.scale(scale, c);
        Ok(slf.into())
    }
    fn transform(&self, _t: &Bound<'_, PyAny>) -> PyResult<()> {
        Err(PyRuntimeError::new_err(
            "[AxisAlignedBoundingBox::Transform] Cannot apply general transform. Convert to OrientedBoundingBox first or use Translate/Scale.",
        ))
    }
    #[pyo3(signature = (r, center = None))]
    fn rotate(&self, r: &Bound<'_, PyAny>, center: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let _ = (r, center);
        Err(PyRuntimeError::new_err(
            "[AxisAlignedBoundingBox::Rotate] Cannot rotate AABB. Convert to OrientedBoundingBox first.",
        ))
    }
}

// ---------------------------------------------------------------- KDTreeSearchParam

#[pyclass(
    subclass,
    name = "KDTreeSearchParam",
    module = "tiny3d.cpu.pybind.geometry"
)]
#[derive(Clone, Copy)]
pub struct KDTreeSearchParam {
    pub param: KdTreeSearchParam,
}

#[pymethods]
impl KDTreeSearchParam {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const KNNSearch: i32 = 0;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const RadiusSearch: i32 = 1;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const HybridSearch: i32 = 2;

    fn get_search_type(&self) -> i32 {
        self.param.search_type()
    }
}

#[pyclass(extends = KDTreeSearchParam, name = "KDTreeSearchParamKNN", module = "tiny3d.cpu.pybind.geometry")]
pub struct KDTreeSearchParamKNN;

#[pymethods]
impl KDTreeSearchParamKNN {
    #[new]
    #[pyo3(signature = (knn = 30))]
    fn new(knn: i32) -> (Self, KDTreeSearchParam) {
        (
            KDTreeSearchParamKNN,
            KDTreeSearchParam {
                param: KdTreeSearchParam::Knn { knn },
            },
        )
    }
    fn __repr__(slf: PyRef<'_, Self>) -> String {
        let base = slf.as_super();
        if let KdTreeSearchParam::Knn { knn } = base.param {
            format!("KDTreeSearchParamKNN(knn={})", knn)
        } else {
            unreachable!()
        }
    }
    #[getter]
    fn knn(slf: PyRef<'_, Self>) -> i32 {
        if let KdTreeSearchParam::Knn { knn } = slf.as_super().param {
            knn
        } else {
            0
        }
    }
    #[setter]
    fn set_knn(mut slf: PyRefMut<'_, Self>, v: i32) {
        slf.as_super().param = KdTreeSearchParam::Knn { knn: v };
    }
}

#[pyclass(extends = KDTreeSearchParam, name = "KDTreeSearchParamRadius", module = "tiny3d.cpu.pybind.geometry")]
pub struct KDTreeSearchParamRadius;

#[pymethods]
impl KDTreeSearchParamRadius {
    #[new]
    fn new(radius: f64) -> (Self, KDTreeSearchParam) {
        (
            KDTreeSearchParamRadius,
            KDTreeSearchParam {
                param: KdTreeSearchParam::Radius { radius },
            },
        )
    }
    fn __repr__(slf: PyRef<'_, Self>) -> String {
        if let KdTreeSearchParam::Radius { radius } = slf.as_super().param {
            format!(
                "KDTreeSearchParamRadius(radius={})",
                crate::fmtutil::shortest(radius)
            )
        } else {
            unreachable!()
        }
    }
    #[getter]
    fn radius(slf: PyRef<'_, Self>) -> f64 {
        if let KdTreeSearchParam::Radius { radius } = slf.as_super().param {
            radius
        } else {
            0.0
        }
    }
    #[setter]
    fn set_radius(mut slf: PyRefMut<'_, Self>, v: f64) {
        slf.as_super().param = KdTreeSearchParam::Radius { radius: v };
    }
}

#[pyclass(extends = KDTreeSearchParam, name = "KDTreeSearchParamHybrid", module = "tiny3d.cpu.pybind.geometry")]
pub struct KDTreeSearchParamHybrid;

#[pymethods]
impl KDTreeSearchParamHybrid {
    #[new]
    fn new(radius: f64, max_nn: i32) -> (Self, KDTreeSearchParam) {
        (
            KDTreeSearchParamHybrid,
            KDTreeSearchParam {
                param: KdTreeSearchParam::Hybrid { radius, max_nn },
            },
        )
    }
    fn __repr__(slf: PyRef<'_, Self>) -> String {
        if let KdTreeSearchParam::Hybrid { radius, max_nn } = slf.as_super().param {
            format!(
                "KDTreeSearchParamHybrid(radius={}, max_nn={})",
                crate::fmtutil::shortest(radius),
                max_nn
            )
        } else {
            unreachable!()
        }
    }
    #[getter]
    fn radius(slf: PyRef<'_, Self>) -> f64 {
        if let KdTreeSearchParam::Hybrid { radius, .. } = slf.as_super().param {
            radius
        } else {
            0.0
        }
    }
    #[getter]
    fn max_nn(slf: PyRef<'_, Self>) -> i32 {
        if let KdTreeSearchParam::Hybrid { max_nn, .. } = slf.as_super().param {
            max_nn
        } else {
            0
        }
    }
}

// ---------------------------------------------------------------- KDTreeFlann

#[pyclass(name = "KDTreeFlann", module = "tiny3d.cpu.pybind.geometry")]
#[derive(Default)]
pub struct KDTreeFlann {
    pub tree: CoreKdTree,
}

impl KDTreeFlann {
    fn search_result(
        py: Python<'_>,
        k: i32,
        indices: Vec<i64>,
        dists: Vec<f64>,
    ) -> PyResult<Py<PyAny>> {
        let iv = IntVector {
            data: indices.iter().map(|&i| i as i32).collect(),
        };
        let dv = DoubleVector { data: dists };
        (k, iv, dv).into_py_any(py)
    }
}

#[pymethods]
impl KDTreeFlann {
    #[new]
    #[pyo3(signature = (data = None))]
    fn new(py: Python<'_>, data: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let mut t = KDTreeFlann::default();
        if let Some(d) = data {
            if let Ok(pcd) = d.extract::<PyRef<'_, PointCloud>>() {
                t.tree.set_points(&pcd.inner.points);
            } else if let Ok(feat) = d.extract::<PyRef<'_, crate::registration::Feature>>() {
                t.tree
                    .set_matrix_data(feat.inner.dim, feat.inner.num, feat.inner.data.clone());
            } else {
                // matrix data (dim x n)
                let _ = py;
                let arr: PyReadonlyArray2<f64> = d.extract()?;
                let view = arr.as_array();
                let dims = view.shape()[0];
                let n = view.shape()[1];
                let mut data = Vec::with_capacity(dims * n);
                for j in 0..n {
                    for i in 0..dims {
                        data.push(view[[i, j]]);
                    }
                }
                t.tree.set_matrix_data(dims, n, data);
            }
        }
        Ok(t)
    }

    fn set_geometry(&mut self, py: Python<'_>, geometry: PyRef<'_, PointCloud>) -> bool {
        let tree = &mut self.tree;
        let pts = &geometry.inner.points;
        py.allow_threads(|| tree.set_points(pts))
    }

    fn set_feature(&mut self, feature: PyRef<'_, crate::registration::Feature>) -> bool {
        self.tree.set_matrix_data(
            feature.inner.dim,
            feature.inner.num,
            feature.inner.data.clone(),
        )
    }

    fn set_matrix_data(&mut self, data: PyReadonlyArray2<f64>) -> bool {
        let view = data.as_array();
        let dims = view.shape()[0];
        let n = view.shape()[1];
        let mut d = Vec::with_capacity(dims * n);
        for j in 0..n {
            for i in 0..dims {
                d.push(view[[i, j]]);
            }
        }
        self.tree.set_matrix_data(dims, n, d)
    }

    fn search_knn_vector_3d(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        knn: i32,
    ) -> PyResult<Py<PyAny>> {
        let q = v3_from_any(query)?;
        let mut idx = Vec::new();
        let mut d2 = Vec::new();
        let k = self.tree.search_knn(&q, knn, &mut idx, &mut d2);
        Self::search_result(py, k, idx, d2)
    }

    fn search_radius_vector_3d(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        radius: f64,
    ) -> PyResult<Py<PyAny>> {
        let q = v3_from_any(query)?;
        let mut idx = Vec::new();
        let mut d2 = Vec::new();
        let k = self.tree.search_radius(&q, radius, &mut idx, &mut d2);
        Self::search_result(py, k, idx, d2)
    }

    fn search_hybrid_vector_3d(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        radius: f64,
        max_nn: i32,
    ) -> PyResult<Py<PyAny>> {
        let q = v3_from_any(query)?;
        let mut idx = Vec::new();
        let mut d2 = Vec::new();
        let k = self
            .tree
            .search_hybrid(&q, radius, max_nn, &mut idx, &mut d2);
        Self::search_result(py, k, idx, d2)
    }

    fn search_vector_3d(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        search_param: PyRef<'_, KDTreeSearchParam>,
    ) -> PyResult<Py<PyAny>> {
        let q = v3_from_any(query)?;
        let mut idx = Vec::new();
        let mut d2 = Vec::new();
        let k = self.tree.search(&q, &search_param.param, &mut idx, &mut d2);
        Self::search_result(py, k, idx, d2)
    }

    fn search_knn_vector_xd(
        &self,
        py: Python<'_>,
        query: PyReadonlyArray1<f64>,
        knn: i32,
    ) -> PyResult<Py<PyAny>> {
        let q = query.as_slice()?.to_vec();
        let mut idx = Vec::new();
        let mut d2 = Vec::new();
        let k = self.tree.search_knn(&q, knn, &mut idx, &mut d2);
        Self::search_result(py, k, idx, d2)
    }

    fn search_radius_vector_xd(
        &self,
        py: Python<'_>,
        query: PyReadonlyArray1<f64>,
        radius: f64,
    ) -> PyResult<Py<PyAny>> {
        let q = query.as_slice()?.to_vec();
        let mut idx = Vec::new();
        let mut d2 = Vec::new();
        let k = self.tree.search_radius(&q, radius, &mut idx, &mut d2);
        Self::search_result(py, k, idx, d2)
    }

    fn search_hybrid_vector_xd(
        &self,
        py: Python<'_>,
        query: PyReadonlyArray1<f64>,
        radius: f64,
        max_nn: i32,
    ) -> PyResult<Py<PyAny>> {
        let q = query.as_slice()?.to_vec();
        let mut idx = Vec::new();
        let mut d2 = Vec::new();
        let k = self
            .tree
            .search_hybrid(&q, radius, max_nn, &mut idx, &mut d2);
        Self::search_result(py, k, idx, d2)
    }

    fn search_vector_xd(
        &self,
        py: Python<'_>,
        query: PyReadonlyArray1<f64>,
        search_param: PyRef<'_, KDTreeSearchParam>,
    ) -> PyResult<Py<PyAny>> {
        let q = query.as_slice()?.to_vec();
        let mut idx = Vec::new();
        let mut d2 = Vec::new();
        let k = self.tree.search(&q, &search_param.param, &mut idx, &mut d2);
        Self::search_result(py, k, idx, d2)
    }
}

// ---------------------------------------------------------------- TriangleMesh

#[pyclass(extends = Geometry3D, name = "MeshBase", module = "tiny3d.cpu.pybind.geometry", subclass)]
pub struct MeshBase;

#[pymethods]
impl MeshBase {}

#[pyclass(extends = MeshBase, name = "TriangleMesh", module = "tiny3d.cpu.pybind.geometry")]
#[derive(Default)]
pub struct TriangleMesh {
    pub inner: cg::TriangleMesh,
}

pub fn mesh_init(inner: cg::TriangleMesh) -> PyClassInitializer<TriangleMesh> {
    PyClassInitializer::from(Geometry)
        .add_subclass(Geometry3D)
        .add_subclass(MeshBase)
        .add_subclass(TriangleMesh { inner })
}

#[pymethods]
impl TriangleMesh {
    #[new]
    #[pyo3(signature = (other = None))]
    fn new(other: Option<PyRef<'_, TriangleMesh>>) -> PyClassInitializer<TriangleMesh> {
        match other {
            Some(o) => mesh_init(o.inner.clone()),
            None => mesh_init(cg::TriangleMesh::new()),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TriangleMesh with {} vertices and {} triangles.",
            self.inner.vertices.len(),
            self.inner.triangles.len()
        )
    }

    fn __copy__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<TriangleMesh>> {
        Py::new(py, mesh_init(slf.inner.clone()))
    }
    fn __deepcopy__(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        _memo: &Bound<'_, PyAny>,
    ) -> PyResult<Py<TriangleMesh>> {
        Py::new(py, mesh_init(slf.inner.clone()))
    }

    #[getter(vertices)]
    fn vertices_getter(slf: PyRef<'_, Self>) -> Vector3dVector {
        let owner: Py<TriangleMesh> = slf.into();
        Vector3dVector {
            data: Vec::new(),
            owner: Some((owner.into_any(), VT_MESH_VERTICES)),
        }
    }
    #[setter(vertices)]
    fn vertices_setter(&mut self, py: Python<'_>, v: &Bound<'_, PyAny>) -> PyResult<()> {
        Vector3dVector::assign_into(py, v, &mut self.inner.vertices)?;
        Ok(())
    }
    #[getter(vertex_normals)]
    fn vertex_normals_getter(slf: PyRef<'_, Self>) -> Vector3dVector {
        let owner: Py<TriangleMesh> = slf.into();
        Vector3dVector {
            data: Vec::new(),
            owner: Some((owner.into_any(), VT_MESH_VERTEX_NORMALS)),
        }
    }
    #[setter(vertex_normals)]
    fn vertex_normals_setter(&mut self, py: Python<'_>, v: &Bound<'_, PyAny>) -> PyResult<()> {
        Vector3dVector::assign_into(py, v, &mut self.inner.vertex_normals)?;
        Ok(())
    }
    #[getter(vertex_colors)]
    fn vertex_colors_getter(slf: PyRef<'_, Self>) -> Vector3dVector {
        let owner: Py<TriangleMesh> = slf.into();
        Vector3dVector {
            data: Vec::new(),
            owner: Some((owner.into_any(), VT_MESH_VERTEX_COLORS)),
        }
    }
    #[setter(vertex_colors)]
    fn vertex_colors_setter(&mut self, py: Python<'_>, v: &Bound<'_, PyAny>) -> PyResult<()> {
        Vector3dVector::assign_into(py, v, &mut self.inner.vertex_colors)?;
        Ok(())
    }
    #[getter(triangles)]
    fn triangles_getter(slf: PyRef<'_, Self>) -> Vector3iVector {
        let owner: Py<TriangleMesh> = slf.into();
        Vector3iVector {
            data: Vec::new(),
            owner: Some((owner.into_any(), VT_MESH_TRIANGLES)),
        }
    }
    #[setter(triangles)]
    fn triangles_setter(&mut self, py: Python<'_>, v: &Bound<'_, PyAny>) -> PyResult<()> {
        Vector3iVector::assign_into(py, v, &mut self.inner.triangles)?;
        Ok(())
    }
    #[getter(triangle_normals)]
    fn triangle_normals_getter(slf: PyRef<'_, Self>) -> Vector3dVector {
        let owner: Py<TriangleMesh> = slf.into();
        Vector3dVector {
            data: Vec::new(),
            owner: Some((owner.into_any(), VT_MESH_TRIANGLE_NORMALS)),
        }
    }
    #[setter(triangle_normals)]
    fn triangle_normals_setter(&mut self, py: Python<'_>, v: &Bound<'_, PyAny>) -> PyResult<()> {
        Vector3dVector::assign_into(py, v, &mut self.inner.triangle_normals)?;
        Ok(())
    }

    fn clear(mut slf: PyRefMut<'_, Self>) -> Py<Self> {
        slf.inner.clear();
        slf.into()
    }
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    fn dimension(&self) -> i32 {
        3
    }
    fn get_geometry_type(&self) -> GeometryType {
        GT_TRIANGLEMESH
    }
    fn has_vertices(&self) -> bool {
        self.inner.has_vertices()
    }
    fn has_vertex_normals(&self) -> bool {
        self.inner.has_vertex_normals()
    }
    fn has_vertex_colors(&self) -> bool {
        self.inner.has_vertex_colors()
    }
    fn has_triangles(&self) -> bool {
        self.inner.has_triangles()
    }
    fn has_triangle_normals(&self) -> bool {
        self.inner.has_triangle_normals()
    }
    fn get_min_bound(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        v3_to_numpy(py, self.inner.get_min_bound())
    }
    fn get_max_bound(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        v3_to_numpy(py, self.inner.get_max_bound())
    }
    fn get_center(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        v3_to_numpy(py, self.inner.get_center())
    }
    fn get_axis_aligned_bounding_box(
        &self,
        py: Python<'_>,
    ) -> PyResult<Py<AxisAlignedBoundingBox>> {
        Py::new(
            py,
            aabb_init(cg::AxisAlignedBoundingBox::new(
                self.inner.get_min_bound(),
                self.inner.get_max_bound(),
            )),
        )
    }
    fn transform(
        mut slf: PyRefMut<'_, Self>,
        transformation: &Bound<'_, PyAny>,
    ) -> PyResult<Py<Self>> {
        let t = m4_from_any(transformation)?;
        slf.inner.transform(&t);
        Ok(slf.into())
    }
    #[pyo3(signature = (translation, relative = true))]
    fn translate(
        mut slf: PyRefMut<'_, Self>,
        translation: &Bound<'_, PyAny>,
        relative: bool,
    ) -> PyResult<Py<Self>> {
        let t = v3_from_any(translation)?;
        slf.inner.translate(t, relative);
        Ok(slf.into())
    }
    fn scale(
        mut slf: PyRefMut<'_, Self>,
        scale: f64,
        center: &Bound<'_, PyAny>,
    ) -> PyResult<Py<Self>> {
        let c = v3_from_any(center)?;
        slf.inner.scale(scale, c);
        Ok(slf.into())
    }
    #[pyo3(signature = (r, center = None))]
    fn rotate(
        mut slf: PyRefMut<'_, Self>,
        r: &Bound<'_, PyAny>,
        center: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<Self>> {
        let rm = m3_from_any(r)?;
        let c = match center {
            Some(c) => v3_from_any(c)?,
            None => slf.inner.get_center(),
        };
        slf.inner.rotate(&rm, c);
        Ok(slf.into())
    }
    fn normalize_normals(mut slf: PyRefMut<'_, Self>) -> Py<Self> {
        slf.inner.normalize_normals();
        slf.into()
    }
    fn paint_uniform_color(
        mut slf: PyRefMut<'_, Self>,
        color: &Bound<'_, PyAny>,
    ) -> PyResult<Py<Self>> {
        let c = v3_from_any(color)?;
        slf.inner.paint_uniform_color(c);
        Ok(slf.into())
    }
    #[pyo3(signature = (normalized = true))]
    fn compute_triangle_normals(mut slf: PyRefMut<'_, Self>, normalized: bool) -> Py<Self> {
        slf.inner.compute_triangle_normals(normalized);
        slf.into()
    }
    #[pyo3(signature = (normalized = true))]
    fn compute_vertex_normals(mut slf: PyRefMut<'_, Self>, normalized: bool) -> Py<Self> {
        slf.inner.compute_vertex_normals(normalized);
        slf.into()
    }
}

// ---------------------------------------------------------------- VoxelGrid

#[pyclass(name = "Voxel", module = "tiny3d.cpu.pybind.geometry")]
#[derive(Clone, Default)]
pub struct Voxel {
    pub inner: cg::Voxel,
}

#[pymethods]
impl Voxel {
    #[new]
    #[pyo3(signature = (grid_index = None, color = None))]
    fn new(
        grid_index: Option<&Bound<'_, PyAny>>,
        color: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut v = cg::Voxel::default();
        if let Some(gi) = grid_index {
            let g: Vec<i32> = if let Ok(arr) = gi.extract::<PyReadonlyArray1<i32>>() {
                arr.as_slice()?.to_vec()
            } else {
                gi.extract()?
            };
            if g.len() == 3 {
                v.grid_index = [g[0], g[1], g[2]];
            }
        }
        if let Some(c) = color {
            v.color = v3_from_any(c)?;
        }
        Ok(Voxel { inner: v })
    }

    fn __repr__(&self) -> String {
        format!(
            "Voxel(grid_index={} {} {}, color={} {} {})",
            self.inner.grid_index[0],
            self.inner.grid_index[1],
            self.inner.grid_index[2],
            ostream_double(self.inner.color[0]),
            ostream_double(self.inner.color[1]),
            ostream_double(self.inner.color[2]),
        )
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }
    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    #[getter(grid_index)]
    fn grid_index_getter(&self, py: Python<'_>) -> Py<PyArray1<i32>> {
        PyArray1::from_slice(py, &self.inner.grid_index).unbind()
    }
    #[setter(grid_index)]
    fn grid_index_setter(&mut self, v: Vec<i32>) -> PyResult<()> {
        if v.len() != 3 {
            return Err(PyTypeError::new_err("expected 3-vector"));
        }
        self.inner.grid_index = [v[0], v[1], v[2]];
        Ok(())
    }
    #[getter(color)]
    fn color_getter(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        v3_to_numpy(py, self.inner.color)
    }
    #[setter(color)]
    fn color_setter(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.color = v3_from_any(v)?;
        Ok(())
    }
}

#[pyclass(extends = Geometry3D, name = "VoxelGrid", module = "tiny3d.cpu.pybind.geometry")]
#[derive(Default)]
pub struct VoxelGrid {
    pub inner: cg::VoxelGrid,
}

pub fn voxelgrid_init(inner: cg::VoxelGrid) -> PyClassInitializer<VoxelGrid> {
    PyClassInitializer::from(Geometry)
        .add_subclass(Geometry3D)
        .add_subclass(VoxelGrid { inner })
}

#[pymethods]
impl VoxelGrid {
    #[new]
    fn new() -> PyClassInitializer<VoxelGrid> {
        voxelgrid_init(cg::VoxelGrid::new())
    }

    fn __repr__(&self) -> String {
        format!("VoxelGrid with {} voxels.", self.inner.voxels.len())
    }

    #[getter(origin)]
    fn origin_getter(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        v3_to_numpy(py, self.inner.origin)
    }
    #[setter(origin)]
    fn origin_setter(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.origin = v3_from_any(v)?;
        Ok(())
    }
    #[getter(voxel_size)]
    fn voxel_size_getter(&self) -> f64 {
        self.inner.voxel_size
    }
    #[setter(voxel_size)]
    fn voxel_size_setter(&mut self, v: f64) {
        self.inner.voxel_size = v;
    }

    fn clear(mut slf: PyRefMut<'_, Self>) -> Py<Self> {
        slf.inner.clear();
        slf.into()
    }
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    fn dimension(&self) -> i32 {
        3
    }
    fn get_geometry_type(&self) -> GeometryType {
        GT_VOXELGRID
    }
    fn has_voxels(&self) -> bool {
        self.inner.has_voxels()
    }
    fn has_colors(&self) -> bool {
        true
    }
    fn get_min_bound(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        v3_to_numpy(py, self.inner.get_min_bound())
    }
    fn get_max_bound(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        v3_to_numpy(py, self.inner.get_max_bound())
    }
    fn get_center(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        v3_to_numpy(py, self.inner.get_center())
    }
    fn get_axis_aligned_bounding_box(
        &self,
        py: Python<'_>,
    ) -> PyResult<Py<AxisAlignedBoundingBox>> {
        Py::new(py, aabb_init(self.inner.get_axis_aligned_bounding_box()))
    }
    fn get_voxels(&self) -> Vec<Voxel> {
        self.inner
            .get_voxels()
            .into_iter()
            .map(|v| Voxel { inner: v })
            .collect()
    }
    fn transform(
        mut slf: PyRefMut<'_, Self>,
        transformation: &Bound<'_, PyAny>,
    ) -> PyResult<Py<Self>> {
        let t = m4_from_any(transformation)?;
        slf.inner.transform(&t).map_err(PyRuntimeError::new_err)?;
        Ok(slf.into())
    }
    #[pyo3(signature = (translation, relative = true))]
    fn translate(
        mut slf: PyRefMut<'_, Self>,
        translation: &Bound<'_, PyAny>,
        relative: bool,
    ) -> PyResult<Py<Self>> {
        let t = v3_from_any(translation)?;
        slf.inner.translate(t, relative);
        Ok(slf.into())
    }
    fn scale(
        mut slf: PyRefMut<'_, Self>,
        scale: f64,
        center: &Bound<'_, PyAny>,
    ) -> PyResult<Py<Self>> {
        let c = v3_from_any(center)?;
        slf.inner.scale(scale, c);
        Ok(slf.into())
    }
    #[pyo3(signature = (r, center = None))]
    fn rotate(
        mut slf: PyRefMut<'_, Self>,
        r: &Bound<'_, PyAny>,
        center: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<Self>> {
        let rm = m3_from_any(r)?;
        let c = match center {
            Some(c) => v3_from_any(c)?,
            None => slf.inner.get_center(),
        };
        slf.inner.rotate(&rm, c).map_err(PyRuntimeError::new_err)?;
        Ok(slf.into())
    }
    #[staticmethod]
    fn create_from_point_cloud(
        py: Python<'_>,
        input: PyRef<'_, PointCloud>,
        voxel_size: f64,
    ) -> PyResult<Py<VoxelGrid>> {
        match cg::VoxelGrid::create_from_point_cloud(&input.inner, voxel_size) {
            Ok(vg) => Py::new(py, voxelgrid_init(vg)),
            Err(e) => Err(PyRuntimeError::new_err(e)),
        }
    }
    #[staticmethod]
    fn create_from_point_cloud_within_bounds(
        py: Python<'_>,
        input: PyRef<'_, PointCloud>,
        voxel_size: f64,
        min_bound: &Bound<'_, PyAny>,
        max_bound: &Bound<'_, PyAny>,
    ) -> PyResult<Py<VoxelGrid>> {
        let mn = v3_from_any(min_bound)?;
        let mx = v3_from_any(max_bound)?;
        match cg::VoxelGrid::create_from_point_cloud_within_bounds(&input.inner, voxel_size, mn, mx)
        {
            Ok(vg) => Py::new(py, voxelgrid_init(vg)),
            Err(e) => Err(PyRuntimeError::new_err(e)),
        }
    }
}

// ---------------------------------------------------------------- module

pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GeometryType>()?;
    m.add_class::<Geometry>()?;
    m.add_class::<Geometry3D>()?;
    m.add_class::<Geometry2D>()?;
    m.add_class::<PointCloud>()?;
    m.add_class::<AxisAlignedBoundingBox>()?;
    m.add_class::<KDTreeSearchParam>()?;
    m.add_class::<KDTreeSearchParamKNN>()?;
    m.add_class::<KDTreeSearchParamRadius>()?;
    m.add_class::<KDTreeSearchParamHybrid>()?;
    m.add_class::<KDTreeFlann>()?;
    m.add_class::<MeshBase>()?;
    m.add_class::<TriangleMesh>()?;
    m.add_class::<Voxel>()?;
    m.add_class::<VoxelGrid>()?;

    // module-level rotation helpers
    let g3d = py.get_type::<Geometry3D>();
    for name in [
        "get_rotation_matrix_from_xyz",
        "get_rotation_matrix_from_yzx",
        "get_rotation_matrix_from_zxy",
        "get_rotation_matrix_from_xzy",
        "get_rotation_matrix_from_zyx",
        "get_rotation_matrix_from_yxz",
        "get_rotation_matrix_from_axis_angle",
        "get_rotation_matrix_from_quaternion",
    ] {
        m.add(name, g3d.getattr(name)?)?;
    }
    Ok(())
}
