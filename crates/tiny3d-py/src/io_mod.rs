//! tiny3d.cpu.pybind.io

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use tiny3d_core::io as cio;

use crate::geometry::{mesh_init, pcd_init_pub, PointCloud, TriangleMesh};

#[pyclass(
    name = "FileGeometry",
    module = "tiny3d.cpu.pybind.io",
    frozen,
    eq,
    hash
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FileGeometry {
    #[pyo3(get)]
    pub value: i32,
    pub name_str: &'static str,
}

#[pymethods]
impl FileGeometry {
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
        format!("<FileGeometry.{}: {}>", self.name_str, self.value)
    }
    #[classattr]
    const CONTENTS_UNKNOWN: FileGeometry = FG_UNKNOWN;
    #[classattr]
    const CONTAINS_POINTS: FileGeometry = FG_POINTS;
    #[classattr]
    const CONTAINS_LINES: FileGeometry = FG_LINES;
    #[classattr]
    const CONTAINS_TRIANGLES: FileGeometry = FG_TRIANGLES;
}

pub const FG_UNKNOWN: FileGeometry = FileGeometry {
    value: 0,
    name_str: "CONTENTS_UNKNOWN",
};
pub const FG_POINTS: FileGeometry = FileGeometry {
    value: 1,
    name_str: "CONTAINS_POINTS",
};
pub const FG_LINES: FileGeometry = FileGeometry {
    value: 2,
    name_str: "CONTAINS_LINES",
};
pub const FG_TRIANGLES: FileGeometry = FileGeometry {
    value: 4,
    name_str: "CONTAINS_TRIANGLES",
};

fn fg_from_core(fg: cio::FileGeometry) -> FileGeometry {
    match fg {
        cio::FileGeometry::ContentsUnknown => FG_UNKNOWN,
        cio::FileGeometry::ContainsPoints => FG_POINTS,
        cio::FileGeometry::ContainsLines => FG_LINES,
        cio::FileGeometry::ContainsTriangles => FG_TRIANGLES,
    }
}

fn path_from_any(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(s) = obj.extract::<String>() {
        return Ok(s);
    }
    let os = obj.py().import("os")?;
    os.getattr("fspath")?.call1((obj,))?.extract()
}

#[pyfunction]
fn read_file_geometry_type(filename: &Bound<'_, PyAny>) -> PyResult<FileGeometry> {
    Ok(fg_from_core(cio::read_file_geometry_type(&path_from_any(
        filename,
    )?)))
}

#[pyfunction]
#[pyo3(signature = (filename, format = "auto", remove_nan_points = false, remove_infinite_points = false, print_progress = false))]
fn read_point_cloud(
    py: Python<'_>,
    filename: &Bound<'_, PyAny>,
    format: &str,
    remove_nan_points: bool,
    remove_infinite_points: bool,
    print_progress: bool,
) -> PyResult<Py<PointCloud>> {
    let _ = print_progress;
    let opts = cio::ReadPointCloudOptions {
        remove_nan_points,
        remove_infinite_points,
    };
    let path = path_from_any(filename)?;
    let pcd = py.allow_threads(|| cio::read_point_cloud(&path, format, &opts));
    Py::new(py, pcd_init_pub(pcd))
}

#[pyfunction]
#[pyo3(signature = (bytes, format = "auto", remove_nan_points = false, remove_infinite_points = false, print_progress = false))]
fn read_point_cloud_from_bytes(
    py: Python<'_>,
    bytes: &[u8],
    format: &str,
    remove_nan_points: bool,
    remove_infinite_points: bool,
    print_progress: bool,
) -> PyResult<Py<PointCloud>> {
    let _ = print_progress;
    let opts = cio::ReadPointCloudOptions {
        remove_nan_points,
        remove_infinite_points,
    };
    let pcd = cio::read_point_cloud_from_bytes(bytes, format, &opts);
    Py::new(py, pcd_init_pub(pcd))
}

#[pyfunction]
#[pyo3(signature = (filename, pointcloud, format = "auto", write_ascii = false, compressed = false, print_progress = false))]
fn write_point_cloud(
    filename: &Bound<'_, PyAny>,
    pointcloud: PyRef<'_, PointCloud>,
    format: &str,
    write_ascii: bool,
    compressed: bool,
    print_progress: bool,
) -> PyResult<bool> {
    let _ = (compressed, print_progress);
    // C++ WritePointCloud(file) keys the writer off the file extension only.
    let _ = format;
    let path = path_from_any(filename)?;
    let inner = &pointcloud.inner;
    let py = pointcloud.py();
    Ok(py.allow_threads(|| cio::write_point_cloud(&path, inner, "auto", write_ascii)))
}

#[pyfunction]
#[pyo3(signature = (pointcloud, format = "auto", write_ascii = false, compressed = false, print_progress = false))]
fn write_point_cloud_to_bytes(
    py: Python<'_>,
    pointcloud: PyRef<'_, PointCloud>,
    format: &str,
    write_ascii: bool,
    compressed: bool,
    print_progress: bool,
) -> PyResult<Py<PyBytes>> {
    let _ = (compressed, print_progress, write_ascii);
    let out = cio::write_point_cloud_to_bytes(&pointcloud.inner, format, write_ascii)
        .ok_or_else(|| PyValueError::new_err(format!("unsupported byte format: {format}")))?;
    Ok(PyBytes::new(py, &out).unbind())
}

#[pyfunction]
#[pyo3(signature = (filename, enable_post_processing = false, print_progress = false))]
fn read_triangle_mesh(
    py: Python<'_>,
    filename: &Bound<'_, PyAny>,
    enable_post_processing: bool,
    print_progress: bool,
) -> PyResult<Py<TriangleMesh>> {
    let _ = (enable_post_processing, print_progress);
    let mesh = cio::read_triangle_mesh(&path_from_any(filename)?);
    Py::new(py, mesh_init(mesh))
}

#[pyfunction]
#[pyo3(signature = (filename, mesh, write_ascii = false, compressed = false, write_vertex_normals = true, write_vertex_colors = false, write_triangle_uvs = false, print_progress = false))]
#[allow(clippy::too_many_arguments)]
fn write_triangle_mesh(
    filename: &Bound<'_, PyAny>,
    mesh: PyRef<'_, TriangleMesh>,
    write_ascii: bool,
    compressed: bool,
    write_vertex_normals: bool,
    write_vertex_colors: bool,
    write_triangle_uvs: bool,
    print_progress: bool,
) -> PyResult<bool> {
    let _ = (compressed, write_triangle_uvs, print_progress);
    Ok(cio::write_triangle_mesh(
        &path_from_any(filename)?,
        &mesh.inner,
        write_ascii,
        write_vertex_normals,
        write_vertex_colors,
    ))
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FileGeometry>()?;
    m.add("CONTENTS_UNKNOWN", FG_UNKNOWN)?;
    m.add("CONTAINS_POINTS", FG_POINTS)?;
    m.add("CONTAINS_LINES", FG_LINES)?;
    m.add("CONTAINS_TRIANGLES", FG_TRIANGLES)?;
    m.add_function(wrap_pyfunction!(read_file_geometry_type, m)?)?;
    m.add_function(wrap_pyfunction!(read_point_cloud, m)?)?;
    m.add_function(wrap_pyfunction!(read_point_cloud_from_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(write_point_cloud, m)?)?;
    m.add_function(wrap_pyfunction!(write_point_cloud_to_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(read_triangle_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(write_triangle_mesh, m)?)?;
    Ok(())
}
