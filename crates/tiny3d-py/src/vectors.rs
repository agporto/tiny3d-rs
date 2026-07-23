//! pybind-style bound std::vector types (Vector3dVector & friends).
//!
//! Like the pybind11 build, vectors returned from geometry properties
//! (`pcd.points`, `mesh.triangles`, ...) are LIVE REFERENCES: element access,
//! mutation, `append`, and `np.asarray()` all operate on the owning
//! geometry's storage (np.asarray shares memory, with the owner as the numpy
//! base object). Free-standing vectors own their data; `np.asarray` on them
//! shares memory with the vector object itself.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PySequence;

/// View-target codes (interpreted by `ViewAccess` impls in geometry.rs).
pub const VT_PCD_POINTS: u8 = 0;
pub const VT_PCD_NORMALS: u8 = 1;
pub const VT_PCD_COLORS: u8 = 2;
pub const VT_MESH_VERTICES: u8 = 3;
pub const VT_MESH_VERTEX_NORMALS: u8 = 4;
pub const VT_MESH_VERTEX_COLORS: u8 = 5;
pub const VT_MESH_TRIANGLE_NORMALS: u8 = 6;
pub const VT_MESH_TRIANGLES: u8 = 10;

/// Element types that can act as views onto geometry storage.
pub trait ViewAccess: Copy + Sized + 'static {
    fn with_view<R>(
        py: Python<'_>,
        owner: &Py<PyAny>,
        target: u8,
        f: impl FnOnce(&Vec<Self>) -> R,
    ) -> PyResult<R>;
    fn with_view_mut<R>(
        py: Python<'_>,
        owner: &Py<PyAny>,
        target: u8,
        f: impl FnOnce(&mut Vec<Self>) -> R,
    ) -> PyResult<R>;
}

macro_rules! no_view_impl {
    ($t:ty) => {
        impl ViewAccess for $t {
            fn with_view<R>(
                _py: Python<'_>,
                _o: &Py<PyAny>,
                _t: u8,
                _f: impl FnOnce(&Vec<Self>) -> R,
            ) -> PyResult<R> {
                Err(PyRuntimeError::new_err(
                    "internal error: not a viewable vector type",
                ))
            }
            fn with_view_mut<R>(
                _py: Python<'_>,
                _o: &Py<PyAny>,
                _t: u8,
                _f: impl FnOnce(&mut Vec<Self>) -> R,
            ) -> PyResult<R> {
                Err(PyRuntimeError::new_err(
                    "internal error: not a viewable vector type",
                ))
            }
        }
    };
}

no_view_impl!([f64; 2]);
no_view_impl!([i32; 2]);
no_view_impl!([i32; 4]);

/// Build a shared-memory (n, dim) numpy array over `ptr`, with `container`
/// as the base object keeping the owner alive. Same aliasing contract as the
/// pybind build: the buffer must not be reallocated while the array is used.
pub unsafe fn shared_array2<'py, T: numpy::Element>(
    py: Python<'py>,
    ptr: *mut T,
    n: usize,
    dim: usize,
    container: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let view = numpy::ndarray::ArrayViewMut2::from_shape_ptr((n, dim), ptr);
    let arr = PyArray2::<T>::borrow_from_array(&view, container.clone());
    let _ = py;
    Ok(arr.into_any())
}

macro_rules! eigen_vec_class {
    ($name:ident, $pyname:literal, $scalar:ty, $dim:expr, $reprname:literal) => {
        #[pyclass(name = $pyname, module = "tiny3d.cpu.pybind.utility")]
        #[derive(Default)]
        pub struct $name {
            pub data: Vec<[$scalar; $dim]>,
            /// When set, this vector is a live reference into the owner's
            /// storage (pybind reference semantics); `data` is unused.
            pub owner: Option<(Py<PyAny>, u8)>,
        }

        impl $name {
            #[inline]
            pub fn read<R>(
                &self,
                py: Python<'_>,
                f: impl FnOnce(&Vec<[$scalar; $dim]>) -> R,
            ) -> PyResult<R> {
                match &self.owner {
                    None => Ok(f(&self.data)),
                    Some((o, t)) => <[$scalar; $dim] as ViewAccess>::with_view(py, o, *t, f),
                }
            }

            #[inline]
            pub fn write<R>(
                &mut self,
                py: Python<'_>,
                f: impl FnOnce(&mut Vec<[$scalar; $dim]>) -> R,
            ) -> PyResult<R> {
                match &self.owner {
                    None => Ok(f(&mut self.data)),
                    Some((o, t)) => <[$scalar; $dim] as ViewAccess>::with_view_mut(py, o, *t, f),
                }
            }

            /// Snapshot the current contents (view or owned) as a Vec.
            pub fn snapshot(&self, py: Python<'_>) -> PyResult<Vec<[$scalar; $dim]>> {
                self.read(py, |d| d.clone())
            }

            /// Assign `obj`'s contents into `dest` with vector-assign
            /// semantics (clear + extend: reuses the existing buffer when
            /// capacity suffices) and a single data copy.
            pub fn assign_into(
                py: Python<'_>,
                obj: &Bound<'_, PyAny>,
                dest: &mut Vec<[$scalar; $dim]>,
            ) -> PyResult<()> {
                if let Ok(cell) = obj.downcast::<Self>() {
                    let v = cell.borrow();
                    return v.read(py, |d| {
                        dest.clear();
                        dest.extend_from_slice(d);
                    });
                }
                if let Ok(arr) = obj.downcast::<PyArray2<$scalar>>() {
                    let ro = arr.readonly();
                    let view = ro.as_array();
                    if view.shape()[1] != $dim {
                        return Err(PyTypeError::new_err("invalid array shape"));
                    }
                    let n = view.shape()[0];
                    if let Some(slice) = view.as_slice() {
                        dest.clear();
                        dest.reserve(n);
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                slice.as_ptr(),
                                dest.as_mut_ptr() as *mut $scalar,
                                n * $dim,
                            );
                            dest.set_len(n);
                        }
                        return Ok(());
                    }
                    dest.clear();
                    dest.reserve(n);
                    for row in view.outer_iter() {
                        let mut a = [<$scalar>::default(); $dim];
                        for (k, x) in row.iter().enumerate() {
                            a[k] = *x;
                        }
                        dest.push(a);
                    }
                    return Ok(());
                }
                // generic path: convert once, then move
                let v = Self::extract_from(py, obj)?;
                dest.clear();
                dest.extend_from_slice(&v.data);
                Ok(())
            }

            pub fn extract_from(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Self> {
                if let Ok(cell) = obj.downcast::<Self>() {
                    let v = cell.borrow();
                    return Ok(Self {
                        data: v.snapshot(py)?,
                        owner: None,
                    });
                }
                // numpy array path
                if let Ok(arr) = obj.downcast::<PyArray2<$scalar>>() {
                    let ro = arr.readonly();
                    let view = ro.as_array();
                    if view.shape()[1] != $dim {
                        return Err(PyTypeError::new_err("invalid array shape"));
                    }
                    let n = view.shape()[0];
                    if let Some(slice) = view.as_slice() {
                        // C-contiguous: single bulk copy
                        let mut data: Vec<[$scalar; $dim]> = Vec::with_capacity(n);
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                slice.as_ptr(),
                                data.as_mut_ptr() as *mut $scalar,
                                n * $dim,
                            );
                            data.set_len(n);
                        }
                        return Ok(Self { data, owner: None });
                    }
                    let mut data = Vec::with_capacity(n);
                    for row in view.outer_iter() {
                        let mut a = [<$scalar>::default(); $dim];
                        for (k, x) in row.iter().enumerate() {
                            a[k] = *x;
                        }
                        data.push(a);
                    }
                    return Ok(Self { data, owner: None });
                }
                // generic: cast via numpy
                let np = PyModule::import(obj.py(), "numpy")?;
                let arr = np.getattr("asarray")?.call1((obj,))?;
                let arr = arr.call_method1("astype", (numpy_dtype::<$scalar>(),))?;
                let arr = arr.downcast::<PyArray2<$scalar>>().map_err(|_| {
                    PyTypeError::new_err("cannot convert argument to a (n, k) array")
                })?;
                let ro = arr.readonly();
                let view = ro.as_array();
                if view.shape()[1] != $dim {
                    return Err(PyTypeError::new_err("invalid array shape"));
                }
                let mut data = Vec::with_capacity(view.shape()[0]);
                for row in view.outer_iter() {
                    let mut a = [<$scalar>::default(); $dim];
                    for (k, x) in row.iter().enumerate() {
                        a[k] = *x;
                    }
                    data.push(a);
                }
                Ok(Self { data, owner: None })
            }
        }

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (arg = None))]
            fn new(py: Python<'_>, arg: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
                match arg {
                    None => Ok(Self::default()),
                    Some(a) => Self::extract_from(py, a),
                }
            }

            fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
                self.read(py, |d| d.len())
            }

            fn __getitem__(&self, py: Python<'_>, idx: isize) -> PyResult<Py<PyArray1<$scalar>>> {
                let row = self.read(py, |d| {
                    let n = d.len() as isize;
                    let i = if idx < 0 { idx + n } else { idx };
                    if i < 0 || i >= n {
                        None
                    } else {
                        Some(d[i as usize])
                    }
                })?;
                match row {
                    Some(r) => Ok(PyArray1::from_slice(py, &r).unbind()),
                    None => Err(PyIndexError::new_err("list index out of range")),
                }
            }

            fn __setitem__(
                &mut self,
                py: Python<'_>,
                idx: isize,
                value: Vec<$scalar>,
            ) -> PyResult<()> {
                if value.len() != $dim {
                    return Err(PyTypeError::new_err("invalid element size"));
                }
                let mut arr = [<$scalar>::default(); $dim];
                arr.copy_from_slice(&value);
                let ok = self.write(py, |d| {
                    let n = d.len() as isize;
                    let i = if idx < 0 { idx + n } else { idx };
                    if i < 0 || i >= n {
                        false
                    } else {
                        d[i as usize] = arr;
                        true
                    }
                })?;
                if ok {
                    Ok(())
                } else {
                    Err(PyIndexError::new_err("list assignment index out of range"))
                }
            }

            fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
                let n = self.read(py, |d| d.len())?;
                Ok(format!(
                    "{} with {} elements.\nUse numpy.asarray() to access data.",
                    $reprname, n
                ))
            }

            fn __copy__(&self, py: Python<'_>) -> PyResult<Self> {
                Ok(Self {
                    data: self.snapshot(py)?,
                    owner: None,
                })
            }
            fn __deepcopy__(&self, py: Python<'_>, _memo: &Bound<'_, PyAny>) -> PyResult<Self> {
                Ok(Self {
                    data: self.snapshot(py)?,
                    owner: None,
                })
            }

            fn append(&mut self, py: Python<'_>, value: Vec<$scalar>) -> PyResult<()> {
                if value.len() != $dim {
                    return Err(PyTypeError::new_err("invalid element size"));
                }
                let mut arr = [<$scalar>::default(); $dim];
                arr.copy_from_slice(&value);
                self.write(py, |d| d.push(arr))
            }

            fn clear(&mut self, py: Python<'_>) -> PyResult<()> {
                self.write(py, |d| d.clear())
            }

            fn extend(&mut self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<()> {
                let o = Self::extract_from(py, other)?;
                self.write(py, move |d| d.extend(o.data))
            }

            fn insert(&mut self, py: Python<'_>, idx: isize, value: Vec<$scalar>) -> PyResult<()> {
                if value.len() != $dim {
                    return Err(PyTypeError::new_err("invalid element size"));
                }
                let mut arr = [<$scalar>::default(); $dim];
                arr.copy_from_slice(&value);
                self.write(py, |d| {
                    let n = d.len() as isize;
                    let i = if idx < 0 {
                        (idx + n).max(0)
                    } else {
                        idx.min(n)
                    };
                    d.insert(i as usize, arr);
                })
            }

            #[pyo3(signature = (idx = None))]
            fn pop(
                &mut self,
                py: Python<'_>,
                idx: Option<isize>,
            ) -> PyResult<Py<PyArray1<$scalar>>> {
                let v = self.write(py, |d| {
                    let n = d.len() as isize;
                    let i = match idx {
                        None => n - 1,
                        Some(v) if v < 0 => v + n,
                        Some(v) => v,
                    };
                    if i < 0 || i >= n {
                        None
                    } else {
                        Some(d.remove(i as usize))
                    }
                })?;
                match v {
                    Some(v) => Ok(PyArray1::from_slice(py, &v).unbind()),
                    None => Err(PyIndexError::new_err("pop index out of range")),
                }
            }

            fn count(&self, py: Python<'_>, value: Vec<$scalar>) -> PyResult<usize> {
                if value.len() != $dim {
                    return Ok(0);
                }
                self.read(py, |d| {
                    d.iter()
                        .filter(|x| x.iter().zip(value.iter()).all(|(a, b)| a == b))
                        .count()
                })
            }

            fn remove(&mut self, py: Python<'_>, value: Vec<$scalar>) -> PyResult<()> {
                let removed = self.write(py, |d| {
                    if let Some(pos) = d
                        .iter()
                        .position(|x| x.iter().zip(value.iter()).all(|(a, b)| a == b))
                    {
                        d.remove(pos);
                        true
                    } else {
                        false
                    }
                })?;
                if removed {
                    Ok(())
                } else {
                    Err(PyTypeError::new_err("value not found"))
                }
            }

            fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
                if let Ok(cell) = other.downcast::<Self>() {
                    let o = cell.borrow();
                    let a = self.snapshot(py)?;
                    let b = o.snapshot(py)?;
                    Ok(a == b)
                } else {
                    Ok(false)
                }
            }

            /// Shared-memory ndarray over the underlying storage (pybind
            /// semantics): mutations write through; the owning geometry (or
            /// this vector, when free-standing) is the numpy base object.
            #[pyo3(signature = (dtype = None, copy = None))]
            fn __array__(
                slf: &Bound<'_, Self>,
                py: Python<'_>,
                dtype: Option<&Bound<'_, PyAny>>,
                copy: Option<bool>,
            ) -> PyResult<Py<PyAny>> {
                let _ = copy;
                let cell = slf.borrow();
                let arr = match &cell.owner {
                    None => {
                        let n = cell.data.len();
                        let ptr = cell.data.as_ptr() as *mut $scalar;
                        drop(cell);
                        unsafe { shared_array2::<$scalar>(py, ptr, n, $dim, slf.as_any())? }
                    }
                    Some((o, t)) => {
                        let (ptr, n) =
                            <[$scalar; $dim] as ViewAccess>::with_view(py, o, *t, |d| {
                                (d.as_ptr() as *mut $scalar, d.len())
                            })?;
                        let base = o.bind(py).clone();
                        drop(cell);
                        unsafe { shared_array2::<$scalar>(py, ptr, n, $dim, &base)? }
                    }
                };
                match dtype {
                    None => Ok(arr.unbind()),
                    Some(dt) => Ok(arr.call_method1("astype", (dt,))?.unbind()),
                }
            }
        }
    };
}

fn numpy_dtype<T: 'static>() -> &'static str {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        "float64"
    } else {
        "int32"
    }
}

eigen_vec_class!(
    Vector3dVector,
    "Vector3dVector",
    f64,
    3,
    "std::vector<Eigen::Vector3d>"
);
eigen_vec_class!(
    Vector2dVector,
    "Vector2dVector",
    f64,
    2,
    "std::vector<Eigen::Vector2d>"
);
eigen_vec_class!(
    Vector3iVector,
    "Vector3iVector",
    i32,
    3,
    "std::vector<Eigen::Vector3i>"
);
eigen_vec_class!(
    Vector2iVector,
    "Vector2iVector",
    i32,
    2,
    "std::vector<Eigen::Vector2i>"
);
eigen_vec_class!(
    Vector4iVector,
    "Vector4iVector",
    i32,
    4,
    "std::vector<Eigen::Vector4i>"
);

// Matrix3dVector / Matrix4dVector: vectors of matrices
macro_rules! eigen_mat_class {
    ($name:ident, $pyname:literal, $dim:expr, $reprname:literal) => {
        #[pyclass(name = $pyname, module = "tiny3d.cpu.pybind.utility")]
        #[derive(Clone, Default)]
        pub struct $name {
            pub data: Vec<[[f64; $dim]; $dim]>,
        }

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (arg = None))]
            fn new(arg: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
                match arg {
                    None => Ok(Self { data: Vec::new() }),
                    Some(_a) => Err(PyTypeError::new_err(
                        "constructing from data is not supported here",
                    )),
                }
            }

            fn __len__(&self) -> usize {
                self.data.len()
            }

            fn __getitem__(&self, py: Python<'_>, idx: isize) -> PyResult<Py<PyAny>> {
                let n = self.data.len() as isize;
                let i = if idx < 0 { idx + n } else { idx };
                if i < 0 || i >= n {
                    return Err(PyIndexError::new_err("list index out of range"));
                }
                let m = &self.data[i as usize];
                let flat: Vec<f64> = m.iter().flatten().copied().collect();
                Ok(PyArray1::from_vec(py, flat)
                    .reshape([$dim, $dim])?
                    .into_any()
                    .unbind())
            }

            fn __repr__(&self) -> String {
                format!(
                    "{} with {} elements.\nUse numpy.asarray() to access data.",
                    $reprname,
                    self.data.len()
                )
            }

            fn clear(&mut self) {
                self.data.clear();
            }
        }
    };
}

eigen_mat_class!(
    Matrix3dVector,
    "Matrix3dVector",
    3,
    "std::vector<Eigen::Matrix3d>"
);
eigen_mat_class!(
    Matrix4dVector,
    "Matrix4dVector",
    4,
    "std::vector<Eigen::Matrix4d>"
);

// IntVector / DoubleVector (pybind bind_vector with default repr "IntVector[...]")
macro_rules! scalar_vec_class {
    ($name:ident, $pyname:literal, $scalar:ty) => {
        #[pyclass(name = $pyname, module = "tiny3d.cpu.pybind.utility", sequence)]
        #[derive(Clone, Default)]
        pub struct $name {
            pub data: Vec<$scalar>,
        }

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (arg = None))]
            fn new(arg: Option<&Bound<'_, PySequence>>) -> PyResult<Self> {
                match arg {
                    None => Ok(Self { data: Vec::new() }),
                    Some(seq) => {
                        let mut data = Vec::new();
                        for i in 0..seq.len()? {
                            data.push(seq.get_item(i)?.extract::<$scalar>()?);
                        }
                        Ok(Self { data })
                    }
                }
            }

            fn __len__(&self) -> usize {
                self.data.len()
            }

            fn __getitem__(&self, idx: isize) -> PyResult<$scalar> {
                let n = self.data.len() as isize;
                let i = if idx < 0 { idx + n } else { idx };
                if i < 0 || i >= n {
                    return Err(PyIndexError::new_err("list index out of range"));
                }
                Ok(self.data[i as usize])
            }

            fn __setitem__(&mut self, idx: isize, value: $scalar) -> PyResult<()> {
                let n = self.data.len() as isize;
                let i = if idx < 0 { idx + n } else { idx };
                if i < 0 || i >= n {
                    return Err(PyIndexError::new_err("list assignment index out of range"));
                }
                self.data[i as usize] = value;
                Ok(())
            }

            fn __repr__(&self) -> String {
                let items: Vec<String> = self.data.iter().map(|x| format!("{}", x)).collect();
                format!("{}[{}]", $pyname, items.join(", "))
            }

            fn __copy__(&self) -> Self {
                self.clone()
            }
            fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
                self.clone()
            }

            fn append(&mut self, value: $scalar) {
                self.data.push(value);
            }
            fn clear(&mut self) {
                self.data.clear();
            }
            fn count(&self, value: $scalar) -> usize {
                self.data.iter().filter(|&&x| x == value).count()
            }
            fn extend(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
                if let Ok(o) = other.extract::<Self>() {
                    self.data.extend(o.data);
                    return Ok(());
                }
                let seq = other.downcast::<PySequence>()?;
                for i in 0..seq.len()? {
                    self.data.push(seq.get_item(i)?.extract::<$scalar>()?);
                }
                Ok(())
            }
            fn insert(&mut self, idx: isize, value: $scalar) {
                let n = self.data.len() as isize;
                let i = if idx < 0 {
                    (idx + n).max(0)
                } else {
                    idx.min(n)
                };
                self.data.insert(i as usize, value);
            }
            #[pyo3(signature = (idx = None))]
            fn pop(&mut self, idx: Option<isize>) -> PyResult<$scalar> {
                let n = self.data.len() as isize;
                let i = match idx {
                    None => n - 1,
                    Some(v) if v < 0 => v + n,
                    Some(v) => v,
                };
                if i < 0 || i >= n {
                    return Err(PyIndexError::new_err("pop index out of range"));
                }
                Ok(self.data.remove(i as usize))
            }
            fn remove(&mut self, value: $scalar) -> PyResult<()> {
                if let Some(pos) = self.data.iter().position(|&x| x == value) {
                    self.data.remove(pos);
                    Ok(())
                } else {
                    Err(PyTypeError::new_err("value not found"))
                }
            }
            fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
                if let Ok(o) = other.extract::<Self>() {
                    self.data == o.data
                } else {
                    false
                }
            }
            #[pyo3(signature = (dtype = None, copy = None))]
            fn __array__(
                &self,
                py: Python<'_>,
                dtype: Option<&Bound<'_, PyAny>>,
                copy: Option<bool>,
            ) -> PyResult<Py<PyAny>> {
                let _ = copy;
                let arr = PyArray1::from_vec(py, self.data.clone());
                match dtype {
                    None => Ok(arr.into_any().unbind()),
                    Some(dt) => Ok(arr.call_method1("astype", (dt,))?.unbind()),
                }
            }
        }
    };
}

scalar_vec_class!(IntVector, "IntVector", i32);
scalar_vec_class!(DoubleVector, "DoubleVector", f64);
