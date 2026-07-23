//! tiny3d.cpu.pybind.utility

use pyo3::prelude::*;
use std::sync::atomic::{AtomicI32, Ordering};

use crate::vectors::*;

static VERBOSITY: AtomicI32 = AtomicI32::new(2); // Info

#[pyclass(
    name = "VerbosityLevel",
    module = "tiny3d.cpu.pybind.utility",
    frozen,
    eq,
    hash
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct VerbosityLevel {
    #[pyo3(get)]
    pub value: i32,
    pub name_str: &'static str,
}

#[pymethods]
impl VerbosityLevel {
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
        format!("<VerbosityLevel.{}: {}>", self.name_str, self.value)
    }
    #[classattr]
    #[allow(non_upper_case_globals)]
    const Error: VerbosityLevel = VL_ERROR;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const Warning: VerbosityLevel = VL_WARNING;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const Info: VerbosityLevel = VL_INFO;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const Debug: VerbosityLevel = VL_DEBUG;
}

pub const VL_ERROR: VerbosityLevel = VerbosityLevel {
    value: 0,
    name_str: "Error",
};
pub const VL_WARNING: VerbosityLevel = VerbosityLevel {
    value: 1,
    name_str: "Warning",
};
pub const VL_INFO: VerbosityLevel = VerbosityLevel {
    value: 2,
    name_str: "Info",
};
pub const VL_DEBUG: VerbosityLevel = VerbosityLevel {
    value: 3,
    name_str: "Debug",
};

fn vl_from_value(v: i32) -> VerbosityLevel {
    match v {
        0 => VL_ERROR,
        1 => VL_WARNING,
        3 => VL_DEBUG,
        _ => VL_INFO,
    }
}

#[pyfunction]
fn set_verbosity_level(verbosity_level: &Bound<'_, PyAny>) -> PyResult<()> {
    let v: i32 = if let Ok(vl) = verbosity_level.extract::<VerbosityLevel>() {
        vl.value
    } else {
        verbosity_level.extract()?
    };
    VERBOSITY.store(v, Ordering::SeqCst);
    Ok(())
}

#[pyfunction]
fn get_verbosity_level() -> VerbosityLevel {
    vl_from_value(VERBOSITY.load(Ordering::SeqCst))
}

#[pyfunction]
fn reset_print_function() {}

#[pyclass(name = "VerbosityContextManager", module = "tiny3d.cpu.pybind.utility")]
pub struct VerbosityContextManager {
    level: i32,
    saved: i32,
}

#[pymethods]
impl VerbosityContextManager {
    #[new]
    fn new(level: &Bound<'_, PyAny>) -> PyResult<Self> {
        let v: i32 = if let Ok(vl) = level.extract::<VerbosityLevel>() {
            vl.value
        } else {
            level.extract()?
        };
        Ok(VerbosityContextManager { level: v, saved: 2 })
    }
    fn __enter__(&mut self) {
        self.saved = VERBOSITY.load(Ordering::SeqCst);
        VERBOSITY.store(self.level, Ordering::SeqCst);
    }
    fn __exit__(&mut self, _t: &Bound<'_, PyAny>, _v: &Bound<'_, PyAny>, _tb: &Bound<'_, PyAny>) {
        VERBOSITY.store(self.saved, Ordering::SeqCst);
    }
}

#[pyfunction]
fn seed(seed: i32) {
    tiny3d_core::random::seed(seed);
}

pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Vector3dVector>()?;
    m.add_class::<Vector2dVector>()?;
    m.add_class::<Vector3iVector>()?;
    m.add_class::<Vector2iVector>()?;
    m.add_class::<Vector4iVector>()?;
    m.add_class::<Matrix3dVector>()?;
    m.add_class::<Matrix4dVector>()?;
    m.add_class::<IntVector>()?;
    m.add_class::<DoubleVector>()?;
    m.add_class::<VerbosityLevel>()?;
    m.add_class::<VerbosityContextManager>()?;
    m.add("Error", VL_ERROR)?;
    m.add("Warning", VL_WARNING)?;
    m.add("Info", VL_INFO)?;
    m.add("Debug", VL_DEBUG)?;
    m.add_function(wrap_pyfunction!(set_verbosity_level, m)?)?;
    m.add_function(wrap_pyfunction!(get_verbosity_level, m)?)?;
    m.add_function(wrap_pyfunction!(reset_print_function, m)?)?;

    // utility.random submodule
    let random = PyModule::new(py, "random")?;
    random.add_function(wrap_pyfunction!(seed, &random)?)?;
    m.add_submodule(&random)?;
    Ok(())
}
