//! tiny3d.cpu.pybind — Rust drop-in replacement for the tiny3D pybind module.

use pyo3::prelude::*;

mod fmtutil;
mod geometry;
mod io_mod;
mod registration;
mod utility;
mod vectors;

#[pymodule]
fn pybind(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let sys_modules = py.import("sys")?.getattr("modules")?;

    let geometry_mod = PyModule::new(py, "geometry")?;
    geometry::register(py, &geometry_mod)?;
    m.add_submodule(&geometry_mod)?;
    sys_modules.set_item("tiny3d.cpu.pybind.geometry", &geometry_mod)?;

    let utility_mod = PyModule::new(py, "utility")?;
    utility::register(py, &utility_mod)?;
    m.add_submodule(&utility_mod)?;
    sys_modules.set_item("tiny3d.cpu.pybind.utility", &utility_mod)?;
    sys_modules.set_item(
        "tiny3d.cpu.pybind.utility.random",
        utility_mod.getattr("random")?,
    )?;

    let io_module = PyModule::new(py, "io")?;
    io_mod::register(py, &io_module)?;
    m.add_submodule(&io_module)?;
    sys_modules.set_item("tiny3d.cpu.pybind.io", &io_module)?;

    let pipelines_mod = PyModule::new(py, "pipelines")?;
    let registration_mod = PyModule::new(py, "registration")?;
    registration::register(py, &registration_mod)?;
    pipelines_mod.add_submodule(&registration_mod)?;
    m.add_submodule(&pipelines_mod)?;
    sys_modules.set_item("tiny3d.cpu.pybind.pipelines", &pipelines_mod)?;
    sys_modules.set_item(
        "tiny3d.cpu.pybind.pipelines.registration",
        &registration_mod,
    )?;

    m.add("__version__", "2.0.0")?;
    Ok(())
}
