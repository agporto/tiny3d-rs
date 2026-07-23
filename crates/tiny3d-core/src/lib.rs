//! Core geometry, spatial search, I/O, and registration algorithms for
//! `tiny3d`.
//!
//! This crate is the implementation layer used by the Python extension. Its
//! public API is primarily intended for that binding and for fidelity tests.
//! Numerical code preserves the operation ordering of the reference tiny3D
//! implementation where observable behavior depends on floating-point
//! rounding.

pub mod eigen_solvers;
pub mod geometry;
pub mod io;
pub mod kdtree;
pub mod linalg;
pub mod random;
pub mod registration;
pub mod stdhash;
pub mod stdsort;
