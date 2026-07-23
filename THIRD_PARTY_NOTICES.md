# Third-party notices

This document describes the lineage, compatibility targets, and dependencies
of tiny3d-rs 2.0.0. It is provided for attribution and release auditing; it
does not replace the license texts shipped by each dependency.

## tiny3D and Open3D lineage

tiny3d-rs is an MIT-licensed Rust reimplementation of
[tiny3D](https://github.com/agporto/tiny3D), which is an MIT-licensed subset
and fork of [Open3D](https://github.com/isl-org/Open3D). The project preserves
their Python API shape and includes behavior derived from that lineage.
Copyright notices and the MIT terms are retained in `LICENSE`.

## Independently implemented compatibility targets

The Rust modules that reproduce observable Eigen, nanoflann, libstdc++, glibc,
and rply behavior were independently implemented from documented behavior and
black-box test probes. No source code from Eigen, nanoflann, libstdc++, glibc,
or rply is vendored or compiled into this repository.

Names such as Eigen, nanoflann, libstdc++, glibc, and rply identify
compatibility targets only. Their respective projects are not dependencies of
the built wheel and do not endorse tiny3d-rs.

## Rust dependencies

Release artifacts are built from the Cargo dependency graph recorded in
`Cargo.lock`. The graph contains the following third-party crates:

- Apache-2.0 or MIT: autocfg, cfg-if, crossbeam-deque, crossbeam-epoch,
  crossbeam-utils, either, heck, indoc, libc, matrixmultiply, ndarray,
  num-complex, num-integer, num-traits, once_cell, portable-atomic,
  portable-atomic-util, proc-macro2, PyO3 and its support crates, quote,
  rawpointer, rayon, rayon-core, rustc-hash, rustversion, syn, and unindent.
- MIT: memoffset.
- BSD-2-Clause: numpy (the Rust crate).
- Apache-2.0 WITH LLVM-exception: target-lexicon.
- (MIT OR Apache-2.0) AND Unicode-3.0: unicode-ident.

Exact versions are recorded in `Cargo.lock`. CI runs `cargo-deny` against
`deny.toml` so new or changed dependency licenses must be reviewed.

## Python runtime dependency

The Python package depends on NumPy, distributed separately under the
BSD-3-Clause license. NumPy is not bundled in tiny3d-rs wheels.

## Obtaining dependency license texts

Crate source distributions downloaded by Cargo include their license files.
The corresponding canonical package pages are available at
`https://crates.io/crates/<crate-name>`. NumPy's license is available at
<https://numpy.org/doc/stable/license.html>.
