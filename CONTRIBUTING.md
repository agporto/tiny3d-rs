# Contributing

Thank you for improving tiny3d-rs.

## Development setup

Install Python 3.9 or newer, the pinned Rust toolchain from
`rust-toolchain.toml`, and maturin. Use a virtual environment that does not
contain the C++ `tiny3d` package.

```bash
python -m pip install maturin numpy
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

Build and test the Python package:

```bash
maturin build --release --manifest-path crates/tiny3d-py/Cargo.toml
python -m pip install --force-reinstall --no-deps target/wheels/tiny3d_rs-*.whl
python golden/smoke.py
```

## Numerical fidelity

Do not reorder floating-point expressions or replace indexed compatibility
loops mechanically. Any numerical change must run the smoke suite and the
Linux x86_64 golden suite. Narrow Clippy allowances in fidelity-sensitive
code must explain why the suggested rewrite is unsafe.

Low-level Eigen compatibility probe tests are optional and ignored by
default. Set `TINY3D_PROBE_DIR` and run the relevant test with `--ignored`.

## Pull requests

Keep changes focused, add regression tests, update user-facing documentation,
and complete the pull-request checklist. By contributing, you agree that your
work is licensed under the repository's MIT License.
