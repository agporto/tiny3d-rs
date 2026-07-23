# tiny3d-rs

`tiny3d-rs` is a Rust implementation of the tiny3D point-cloud API. Install
the distribution as `tiny3d-rs` and import it as `tiny3d`:

```bash
python -m pip install tiny3d-rs
```

```python
import tiny3d as t3d

cloud = t3d.geometry.PointCloud()
```

## Package conflict

The C++ `tiny3d` distribution and `tiny3d-rs` both install the same
`tiny3d/` import package. Do not install both in one environment. Uninstall
the C++ distribution first or use separate virtual environments.

## Supported platforms

Release wheels target CPython 3.9 and newer through the stable ABI:

- Linux x86_64 and aarch64;
- macOS x86_64 and Apple Silicon;
- Windows x86_64.

An sdist is also published for platforms with a Rust 1.96 toolchain and a
supported Python development environment.

## Fidelity and limits

Valid-input behavior is checked against a single-threaded reference tiny3D
build. Linux x86_64 is the strict bit-exact reference platform; other
platforms are functionally tested because system math libraries can differ in
the final bits.

The implementation intentionally fixes several invalid-input and
nondeterministic behaviors. In particular, unsupported VoxelGrid rotations
and general transforms are rejected instead of partially changing the grid.
See [DIVERGENCES.md](https://github.com/agporto/tiny3d-rs/blob/main/DIVERGENCES.md)
for the complete list.

## Project links

- [Source and documentation](https://github.com/agporto/tiny3d-rs)
- [Issue tracker](https://github.com/agporto/tiny3d-rs/issues)
- [Changelog](https://github.com/agporto/tiny3d-rs/blob/main/CHANGELOG.md)
- [Upstream tiny3D](https://github.com/agporto/tiny3D)

## License

MIT. See the packaged `LICENSE` and `THIRD_PARTY_NOTICES.md` files for
tiny3D/Open3D lineage, independently implemented compatibility targets, and
dependency notices.
