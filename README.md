# tiny3d-rs — Rust implementation of tiny3D

A pure-Rust reimplementation of the [tiny3D](https://github.com/agporto/tiny3D)
point-cloud library (an Open3D subset), pip-installable as a drop-in
replacement for the C++ `tiny3d` Python package:

```bash
pip install tiny3d-rs        # imports as `tiny3d`; uninstall the C++ `tiny3d` first
```

```python
import tiny3d as t3d          # same import, same API
pcd = t3d.geometry.PointCloud()
```

Wheels ship for Linux (x86_64, aarch64), macOS (x86_64, Apple Silicon) and
Windows (x64), one abi3 wheel per platform covering Python ≥ 3.9. Releases
are built and published automatically — see [RELEASING.md](RELEASING.md).

## What's inside

| Crate | Contents |
|---|---|
| `crates/tiny3d-core` | The library: geometry (PointCloud, TriangleMesh, VoxelGrid, AABB), compatible independent KD-tree implementation, registration (ICP, RANSAC, FPFH), PLY/XYZ I/O, utilities |
| `crates/tiny3d-py` | PyO3 bindings + maturin packaging replicating the `tiny3d.cpu.pybind` module layout (`geometry`, `io`, `pipelines.registration`, `utility`) |

## Output fidelity

The port is **bit-exact** with the reference C++ build (GCC 13 `-O3`,
x86-64/SSE2, pinned Eigen, `OMP_NUM_THREADS=1`) across a golden test suite
covering the whole Python API surface, plus randomized differential batteries
(byte-identical dumps across seeds). That required independently reproducing
the observable behavior of:

- Eigen 3.4 operation orderings under SSE2 (verified with bit-level probes),
  including JacobiSVD (3x3), SelfAdjointEigenSolver (3x3), pivoted LDLT (6x6),
  Eigen's own `hypot`, `stableNormalize`, quaternion/angle-axis kernels, and
  the vectorized-redux summation orders;
- nanoflann 1.5.0 tree construction and search;
- libstdc++ `std::mt19937` + `uniform_int_distribution` (Lemire downscaling);
- libstdc++ `unordered_map`/`unordered_set` bucket growth and iteration order
  (observable in voxel operations);
- libstdc++ `std::sort` (introsort) tie ordering (observable in radius search);
- glibc `printf` `%g` / `%.10f` formatting and the exact rply PLY layout
  (headers, ASCII formatting, binary layout) for byte-identical files.

The only intentional output differences are fixes for correctness defects in
the original — see [DIVERGENCES.md](DIVERGENCES.md).

## Building

```bash
# wheel
cd crates/tiny3d-py && maturin build --release

# portable default tests
cargo test --workspace

# optional low-level probe suites
TINY3D_PROBE_DIR=/path/to/probe cargo test -p tiny3d-core --test probe_match -- --ignored
```

## Verification harness

- `golden/golden_gen.py gen` — run with the C++ package on `PYTHONPATH` to
  produce golden outputs (use `OMP_NUM_THREADS=1`).
- `golden/golden_gen.py check` — run with this package installed to compare
  everything bit-exactly (27 cases, whole API surface).
- `golden/differential.py <seed> <out>` — randomized differential battery;
  run under both packages and `cmp` the dumps.

## Notes

- Serial execution (matches `OMP_NUM_THREADS=1` semantics — the only
  reproducible configuration of the original).
- `np.asarray(...)` on geometry vector properties is a shared-memory view
  with pybind-identical write-through semantics (see DIVERGENCES.md for the
  buffer-invalidation caveat inherited from the original).

## License and notices

tiny3d-rs is distributed under the [MIT License](LICENSE). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for tiny3D/Open3D lineage,
independently implemented compatibility targets, and Rust/Python dependency
notices.

## Performance

Benchmarked against the C++ build on the same machine (2 cores; larger
machines scale the parallel sections further). Geometric mean across the
suite: **1.8x faster than single-thread C++, 1.5x faster than 2-thread C++**,
with bit-identical outputs (verified at 1 and 2 rayon threads).

Highlights (2-core machine, optimized wheel vs C++):

| Operation | vs C++ 1t | vs C++ 2t |
|---|---|---|
| np.asarray(pcd.points) getter | 2.3x (zero-copy view) | 2.0x |
| XYZ read (100k) | 6.2x | 6.3x |
| PLY binary write / read (100k) | 4.1x / 3.7x | 4.2x / 3.6x |
| PLY ascii read / write (100k) | 4.3x / 1.7x | 4.0x / 1.9x |
| KD-tree build (100k) | 2.0x | 2.0x |
| estimate_normals (50k, knn30) | 1.9x | 1.0x |
| FPFH (20k) | 1.8x | 1.0x |
| ICP p2p / p2l (20k) | 1.75x / 1.8x | 1.0x |
| RANSAC feature (2k) | 1.7x | 1.3x |
| evaluate_registration (20k) | 1.8x | 1.4x |
| voxel_down_sample (200k) | 1.6x | 1.7x |

How speed was gained **without changing outputs**:

- **Order-preserving parallelism (rayon)**: per-point loops (normals, FPFH,
  feature correspondences) write disjoint indices — results are independent of
  thread count by construction. Order-sensitive reductions (registration
  error sums, JTJ accumulation) run their expensive per-element work
  (KD-tree queries, Jacobians) in parallel but accumulate **serially in index
  order**, reproducing the serial bit pattern exactly. Verified: goldens pass
  identically with `RAYON_NUM_THREADS=1` and `=2`.
- **Deterministic batch-parallel RANSAC**: iterations are evaluated in
  parallel batches — samples are pre-drawn serially (with exact RNG rewind at
  the early-exit), candidate evaluation fans out across threads, and results
  fold serially in iteration order. Unlike the C++ OpenMP RANSAC (whose
  results depend on thread scheduling), output is bit-identical to the serial
  loop at every thread count.
- **Zero-allocation KD-tree queries**: thread-local scratch buffers, a
  specialized 3D distance kernel, unchecked indexing on internally-generated
  indices.
- **Allocation-free ASCII formatting** (`%g`, `%.10f`) with reusable buffers.
- **numpy interop**: single-memcpy `__array__`/setter paths; GIL released
  during heavy operations (so Python threads can overlap with compute).
- **Fat LTO + codegen-units=1.**

Serial-equivalent semantics are preserved: the library behaves like the C++
built with `OMP_NUM_THREADS=1` (the only reproducible configuration of the
original), just faster. Thread count (`RAYON_NUM_THREADS`) affects speed only,
never results.
