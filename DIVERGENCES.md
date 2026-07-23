# Intentional divergences from the C++ tiny3D build

The Rust port is bit-exact with the C++ build (GCC 13, `-O3`, SSE2, Eigen
pinned at `da79095923`, run with `OMP_NUM_THREADS=1`) for every API in the
golden test suite, **except** where the C++ has correctness defects. Each
divergence below is deliberate, and each is covered by a test.

## 1. `PointCloud.estimate_normals()` — uninitialized-memory normal orientation

**C++ bug** (`cpp/tiny3d/geometry/PointCloud.cpp`, `EstimateNormals`): the
code resizes `normals_` to `points_.size()` *before* checking `HasNormals()`.
As a result `has_original_normals` is always true, and when the cloud had no
normals the orientation step compares each computed normal against an
**uninitialized** `Eigen::Vector3d` (Eigen leaves memory uninitialized on
resize). Normal *signs* therefore depend on heap garbage: the same input in
the same process can return different signs after unrelated allocations. This
was directly observed (sign flips between identical calls) and is what made
initial golden-data generation non-reproducible. Upstream Open3D checks
`HasNormals()` before resizing.

**Rust behavior**: orientation against prior normals happens only when the
cloud actually had normals before the call (Open3D semantics). With prior
normals present, output is bit-exact with the C++. Without prior normals the
raw eigenvector direction is kept — deterministic, and equal to the C++ output
up to per-point sign.

**Test**: `normals_no_prior_SIGNFLIP` golden case (compared per-row up to
sign); all other `normals_*` cases seed prior normals and are bit-exact.

## 2. `read_point_cloud(..., remove_nan_points=…, remove_infinite_points=…)` — flags silently ignored

**C++ bug** (`cpp/tiny3d/io/PointCloudIO.cpp`): the post-read call to
`RemoveNonFinitePoints` is commented out, so both parameters are accepted and
do nothing — a cloud with NaN/inf points is returned unfiltered even when
removal was requested.

**Rust behavior**: the flags work (Open3D semantics): NaN rows are removed
when `remove_nan_points=True`, infinite rows when
`remove_infinite_points=True`, preserving order and any normals/colors.

**Test**: `io_nan` golden case (`clean_points` is checked against the
correctly filtered expectation; `raw_points` remains bit-exact).

## 3. `TriangleMesh` normal normalization — heap-alignment-dependent rounding (±1 ulp)

**C++ defect** (build-level, not algorithmic):
`MeshBase::NormalizeNormals` / `TriangleMesh::NormalizeNormals` are defined
inline in headers and get a second, LTO-compiled instantiation inside the
pybind module (built with `-flto`). That instantiation's auto-vectorized loop
takes runtime-alignment-dependent code paths, so `compute_vertex_normals()` /
`normalize_normals()` on a mesh round the last bit differently depending on
where the heap placed the normal arrays. The same `.so` was observed to
return two different results (±1 ulp) for identical input depending on what
was allocated earlier in the process.

**Rust behavior**: one deterministic result, equal to the C++ static-library
code path (`libTiny3D.a`, non-LTO) output.

**Test**: `mesh_ops` golden case compares those keys with a ≤4-ulp tolerance;
everything upstream of the final normalize (raw cross products, vertex
accumulations) is bit-exact.

## 4. `correspondences_from_features(..., mutual_filter=True)` — mutual check reads the wrong element

**C++ bug** (`cpp/tiny3d/pipelines/registration/Feature.cpp`,
`CorrespondencesFromFeatures`): the mutuality test is
`corres[1][j](0) == i`, but element `(0)` of the reverse correspondence at
index `j` is **always `j` itself** (each entry is stored as
`(query_index, match_index)`). Upstream Open3D checks `corres[1][j](1) == i`
— i.e. whether the target→source nearest neighbor of `j` points back at `i`.
The tiny3D check therefore reduces to `j == i`, which keeps essentially
nothing on real data, so the `mutual_consistency_ratio` fallback silently
returns the **unfiltered** forward set. Net effect: `mutual_filter=True` is
a no-op (plus a warning), which degrades feature-based RANSAC — with
low-overlap pairs it can mean failed registration where a correct mutual
filter succeeds.

This also affects `registration_ransac_based_on_feature_matching(...,
mutual_filter=True, ...)`, which routes through the same function.

**Rust behavior**: the Open3D check (`corres[1][j](1) == i`), including the
same ratio fallback (`len(mutual) >= int(ratio * num_source)`, f32 math) and
tiny3D's bounds guard on `j`.

**Test**: `feature_corres` golden keys `mutual` / `mutual_r5` are verified
against an independently (numpy-)computed mutual set with the ratio
fallback applied; the one-way sets (`plain`, `rev`) remain bit-exact against
the C++. `golden/smoke.py` re-derives the expected mutual set on every CI
platform.

## 5. Invalid input handling — explicit errors and atomic reads

Several invalid-input paths in the C++-compatible API either fail silently or
allow invalid indices and dimensions to reach unchecked numerical code.

**Rust behavior**:

- registration correspondence indices are validated against both point clouds
  by all public estimation and checking entry points and raise `ValueError`
  instead of triggering a Rust indexing panic;
- negative `Feature.resize()` dimensions and zero/non-finite quaternions raise
  `ValueError`;
- `write_point_cloud_to_bytes()` raises `ValueError` for unsupported formats
  instead of returning empty bytes;
- PLY point-cloud and mesh loading commits parsed geometry only after the
  complete input succeeds; mesh loading also validates face indices and every
  face triangulation;
- `VoxelGrid.transform()` accepts only identity/pure translation matrices and
  `VoxelGrid.rotate()` accepts only identity. Other transforms raise
  `RuntimeError` without changing the grid because rotating or scaling an
  axis-aligned voxel lattice requires explicit revoxelization.

**Tests**: core unit tests cover correspondence, PLY, and VoxelGrid validation;
`golden/smoke.py` covers all Python-facing error contracts and atomic failed
reads. The `io_xyz` golden case accepts the explicit unsupported-format error
while preserving comparison with the C++ empty-byte reference.

## Behavioral notes (not correctness divergences)

- **Threading**: the Rust port matches the C++ run with `OMP_NUM_THREADS=1`.
  The C++ OpenMP build gives thread-count-dependent floating-point results
  (parallel reductions with `#pragma omp critical` merges); single-thread is
  the only reproducible reference.
- **`np.asarray(pcd.points)` returns a live, shared-memory view — same as
  the pybind build.** Mutating the array writes through to the cloud, and
  vector objects returned by geometry properties (`pcd.points`,
  `mesh.triangles`, …) are live references: `v[i] = …` and `v.append(…)`
  modify the geometry. The pybind caveat carries over too: a numpy array
  captured before an operation that reallocates the underlying buffer
  (property assignment with growth, `estimate_normals` allocating normals,
  `clear()`, …) refers to the old storage — call `.copy()` if you need a
  snapshot. Verified behavior-identical to the C++ package across a
  10-scenario parity test. Exceptions that remain copies:
  `RegistrationResult.correspondence_set` and `Feature.data`.
- **Docstrings** are not replicated verbatim.
- The libstdc++ containers whose iteration order is observable
  (`unordered_map`/`unordered_set` used by voxel ops) and `std::sort` tie
  ordering (radius search) are **emulated exactly**, so voxel output order and
  tied-distance neighbor order match the C++ build bit-for-bit.
