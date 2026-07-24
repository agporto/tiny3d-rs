# Changelog

All notable changes to tiny3d-rs are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [2.1.0] - 2026-07-24

### Added

- `PointCloud.orient_normals_to_align_with_direction` (Open3D-compatible)
- `PointCloud.orient_normals_towards_camera_location` (Open3D-compatible)
- `PointCloud.orient_normals_consistent_tangent_plane` (Hoppe '92 MST propagation; Delaunay-free EMST approximation, documented divergences)

## [2.0.0] - 2026-07-23

Initial public beta of the Rust implementation:

- geometry, KD-tree, registration, FPFH, RANSAC, PLY/XYZ I/O, and utility
  APIs exposed through the `tiny3d` Python package;
- abi3 wheels for Python 3.9 and newer;
- bit-exact Linux x86_64 golden verification for valid reference behavior;
- documented correctness fixes and explicit invalid-input errors.

[Unreleased]: https://github.com/agporto/tiny3d-rs/compare/v2.1.0...HEAD
[2.1.0]: https://github.com/agporto/tiny3d-rs/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/agporto/tiny3d-rs/releases/tag/v2.0.0
