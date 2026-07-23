# Releasing `tiny3d-rs` to PyPI

Every published GitHub release triggers `.github/workflows/release.yml`,
which builds, tests, and uploads to PyPI:

- **Wheels** (abi3 — one wheel per platform covers CPython ≥ 3.9):
  Linux x86_64 + aarch64 (manylinux), macOS x86_64 + arm64 (Apple Silicon),
  Windows x64
- **sdist** (builds anywhere with a Rust toolchain via `pip install`)

Before upload, every wheel is smoke-tested on its own platform
(`golden/smoke.py`), and the Linux x86_64 wheel must additionally pass the
full **bit-exactness golden suite** (`golden/golden_gen.py check`, 27 cases).
On the other platforms the golden check runs advisory-only
(`continue-on-error`): system libm differences (sin/cos/atan2/…) can shift
last-bit results, so those platforms are held to functional correctness, not
Linux-x86_64 bit patterns.

## One-time setup: PyPI Trusted Publishing

No API tokens — the workflow authenticates via OIDC. Configure once:

1. Log in to PyPI with the account that will own `tiny3d-rs`.
2. Since the project doesn't exist yet, go to
   **Account → Publishing → Add a new pending publisher** and enter:
   - **PyPI project name**: `tiny3d-rs`
   - **Owner**: `agporto`
   - **Repository name**: `tiny3d-rs`
   - **Workflow name**: `release.yml`
   - **Environment name**: `pypi`
3. In the GitHub repo: **Settings → Environments → New environment** named
   `pypi`. (Optional but recommended: add yourself as a required reviewer so
   publishing waits for a manual approval click.)

The first successful publish creates the PyPI project and converts the
pending publisher into a normal trusted publisher. Nothing else to manage.

> If you later rename the repo or the workflow file, update the trusted
> publisher on PyPI to match, or publishing will fail with an OIDC error.

## Cutting a release

1. **Bump the version** — it currently lives in 6 places (keep them equal):

   ```bash
   NEW=2.1.0
   sed -i "s/^version = \".*\"/version = \"$NEW\"/" \
       crates/tiny3d-py/pyproject.toml crates/tiny3d-py/Cargo.toml crates/tiny3d-core/Cargo.toml
   sed -i "s/__version__ = \".*\"/__version__ = \"$NEW\"/" crates/tiny3d-py/python/tiny3d/__init__.py
   sed -i "s/\"version\": \".*\"/\"version\": \"$NEW\"/" crates/tiny3d-py/python/tiny3d/_build_config.py
   sed -i "s/m.add(\"__version__\", \".*\")/m.add(\"__version__\", \"$NEW\")/" crates/tiny3d-py/src/lib.rs
   ```

2. Run the local release gates:

   ```bash
   cargo fmt --all -- --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace
   RUSTDOCFLAGS=-Dwarnings cargo doc -p tiny3d-core --no-deps
   cargo deny check licenses
   ```

3. Commit, push, and create a GitHub release with tag `v$NEW` (a bare
   `$NEW` tag also works). The workflow **fails fast if the tag doesn't
   match the `pyproject.toml` version**, so a mismatched bump can't ship.

4. Watch the Actions run. Wheels build in parallel (~15 min, dominated by
   fat-LTO), tests gate publishing, then the `publish` job uploads
   everything to PyPI (pausing for approval if you set a required reviewer).

Dry run: **Actions → Release → Run workflow** (`workflow_dispatch`) builds
and tests everything on all platforms but skips the publish job. Do this
before the first real release to shake out platform issues.

## Notes

- **Users install `tiny3d-rs` but import `tiny3d`** — it's a drop-in
  replacement. It therefore conflicts with the C++ `tiny3d` package: both
  install a `tiny3d/` package directory, so `pip uninstall tiny3d` before
  installing `tiny3d-rs` (or use separate environments).
- Python support is `>=3.9` via a single abi3 wheel per platform — new
  CPython releases work without rebuilding.
- Portable Cargo tests run by default. Low-level bit-level probe tests are
  ignored unless `TINY3D_PROBE_DIR` is set and `--ignored` is requested.
- Linux wheels are built in manylinux containers via `PyO3/maturin-action`
  (`manylinux: auto`), so they work on any mainstream glibc distro.
