## Summary

## Test plan

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `cargo test --workspace`
- [ ] `RUSTDOCFLAGS=-Dwarnings cargo doc -p tiny3d-core --no-deps`
- [ ] Optimized wheel and `python golden/smoke.py`
- [ ] Linux golden suite, if numerical behavior changed

## Checklist

- [ ] I added or updated regression tests.
- [ ] I updated user-facing documentation.
- [ ] I preserved floating-point evaluation order or documented the change.
- [ ] I reviewed dependency and license changes.
