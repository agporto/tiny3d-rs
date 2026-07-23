#![allow(clippy::needless_range_loop)]

use tiny3d_core::eigen_solvers::self_adjoint_eigen3;

fn probe_file(name: &str) -> std::path::PathBuf {
    let directory = std::env::var_os("TINY3D_PROBE_DIR")
        .expect("set TINY3D_PROBE_DIR to the directory containing C++ probe dumps");
    std::path::PathBuf::from(directory).join(name)
}

fn hx(t: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(t, 16).unwrap())
}

#[test]
#[ignore = "requires C++ reference dumps in TINY3D_PROBE_DIR"]
fn saes_stages_match() {
    let path = probe_file("saes_probe.txt");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    let mut lines = text.lines().peekable();
    let mut block = 0;
    while lines.peek().is_some() {
        let mut rows = std::collections::HashMap::<String, Vec<Vec<f64>>>::new();
        loop {
            let l = lines.next().unwrap();
            if l == "END" {
                break;
            }
            let toks: Vec<&str> = l.split_whitespace().collect();
            rows.entry(toks[0].to_string())
                .or_default()
                .push(toks[1..].iter().map(|t| hx(t)).collect());
        }
        let s_rows = &rows["in_S"];
        let mut s = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] = s_rows[i][j];
            }
        }
        let es = self_adjoint_eigen3(&s);
        // stage checks first: tridiag via internal debug
        let (diag, subdiag, q) = tiny3d_core::eigen_solvers::debug_tridiag3(&s);
        for k in 0..3 {
            assert_eq!(
                diag[k].to_bits(),
                rows["tridiag_diag"][0][k].to_bits(),
                "tridiag diag[{}] block {}",
                k,
                block
            );
        }
        for k in 0..2 {
            assert_eq!(
                subdiag[k].to_bits(),
                rows["tridiag_sub"][0][k].to_bits(),
                "tridiag sub[{}] block {}",
                k,
                block
            );
        }
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(
                    q[i][j].to_bits(),
                    rows["tridiag_q"][i][j].to_bits(),
                    "tridiag q[{}][{}] block {}",
                    i,
                    j,
                    block
                );
            }
        }
        for k in 0..3 {
            assert_eq!(
                es.eigenvalues[k].to_bits(),
                rows["eval"][0][k].to_bits(),
                "eval[{}] block {}",
                k,
                block
            );
        }
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(
                    es.eigenvectors[i][j].to_bits(),
                    rows["evec"][i][j].to_bits(),
                    "evec[{}][{}] block {}",
                    i,
                    j,
                    block
                );
            }
        }
        block += 1;
    }
    assert!(block >= 40);
}
