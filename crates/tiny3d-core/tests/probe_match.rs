#![allow(clippy::needless_range_loop)]

//! Bit-exact comparison of compatibility solvers against C++/Eigen probe data.
use std::collections::HashMap;
use tiny3d_core::eigen_solvers::*;
use tiny3d_core::linalg::*;

fn probe_file(name: &str) -> std::path::PathBuf {
    let directory = std::env::var_os("TINY3D_PROBE_DIR")
        .expect("set TINY3D_PROBE_DIR to the directory containing C++ probe dumps");
    std::path::PathBuf::from(directory).join(name)
}

fn parse_probe() -> Vec<HashMap<String, Vec<Vec<f64>>>> {
    let path = probe_file("probe_out.txt");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    let mut blocks = Vec::new();
    let mut cur: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let key = parts.next().unwrap();
        if key == "END" {
            blocks.push(std::mem::take(&mut cur));
            continue;
        }
        let vals: Vec<f64> = parts
            .map(|h| {
                if h.len() == 16 && h.chars().all(|c| c.is_ascii_hexdigit()) {
                    f64::from_bits(u64::from_str_radix(h, 16).unwrap())
                } else {
                    h.parse::<f64>().unwrap()
                }
            })
            .collect();
        cur.entry(key.to_string()).or_default().push(vals);
    }
    blocks
}

fn get_m3(b: &HashMap<String, Vec<Vec<f64>>>, key: &str) -> M3 {
    let rows = &b[key];
    [
        [rows[0][0], rows[0][1], rows[0][2]],
        [rows[1][0], rows[1][1], rows[1][2]],
        [rows[2][0], rows[2][1], rows[2][2]],
    ]
}

fn assert_bits(a: f64, b: f64, what: &str, block: usize) {
    assert!(
        a.to_bits() == b.to_bits(),
        "{} mismatch in block {}: {:e} ({:016x}) vs {:e} ({:016x})",
        what,
        block,
        a,
        a.to_bits(),
        b,
        b.to_bits()
    );
}

#[test]
#[ignore = "requires C++ reference dumps in TINY3D_PROBE_DIR"]
fn svd_matches() {
    for (bi, b) in parse_probe().iter().enumerate() {
        let m = get_m3(b, "in_M");
        let svd = jacobi_svd3(&m);
        for k in 0..3 {
            assert_bits(svd.singular_values[k], b["svd_s"][0][k], "svd sigma", bi);
        }
        assert_bits(svd.u[0][0], b["svd_u"][0][0], "svd u00", bi);
        assert_bits(svd.u[2][1], b["svd_u"][0][1], "svd u21", bi);
        assert_bits(svd.v[0][0], b["svd_v"][0][0], "svd v00", bi);
        assert_bits(svd.v[2][1], b["svd_v"][0][1], "svd v21", bi);
    }
}

#[test]
#[ignore = "requires C++ reference dumps in TINY3D_PROBE_DIR"]
fn saes_matches() {
    for (bi, b) in parse_probe().iter().enumerate() {
        let m = get_m3(b, "in_M");
        // S = M + M^T
        let mut s = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] = m[i][j] + m[j][i];
            }
        }
        let es = self_adjoint_eigen3(&s);
        for k in 0..3 {
            assert_bits(es.eigenvalues[k], b["saes_val"][0][k], "saes eval", bi);
        }
        for k in 0..3 {
            assert_bits(
                es.eigenvectors[k][0],
                b["saes_vec"][0][k],
                "saes evec col0",
                bi,
            );
        }
    }
}

#[test]
#[ignore = "requires C++ reference dumps in TINY3D_PROBE_DIR"]
fn inverse_matches() {
    for (bi, b) in parse_probe().iter().enumerate() {
        let m = get_m3(b, "in_M");
        let (inv, ok) = compute_inverse3_with_check(&m);
        assert_eq!(ok, b["inv3"][0][0] != 0.0, "invertible flag block {}", bi);
        if ok {
            assert_bits(inv[0][0], b["inv3"][0][1], "inv 00", bi);
            assert_bits(inv[2][1], b["inv3"][0][2], "inv 21", bi);
        }
    }
}

#[test]
#[ignore = "requires C++ reference dumps in TINY3D_PROBE_DIR"]
fn basic_ops_match() {
    for (bi, b) in parse_probe().iter().enumerate() {
        let a: V3 = [b["in_a"][0][0], b["in_a"][0][1], b["in_a"][0][2]];
        let bb: V3 = [b["in_b"][0][0], b["in_b"][0][1], b["in_b"][0][2]];
        assert_bits(dot3(a, bb), b["dot"][0][0], "dot", bi);
        assert_bits(squared_norm3(a), b["sqn"][0][0], "sqn", bi);
        assert_bits(norm3(a), b["norm"][0][0], "norm", bi);
        let c = cross3(a, bb);
        for k in 0..3 {
            assert_bits(c[k], b["cross"][0][k], "cross", bi);
        }
        let m = get_m3(b, "in_M");
        let mv = m3v3(&m, a);
        for k in 0..3 {
            assert_bits(mv[k], b["m3v3"][0][k], "m3v3", bi);
        }
        let m2 = get_m3(b, "in_M2");
        let mm = m3m3(&m, &m2);
        assert_bits(mm[0][0], b["m3m3"][0][0], "m3m3 00", bi);
        assert_bits(mm[1][2], b["m3m3"][0][1], "m3m3 12", bi);
        assert_bits(mm[2][1], b["m3m3"][0][2], "m3m3 21", bi);
        assert_bits(mm[2][2], b["m3m3"][0][3], "m3m3 22", bi);
        let t_rows = &b["in_T"];
        let mut t = [[0.0f64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                t[i][j] = t_rows[i][j];
            }
        }
        let p: V4 = [
            b["in_p"][0][0],
            b["in_p"][0][1],
            b["in_p"][0][2],
            b["in_p"][0][3],
        ];
        let tv = m4v4(&t, p);
        for k in 0..4 {
            assert_bits(tv[k], b["m4v4"][0][k], "m4v4", bi);
        }
        let n = normalized3(a);
        for k in 0..3 {
            assert_bits(n[k], b["normd"][0][k], "normalized", bi);
        }
        let mut sn = a;
        stable_normalize3(&mut sn);
        for k in 0..3 {
            assert_bits(sn[k], b["stnorm"][0][k], "stableNormalize", bi);
        }
        let amb = sub3(a, bb);
        assert_bits(squared_norm3(amb), b["dsqn"][0][0], "dsqn", bi);
        // jtj / jtr
        let jv = &b["in_J"][0];
        let j6: V6 = [jv[0], jv[1], jv[2], jv[3], jv[4], jv[5]];
        let w = b["in_wr"][0][0];
        let r = b["in_wr"][0][1];
        let mut jtj = m6_zero();
        m6_add_scaled_outer(&mut jtj, &j6, w);
        assert_bits(jtj[0][0], b["jtj"][0][0], "jtj 00", bi);
        assert_bits(jtj[3][2], b["jtj"][0][1], "jtj 32", bi);
        assert_bits(jtj[5][5], b["jtj"][0][2], "jtj 55", bi);
        let mut jtr = v6_zero();
        v6_add_scaled(&mut jtr, &j6, w, r);
        assert_bits(jtr[0], b["jtr"][0][0], "jtr 0", bi);
        assert_bits(jtr[5], b["jtr"][0][1], "jtr 5", bi);
    }
}
