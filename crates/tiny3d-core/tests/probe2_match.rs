use std::collections::HashMap;
use tiny3d_core::geometry::point_cloud::fast_eigen_3x3;
use tiny3d_core::linalg::*;

fn probe_file(name: &str) -> std::path::PathBuf {
    let directory = std::env::var_os("TINY3D_PROBE_DIR")
        .expect("set TINY3D_PROBE_DIR to the directory containing C++ probe dumps");
    std::path::PathBuf::from(directory).join(name)
}

fn hx(t: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(t, 16).unwrap())
}

fn parse() -> Vec<HashMap<String, Vec<Vec<f64>>>> {
    let path = probe_file("probe2.txt");
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
        cur.entry(key.to_string())
            .or_default()
            .push(parts.map(hx).collect());
    }
    blocks
}

fn v3(b: &HashMap<String, Vec<Vec<f64>>>, k: &str) -> V3 {
    let r = &b[k][0];
    [r[0], r[1], r[2]]
}

fn ab(x: f64, y: f64, what: &str, bi: usize) {
    assert!(
        x.to_bits() == y.to_bits(),
        "{} block {}: {:016x} vs {:016x}",
        what,
        bi,
        x.to_bits(),
        y.to_bits()
    );
}

#[test]
#[ignore = "requires C++ reference dumps in TINY3D_PROBE_DIR"]
fn probe2_ops() {
    for (bi, b) in parse().iter().enumerate() {
        let a = v3(b, "in_a");
        let bb = v3(b, "in_b");
        let c = v3(b, "in_c");
        ab(dot3(sub3(a, bb), c), b["diffdot"][0][0], "diffdot", bi);
        let v01 = sub3(a, bb);
        let v02 = sub3(c, bb);
        let tn = cross3(v01, v02);
        for k in 0..3 {
            ab(tn[k], b["tnorm"][0][k], "tnorm", bi);
        }
        let mut sn = tn;
        stable_normalize3(&mut sn);
        for k in 0..3 {
            ab(sn[k], b["tnormn"][0][k], "tnormn", bi);
        }
        let cr = cross3(a, c);
        for k in 0..3 {
            ab(cr[k], b["crossac"][0][k], "crossac", bi);
        }
        let rows = &b["in_cov"];
        let cov: M3 = [
            [rows[0][0], rows[0][1], rows[0][2]],
            [rows[1][0], rows[1][1], rows[1][2]],
            [rows[2][0], rows[2][1], rows[2][2]],
        ];
        let ev = fast_eigen_3x3(&cov);
        for k in 0..3 {
            ab(ev[k], b["fe"][0][k], "fast_eigen", bi);
        }
    }
}
