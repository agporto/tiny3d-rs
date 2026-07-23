use std::collections::HashMap;
fn hx(t: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(t, 16).unwrap())
}

fn input_path(filename: &str) -> std::path::PathBuf {
    if let Some(path) = std::env::args_os().nth(1) {
        return path.into();
    }
    if let Some(directory) = std::env::var_os("TINY3D_PROBE_DIR") {
        return std::path::PathBuf::from(directory).join(filename);
    }
    eprintln!(
        "usage: cargo run -p tiny3d-core --example fe_dbg -- <probe2.txt>\n\
         or set TINY3D_PROBE_DIR"
    );
    std::process::exit(2);
}

fn main() {
    let path = input_path("probe2.txt");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        eprintln!("failed to read {}: {error}", path.display());
        std::process::exit(2);
    });
    let mut blocks: Vec<HashMap<String, Vec<Vec<f64>>>> = Vec::new();
    let mut cur: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
    for line in text.lines() {
        let mut p = line.split_whitespace();
        let k = p.next().unwrap();
        if k == "END" {
            blocks.push(std::mem::take(&mut cur));
            continue;
        }
        cur.entry(k.to_string())
            .or_default()
            .push(p.map(hx).collect());
    }
    let b = &blocks[4];
    let rows = &b["in_cov"];
    let cov = [
        [rows[0][0], rows[0][1], rows[0][2]],
        [rows[1][0], rows[1][1], rows[1][2]],
        [rows[2][0], rows[2][1], rows[2][2]],
    ];
    // replicate my fast_eigen with midpoint dump
    let mut a = cov;
    let mut mc = 0.0f64;
    for r in a.iter() {
        for &x in r.iter() {
            if x.abs() > mc {
                mc = x.abs();
            }
        }
    }
    for r in a.iter_mut() {
        for x in r.iter_mut() {
            *x /= mc;
        }
    }
    let norm = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
    let q = ((a[0][0] + a[1][1]) + a[2][2]) / 3.0;
    let b00 = a[0][0] - q;
    let b11 = a[1][1] - q;
    let b22 = a[2][2] - q;
    let p = ((b00 * b00 + b11 * b11 + b22 * b22 + norm * 2.0) / 6.0).sqrt();
    let c00 = b11 * b22 - a[1][2] * a[1][2];
    let c01 = a[0][1] * b22 - a[1][2] * a[0][2];
    let c02 = a[0][1] * a[1][2] - b11 * a[0][2];
    let mut det = b00 * c00 - a[0][1] * c01 + a[0][2] * c02;
    det /= p * p * p;
    let half = (det * 0.5).clamp(-1.0, 1.0);
    let angle = half.acos() / 3.0;
    let ttp = 2.0 * std::f64::consts::PI / 3.0;
    let beta2 = angle.cos() * 2.0;
    let beta0 = (angle + ttp).cos() * 2.0;
    let beta1 = -(beta0 + beta2);
    let ev = [q + p * beta0, q + p * beta1, q + p * beta2];
    let cm = &b["fe_mid"][0];
    println!("q   {:016x} vs {:016x}", q.to_bits(), cm[0].to_bits());
    println!("p   {:016x} vs {:016x}", p.to_bits(), cm[1].to_bits());
    println!("det {:016x} vs {:016x}", det.to_bits(), cm[2].to_bits());
    for k in 0..3 {
        println!(
            "ev{} {:016x} vs {:016x}",
            k,
            ev[k].to_bits(),
            cm[3 + k].to_bits()
        );
    }
    println!("angle={:e} beta0={:e}", angle, beta0);
}
