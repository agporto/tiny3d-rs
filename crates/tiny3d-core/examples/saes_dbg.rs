#![allow(clippy::needless_range_loop)]

// debug binary: compare per-step QL against C++ dump
use tiny3d_core::eigen_solvers::debug_tridiag3;

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
        "usage: cargo run -p tiny3d-core --example saes_dbg -- <saes_steps.txt>\n\
         or set TINY3D_PROBE_DIR"
    );
    std::process::exit(2);
}

fn main() {
    let path = input_path("saes_steps.txt");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        eprintln!("failed to read {}: {error}", path.display());
        std::process::exit(2);
    });
    let mut lines = text.lines().peekable();
    let mut block = 0;
    while lines.peek().is_some() {
        let mut s = [[0.0f64; 3]; 3];
        for i in 0..3 {
            let t: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
            assert_eq!(t[0], "in_S");
            for j in 0..3 {
                s[i][j] = hx(t[1 + j]);
            }
        }
        let mut steps = Vec::new();
        loop {
            let l = lines.next().unwrap();
            if l.starts_with("iters") {
                let _ = lines.next();
                assert_eq!(lines.next().unwrap(), "END");
                break;
            }
            let t: Vec<&str> = l.split_whitespace().collect();
            assert_eq!(t[0], "step");
            let vals: Vec<f64> = [4, 5, 6, 8, 9].iter().map(|&i| hx(t[i])).collect();
            let se: (usize, usize) = (t[1].parse().unwrap(), t[2].parse().unwrap());
            steps.push((se, vals));
        }
        // run rust QL with per-step compare
        let (mut diag, mut subdiag, mut _q) = debug_tridiag3(&s);
        let n = 3usize;
        let mut end = n - 1;
        let mut start = 0usize;
        let mut iter = 0usize;
        let caz = f64::MIN_POSITIVE;
        let pinv = 1.0 / f64::EPSILON;
        let mut si = 0usize;
        loop {
            if end == 0 {
                break;
            }
            for i in start..end {
                if subdiag[i].abs() < caz {
                    subdiag[i] = 0.0;
                } else {
                    let ss = pinv * subdiag[i];
                    if ss * ss <= diag[i].abs() + diag[i + 1].abs() {
                        subdiag[i] = 0.0;
                    }
                }
            }
            while end > 0 && subdiag[end - 1] == 0.0 {
                end -= 1;
            }
            if end == 0 {
                break;
            }
            iter += 1;
            if iter > 30 * n {
                break;
            }
            start = end - 1;
            while start > 0 && subdiag[start - 1] != 0.0 {
                start -= 1;
            }
            tiny3d_core::eigen_solvers::debug_qr_step(&mut diag, &mut subdiag, start, end, &mut _q);
            if si < steps.len() {
                let (se, vals) = &steps[si];
                if se.0 != start || se.1 != end {
                    println!(
                        "block {} step {} start/end differ: rust {},{} cpp {},{}",
                        block, si, start, end, se.0, se.1
                    );
                }
                let mine = [diag[0], diag[1], diag[2], subdiag[0], subdiag[1]];
                for k in 0..5 {
                    if mine[k].to_bits() != vals[k].to_bits() {
                        println!(
                            "block {} step {} val{} differ: {:016x} vs {:016x}",
                            block,
                            si,
                            k,
                            mine[k].to_bits(),
                            vals[k].to_bits()
                        );
                    }
                }
            } else {
                println!("block {} extra rust step {}", block, si);
            }
            si += 1;
        }
        if si != steps.len() {
            println!(
                "block {} step count: rust {} cpp {}",
                block,
                si,
                steps.len()
            );
        }
        block += 1;
    }
    println!("done {} blocks", block);
}
