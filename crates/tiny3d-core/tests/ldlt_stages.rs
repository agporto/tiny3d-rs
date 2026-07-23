use tiny3d_core::eigen_solvers::ldlt6;

fn probe_file(name: &str) -> std::path::PathBuf {
    let directory = std::env::var_os("TINY3D_PROBE_DIR")
        .expect("set TINY3D_PROBE_DIR to the directory containing C++ probe dumps");
    std::path::PathBuf::from(directory).join(name)
}

#[test]
#[ignore = "requires C++ reference dumps in TINY3D_PROBE_DIR"]
fn ldlt_stages_match() {
    let path = probe_file("ldlt_probe.txt");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    let mut lines = text.lines().peekable();
    let mut block = 0;
    while lines.peek().is_some() {
        let mut psd = [[0.0f64; 6]; 6];
        for row in psd.iter_mut() {
            let l = lines.next().unwrap();
            let toks: Vec<&str> = l.split_whitespace().collect();
            assert_eq!(toks[0], "in_psd");
            for (j, x) in row.iter_mut().enumerate() {
                *x = f64::from_bits(u64::from_str_radix(toks[1 + j], 16).unwrap());
            }
        }
        let bl = lines.next().unwrap();
        let btoks: Vec<&str> = bl.split_whitespace().collect();
        let mut b = [0.0f64; 6];
        for (i, x) in b.iter_mut().enumerate() {
            *x = f64::from_bits(u64::from_str_radix(btoks[1 + i], 16).unwrap());
        }
        let mut fac = [[0.0f64; 6]; 6];
        for row in fac.iter_mut() {
            let l = lines.next().unwrap();
            let toks: Vec<&str> = l.split_whitespace().collect();
            assert_eq!(toks[0], "fac");
            for (j, x) in row.iter_mut().enumerate() {
                *x = f64::from_bits(u64::from_str_radix(toks[1 + j], 16).unwrap());
            }
        }
        let tl = lines.next().unwrap();
        let ttoks: Vec<&str> = tl.split_whitespace().collect();
        let trans: Vec<usize> = ttoks[1..].iter().map(|t| t.parse().unwrap()).collect();
        let xl = lines.next().unwrap();
        let xtoks: Vec<&str> = xl.split_whitespace().collect();
        let mut xexp = [0.0f64; 6];
        for (i, x) in xexp.iter_mut().enumerate() {
            *x = f64::from_bits(u64::from_str_radix(xtoks[1 + i], 16).unwrap());
        }
        assert_eq!(lines.next().unwrap(), "END");

        let f = ldlt6(&psd);
        let (rfac, rtrans) = f.debug_internals();
        for i in 0..6 {
            assert_eq!(rtrans[i], trans[i], "trans[{}] block {}", i, block);
        }
        // compare lower triangle + diagonal only (upper is untouched input)
        for i in 0..6 {
            for j in 0..=i {
                assert!(
                    rfac[i][j].to_bits() == fac[i][j].to_bits(),
                    "fac[{}][{}] block {}: {:016x} vs {:016x}",
                    i,
                    j,
                    block,
                    rfac[i][j].to_bits(),
                    fac[i][j].to_bits()
                );
            }
        }
        let x = f.solve(&b);
        for i in 0..6 {
            assert!(
                x[i].to_bits() == xexp[i].to_bits(),
                "x[{}] block {}: {:016x} vs {:016x}",
                i,
                block,
                x[i].to_bits(),
                xexp[i].to_bits()
            );
        }
        block += 1;
    }
}
