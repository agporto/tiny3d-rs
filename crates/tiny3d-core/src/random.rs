//! Global RNG matching `tiny3d::utility::random`: std::mt19937 (libstdc++)
//! plus libstdc++'s `uniform_int_distribution` (Lemire downscaling, since the
//! mt19937 range is exactly 2^32-1).

use std::sync::Mutex;

#[derive(Clone)]
pub struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = (1812433253u32.wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30)))
                .wrapping_add(i as u32);
        }
        Mt19937 { state, index: 624 }
    }

    fn generate(&mut self) {
        const M: usize = 397;
        const MATRIX_A: u32 = 0x9908b0df;
        const UPPER_MASK: u32 = 0x80000000;
        const LOWER_MASK: u32 = 0x7fffffff;
        for i in 0..624 {
            let y = (self.state[i] & UPPER_MASK) | (self.state[(i + 1) % 624] & LOWER_MASK);
            let mut next = self.state[(i + M) % 624] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= MATRIX_A;
            }
            self.state[i] = next;
        }
        self.index = 0;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.generate();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }
}

static ENGINE: Mutex<Option<Mt19937>> = Mutex::new(None);

fn with_engine<R>(f: impl FnOnce(&mut Mt19937) -> R) -> R {
    let mut guard = ENGINE.lock().unwrap();
    if guard.is_none() {
        // Match C++: randomly seeded by std::random_device on first use.
        let mut buf = [0u8; 4];
        getrandom(&mut buf);
        *guard = Some(Mt19937::new(u32::from_ne_bytes(buf)));
    }
    f(guard.as_mut().unwrap())
}

fn getrandom(buf: &mut [u8]) {
    // Use /dev/urandom; fall back to a time-derived value.
    use std::io::Read;
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        if f.read_exact(buf).is_ok() {
            return;
        }
    }
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    buf.copy_from_slice(&t.to_ne_bytes());
}

/// Snapshot the global engine state (initializing it exactly as a first
/// draw would). Used by deterministic batch-parallel RANSAC to rewind
/// speculative draws so the engine ends in the exact serial state.
pub fn snapshot_engine() -> Mt19937 {
    with_engine(|e| e.clone())
}

/// Restore a previously snapshotted engine state.
pub fn restore_engine(s: Mt19937) {
    *ENGINE.lock().unwrap() = Some(s);
}

/// `tiny3d.utility.random.seed(seed)`
pub fn seed(seed_value: i32) {
    let mut guard = ENGINE.lock().unwrap();
    *guard = Some(Mt19937::new(seed_value as u32));
}

/// libstdc++ `uniform_int_distribution<int>` (low, high) over mt19937.
/// Uses Lemire's nearly-divisionless algorithm (`_S_nd<u64>`), which is the
/// path taken when the generator range is exactly 2^32-1.
pub struct UniformIntGenerator {
    low: i64,
    range_plus_one: u32, // (high - low + 1)
}

impl UniformIntGenerator {
    pub fn new(low: i64, high: i64) -> Self {
        assert!(low >= 0, "low must be >= 0");
        assert!(low <= high, "low must be <= high");
        UniformIntGenerator {
            low,
            range_plus_one: (high - low + 1) as u32,
        }
    }

    pub fn next(&self) -> i64 {
        let range = self.range_plus_one;
        let val = with_engine(|e| {
            let mut product = (e.next_u32() as u64) * (range as u64);
            let mut low32 = product as u32;
            if low32 < range {
                let threshold = (range.wrapping_neg()) % range;
                while low32 < threshold {
                    product = (e.next_u32() as u64) * (range as u64);
                    low32 = product as u32;
                }
            }
            (product >> 32) as u32
        });
        self.low + val as i64
    }
}
