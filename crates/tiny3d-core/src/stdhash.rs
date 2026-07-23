//! Emulation of libstdc++'s `std::unordered_map` / `std::unordered_set`
//! (GCC 13, _Prime_rehash_policy, max_load_factor 1.0) keyed by
//! `Eigen::Vector3i` with tiny3d's `utility::hash_eigen`.
//!
//! tiny3d's voxel operations iterate these containers, so their exact
//! iteration order is observable in outputs (e.g. the point order produced by
//! `voxel_down_sample`). This module reproduces bucket growth, per-bucket
//! insertion, and rehash node ordering so iteration order matches the C++.

/// Probed from GCC 13's libstdc++ (`__prime_list`); entries ≤ ~21M.
const PRIME_LIST: &[usize] = &[
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    103, 109, 113, 127, 137, 139, 149, 157, 167, 179, 193, 199, 211, 227, 241, 257, 277, 293, 313,
    337, 359, 383, 409, 439, 467, 503, 541, 577, 619, 661, 709, 761, 823, 887, 953, 1031, 1109,
    1193, 1289, 1381, 1493, 1613, 1741, 1879, 2029, 2179, 2357, 2549, 2753, 2971, 3209, 3469, 3739,
    4027, 4349, 4703, 5087, 5503, 5953, 6427, 6949, 7517, 8123, 8783, 9497, 10273, 11113, 12011,
    12983, 14033, 15173, 16411, 17749, 19183, 20753, 22447, 24281, 26267, 28411, 30727, 33223,
    35933, 38873, 42043, 45481, 49201, 53201, 57557, 62233, 67307, 72817, 78779, 85229, 92203,
    99733, 107897, 116731, 126271, 136607, 147793, 159871, 172933, 187091, 202409, 218971, 236897,
    256279, 277261, 299951, 324503, 351061, 379787, 410857, 444487, 480881, 520241, 562841, 608903,
    658753, 712697, 771049, 834181, 902483, 976369, 1056323, 1142821, 1236397, 1337629, 1447153,
    1565659, 1693859, 1832561, 1982627, 2144977, 2320627, 2510653, 2716249, 2938679, 3179303,
    3439651, 3721303, 4026031, 4355707, 4712381, 5098259, 5515729, 5967347, 6456007, 6984629,
    7556579, 8175383, 8844859, 9569143, 10352717, 11200489, 12117689, 13109983, 14183539, 15345007,
    16601593, 17961079, 19431899, 21023161,
];

/// `__fast_bkt` from hashtable_c++0x.cc: for n < 13 use this table.
const FAST_BKT: [usize; 13] = [1, 2, 2, 3, 5, 5, 7, 7, 11, 11, 11, 11, 13];

fn next_bkt(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    if n < 13 {
        return FAST_BKT[n];
    }
    match PRIME_LIST.binary_search(&n) {
        Ok(i) => PRIME_LIST[i],
        Err(i) if i < PRIME_LIST.len() => PRIME_LIST[i],
        _ => {
            // Beyond probed range: fall back to next prime >= n.
            let mut c = n | 1;
            loop {
                if is_prime(c) {
                    return c;
                }
                c += 2;
            }
        }
    }
}

fn is_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    if n.is_multiple_of(2) {
        return n == 2;
    }
    let mut d = 3usize;
    while d * d <= n {
        if n.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

/// tiny3d `utility::hash_eigen<Eigen::Vector3i>`.
pub fn hash_vector3i(v: &[i32; 3]) -> u64 {
    let mut seed: u64 = 0;
    for &e in v.iter() {
        // std::hash<int> = static_cast<size_t>(int) (sign-extending)
        let h = e as i64 as u64;
        seed ^= h
            .wrapping_add(0x9e3779b9)
            .wrapping_add(seed << 6)
            .wrapping_add(seed >> 2);
    }
    seed
}

struct HtNode<V> {
    key: [i32; 3],
    value: V,
    hash: u64,
    next: Option<usize>, // index into nodes arena, global singly-linked list
}

/// Order-faithful emulation of libstdc++ _Hashtable with unique keys.
pub struct StdUnorderedMap<V> {
    nodes: Vec<HtNode<V>>,
    /// buckets[b] = Some(node index of the node BEFORE the first node of
    /// bucket b in the global list); usize::MAX encodes `&_M_before_begin`.
    buckets: Vec<Option<usize>>,
    head: Option<usize>, // _M_before_begin._M_nxt
    bucket_count: usize,
    element_count: usize,
    next_resize: usize,
}

const BEFORE_BEGIN: usize = usize::MAX;

impl<V> Default for StdUnorderedMap<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V> StdUnorderedMap<V> {
    pub fn new() -> Self {
        StdUnorderedMap {
            nodes: Vec::new(),
            buckets: vec![None; 1],
            head: None,
            bucket_count: 1,
            element_count: 0,
            next_resize: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.element_count
    }
    pub fn is_empty(&self) -> bool {
        self.element_count == 0
    }

    /// `reserve(n)`: like C++ reserve — rehash to bucket count next_bkt(ceil(n)).
    pub fn reserve(&mut self, n: usize) {
        let bkts = next_bkt(n);
        if bkts > self.bucket_count {
            self.do_rehash(bkts);
        }
        // _M_next_resize updated by _M_next_bkt: floor(bkt * load_factor)
        self.next_resize = bkts;
    }

    fn bucket_index(hash: u64, count: usize) -> usize {
        (hash % count as u64) as usize
    }

    fn node_first_of_bucket(&self, b: usize) -> Option<usize> {
        match self.buckets[b] {
            None => None,
            Some(BEFORE_BEGIN) => self.head,
            Some(prev) => self.nodes[prev].next,
        }
    }

    pub fn get(&self, key: &[i32; 3]) -> Option<&V> {
        let h = hash_vector3i(key);
        let b = Self::bucket_index(h, self.bucket_count);
        let mut cur = self.node_first_of_bucket(b);
        while let Some(i) = cur {
            let n = &self.nodes[i];
            if Self::bucket_index(n.hash, self.bucket_count) != b {
                break;
            }
            if n.key == *key {
                return Some(&n.value);
            }
            cur = n.next;
        }
        None
    }

    pub fn get_mut_or_insert_with(
        &mut self,
        key: &[i32; 3],
        default: impl FnOnce() -> V,
    ) -> &mut V {
        let h = hash_vector3i(key);
        let b = Self::bucket_index(h, self.bucket_count);
        let mut cur = self.node_first_of_bucket(b);
        while let Some(i) = cur {
            let n = &self.nodes[i];
            if Self::bucket_index(n.hash, self.bucket_count) != b {
                break;
            }
            if n.key == *key {
                return &mut self.nodes[i].value;
            }
            cur = n.next;
        }
        let idx = self.insert_new(*key, h, default());
        &mut self.nodes[idx].value
    }

    /// insert if absent; returns true if inserted.
    pub fn insert(&mut self, key: [i32; 3], value: V) -> bool {
        let h = hash_vector3i(&key);
        let b = Self::bucket_index(h, self.bucket_count);
        let mut cur = self.node_first_of_bucket(b);
        while let Some(i) = cur {
            let n = &self.nodes[i];
            if Self::bucket_index(n.hash, self.bucket_count) != b {
                break;
            }
            if n.key == key {
                return false;
            }
            cur = n.next;
        }
        self.insert_new(key, h, value);
        true
    }

    fn insert_new(&mut self, key: [i32; 3], hash: u64, value: V) -> usize {
        // _M_need_rehash(bucket_count, element_count, 1)
        if self.element_count + 1 > self.next_resize {
            let min_bkts_num = std::cmp::max(
                self.element_count + 1,
                if self.next_resize != 0 { 0 } else { 11 },
            );
            let min_bkts = min_bkts_num as f64; // / max_load_factor (1.0)
            if min_bkts >= self.bucket_count as f64 {
                let target = std::cmp::max(min_bkts.floor() as usize + 1, self.bucket_count * 2);
                let new_count = next_bkt(target);
                self.next_resize = new_count; // floor(bkt * 1.0)
                self.do_rehash(new_count);
            } else {
                self.next_resize = self.bucket_count; // floor(bkt * load)
            }
        }

        let node_idx = self.nodes.len();
        self.nodes.push(HtNode {
            key,
            value,
            hash,
            next: None,
        });
        let bkt = Self::bucket_index(hash, self.bucket_count);
        self.insert_bucket_begin(bkt, node_idx);
        self.element_count += 1;
        node_idx
    }

    fn insert_bucket_begin(&mut self, bkt: usize, node: usize) {
        if let Some(prev) = self.buckets[bkt] {
            // bucket not empty: insert after bucket's before-begin
            let first = if prev == BEFORE_BEGIN {
                self.head
            } else {
                self.nodes[prev].next
            };
            self.nodes[node].next = first;
            if prev == BEFORE_BEGIN {
                self.head = Some(node);
            } else {
                self.nodes[prev].next = Some(node);
            }
        } else {
            // bucket empty: insert at global begin
            self.nodes[node].next = self.head;
            let old_head = self.head;
            self.head = Some(node);
            if let Some(next) = old_head {
                let next_bktidx = Self::bucket_index(self.nodes[next].hash, self.bucket_count);
                self.buckets[next_bktidx] = Some(node);
            }
            self.buckets[bkt] = Some(BEFORE_BEGIN);
        }
    }

    fn do_rehash(&mut self, new_count: usize) {
        // _M_rehash_aux(unique keys): walk global list, reinsert into new buckets
        let mut new_buckets: Vec<Option<usize>> = vec![None; new_count];
        let mut p = self.head;
        self.head = None;
        let mut bbegin_bkt = 0usize;
        while let Some(i) = p {
            let next = self.nodes[i].next;
            let bkt = Self::bucket_index(self.nodes[i].hash, new_count);
            if new_buckets[bkt].is_none() {
                self.nodes[i].next = self.head;
                let had_next = self.nodes[i].next.is_some();
                self.head = Some(i);
                new_buckets[bkt] = Some(BEFORE_BEGIN);
                if had_next {
                    new_buckets[bbegin_bkt] = Some(i);
                }
                bbegin_bkt = bkt;
            } else {
                let prev = new_buckets[bkt].unwrap();
                let first = if prev == BEFORE_BEGIN {
                    self.head
                } else {
                    self.nodes[prev].next
                };
                self.nodes[i].next = first;
                if prev == BEFORE_BEGIN {
                    self.head = Some(i);
                } else {
                    self.nodes[prev].next = Some(i);
                }
            }
            p = next;
        }
        self.buckets = new_buckets;
        self.bucket_count = new_count;
    }

    /// Iterate in C++ iteration order (the global singly-linked list).
    pub fn iter(&self) -> StdMapIter<'_, V> {
        StdMapIter {
            map: self,
            cur: self.head,
        }
    }
}

pub struct StdMapIter<'a, V> {
    map: &'a StdUnorderedMap<V>,
    cur: Option<usize>,
}

impl<'a, V> Iterator for StdMapIter<'a, V> {
    type Item = (&'a [i32; 3], &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        let i = self.cur?;
        let n = &self.map.nodes[i];
        self.cur = n.next;
        Some((&n.key, &n.value))
    }
}

/// `unordered_set<Vector3i>` emulation.
pub type StdUnorderedSet = StdUnorderedMap<()>;
