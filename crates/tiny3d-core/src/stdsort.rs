//! Independent emulation of libstdc++ (GCC 13) `std::sort` (introsort) so
//! that the relative order of tied elements matches the C++ build exactly
//! (observable
//! in nanoflann radius-search results when distances tie, e.g. duplicated
//! points).

const S_THRESHOLD: usize = 16;

#[inline]
fn lg(n: usize) -> u32 {
    // std::__lg — floor(log2(n))
    (usize::BITS - 1) - n.leading_zeros()
}

/// std::sort(v.begin(), v.end(), comp) with comp a strict-weak "less".
pub fn std_sort<T: Clone, F: Fn(&T, &T) -> bool + Copy>(v: &mut [T], comp: F) {
    if v.is_empty() {
        return;
    }
    let depth = lg(v.len()) * 2;
    introsort_loop(v, 0, v.len(), depth, comp);
    final_insertion_sort(v, 0, v.len(), comp);
}

fn introsort_loop<T: Clone, F: Fn(&T, &T) -> bool + Copy>(
    v: &mut [T],
    first: usize,
    mut last: usize,
    mut depth_limit: u32,
    comp: F,
) {
    while last - first > S_THRESHOLD {
        if depth_limit == 0 {
            // partial_sort(first, last, last) == heapsort of the range
            heap_sort(v, first, last, comp);
            return;
        }
        depth_limit -= 1;
        let cut = unguarded_partition_pivot(v, first, last, comp);
        introsort_loop(v, cut, last, depth_limit, comp);
        last = cut;
    }
}

fn unguarded_partition_pivot<T: Clone, F: Fn(&T, &T) -> bool + Copy>(
    v: &mut [T],
    first: usize,
    last: usize,
    comp: F,
) -> usize {
    let mid = first + (last - first) / 2;
    move_median_to_first(v, first, first + 1, mid, last - 1, comp);
    unguarded_partition(v, first + 1, last, first, comp)
}

fn move_median_to_first<T: Clone, F: Fn(&T, &T) -> bool + Copy>(
    v: &mut [T],
    result: usize,
    a: usize,
    b: usize,
    c: usize,
    comp: F,
) {
    if comp(&v[a], &v[b]) {
        if comp(&v[b], &v[c]) {
            v.swap(result, b);
        } else if comp(&v[a], &v[c]) {
            v.swap(result, c);
        } else {
            v.swap(result, a);
        }
    } else if comp(&v[a], &v[c]) {
        v.swap(result, a);
    } else if comp(&v[b], &v[c]) {
        v.swap(result, c);
    } else {
        v.swap(result, b);
    }
}

fn unguarded_partition<T: Clone, F: Fn(&T, &T) -> bool + Copy>(
    v: &mut [T],
    mut first: usize,
    mut last: usize,
    pivot: usize,
    comp: F,
) -> usize {
    loop {
        while comp(&v[first], &v[pivot]) {
            first += 1;
        }
        last -= 1;
        while comp(&v[pivot], &v[last]) {
            last -= 1;
        }
        if first >= last {
            return first;
        }
        v.swap(first, last);
        first += 1;
    }
}

fn final_insertion_sort<T: Clone, F: Fn(&T, &T) -> bool + Copy>(
    v: &mut [T],
    first: usize,
    last: usize,
    comp: F,
) {
    if last - first > S_THRESHOLD {
        insertion_sort(v, first, first + S_THRESHOLD, comp);
        unguarded_insertion_sort(v, first + S_THRESHOLD, last, comp);
    } else {
        insertion_sort(v, first, last, comp);
    }
}

fn insertion_sort<T: Clone, F: Fn(&T, &T) -> bool + Copy>(
    v: &mut [T],
    first: usize,
    last: usize,
    comp: F,
) {
    if first == last {
        return;
    }
    for i in (first + 1)..last {
        if comp(&v[i], &v[first]) {
            // move val to front, shifting [first, i) right
            let val = v[i].clone();
            for j in (first..i).rev() {
                v[j + 1] = v[j].clone();
            }
            v[first] = val;
        } else {
            unguarded_linear_insert(v, i, comp);
        }
    }
}

fn unguarded_insertion_sort<T: Clone, F: Fn(&T, &T) -> bool + Copy>(
    v: &mut [T],
    first: usize,
    last: usize,
    comp: F,
) {
    for i in first..last {
        unguarded_linear_insert(v, i, comp);
    }
}

fn unguarded_linear_insert<T: Clone, F: Fn(&T, &T) -> bool + Copy>(
    v: &mut [T],
    last: usize,
    comp: F,
) {
    let val = v[last].clone();
    let mut last = last;
    let mut next = last - 1;
    while comp(&val, &v[next]) {
        v[last] = v[next].clone();
        last = next;
        if next == 0 {
            break;
        }
        next -= 1;
    }
    v[last] = val;
}

// ---------------- heap ops (for the depth-limit fallback) ----------------

fn heap_sort<T: Clone, F: Fn(&T, &T) -> bool + Copy>(
    v: &mut [T],
    first: usize,
    last: usize,
    comp: F,
) {
    make_heap(v, first, last, comp);
    let mut l = last;
    while l - first > 1 {
        l -= 1;
        // __pop_heap(first, l, l)
        let value = v[l].clone();
        v[l] = v[first].clone();
        adjust_heap(v, first, 0, (l - first) as isize, value, comp);
    }
}

fn make_heap<T: Clone, F: Fn(&T, &T) -> bool + Copy>(
    v: &mut [T],
    first: usize,
    last: usize,
    comp: F,
) {
    let len = (last - first) as isize;
    if len < 2 {
        return;
    }
    let mut parent = (len - 2) / 2;
    loop {
        let value = v[first + parent as usize].clone();
        adjust_heap(v, first, parent, len, value, comp);
        if parent == 0 {
            return;
        }
        parent -= 1;
    }
}

fn adjust_heap<T: Clone, F: Fn(&T, &T) -> bool + Copy>(
    v: &mut [T],
    first: usize,
    hole_index: isize,
    len: isize,
    value: T,
    comp: F,
) {
    let top_index = hole_index;
    let mut hole_index = hole_index;
    let mut second_child = hole_index;
    while second_child < (len - 1) / 2 {
        second_child = 2 * (second_child + 1);
        if comp(
            &v[first + second_child as usize],
            &v[first + (second_child - 1) as usize],
        ) {
            second_child -= 1;
        }
        v[first + hole_index as usize] = v[first + second_child as usize].clone();
        hole_index = second_child;
    }
    if (len & 1) == 0 && second_child == (len - 2) / 2 {
        second_child = 2 * (second_child + 1);
        v[first + hole_index as usize] = v[first + (second_child - 1) as usize].clone();
        hole_index = second_child - 1;
    }
    // push_heap
    let mut parent = (hole_index - 1) / 2;
    while hole_index > top_index && comp(&v[first + parent as usize], &value) {
        v[first + hole_index as usize] = v[first + parent as usize].clone();
        hole_index = parent;
        parent = (hole_index - 1) / 2;
    }
    v[first + hole_index as usize] = value;
}
