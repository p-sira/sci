//! Prints heap allocation counts and bytes for one call of each case in
//! `linalg_triangular.rs` (after a warm-up call), as CSV.

mod common;

use common::values;
use sci::linalg;
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct Counting;
static COUNT: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        COUNT.fetch_add(1, Relaxed);
        BYTES.fetch_add(layout.size(), Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        COUNT.fetch_add(1, Relaxed);
        BYTES.fetch_add(new_size, Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn measure<T>(name: &str, n: usize, mut f: impl FnMut() -> T) {
    black_box(f());
    let (c0, b0) = (COUNT.load(Relaxed), BYTES.load(Relaxed));
    black_box(f());
    let (c1, b1) = (COUNT.load(Relaxed), BYTES.load(Relaxed));
    println!("{name},{n},{},{}", c1 - c0, b1 - b0);
}

fn triangular_rows(n: usize) -> Vec<Vec<f64>> {
    let v = values(n * n, 21);
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let x = v[i * n + j];
                    if i == j { 1.5 + 0.5 * x } else { x / n as f64 }
                })
                .collect()
        })
        .collect()
}

fn main() {
    println!("case,n,allocations,bytes");
    let flags = [
        ("lower", true, false, false),
        ("lower_unit", true, true, false),
        ("lower_trans", true, false, true),
        ("lower_unit_trans", true, true, true),
        ("upper", false, false, false),
        ("upper_unit", false, true, false),
        ("upper_trans", false, false, true),
        ("upper_unit_trans", false, true, true),
    ];
    for n in [1usize, 16, 64, 128, 256] {
        let rows = triangular_rows(n);
        let row_major = ndarray::Array2::from_shape_fn((n, n), |(i, j)| rows[i][j]);
        let col_major = faer::Mat::from_fn(n, n, |i, j| rows[i][j]);
        let rev = faer::Mat::from_fn(n, n, |i, j| rows[n - 1 - i][n - 1 - j]);
        let reversed = rev.as_ref().reverse_rows_and_cols();
        let wide = ndarray::Array2::from_shape_fn((2 * n, 2 * n), |(i, j)| {
            if i % 2 == 0 && j % 2 == 0 {
                rows[i / 2][j / 2]
            } else {
                f64::NAN
            }
        });
        let strided = wide.slice(ndarray::s![..;2, ..;2]);
        let b = values(n, 3);
        for (name, lower, unit, trans) in flags {
            measure(&format!("triangular::{name}::shared"), n, || {
                linalg::solve_triangular(&rows, &b, lower, unit, trans).unwrap()
            });
            measure(&format!("triangular::{name}::row_major"), n, || {
                linalg::solve_triangular(&row_major, &b, lower, unit, trans).unwrap()
            });
            measure(&format!("triangular::{name}::col_major"), n, || {
                linalg::solve_triangular(&col_major, &b, lower, unit, trans).unwrap()
            });
            measure(&format!("triangular::{name}::reversed"), n, || {
                linalg::solve_triangular(&reversed, &b, lower, unit, trans).unwrap()
            });
            measure(&format!("triangular::{name}::strided"), n, || {
                linalg::solve_triangular(&strided, &b, lower, unit, trans).unwrap()
            });
        }
        let mut sq = common::points(n, n, 7);
        for (i, row) in sq.iter_mut().enumerate() {
            row[i] += n as f64;
        }
        let spd: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| 0.5 * (sq[i][j] + sq[j][i]) + if i == j { n as f64 } else { 0.0 })
                    .collect()
            })
            .collect();
        let lu = linalg::lu_factor(&sq).unwrap();
        measure("factor_solve::lu_solve", n, || {
            linalg::lu_solve(&lu, &b, false).unwrap()
        });
        measure("factor_solve::lu_solve_trans", n, || {
            linalg::lu_solve(&lu, &b, true).unwrap()
        });
        let cl = linalg::cho_factor(&spd, true).unwrap();
        measure("factor_solve::cho_solve", n, || {
            linalg::cho_solve(&cl, &b).unwrap()
        });
        let cu = linalg::cho_factor(&spd, false).unwrap();
        measure("factor_solve::cho_solve_upper", n, || {
            linalg::cho_solve(&cu, &b).unwrap()
        });
        let band = if n == 1 {
            vec![vec![4.0]]
        } else {
            let mut sub = values(n, 15);
            sub[n - 1] = 0.0;
            vec![vec![4.0; n], sub]
        };
        let cb = linalg::cholesky_banded(&band, true).unwrap();
        measure("factor_solve::cho_solve_banded", n, || {
            linalg::cho_solve_banded(&cb, &b, true).unwrap()
        });
    }
}
