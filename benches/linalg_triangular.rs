//! Size and storage-layout sweep for triangular solves and the factor solvers
//! built on them.
//!
//! Only public APIs available since `sci-linalg` 0.2.1 are used, so the same
//! file can be run against older trees to compare implementations.

mod common;

use common::{values, warm};
use divan::{Bencher, black_box};
use sci::linalg;

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[1, 16, 64, 128, 256];

/// Well-conditioned matrix for every triangle and flag combination: small
/// off-diagonal entries and a diagonal in `[1, 2)`.
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

fn row_major(n: usize) -> ndarray::Array2<f64> {
    let a = triangular_rows(n);
    ndarray::Array2::from_shape_fn((n, n), |(i, j)| a[i][j])
}

fn col_major(n: usize) -> faer::Mat<f64> {
    let a = triangular_rows(n);
    faer::Mat::from_fn(n, n, |i, j| a[i][j])
}

/// Storage whose reversed view (negative row and column strides) is the matrix.
fn reversed_storage(n: usize) -> faer::Mat<f64> {
    let a = triangular_rows(n);
    faer::Mat::from_fn(n, n, |i, j| a[n - 1 - i][n - 1 - j])
}

/// Storage whose `[..;2, ..;2]` view (both strides non-unit) is the matrix.
fn strided_storage(n: usize) -> ndarray::Array2<f64> {
    let a = triangular_rows(n);
    ndarray::Array2::from_shape_fn((2 * n, 2 * n), |(i, j)| {
        if i % 2 == 0 && j % 2 == 0 {
            a[i / 2][j / 2]
        } else {
            f64::NAN
        }
    })
}

macro_rules! triangular_flags {
    ($($name:ident: $lower:expr, $unit:expr, $trans:expr;)*) => {$(
        mod $name {
            use super::*;

            fn run(bencher: Bencher, a: &(impl linalg::MatrixInput + ?Sized), n: usize) {
                let b = values(n, 3);
                warm(bencher, || {
                    linalg::solve_triangular(black_box(a), black_box(&b), $lower, $unit, $trans)
                        .unwrap()
                });
            }

            #[divan::bench(args = SIZES)]
            fn shared(bencher: Bencher, n: usize) {
                run(bencher, &triangular_rows(n), n);
            }

            #[divan::bench(args = SIZES)]
            fn row_major(bencher: Bencher, n: usize) {
                run(bencher, &super::row_major(n), n);
            }

            #[divan::bench(args = SIZES)]
            fn col_major(bencher: Bencher, n: usize) {
                run(bencher, &super::col_major(n), n);
            }

            #[divan::bench(args = SIZES)]
            fn reversed(bencher: Bencher, n: usize) {
                let storage = reversed_storage(n);
                let view = storage.as_ref().reverse_rows_and_cols();
                run(bencher, &view, n);
            }

            #[divan::bench(args = SIZES)]
            fn strided(bencher: Bencher, n: usize) {
                let storage = strided_storage(n);
                let view = storage.slice(ndarray::s![..;2, ..;2]);
                run(bencher, &view, n);
            }
        }
    )*};
}

/// `solve_triangular` for every (lower, unit_diagonal, transpose) combination.
mod triangular {
    use super::*;

    triangular_flags! {
        lower: true, false, false;
        lower_unit: true, true, false;
        lower_trans: true, false, true;
        lower_unit_trans: true, true, true;
        upper: false, false, false;
        upper_unit: false, true, false;
        upper_trans: false, false, true;
        upper_unit_trans: false, true, true;
    }
}

/// faer's triangular kernels called directly, without any `sci` validation,
/// on a borrowed column-major matrix, a borrowed row-major matrix, and a
/// row-major matrix first copied into column-major storage.
mod faer_kernel {
    use super::*;
    use faer::MatRef;

    fn kernel(a: MatRef<'_, f64>, x: &mut faer::Col<f64>, lower: bool, unit: bool, trans: bool) {
        let a = if trans { a.transpose() } else { a };
        let x = x.as_mat_mut();
        match (lower != trans, unit) {
            (true, false) => a.solve_lower_triangular_in_place(x),
            (false, false) => a.solve_upper_triangular_in_place(x),
            (true, true) => a.solve_unit_lower_triangular_in_place(x),
            (false, true) => a.solve_unit_upper_triangular_in_place(x),
        }
    }

    const FLAGS: &[(bool, bool, bool)] = &[
        (true, false, false),
        (true, true, false),
        (true, false, true),
        (true, true, true),
        (false, false, false),
        (false, true, false),
        (false, false, true),
        (false, true, true),
    ];

    fn sweep(bencher: Bencher, n: usize, a: MatRef<'_, f64>, copy: bool) {
        let b = values(n, 3);
        warm(bencher, || {
            for &(lower, unit, trans) in FLAGS {
                let mut x = faer::Col::from_fn(n, |i| b[i]);
                if copy {
                    let owned = black_box(a).to_owned();
                    kernel(owned.as_ref(), &mut x, lower, unit, trans);
                } else {
                    kernel(black_box(a), &mut x, lower, unit, trans);
                }
                black_box(&x);
            }
        });
    }

    /// All eight flag combinations per iteration (divide by 8 for one solve).
    #[divan::bench(args = SIZES)]
    fn col_major_borrowed(bencher: Bencher, n: usize) {
        let a = super::col_major(n);
        sweep(bencher, n, a.as_ref(), false);
    }

    #[divan::bench(args = SIZES)]
    fn row_major_borrowed(bencher: Bencher, n: usize) {
        let a = super::col_major(n).transpose().to_owned();
        sweep(bencher, n, a.as_ref().transpose(), false);
    }

    #[divan::bench(args = SIZES)]
    fn row_major_copied(bencher: Bencher, n: usize) {
        let a = super::col_major(n).transpose().to_owned();
        sweep(bencher, n, a.as_ref().transpose(), true);
    }
}

/// Factor solvers that are built on triangular substitution.
mod factor_solve {
    use super::*;

    fn square(n: usize) -> Vec<Vec<f64>> {
        let mut a = common::points(n, n, 7);
        for (i, row) in a.iter_mut().enumerate() {
            row[i] += n as f64;
        }
        a
    }

    fn spd(n: usize) -> Vec<Vec<f64>> {
        let a = square(n);
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| 0.5 * (a[i][j] + a[j][i]) + if i == j { n as f64 } else { 0.0 })
                    .collect()
            })
            .collect()
    }

    /// Tridiagonal SPD band; a 1×1 system only has the diagonal row.
    fn symmetric_band(n: usize) -> Vec<Vec<f64>> {
        if n == 1 {
            return vec![vec![4.0]];
        }
        let mut sub = values(n, 15);
        sub[n - 1] = 0.0;
        vec![vec![4.0; n], sub]
    }

    #[divan::bench(args = SIZES)]
    fn lu_solve(bencher: Bencher, n: usize) {
        let factor = linalg::lu_factor(&square(n)).unwrap();
        let b = values(n, 3);
        warm(bencher, || {
            linalg::lu_solve(black_box(&factor), black_box(&b), false).unwrap()
        });
    }

    #[divan::bench(args = SIZES)]
    fn lu_solve_trans(bencher: Bencher, n: usize) {
        let factor = linalg::lu_factor(&square(n)).unwrap();
        let b = values(n, 3);
        warm(bencher, || {
            linalg::lu_solve(black_box(&factor), black_box(&b), true).unwrap()
        });
    }

    #[divan::bench(args = SIZES)]
    fn cho_solve(bencher: Bencher, n: usize) {
        let factor = linalg::cho_factor(&spd(n), true).unwrap();
        let b = values(n, 3);
        warm(bencher, || {
            linalg::cho_solve(black_box(&factor), black_box(&b)).unwrap()
        });
    }

    #[divan::bench(args = SIZES)]
    fn cho_solve_upper(bencher: Bencher, n: usize) {
        let factor = linalg::cho_factor(&spd(n), false).unwrap();
        let b = values(n, 3);
        warm(bencher, || {
            linalg::cho_solve(black_box(&factor), black_box(&b)).unwrap()
        });
    }

    #[divan::bench(args = SIZES)]
    fn cho_solve_banded(bencher: Bencher, n: usize) {
        let cb = linalg::cholesky_banded(&symmetric_band(n), true).unwrap();
        let b = values(n, 3);
        warm(bencher, || {
            linalg::cho_solve_banded(black_box(&cb), black_box(&b), true).unwrap()
        });
    }
}

/// Kernel-only comparison on borrowed column-major storage, without `sci`
/// validation: faer's blocked triangular solve versus the direct
/// column-oriented substitution used by `solve_triangular` (copied here).
mod kernel_only {
    use super::*;
    use faer::MatRef;

    fn direct(a: MatRef<'_, f64>, x: &mut [f64], lower: bool, trans: bool) {
        let n = a.nrows();
        let column = |j: usize| a.col(j).try_as_col_major().unwrap().as_slice();
        let effective_lower = lower != trans;
        if trans {
            for k in 0..n {
                let i = if effective_lower { k } else { n - 1 - k };
                let c = column(i);
                let range = if effective_lower { 0..i } else { i + 1..n };
                let mut value = x[i];
                for (&aji, &xj) in c[range.clone()].iter().zip(&x[range]) {
                    value -= aji * xj;
                }
                x[i] = value / c[i];
            }
        } else {
            for k in 0..n {
                let j = if effective_lower { k } else { n - 1 - k };
                let c = column(j);
                x[j] /= c[j];
                let xj = x[j];
                let range = if effective_lower { j + 1..n } else { 0..j };
                for (xi, &aij) in x[range.clone()].iter_mut().zip(&c[range]) {
                    *xi -= aij * xj;
                }
            }
        }
    }

    fn faer_solve(a: MatRef<'_, f64>, x: &mut [f64], lower: bool, trans: bool) {
        let a = if trans { a.transpose() } else { a };
        let n = x.len();
        let x = faer::MatMut::from_column_major_slice_mut(x, n, 1);
        if lower != trans {
            a.solve_lower_triangular_in_place(x);
        } else {
            a.solve_upper_triangular_in_place(x);
        }
    }

    fn run(
        bencher: Bencher,
        n: usize,
        trans: bool,
        f: fn(MatRef<'_, f64>, &mut [f64], bool, bool),
    ) {
        let a = super::col_major(n);
        let b = values(n, 3);
        let mut x = b.clone();
        warm(bencher, || {
            x.copy_from_slice(&b);
            f(black_box(a.as_ref()), black_box(&mut x), true, trans);
            black_box(x[0])
        });
    }

    #[divan::bench(args = SIZES)]
    fn direct_lower(bencher: Bencher, n: usize) {
        run(bencher, n, false, direct);
    }

    #[divan::bench(args = SIZES)]
    fn faer_lower(bencher: Bencher, n: usize) {
        run(bencher, n, false, faer_solve);
    }

    #[divan::bench(args = SIZES)]
    fn direct_lower_trans(bencher: Bencher, n: usize) {
        run(bencher, n, true, direct);
    }

    #[divan::bench(args = SIZES)]
    fn faer_lower_trans(bencher: Bencher, n: usize) {
        run(bencher, n, true, faer_solve);
    }
}
