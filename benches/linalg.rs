//! Benchmarks covering the public API of `sci::linalg`.
//!
//! Inputs are deterministic, well conditioned and small so that every
//! benchmark stays cheap under simulation. Linear-algebra calls are warmed up
//! once to exclude one-time backend initialization.

mod common;

use common::{points, spd_matrix, square_matrix, symmetric_matrix, values, warm};
use divan::{Bencher, black_box};
use sci::linalg;

fn main() {
    divan::main();
}

/// Default order for dense decompositions.
const N: usize = 16;
/// Order for expensive algorithms (matrix functions, Riccati, QZ).
const SMALL: usize = 8;

/// Matrix with diagonal entries of alternating sign, eigenvalues away from the
/// imaginary axis on both half planes.
fn mixed_matrix(n: usize) -> Vec<Vec<f64>> {
    let mut a = square_matrix(n);
    for (i, row) in a.iter_mut().enumerate() {
        if i % 2 == 1 {
            row[i] -= 2.0 * n as f64;
        }
    }
    a
}

/// Scales every entry of `a` by `factor`.
fn scaled(a: &[Vec<f64>], factor: f64) -> Vec<Vec<f64>> {
    a.iter()
        .map(|row| row.iter().map(|v| v * factor).collect())
        .collect()
}

/// Rank-`k` `m × n` matrix.
fn low_rank(m: usize, n: usize, k: usize) -> Vec<Vec<f64>> {
    let left = points(m, k, 21);
    let right = points(k, n, 22);
    linalg::matmul(&left, &right).unwrap()
}

/// Identity matrix of order `n`.
fn identity(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..n).map(|j| f64::from(u8::from(i == j))).collect())
        .collect()
}

mod basic_benches {
    use super::*;

    #[divan::bench]
    fn dot(bencher: Bencher) {
        let x = values(1_000, 1);
        let y = values(1_000, 2);
        warm(bencher, || {
            linalg::dot(black_box(&x), black_box(&y)).unwrap()
        });
    }

    #[divan::bench]
    fn matvec(bencher: Bencher) {
        let a = square_matrix(N);
        let x = values(N, 3);
        warm(bencher, || {
            linalg::matvec(black_box(&a), black_box(&x)).unwrap()
        });
    }

    #[divan::bench]
    fn matmul(bencher: Bencher) {
        let a = square_matrix(N);
        let b = symmetric_matrix(N);
        warm(bencher, || {
            linalg::matmul(black_box(&a), black_box(&b)).unwrap()
        });
    }

    #[divan::bench]
    fn transpose(bencher: Bencher) {
        let a = square_matrix(N);
        warm(bencher, || linalg::transpose(black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn norm_frobenius(bencher: Bencher) {
        let a = square_matrix(N);
        warm(bencher, || {
            linalg::norm(black_box(&a), linalg::MatrixNorm::Frobenius).unwrap()
        });
    }

    #[divan::bench]
    fn norm_spectral(bencher: Bencher) {
        let a = square_matrix(N);
        warm(bencher, || {
            linalg::norm(black_box(&a), linalg::MatrixNorm::Spectral).unwrap()
        });
    }

    #[divan::bench]
    fn bandwidth(bencher: Bencher) {
        let a = linalg::toeplitz(&values(N, 4), None).unwrap();
        warm(bencher, || linalg::bandwidth(black_box(&a)).unwrap());
    }

    /// Also covers the `ishermitian` alias.
    #[divan::bench]
    fn issymmetric(bencher: Bencher) {
        let a = symmetric_matrix(N);
        warm(bencher, || {
            linalg::issymmetric(black_box(&a), None, None).unwrap()
        });
    }

    #[divan::bench]
    fn khatri_rao(bencher: Bencher) {
        let a = points(4, N, 5);
        let b = points(4, N, 6);
        warm(bencher, || {
            linalg::khatri_rao(black_box(&a), black_box(&b)).unwrap()
        });
    }

    #[divan::bench]
    fn clarkson_woodruff_transform(bencher: Bencher) {
        let a = points(64, N, 7);
        let buckets: Vec<usize> = (0..64).map(|i| (i * 5) % 8).collect();
        let signs: Vec<f64> = (0..64)
            .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
            .collect();
        warm(bencher, || {
            linalg::clarkson_woodruff_transform(
                black_box(&a),
                8,
                black_box(&buckets),
                black_box(&signs),
            )
            .unwrap()
        });
    }
}

mod solve_benches {
    use super::*;

    #[divan::bench]
    fn solve(bencher: Bencher) {
        let a = square_matrix(N);
        let b = values(N, 3);
        warm(bencher, || {
            linalg::solve(black_box(&a), black_box(&b)).unwrap()
        });
    }

    #[divan::bench]
    fn inv(bencher: Bencher) {
        let a = square_matrix(N);
        warm(bencher, || linalg::inv(black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn det(bencher: Bencher) {
        let a = square_matrix(N);
        warm(bencher, || linalg::det(black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn solve_triangular(bencher: Bencher) {
        let a = square_matrix(N);
        let b = values(N, 3);
        warm(bencher, || {
            linalg::solve_triangular(black_box(&a), black_box(&b), true, false, false).unwrap()
        });
    }

    fn toeplitz_column() -> Vec<f64> {
        let mut c = values(N, 8);
        c[0] = 2.0 * N as f64;
        c
    }

    #[divan::bench]
    fn solve_toeplitz(bencher: Bencher) {
        let c = toeplitz_column();
        let mut r = values(N, 9);
        r[0] = c[0];
        let b = values(N, 3);
        warm(bencher, || {
            linalg::solve_toeplitz(black_box(&c), Some(black_box(&r)), black_box(&b)).unwrap()
        });
    }

    #[divan::bench]
    fn matmul_toeplitz(bencher: Bencher) {
        let c = toeplitz_column();
        let mut r = values(N, 9);
        r[0] = c[0];
        let x = points(N, 4, 10);
        warm(bencher, || {
            linalg::matmul_toeplitz(black_box(&c), Some(black_box(&r)), black_box(&x)).unwrap()
        });
    }

    #[divan::bench]
    fn solve_circulant(bencher: Bencher) {
        let c = toeplitz_column();
        let b = values(N, 3);
        warm(bencher, || {
            linalg::solve_circulant(black_box(&c), black_box(&b)).unwrap()
        });
    }

    #[divan::bench]
    fn lu_factor(bencher: Bencher) {
        let a = square_matrix(N);
        warm(bencher, || linalg::lu_factor(black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn lu_solve(bencher: Bencher) {
        let factor = linalg::lu_factor(&square_matrix(N)).unwrap();
        let b = values(N, 3);
        warm(bencher, || {
            linalg::lu_solve(black_box(&factor), black_box(&b), false).unwrap()
        });
    }

    #[divan::bench]
    fn cho_factor(bencher: Bencher) {
        let a = spd_matrix(N);
        warm(bencher, || linalg::cho_factor(black_box(&a), true).unwrap());
    }

    #[divan::bench]
    fn cho_solve(bencher: Bencher) {
        let factor = linalg::cho_factor(&spd_matrix(N), true).unwrap();
        let b = values(N, 3);
        warm(bencher, || {
            linalg::cho_solve(black_box(&factor), black_box(&b)).unwrap()
        });
    }

    #[divan::bench]
    fn lstsq(bencher: Bencher) {
        let a = points(2 * N, N, 11);
        let b = values(2 * N, 12);
        warm(bencher, || {
            linalg::lstsq(black_box(&a), black_box(&b), None).unwrap()
        });
    }

    #[divan::bench]
    fn pinv(bencher: Bencher) {
        let a = square_matrix(N);
        warm(bencher, || linalg::pinv(black_box(&a), 0.0, None).unwrap());
    }

    #[divan::bench]
    fn pinvh(bencher: Bencher) {
        let a = symmetric_matrix(N);
        warm(bencher, || {
            linalg::pinvh(black_box(&a), 0.0, None, true).unwrap()
        });
    }
}

mod banded_benches {
    use super::*;

    /// General band storage with one sub- and one super-diagonal.
    fn general_band() -> Vec<Vec<f64>> {
        vec![values(N, 13), vec![4.0; N], values(N, 14)]
    }

    /// Lower symmetric band storage of an SPD tridiagonal matrix.
    fn symmetric_band() -> Vec<Vec<f64>> {
        let mut sub = values(N, 15);
        sub[N - 1] = 0.0;
        vec![vec![4.0; N], sub]
    }

    #[divan::bench]
    fn solve_banded(bencher: Bencher) {
        let ab = general_band();
        let b = values(N, 3);
        warm(bencher, || {
            linalg::solve_banded(1, 1, black_box(&ab), black_box(&b)).unwrap()
        });
    }

    #[divan::bench]
    fn solveh_banded(bencher: Bencher) {
        let ab = symmetric_band();
        let b = values(N, 3);
        warm(bencher, || {
            linalg::solveh_banded(black_box(&ab), black_box(&b), true).unwrap()
        });
    }

    #[divan::bench]
    fn cholesky_banded(bencher: Bencher) {
        let ab = symmetric_band();
        warm(bencher, || {
            linalg::cholesky_banded(black_box(&ab), true).unwrap()
        });
    }

    #[divan::bench]
    fn cho_solve_banded(bencher: Bencher) {
        let cb = linalg::cholesky_banded(&symmetric_band(), true).unwrap();
        let b = values(N, 3);
        warm(bencher, || {
            linalg::cho_solve_banded(black_box(&cb), black_box(&b), true).unwrap()
        });
    }

    #[divan::bench]
    fn eig_banded(bencher: Bencher) {
        let ab = symmetric_band();
        warm(bencher, || {
            linalg::eig_banded(black_box(&ab), true).unwrap()
        });
    }

    #[divan::bench]
    fn eigvals_banded(bencher: Bencher) {
        let ab = symmetric_band();
        warm(bencher, || {
            linalg::eigvals_banded(black_box(&ab), true).unwrap()
        });
    }

    #[divan::bench]
    fn eigh_tridiagonal(bencher: Bencher) {
        let d = vec![4.0; N];
        let e = values(N - 1, 15);
        warm(bencher, || {
            linalg::eigh_tridiagonal(black_box(&d), black_box(&e)).unwrap()
        });
    }

    #[divan::bench]
    fn eigvalsh_tridiagonal(bencher: Bencher) {
        let d = vec![4.0; N];
        let e = values(N - 1, 15);
        warm(bencher, || {
            linalg::eigvalsh_tridiagonal(black_box(&d), black_box(&e)).unwrap()
        });
    }
}

mod decomposition_benches {
    use super::*;

    #[divan::bench]
    fn cholesky(bencher: Bencher) {
        let a = spd_matrix(N);
        warm(bencher, || linalg::cholesky(black_box(&a), true).unwrap());
    }

    #[divan::bench]
    fn lu(bencher: Bencher) {
        let a = square_matrix(N);
        warm(bencher, || linalg::lu(black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn ldl(bencher: Bencher) {
        let a = symmetric_matrix(N);
        warm(bencher, || linalg::ldl(black_box(&a), true).unwrap());
    }

    #[divan::bench]
    fn qr(bencher: Bencher) {
        let a = square_matrix(N);
        warm(bencher, || linalg::qr(black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn rq(bencher: Bencher) {
        let a = square_matrix(N);
        warm(bencher, || linalg::rq(black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn qr_multiply(bencher: Bencher) {
        let a = square_matrix(N);
        let c = symmetric_matrix(N);
        warm(bencher, || {
            linalg::qr_multiply(black_box(&a), black_box(&c), true).unwrap()
        });
    }

    #[divan::bench]
    fn qr_update(bencher: Bencher) {
        let f = linalg::qr(&square_matrix(N)).unwrap();
        let u = values(N, 16);
        let v = values(N, 17);
        warm(bencher, || {
            linalg::qr_update(
                black_box(&f.q),
                black_box(&f.r),
                black_box(&u),
                black_box(&v),
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn qr_insert(bencher: Bencher) {
        let f = linalg::qr(&square_matrix(N)).unwrap();
        let row = points(1, N, 18);
        warm(bencher, || {
            linalg::qr_insert(black_box(&f.q), black_box(&f.r), black_box(&row), N, true).unwrap()
        });
    }

    #[divan::bench]
    fn qr_delete(bencher: Bencher) {
        let f = linalg::qr(&square_matrix(N)).unwrap();
        warm(bencher, || {
            linalg::qr_delete(black_box(&f.q), black_box(&f.r), 0, 1, true).unwrap()
        });
    }

    #[divan::bench]
    fn svd(bencher: Bencher) {
        let a = square_matrix(N);
        warm(bencher, || linalg::svd(black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn svdvals(bencher: Bencher) {
        let a = square_matrix(N);
        warm(bencher, || linalg::svdvals(black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn orth(bencher: Bencher) {
        let a = low_rank(N, N, 6);
        warm(bencher, || linalg::orth(black_box(&a), None).unwrap());
    }

    #[divan::bench]
    fn null_space(bencher: Bencher) {
        let a = points(N / 2, N, 19);
        warm(bencher, || linalg::null_space(black_box(&a), None).unwrap());
    }

    #[divan::bench]
    fn polar(bencher: Bencher) {
        let a = square_matrix(N);
        warm(bencher, || linalg::polar(black_box(&a), true).unwrap());
    }

    #[divan::bench]
    fn orthogonal_procrustes(bencher: Bencher) {
        let a = square_matrix(N);
        let b = symmetric_matrix(N);
        warm(bencher, || {
            linalg::orthogonal_procrustes(black_box(&a), black_box(&b)).unwrap()
        });
    }

    #[divan::bench]
    fn subspace_angles(bencher: Bencher) {
        let a = points(N, 4, 20);
        let b = points(N, 4, 21);
        warm(bencher, || {
            linalg::subspace_angles(black_box(&a), black_box(&b)).unwrap()
        });
    }

    #[divan::bench]
    fn cossin(bencher: Bencher) {
        let x = linalg::qr(&square_matrix(SMALL)).unwrap().q;
        warm(bencher, || {
            linalg::cossin(black_box(&x), SMALL / 2, SMALL / 2).unwrap()
        });
    }
}

mod eigen_benches {
    use super::*;

    #[divan::bench]
    fn eigh(bencher: Bencher) {
        let a = symmetric_matrix(N);
        warm(bencher, || linalg::eigh(black_box(&a), true).unwrap());
    }

    #[divan::bench]
    fn eigvalsh(bencher: Bencher) {
        let a = symmetric_matrix(N);
        warm(bencher, || linalg::eigvalsh(black_box(&a), true).unwrap());
    }

    #[divan::bench]
    fn eig(bencher: Bencher) {
        let a = square_matrix(N);
        warm(bencher, || linalg::eig(black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn eigvals(bencher: Bencher) {
        let a = square_matrix(N);
        warm(bencher, || linalg::eigvals(black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn cdf2rdf(bencher: Bencher) {
        let e = linalg::eig(&square_matrix(N)).unwrap();
        warm(bencher, || {
            linalg::cdf2rdf(black_box(&e.eigenvalues), black_box(&e.eigenvectors)).unwrap()
        });
    }

    #[divan::bench]
    fn hessenberg(bencher: Bencher) {
        let a = square_matrix(N);
        warm(bencher, || linalg::hessenberg(black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn schur(bencher: Bencher) {
        let a = square_matrix(N);
        warm(bencher, || linalg::schur(black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn rsf2csf(bencher: Bencher) {
        let (t, z) = linalg::schur(&square_matrix(N)).unwrap();
        warm(bencher, || {
            linalg::rsf2csf(black_box(&t), black_box(&z)).unwrap()
        });
    }

    #[divan::bench]
    fn matrix_balance(bencher: Bencher) {
        let a: Vec<Vec<f64>> = square_matrix(N)
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .map(|(j, v)| v * 10f64.powi(i as i32 - j as i32))
                    .collect()
            })
            .collect();
        warm(bencher, || {
            linalg::matrix_balance(black_box(&a), true, true).unwrap()
        });
    }

    #[divan::bench]
    fn qz(bencher: Bencher) {
        let a = mixed_matrix(SMALL);
        let b = spd_matrix(SMALL);
        warm(bencher, || {
            linalg::qz(black_box(&a), black_box(&b)).unwrap()
        });
    }

    #[divan::bench]
    fn ordqz(bencher: Bencher) {
        let a = mixed_matrix(SMALL);
        let b = spd_matrix(SMALL);
        warm(bencher, || {
            linalg::ordqz(black_box(&a), black_box(&b), |alpha, beta| {
                alpha.re * beta.re + alpha.im * beta.im < 0.0
            })
            .unwrap()
        });
    }
}

mod matrix_function_benches {
    use super::*;

    /// Matrix with spectral radius well below one.
    fn small_norm() -> Vec<Vec<f64>> {
        scaled(&square_matrix(SMALL), 0.5 / SMALL as f64)
    }

    /// SPD matrix with eigenvalues of order one.
    fn spd() -> Vec<Vec<f64>> {
        scaled(&spd_matrix(SMALL), 1.0 / SMALL as f64)
    }

    #[divan::bench]
    fn expm(bencher: Bencher) {
        let a = small_norm();
        warm(bencher, || linalg::expm(black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn cosm(bencher: Bencher) {
        let a = small_norm();
        warm(bencher, || linalg::cosm(black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn sinm(bencher: Bencher) {
        let a = small_norm();
        warm(bencher, || linalg::sinm(black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn coshm(bencher: Bencher) {
        let a = small_norm();
        warm(bencher, || linalg::coshm(black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn sinhm(bencher: Bencher) {
        let a = small_norm();
        warm(bencher, || linalg::sinhm(black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn tanm(bencher: Bencher) {
        let a = small_norm();
        warm(bencher, || linalg::tanm(black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn tanhm(bencher: Bencher) {
        let a = small_norm();
        warm(bencher, || linalg::tanhm(black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn expm_frechet(bencher: Bencher) {
        let a = small_norm();
        let e = scaled(&symmetric_matrix(SMALL), 0.1);
        warm(bencher, || {
            linalg::expm_frechet(black_box(&a), black_box(&e)).unwrap()
        });
    }

    /// Uses order 4: the condition estimate needs one Fréchet derivative per entry.
    #[divan::bench]
    fn expm_cond(bencher: Bencher) {
        let a = scaled(&square_matrix(4), 0.5 / 4.0);
        warm(bencher, || linalg::expm_cond(black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn sqrtm(bencher: Bencher) {
        let a = spd();
        warm(bencher, || linalg::sqrtm(black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn logm(bencher: Bencher) {
        let a = spd();
        warm(bencher, || linalg::logm(black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn fractional_matrix_power(bencher: Bencher) {
        let a = spd();
        warm(bencher, || {
            linalg::fractional_matrix_power(black_box(&a), 0.3).unwrap()
        });
    }

    #[divan::bench]
    fn funm(bencher: Bencher) {
        let a = small_norm();
        warm(bencher, || {
            linalg::funm(black_box(&a), sci::complex::exp).unwrap()
        });
    }

    #[divan::bench]
    fn signm(bencher: Bencher) {
        let a = mixed_matrix(SMALL);
        warm(bencher, || linalg::signm(black_box(&a)).unwrap());
    }
}

mod equation_benches {
    use super::*;

    #[divan::bench]
    fn solve_sylvester(bencher: Bencher) {
        let a = square_matrix(SMALL);
        let b = spd_matrix(SMALL);
        let q = points(SMALL, SMALL, 23);
        warm(bencher, || {
            linalg::solve_sylvester(black_box(&a), black_box(&b), black_box(&q)).unwrap()
        });
    }

    /// Also covers the `solve_lyapunov` alias.
    #[divan::bench]
    fn solve_continuous_lyapunov(bencher: Bencher) {
        let a = scaled(&square_matrix(SMALL), -1.0);
        let q = spd_matrix(SMALL);
        warm(bencher, || {
            linalg::solve_continuous_lyapunov(black_box(&a), black_box(&q)).unwrap()
        });
    }

    #[divan::bench]
    fn solve_discrete_lyapunov(bencher: Bencher) {
        let a = scaled(&square_matrix(SMALL), 0.5 / SMALL as f64);
        let q = spd_matrix(SMALL);
        warm(bencher, || {
            linalg::solve_discrete_lyapunov(black_box(&a), black_box(&q)).unwrap()
        });
    }

    #[divan::bench]
    fn solve_continuous_are(bencher: Bencher) {
        let a = scaled(&square_matrix(SMALL), -1.0 / SMALL as f64);
        let b = identity(SMALL);
        let q = identity(SMALL);
        let r = identity(SMALL);
        warm(bencher, || {
            linalg::solve_continuous_are(black_box(&a), black_box(&b), black_box(&q), black_box(&r))
                .unwrap()
        });
    }

    #[divan::bench]
    fn solve_discrete_are(bencher: Bencher) {
        let a = scaled(&square_matrix(SMALL), 0.5 / SMALL as f64);
        let b = identity(SMALL);
        let q = identity(SMALL);
        let r = identity(SMALL);
        warm(bencher, || {
            linalg::solve_discrete_are(black_box(&a), black_box(&b), black_box(&q), black_box(&r))
                .unwrap()
        });
    }
}

mod structured_benches {
    use super::*;
    use linalg::{DftScale, PascalKind};

    #[divan::bench]
    fn toeplitz(bencher: Bencher) {
        let c = values(N, 1);
        let r = values(N, 2);
        bencher.bench(|| linalg::toeplitz(black_box(&c), Some(black_box(&r))).unwrap());
    }

    #[divan::bench]
    fn hankel(bencher: Bencher) {
        let c = values(N, 1);
        let r = values(N, 2);
        bencher.bench(|| linalg::hankel(black_box(&c), Some(black_box(&r))).unwrap());
    }

    #[divan::bench]
    fn circulant(bencher: Bencher) {
        let c = values(N, 1);
        bencher.bench(|| linalg::circulant(black_box(&c)).unwrap());
    }

    #[divan::bench]
    fn hilbert(bencher: Bencher) {
        bencher.bench(|| linalg::hilbert(black_box(N)).unwrap());
    }

    #[divan::bench]
    fn invhilbert(bencher: Bencher) {
        bencher.bench(|| linalg::invhilbert(black_box(SMALL)).unwrap());
    }

    #[divan::bench]
    fn pascal(bencher: Bencher) {
        bencher.bench(|| linalg::pascal(black_box(N), PascalKind::Symmetric).unwrap());
    }

    #[divan::bench]
    fn invpascal(bencher: Bencher) {
        bencher.bench(|| linalg::invpascal(black_box(N), PascalKind::Lower).unwrap());
    }

    #[divan::bench]
    fn hadamard(bencher: Bencher) {
        bencher.bench(|| linalg::hadamard(black_box(N)).unwrap());
    }

    #[divan::bench]
    fn helmert(bencher: Bencher) {
        bencher.bench(|| linalg::helmert(black_box(N), true).unwrap());
    }

    #[divan::bench]
    fn fiedler(bencher: Bencher) {
        let a = values(N, 1);
        bencher.bench(|| linalg::fiedler(black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn companion(bencher: Bencher) {
        let mut a = values(N + 1, 1);
        a[0] = 1.0;
        bencher.bench(|| linalg::companion(black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn fiedler_companion(bencher: Bencher) {
        let mut a = values(N + 1, 1);
        a[0] = 1.0;
        bencher.bench(|| linalg::fiedler_companion(black_box(&a)).unwrap());
    }

    #[divan::bench]
    fn leslie(bencher: Bencher) {
        let f = common::unit_values(N, 1);
        let s = common::unit_values(N - 1, 2);
        bencher.bench(|| linalg::leslie(black_box(&f), black_box(&s)).unwrap());
    }

    #[divan::bench]
    fn block_diag(bencher: Bencher) {
        let blocks = vec![square_matrix(4), symmetric_matrix(5), spd_matrix(7)];
        bencher.bench(|| linalg::block_diag(black_box(&blocks)).unwrap());
    }

    #[divan::bench]
    fn convolution_matrix(bencher: Bencher) {
        let a = values(SMALL, 1);
        bencher.bench(|| linalg::convolution_matrix(black_box(&a), N, sci::Mode::Full).unwrap());
    }

    #[divan::bench]
    fn dft(bencher: Bencher) {
        bencher.bench(|| linalg::dft(black_box(N), DftScale::SqrtN).unwrap());
    }

    #[divan::bench]
    fn diagsvd(bencher: Bencher) {
        let s = common::positive_values(SMALL, 1);
        bencher.bench(|| linalg::diagsvd(black_box(&s), N, SMALL).unwrap());
    }
}

mod interpolative_benches {
    use super::*;
    use linalg::interpolative::{self, Rank};

    const RANK: usize = 4;

    #[divan::bench]
    fn interp_decomp(bencher: Bencher) {
        let a = low_rank(N, N, RANK);
        warm(bencher, || {
            interpolative::interp_decomp(black_box(&a), Rank::Fixed(RANK)).unwrap()
        });
    }

    #[divan::bench]
    fn reconstruct_interp_matrix(bencher: Bencher) {
        let id = interpolative::interp_decomp(&low_rank(N, N, RANK), Rank::Fixed(RANK)).unwrap();
        warm(bencher, || {
            interpolative::reconstruct_interp_matrix(
                black_box(&id.indices),
                black_box(&id.projection),
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn reconstruct_skel_matrix(bencher: Bencher) {
        let a = low_rank(N, N, RANK);
        let id = interpolative::interp_decomp(&a, Rank::Fixed(RANK)).unwrap();
        warm(bencher, || {
            interpolative::reconstruct_skel_matrix(black_box(&a), id.rank, black_box(&id.indices))
                .unwrap()
        });
    }

    #[divan::bench]
    fn reconstruct_matrix_from_id(bencher: Bencher) {
        let a = low_rank(N, N, RANK);
        let id = interpolative::interp_decomp(&a, Rank::Fixed(RANK)).unwrap();
        let b = interpolative::reconstruct_skel_matrix(&a, id.rank, &id.indices).unwrap();
        warm(bencher, || {
            interpolative::reconstruct_matrix_from_id(
                black_box(&b),
                black_box(&id.indices),
                black_box(&id.projection),
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn svd(bencher: Bencher) {
        let a = low_rank(N, N, RANK);
        warm(bencher, || {
            interpolative::svd(black_box(&a), Rank::Fixed(RANK)).unwrap()
        });
    }

    #[divan::bench]
    fn id_to_svd(bencher: Bencher) {
        let a = low_rank(N, N, RANK);
        let id = interpolative::interp_decomp(&a, Rank::Fixed(RANK)).unwrap();
        let b = interpolative::reconstruct_skel_matrix(&a, id.rank, &id.indices).unwrap();
        warm(bencher, || {
            interpolative::id_to_svd(
                black_box(&b),
                black_box(&id.indices),
                black_box(&id.projection),
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn estimate_rank(bencher: Bencher) {
        let a = low_rank(N, N, RANK);
        warm(bencher, || {
            interpolative::estimate_rank(black_box(&a), 1e-10).unwrap()
        });
    }

    #[divan::bench]
    fn estimate_spectral_norm(bencher: Bencher) {
        let a = square_matrix(N);
        warm(bencher, || {
            interpolative::estimate_spectral_norm(black_box(&a)).unwrap()
        });
    }

    #[divan::bench]
    fn estimate_spectral_norm_diff(bencher: Bencher) {
        let a = square_matrix(N);
        let b = symmetric_matrix(N);
        warm(bencher, || {
            interpolative::estimate_spectral_norm_diff(black_box(&a), black_box(&b)).unwrap()
        });
    }
}

mod dispatch_benches {
    use super::*;

    #[divan::bench]
    fn find_best_blas_type(bencher: Bencher) {
        let arrays = vec![square_matrix(N), symmetric_matrix(N)];
        bencher.bench(|| linalg::find_best_blas_type(black_box(&arrays)).unwrap());
    }

    #[divan::bench]
    fn get_blas_funcs(bencher: Bencher) {
        let names = ["dot", "nrm2", "gemv", "gemm"];
        bencher.bench(|| linalg::get_blas_funcs(black_box(&names)).unwrap());
    }

    #[divan::bench]
    fn get_lapack_funcs(bencher: Bencher) {
        let names = [
            "gesv", "getrf", "getrs", "potrf", "potrs", "geev", "syev", "gesvd", "gges",
        ];
        bencher.bench(|| linalg::get_lapack_funcs(black_box(&names)).unwrap());
    }
}

mod interop_benches {
    use super::*;
    use linalg::{MatrixInput, VectorInput};

    #[divan::bench]
    fn to_matrix(bencher: Bencher) {
        let a = square_matrix(N).to_faer().unwrap();
        bencher.bench(|| black_box(&a).to_matrix().unwrap());
    }

    #[divan::bench]
    fn to_ndarray(bencher: Bencher) {
        let a = square_matrix(N);
        bencher.bench(|| black_box(&a).to_ndarray().unwrap());
    }

    #[divan::bench]
    fn to_nalgebra(bencher: Bencher) {
        let a = square_matrix(N);
        bencher.bench(|| black_box(&a).to_nalgebra().unwrap());
    }

    #[divan::bench]
    fn to_faer(bencher: Bencher) {
        let a = square_matrix(N);
        bencher.bench(|| black_box(&a).to_faer().unwrap());
    }

    #[divan::bench]
    fn to_vector(bencher: Bencher) {
        let x = values(256, 1).to_faer_col().unwrap();
        bencher.bench(|| black_box(&x).to_vector().unwrap());
    }

    #[divan::bench]
    fn to_ndarray_vector(bencher: Bencher) {
        let x = values(256, 1);
        bencher.bench(|| black_box(&x).to_ndarray_vector().unwrap());
    }

    #[divan::bench]
    fn to_nalgebra_vector(bencher: Bencher) {
        let x = values(256, 1);
        bencher.bench(|| black_box(&x).to_nalgebra_vector().unwrap());
    }

    #[divan::bench]
    fn to_faer_col(bencher: Bencher) {
        let x = values(256, 1);
        bencher.bench(|| black_box(&x).to_faer_col().unwrap());
    }

    /// Products on library-native inputs (ndarray × nalgebra).
    #[divan::bench]
    fn matmul_mixed_inputs(bencher: Bencher) {
        let a = square_matrix(N).to_ndarray().unwrap();
        let b = symmetric_matrix(N).to_nalgebra().unwrap();
        warm(bencher, || {
            linalg::matmul(black_box(&a), black_box(&b)).unwrap()
        });
    }
}
