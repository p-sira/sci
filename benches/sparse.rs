//! Deterministic CSR construction and arithmetic workloads.

use divan::{Bencher, black_box};
use sci::SparseMatrix;
use sci::sparse::{construct, linalg, operations};

fn main() {
    divan::main();
}

fn matrix(n: usize) -> SparseMatrix {
    let entries: Vec<_> = (0..n)
        .flat_map(|r| {
            (0..10).map(move |k| {
                (
                    r,
                    (r * 17 + k * 29) % n,
                    if k == 0 { 0. } else { (k + 1) as f64 / 8. },
                )
            })
        })
        .collect();
    construct::from_coo(n, n, &entries).unwrap()
}

#[divan::bench(args = [128, 256, 512])]
fn canonical_csr_conversion(bencher: Bencher, n: usize) {
    let a = matrix(n);
    bencher.bench(|| {
        construct::from_csr(
            n,
            n,
            black_box(&a.indptr),
            black_box(&a.indices),
            black_box(&a.data),
        )
        .unwrap()
    });
}

#[divan::bench(args = [128, 256, 512])]
fn canonical_csc_conversion(bencher: Bencher, n: usize) {
    let a = operations::transpose(&matrix(n)).unwrap();
    bencher.bench(|| {
        construct::from_csc(
            n,
            n,
            black_box(&a.indptr),
            black_box(&a.indices),
            black_box(&a.data),
        )
        .unwrap()
    });
}

#[divan::bench(args = [128, 256])]
fn dense_to_csr(bencher: Bencher, n: usize) {
    let a = construct::to_dense(&matrix(n)).unwrap();
    bencher.bench(|| construct::from_dense(black_box(&a)).unwrap());
}

#[divan::bench(args = [128, 256, 512])]
fn transpose(bencher: Bencher, n: usize) {
    let a = matrix(n);
    bencher.bench(|| operations::transpose(black_box(&a)).unwrap());
}

#[divan::bench(args = [128, 256, 512])]
fn eliminate_zeros(bencher: Bencher, n: usize) {
    let a = matrix(n);
    bencher.bench(|| operations::eliminate_zeros(black_box(&a)).unwrap());
}

#[divan::bench(args = [128, 256, 512])]
fn lower_triangle(bencher: Bencher, n: usize) {
    let a = matrix(n);
    bencher.bench(|| operations::tril(black_box(&a), black_box(0)).unwrap());
}

#[divan::bench(args = [128, 256, 512])]
fn csr_matmul(bencher: Bencher, n: usize) {
    let a = matrix(n);
    let b = operations::transpose(&a).unwrap();
    bencher.bench(|| operations::matmul(black_box(&a), black_box(&b)).unwrap());
}

#[divan::bench]
fn wide_sparse_matmul(bencher: Bencher) {
    let a = construct::from_coo(1, 3, &[(0, 0, 2.), (0, 2, 3.)]).unwrap();
    let b = construct::from_coo(3, 100_000, &[(0, 99_999, 1.), (2, 4, 4.)]).unwrap();
    bencher.bench(|| operations::matmul(black_box(&a), black_box(&b)).unwrap());
}

#[divan::bench(args = [64, 128])]
fn sparse_cg(bencher: Bencher, n: usize) {
    let a = construct::identity(n).unwrap();
    let rhs = vec![1.0; n];
    let options = linalg::SolverOptions::default();
    bencher.bench(|| linalg::cg(black_box(&a), black_box(&rhs), black_box(&options)).unwrap());
}
