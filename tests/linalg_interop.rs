mod common;
use common::{matrix_rows as matrix, vector};

use sci::{
    Error, Result,
    linalg::{self as l, MatrixInput, VectorInput},
};

fn close(a: f64, b: f64) {
    // BLAS reductions can use different summation orders. A relative tolerance
    // also checks the very small nonzero reference values in the third case.
    assert!(
        (a - b).abs() <= 2e-14 * b.abs().max(f64::MIN_POSITIVE),
        "{a} != {b}"
    );
}
fn check_vector(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        close(a, b);
    }
}
fn check_matrix(a: &[Vec<f64>], b: &[Vec<f64>]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        check_vector(a, b);
    }
}

fn check_basic(a: &dyn MatrixInput, b: &dyn MatrixInput, x: &dyn VectorInput, expected: &[&str]) {
    check_matrix(&l::matmul(a, b).unwrap(), &matrix(expected[1]));
    check_vector(&l::matvec(a, x).unwrap(), &vector(expected[2]));
    check_matrix(&l::transpose(a).unwrap(), &matrix(expected[3]));
    let reverse: Vec<_> = x.to_vector().unwrap().into_iter().rev().collect();
    close(l::dot(x, &reverse).unwrap(), expected[4].parse().unwrap());
}

#[test]
fn mixed_library_products_and_strided_views_match_scipy() {
    let inputs = common::lines(include_str!("fixtures/linalg_interop_inputs.csv"));
    let expected = common::lines(include_str!("fixtures/linalg_interop_expected.csv"));
    for (input, expected) in inputs.zip(expected) {
        let c: Vec<_> = common::fields(input);
        let e: Vec<_> = common::fields(expected);
        assert_eq!(c[0], e[0]);
        let (a, b, x) = (matrix(c[1]), matrix(c[2]), vector(c[3]));
        let (m, n) = a.matrix_shape().unwrap();
        // Padding forces noncontiguous views; nonsymmetric rectangular matrices
        // detect accidental interpretation of column-major memory as row-major.
        let mut nd = ndarray::Array2::from_elem((m * 2, n * 2), 99.0);
        for i in 0..m {
            for j in 0..n {
                nd[(i * 2, j * 2)] = a[i][j];
            }
        }
        let nd_view = nd.slice(ndarray::s![..;2, ..;2]);
        let mut nx = ndarray::Array1::from_elem(x.len() * 2, 99.0);
        for (i, &v) in x.iter().enumerate() {
            nx[i * 2] = v;
        }
        let nx_view = nx.slice(ndarray::s![..;2]);
        let nb = b.to_nalgebra().unwrap();
        check_basic(&nd_view, &nb, &nx_view, &e);

        let na = nd.to_nalgebra().unwrap();
        let na_view = na.view_with_steps((0, 0), (m, n), (1, 1));
        let fb = b.to_faer().unwrap();
        let fx = x.to_faer_col().unwrap();
        check_basic(&na_view, &fb, &fx.as_ref(), &e);
        check_basic(&na_view, &fb.as_ref(), &fx.as_ref().transpose(), &e);

        let fa = a.to_faer().unwrap();
        let reversed = faer::Mat::from_fn(m, n, |i, j| a[m - 1 - i][n - 1 - j]);
        let reversed_x = faer::Col::from_fn(x.len(), |i| x[x.len() - 1 - i]);
        check_basic(
            &reversed.as_ref().reverse_rows().reverse_cols(),
            &nb,
            &reversed_x.as_ref().reverse_rows(),
            &e,
        );
        let ft = l::transpose(&a).unwrap().to_faer().unwrap();
        let nx = x.to_nalgebra_vector().unwrap();
        check_basic(&ft.as_ref().transpose(), &b.to_ndarray().unwrap(), &nx, &e);
        check_basic(&fa, &nb, &x, &e);
        check_matrix(&nd_view.to_matrix().unwrap(), &a);
        check_matrix(&na_view.to_matrix().unwrap(), &a);
        check_matrix(&fa.to_matrix().unwrap(), &a);
    }
}

#[test]
fn mutable_views_are_read_only_and_fixed_storage_is_supported() {
    let mut nd = ndarray::arr2(&[[2.0, 1.0], [0.0, 3.0]]);
    let original = nd.clone();
    let mut na = nalgebra::Matrix2::new(2.0, 1.0, 0.0, 3.0);
    let mut fa = nd.to_faer().unwrap();
    let rhs = nalgebra::Vector2::new(5.0, 6.0);
    assert_eq!(l::solve(&nd.view_mut(), &rhs).unwrap(), vec![1.5, 2.0]);
    assert_eq!(
        l::solve(&na.view_mut((0, 0), (2, 2)), &rhs).unwrap(),
        vec![1.5, 2.0]
    );
    assert_eq!(l::solve(&fa.as_mut(), &rhs).unwrap(), vec![1.5, 2.0]);
    assert_eq!(nd, original);
    assert_eq!(na.to_ndarray().unwrap(), original);
    assert_eq!(fa.to_ndarray().unwrap(), original);
    assert_eq!(
        l::solve(&[[2.0, 1.0], [0.0, 3.0]], &rhs).unwrap(),
        vec![1.5, 2.0]
    );
    let mut col = [5.0, 6.0].to_faer_col().unwrap();
    assert_eq!(l::solve(&nd, &col.as_mut()).unwrap(), vec![1.5, 2.0]);
    let mut row = faer::Row::from_fn(2, |i| [5.0, 6.0][i]);
    assert_eq!(l::solve(&nd, &row).unwrap(), vec![1.5, 2.0]);
    assert_eq!(l::solve(&nd, &row.as_mut()).unwrap(), vec![1.5, 2.0]);
    let mut nx = ndarray::arr1(&[5.0, 6.0]);
    assert_eq!(l::solve(&nd, &nx.view_mut()).unwrap(), vec![1.5, 2.0]);
    let mut nv = rhs;
    assert_eq!(l::solve(&nd, &nv.rows_mut(0, 2)).unwrap(), vec![1.5, 2.0]);
}

#[test]
fn conversions_preserve_ieee_values_and_empty_axes() {
    let a = [[-0.0, f64::NAN], [f64::INFINITY, f64::NEG_INFINITY]];
    for converted in [
        a.to_ndarray().unwrap().to_matrix().unwrap(),
        a.to_nalgebra().unwrap().to_matrix().unwrap(),
        a.to_faer().unwrap().to_matrix().unwrap(),
    ] {
        assert_eq!(converted[0][0].to_bits(), (-0.0_f64).to_bits());
        assert!(converted[0][1].is_nan());
        assert_eq!(converted[1], vec![f64::INFINITY, f64::NEG_INFINITY]);
    }
    let v = [-0.0, f64::NAN, f64::INFINITY];
    for converted in [
        v.to_ndarray_vector().unwrap().to_vector().unwrap(),
        v.to_nalgebra_vector().unwrap().to_vector().unwrap(),
        v.to_faer_col().unwrap().to_vector().unwrap(),
    ] {
        assert_eq!(converted[0].to_bits(), v[0].to_bits());
        assert!(converted[1].is_nan());
        assert_eq!(converted[2], f64::INFINITY);
    }
    for (m, n) in [(0, 0), (0, 3), (3, 0)] {
        let nd = ndarray::Array2::<f64>::zeros((m, n));
        let na = nd.to_nalgebra().unwrap();
        let fa = na.to_faer().unwrap();
        assert_eq!(fa.to_ndarray().unwrap().dim(), (m, n));
        assert_eq!(na.matrix_shape().unwrap(), (m, n));
        assert_eq!(fa.matrix_shape().unwrap(), (m, n));
        assert_eq!(l::matvec(&fa, &vec![0.0; n]).unwrap(), vec![0.0; m]);
        assert_eq!(l::transpose(&na).unwrap(), vec![vec![0.0; m]; n]);
        if m == 0 && n > 0 {
            assert_eq!(fa.to_matrix(), Err(Error::DimensionMismatch));
        } else {
            assert_eq!(fa.to_matrix().unwrap(), vec![vec![0.0; n]; m]);
        }
    }
    assert_eq!(
        l::matmul(
            &ndarray::Array2::<f64>::zeros((2, 0)),
            &faer::Mat::<f64>::zeros(0, 3)
        )
        .unwrap(),
        vec![vec![0.0; 3]; 2]
    );
    let empty: Vec<f64> = vec![];
    assert_eq!(empty.to_faer_col().unwrap().vector_len(), 0);
    assert_eq!(
        l::dot(
            &empty.to_ndarray_vector().unwrap(),
            &empty.to_nalgebra_vector().unwrap()
        )
        .unwrap(),
        0.0
    );
}

#[test]
fn errors_and_shared_results_interoperate_across_libraries() {
    let ragged = vec![vec![1.0], vec![2.0, 3.0]];
    assert_eq!(ragged.to_faer().unwrap_err(), Error::DimensionMismatch);
    let singular = ndarray::arr2(&[[1.0, 2.0], [2.0, 4.0]]);
    for a in [
        &singular as &dyn MatrixInput,
        &singular.to_faer().unwrap(),
        &singular.to_nalgebra().unwrap(),
    ] {
        assert!(matches!(l::inv(a), Err(Error::NumericalFailure(_))));
        assert_eq!(l::matvec(a, &[1.0]), Err(Error::DimensionMismatch));
    }
    let bad = faer::Mat::from_fn(1, 1, |_, _| f64::NAN);
    assert!(l::inv(&bad).is_err());
    assert_eq!(l::matmul(&singular, &bad), Err(Error::DimensionMismatch));
    let jac =
        sci::differentiate::jacobian(|x| vec![2.0 * x[0] + x[1], 3.0 * x[1]], &[0.0, 0.0], 1e-4)
            .unwrap();
    let x = l::solve(&jac.to_faer().unwrap(), &nalgebra::Vector2::new(5.0, 6.0)).unwrap();
    check_vector(&x, &[1.5, 2.0]);
    let svd = l::svd(&jac.to_ndarray().unwrap()).unwrap();
    let s = l::diagsvd(&svd.singular_values.to_nalgebra_vector().unwrap(), 2, 2).unwrap();
    let us = l::matmul(&svd.u.to_faer().unwrap(), &s.to_nalgebra().unwrap()).unwrap();
    let reconstructed = l::matmul(&us, &svd.vh.to_ndarray().unwrap()).unwrap();
    // Reconstruction includes floating-point factorization error. Bound the
    // absolute residual by the matrix scale, including entries expected to be zero.
    let scale = jac.iter().flatten().fold(0.0_f64, |s, &v| s.max(v.abs()));
    for (actual, expected) in reconstructed.iter().flatten().zip(jac.iter().flatten()) {
        assert!((actual - expected).abs() <= 2e-14 * scale);
    }
    let a = ndarray::arr2(&[[1.0, 2.0]]);
    let b = nalgebra::Matrix2::identity();
    let blocks: [&dyn MatrixInput; 2] = [&a, &b];
    assert_eq!(
        l::block_diag(&blocks).unwrap(),
        vec![
            vec![1.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0]
        ]
    );
    assert!(!l::find_best_blas_type(&blocks).unwrap().prefer_fortran);
}

struct Oversized;
impl MatrixInput for Oversized {
    fn matrix_shape(&self) -> Result<(usize, usize)> {
        Ok((usize::MAX, 2))
    }
    fn matrix_entry(&self, _: usize, _: usize) -> f64 {
        panic!("must validate dimensions before reading")
    }
}
impl VectorInput for Oversized {
    fn vector_len(&self) -> usize {
        usize::MAX
    }
    fn vector_entry(&self, _: usize) -> f64 {
        panic!("must validate dimensions before reading")
    }
}

#[test]
fn conversion_work_is_bounded_before_reading_or_allocating() {
    assert!(Oversized.to_matrix().is_err());
    assert!(Oversized.to_ndarray().is_err());
    assert!(Oversized.to_nalgebra().is_err());
    assert!(Oversized.to_faer().is_err());
    assert!(Oversized.to_vector().is_err());
    assert!(Oversized.to_ndarray_vector().is_err());
    assert!(Oversized.to_nalgebra_vector().is_err());
    assert!(Oversized.to_faer_col().is_err());
    assert!(l::matvec(&Oversized, &[1.0]).is_err());
    assert!(l::inv(&Oversized).is_err());
    assert!(l::dot(&Oversized, &[1.0]).is_err());
    let huge = ndarray::Array2::<f64>::zeros((1001, 1000));
    assert!(huge.to_matrix().is_err());
    assert!(
        l::matmul(
            &ndarray::Array2::<f64>::zeros((1001, 1)),
            &faer::Mat::<f64>::zeros(1, 1000)
        )
        .is_err()
    );
}
