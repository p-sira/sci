mod common;
use common::relative_close;
use common::{matrix_rows as matrix, vector};

use sci::{
    Error, Result,
    linalg::{self as l, MatrixInput, VectorInput},
};

// BLAS reductions can use different summation orders. A relative tolerance
// also checks the very small nonzero reference values in the fixtures.
fn check_vector(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        relative_close(a, b, 2e-14);
    }
}
fn check_matrix(a: &[Vec<f64>], b: &[Vec<f64>]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        check_vector(a, b);
    }
}

fn check_basic(a: &dyn MatrixInput, b: &dyn MatrixInput, x: &dyn VectorInput, expected: &[&str]) {
    check_matrix(&l::basic::matmul(a, b).unwrap(), &matrix(expected[1]));
    check_vector(&l::basic::matvec(a, x).unwrap(), &vector(expected[2]));
    check_matrix(&l::basic::transpose(a).unwrap(), &matrix(expected[3]));
    let reverse: Vec<_> = x.to_vector().unwrap().into_iter().rev().collect();
    relative_close(
        l::basic::dot(x, &reverse).unwrap(),
        expected[4].parse().unwrap(),
        2e-14,
    );
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
        let ft = l::basic::transpose(&a).unwrap().to_faer().unwrap();
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
    assert_eq!(
        l::solvers::solve(&nd.view_mut(), &rhs).unwrap(),
        vec![1.5, 2.0]
    );
    assert_eq!(
        l::solvers::solve(&na.view_mut((0, 0), (2, 2)), &rhs).unwrap(),
        vec![1.5, 2.0]
    );
    assert_eq!(
        l::solvers::solve(&fa.as_mut(), &rhs).unwrap(),
        vec![1.5, 2.0]
    );
    assert_eq!(nd, original);
    assert_eq!(na.to_ndarray().unwrap(), original);
    assert_eq!(fa.to_ndarray().unwrap(), original);
    assert_eq!(
        l::solvers::solve(&[[2.0, 1.0], [0.0, 3.0]], &rhs).unwrap(),
        vec![1.5, 2.0]
    );
    let mut col = [5.0, 6.0].to_faer_col().unwrap();
    assert_eq!(
        l::solvers::solve(&nd, &col.as_mut()).unwrap(),
        vec![1.5, 2.0]
    );
    let mut row = faer::Row::from_fn(2, |i| [5.0, 6.0][i]);
    assert_eq!(l::solvers::solve(&nd, &row).unwrap(), vec![1.5, 2.0]);
    assert_eq!(
        l::solvers::solve(&nd, &row.as_mut()).unwrap(),
        vec![1.5, 2.0]
    );
    let mut nx = ndarray::arr1(&[5.0, 6.0]);
    assert_eq!(
        l::solvers::solve(&nd, &nx.view_mut()).unwrap(),
        vec![1.5, 2.0]
    );
    let mut nv = rhs;
    assert_eq!(
        l::solvers::solve(&nd, &nv.rows_mut(0, 2)).unwrap(),
        vec![1.5, 2.0]
    );
}

#[test]
fn conversions_preserve_ieee_values_and_empty_axes() {
    let a = [[-0.0, f64::NAN], [f64::INFINITY, f64::NEG_INFINITY]];
    let transposed = l::basic::transpose(&a).unwrap();
    assert_eq!(transposed[0][0].to_bits(), (-0.0_f64).to_bits());
    assert_eq!(transposed[0][1], f64::INFINITY);
    assert!(transposed[1][0].is_nan());
    assert_eq!(transposed[1][1], f64::NEG_INFINITY);
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
        assert_eq!(l::basic::matvec(&fa, &vec![0.0; n]).unwrap(), vec![0.0; m]);
        assert_eq!(l::basic::transpose(&fa).unwrap(), vec![vec![0.0; m]; n]);
        assert_eq!(l::basic::transpose(&na).unwrap(), vec![vec![0.0; m]; n]);
        if m == 0 && n > 0 {
            assert_eq!(fa.to_matrix(), Err(Error::DimensionMismatch));
        } else {
            assert_eq!(fa.to_matrix().unwrap(), vec![vec![0.0; n]; m]);
        }
    }
    assert_eq!(
        l::basic::matmul(
            &ndarray::Array2::<f64>::zeros((2, 0)),
            &faer::Mat::<f64>::zeros(0, 3)
        )
        .unwrap(),
        vec![vec![0.0; 3]; 2]
    );
    let empty: Vec<f64> = vec![];
    assert_eq!(empty.to_faer_col().unwrap().vector_len(), 0);
    assert_eq!(
        l::basic::dot(
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
        assert!(matches!(l::basic::inv(a), Err(Error::NumericalFailure(_))));
        assert_eq!(l::basic::matvec(a, &[1.0]), Err(Error::DimensionMismatch));
    }
    let bad = faer::Mat::from_fn(1, 1, |_, _| f64::NAN);
    assert!(l::basic::inv(&bad).is_err());
    assert_eq!(
        l::basic::matmul(&singular, &bad),
        Err(Error::DimensionMismatch)
    );
    let jac =
        sci::differentiate::jacobian(|x| vec![2.0 * x[0] + x[1], 3.0 * x[1]], &[0.0, 0.0], 1e-4)
            .unwrap();
    let x = l::solvers::solve(&jac.to_faer().unwrap(), &nalgebra::Vector2::new(5.0, 6.0)).unwrap();
    check_vector(&x, &[1.5, 2.0]);
    let svd = l::decompositions::svd(&jac.to_ndarray().unwrap()).unwrap();
    let s =
        l::structured::diagsvd(&svd.singular_values.to_nalgebra_vector().unwrap(), 2, 2).unwrap();
    let us = l::basic::matmul(&svd.u.to_faer().unwrap(), &s.to_nalgebra().unwrap()).unwrap();
    let reconstructed = l::basic::matmul(&us, &svd.vh.to_ndarray().unwrap()).unwrap();
    // Reconstruction includes floating-point factorization error. Bound the
    // absolute residual by the matrix scale, including entries expected to be zero.
    let scale = jac.iter().flatten().fold(0.0_f64, |s, &v| s.max(v.abs()));
    for (actual, expected) in reconstructed.iter().flatten().zip(jac.iter().flatten()) {
        assert!(approx::abs_diff_eq!(
            actual,
            expected,
            epsilon = 2e-14 * scale
        ));
    }
    let a = ndarray::arr2(&[[1.0, 2.0]]);
    let b = nalgebra::Matrix2::identity();
    let blocks: [&dyn MatrixInput; 2] = [&a, &b];
    assert_eq!(
        l::structured::block_diag(&blocks).unwrap(),
        vec![
            vec![1.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0]
        ]
    );
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
    assert!(l::basic::matvec(&Oversized, &[1.0]).is_err());
    assert!(l::basic::inv(&Oversized).is_err());
    assert!(l::basic::dot(&Oversized, &[1.0]).is_err());
    let huge = ndarray::Array2::<f64>::zeros((1001, 1000));
    assert!(huge.to_matrix().is_err());
    assert!(
        l::basic::matmul(
            &ndarray::Array2::<f64>::zeros((1001, 1)),
            &faer::Mat::<f64>::zeros(1, 1000)
        )
        .is_err()
    );
}

#[test]
fn borrowed_storage_bypasses_element_conversion_and_preserves_inputs() {
    struct ViewOnly<'a>(faer::MatRef<'a, f64>);
    impl MatrixInput for ViewOnly<'_> {
        fn matrix_shape(&self) -> Result<(usize, usize)> {
            Ok((self.0.nrows(), self.0.ncols()))
        }
        fn matrix_entry(&self, _: usize, _: usize) -> f64 {
            panic!("compatible storage should be borrowed directly")
        }
        fn faer_view(&self) -> Option<faer::MatRef<'_, f64>> {
            Some(self.0)
        }
    }
    // An asymmetric matrix detects row/column confusion during refinement.
    let a = [[4.0, 1.0], [2.0, 3.0]].to_faer().unwrap();
    for view in [
        a.as_ref(),
        a.as_ref().transpose(),
        a.as_ref().reverse_rows(),
    ] {
        let expected = view.to_matrix().unwrap();
        let borrowed = ViewOnly(view);
        check_matrix(
            &l::basic::matmul(&borrowed, &[[1., 0.], [0., 1.]]).unwrap(),
            &expected,
        );
        check_vector(
            &l::basic::matvec(&borrowed, &[1., 0.]).unwrap(),
            &[expected[0][0], expected[1][0]],
        );
        let rhs = [
            expected[0][0] + expected[0][1],
            expected[1][0] + expected[1][1],
        ];
        check_vector(&l::solvers::solve(&borrowed, &rhs).unwrap(), &[1., 1.]);
        relative_close(
            l::basic::det(&borrowed).unwrap(),
            if view.row_stride() < 0 { -10. } else { 10. },
            2e-14,
        );
        relative_close(
            l::basic::norm(&borrowed, l::basic::MatrixNorm::One).unwrap(),
            l::basic::norm(&expected, l::basic::MatrixNorm::One).unwrap(),
            2e-14,
        );
        assert_eq!(view.to_matrix().unwrap(), expected);
    }
    assert_eq!(a.to_matrix().unwrap(), vec![vec![4., 1.], vec![2., 3.]]);
}

#[test]
fn zero_rank_projection_validates_native_column_count() {
    let indices = [0, 1, 2];
    let valid = ndarray::Array2::<f64>::zeros((0, 3));
    let invalid = ndarray::Array2::<f64>::zeros((0, 2));
    assert!(
        l::interpolative::reconstruct_interp_matrix(&indices, &valid)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        l::interpolative::reconstruct_interp_matrix(&indices, &invalid),
        Err(Error::DimensionMismatch)
    );
}

#[test]
fn triangular_backend_handles_all_flags_on_reversed_storage() {
    // Asymmetric entries detect a triangle or transpose mix-up.
    let a = faer::Mat::from_fn(9, 9, |i, j| {
        if i == j {
            4.0 + i as f64 / 10.0
        } else {
            (i + 2 * j) as f64 / 50.0
        }
    });
    // The same logical matrix over reversed column-major and row-major storage.
    let row_major = a.transpose().to_owned();
    let reversed_rows = a.as_ref().reverse_rows().to_owned();
    let reversed_columns = row_major.as_ref().reverse_rows().to_owned();
    let views = [
        a.as_ref().reverse_rows().reverse_cols(),
        row_major.as_ref().transpose().reverse_rows().reverse_cols(),
        // One reversed axis has neither unit stride, requiring the owned
        // triangular workspace. Keep both orientations and transpose paths.
        reversed_rows.as_ref().reverse_rows(),
        reversed_columns.as_ref().transpose().reverse_cols(),
    ];
    assert_eq!(views[0].to_matrix(), views[1].to_matrix());
    let solution: Vec<_> = (0..9).map(|i| (i as f64 + 1.0) / 3.0).collect();
    for view in views {
        for lower in [true, false] {
            for unit in [true, false] {
                for transpose in [true, false] {
                    let rhs: Vec<_> = (0..9)
                        .map(|i| {
                            (0..9)
                                .map(|j| {
                                    let (r, c) = if transpose { (j, i) } else { (i, j) };
                                    let value = if unit && r == c {
                                        1.0
                                    } else if (lower && r >= c) || (!lower && r <= c) {
                                        view[(r, c)]
                                    } else {
                                        0.0
                                    };
                                    value * solution[j]
                                })
                                .sum()
                        })
                        .collect();
                    check_vector(
                        &l::solvers::solve_triangular(&view, &rhs, lower, unit, transpose).unwrap(),
                        &solution,
                    );
                }
            }
        }
    }
}

#[test]
fn fixed_slices_and_nested_references_preserve_rectangular_layout() {
    let a = [[1., 2., 3.], [4., 5., 6.]];
    let slice = a.as_slice();
    let reference = &slice;
    let identity = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
    let expected = a.map(Vec::from);
    assert_eq!(l::basic::matmul(slice, &identity).unwrap(), expected);
    assert_eq!(l::basic::matmul(&reference, &identity).unwrap(), expected);
    assert_eq!(reference.to_matrix().unwrap(), expected);
}
