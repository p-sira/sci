mod common;
// Small, well-conditioned reconstruction identities; allow rounding
// accumulated by factorization and matrix products with 2e-12 bounds.
use common::matrix_close;
use sci::{Error, linalg as l};

#[test]
fn eigenvalues_only_paths_preserve_order_trace_and_validation() {
    // The analytic spectrum of this tridiagonal Toeplitz matrix is
    // 2 + 2 cos(k pi/(n+1)); exercise the inclusive dense order limit.
    for n in [1, 16, 256] {
        let d = vec![2.0; n];
        let e = vec![1.0; n - 1];
        let values = l::eigvalsh_tridiagonal(&d, &e).unwrap();
        assert_eq!(values.len(), n);
        assert!(values.windows(2).all(|w| w[0] <= w[1]));
        for (i, &value) in values.iter().enumerate() {
            let expected =
                2.0 - 2.0 * (((i + 1) as f64 * std::f64::consts::PI) / (n + 1) as f64).cos();
            // An absolute bound handles eigenvalues close to zero.
            assert!((value - expected).abs() < 2e-13);
        }
        assert!((values.iter().sum::<f64>() - 2.0 * n as f64).abs() < 2e-11);
    }
    assert!(l::eigvalsh_tridiagonal(&[1.0, 2.0], &[]).is_err());
    assert!(l::eigvalsh_tridiagonal(&vec![1.0; 257], &vec![0.0; 256]).is_err());
    for nonfinite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(l::eigvalsh_tridiagonal(&[nonfinite], &[]).is_err());
        assert!(l::eigvalsh_tridiagonal(&[1.0, 2.0], &[nonfinite]).is_err());
        for lower in [false, true] {
            // Validate even the triangle not selected for computation.
            assert!(l::eigvalsh(&[[1.0, nonfinite], [nonfinite, 2.0]], lower).is_err());
            assert!(l::eigvals_banded(&[[nonfinite]], lower).is_err());
        }
    }
}

#[test]
fn eigenvalues_only_paths_scale_extreme_inputs_without_losing_relative_accuracy() {
    // A symmetric 2x2 matrix with diagonal 2s and off-diagonal s has
    // eigenvalues s and 3s. Normalize before comparing to test tiny scales.
    for scale in [1e-320, 1e-200, 1.0, 1e200, 1e307] {
        let a = [[2.0 * scale, scale], [scale, 2.0 * scale]];
        for lower in [false, true] {
            let ab = if lower {
                [[2.0 * scale, 2.0 * scale], [scale, 0.0]]
            } else {
                [[0.0, scale], [2.0 * scale, 2.0 * scale]]
            };
            for values in [
                l::eigvalsh(&a, lower).unwrap(),
                l::eigvals_banded(&ab, lower).unwrap(),
                l::eigvalsh_tridiagonal(&[2.0 * scale; 2], &[scale]).unwrap(),
            ] {
                for (value, expected) in values.into_iter().zip([1.0, 3.0]) {
                    assert!((value / scale - expected).abs() < 2e-14);
                }
            }
        }
    }
}

fn identity(n: usize) -> l::Matrix {
    (0..n)
        .map(|i| (0..n).map(|j| f64::from(i == j)).collect())
        .collect()
}

fn projector(a: &l::Matrix) -> l::Matrix {
    // Shared rows cannot retain the column count of a 0 × n transpose.
    if a[0].is_empty() {
        vec![vec![0.; a.len()]; a.len()]
    } else {
        l::matmul(a, &l::transpose(a).unwrap()).unwrap()
    }
}

#[test]
fn singular_lu_preserves_rectangular_reconstruction_and_rejects_solves() {
    // A zero first pivot followed by a nonzero pivot exercises faer's singular
    // LU recovery, including later row swaps of previously computed L columns.
    for a in [
        vec![vec![0., 2., 1.], vec![0., 0., 3.], vec![0., 5., 4.]],
        vec![vec![1., 2., 3.], vec![0., 0., 4.], vec![2., 4., 1.]],
        vec![vec![0., 1.], vec![0., 3.], vec![0., 2.]],
        vec![vec![0., 2., 3.], vec![0., 1., 4.]],
    ] {
        let f = l::lu(&a).unwrap();
        let k = a.len().min(a[0].len());
        assert_eq!((f.l.len(), f.l[0].len()), (a.len(), k));
        assert_eq!((f.u.len(), f.u[0].len()), (k, a[0].len()));
        matrix_close(
            &l::matmul(&l::matmul(&f.p, &f.l).unwrap(), &f.u).unwrap(),
            &a,
            2e-12,
            2e-12,
        );
        if a.len() == a[0].len() {
            let f = l::lu_factor(&a).unwrap();
            assert!(f.packed.iter().flatten().all(|x| x.is_finite()));
            for transpose in [false, true] {
                assert!(matches!(
                    l::lu_solve(&f, &vec![1.; a.len()], transpose),
                    Err(Error::NumericalFailure(_))
                ));
            }
        }
    }
}

#[test]
fn rank_cutoff_is_strict_and_projectors_partition_the_space() {
    let a = [[4., 0.], [0., 1.]];
    for (cutoff, rank) in [(0.25 - f64::EPSILON, 2), (0.25, 1), (1., 0)] {
        let (inverse, actual) = l::pinv(&a, 0., Some(cutoff)).unwrap();
        assert_eq!(actual, rank);
        let range = l::orth(&a, Some(cutoff)).unwrap();
        let null = l::null_space(&a, Some(cutoff)).unwrap();
        assert_eq!(range[0].len(), rank);
        assert_eq!(null[0].len(), 2 - rank);
        let p = projector(&range);
        let q = projector(&null);
        let sum: l::Matrix = p
            .iter()
            .zip(&q)
            .map(|(p, q)| p.iter().zip(q).map(|(p, q)| p + q).collect())
            .collect();
        matrix_close(&sum, &identity(2), 2e-12, 2e-12);
        matrix_close(&l::matmul(&a, &inverse).unwrap(), &p, 2e-12, 2e-12);
        let fit = l::lstsq(&a, &[4., 1.], Some(cutoff)).unwrap();
        assert_eq!(fit.rank, rank);
        assert!(fit.residuals.is_empty());
    }
}

#[test]
fn dense_axis_limit_accepts_256_and_rejects_257() {
    // Exact identity references avoid storing a large redundant fixture.
    let a = identity(256);
    let b = vec![1.; 256];
    assert_eq!(l::solve(&a, &b).unwrap(), b);
    assert_eq!(l::det(&a).unwrap(), 1.);
    matrix_close(&l::cholesky(&a, true).unwrap(), &a, 2e-12, 2e-12);
    let too_large = identity(257);
    for result in [l::inv(&too_large), l::cholesky(&too_large, true)] {
        assert!(matches!(result, Err(Error::InvalidInput(_))));
    }
    assert!(matches!(
        l::solve(&too_large, &vec![1.; 257]),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn overflow_is_reported_in_rank_thresholds_and_factor_outputs() {
    let a = [[2., 0.], [0., 1.]];
    assert_eq!(
        l::pinv(&a, f64::MAX, Some(f64::MAX)),
        Err(Error::NonFiniteEvaluation)
    );
    assert_eq!(
        l::pinvh(&a, f64::MAX, Some(f64::MAX), true),
        Err(Error::NonFiniteEvaluation)
    );
    assert_eq!(l::orth(&a, Some(f64::MAX)), Err(Error::NonFiniteEvaluation));
    assert_eq!(
        l::null_space(&a, Some(f64::MAX)),
        Err(Error::NonFiniteEvaluation)
    );
    assert!(matches!(
        l::lstsq(&a, &[1., 1.], Some(f64::MAX)),
        Err(Error::NonFiniteEvaluation)
    ));
    let overflowing = [[f64::MAX, f64::MAX], [f64::MAX, -f64::MAX]];
    assert!(matches!(
        l::qr(&overflowing),
        Err(Error::NonFiniteEvaluation)
    ));
    assert!(matches!(
        l::svdvals(&overflowing),
        Err(Error::NonFiniteEvaluation)
    ));
    assert!(matches!(
        l::eigvalsh(&overflowing, true),
        Err(Error::NonFiniteEvaluation)
    ));
    assert!(matches!(
        l::norm(&overflowing, l::MatrixNorm::Frobenius),
        Err(Error::NonFiniteEvaluation)
    ));
}

#[test]
fn principal_functions_validate_callbacks_and_singular_branches() {
    let jordan = [[2., 1.], [0., 2.]];
    assert!(matches!(
        l::funm(&jordan, |z| z),
        Err(Error::NumericalFailure(_))
    ));
    assert_eq!(
        l::funm(&[[1.]], |mut z| {
            z.re = f64::NAN;
            z
        }),
        Err(Error::NonFiniteEvaluation)
    );
    assert_eq!(
        l::funm(&[[1.]], |mut z| {
            z.im = f64::INFINITY;
            z
        }),
        Err(Error::NonFiniteEvaluation)
    );
    let singular = [[0., 0.], [0., 1.]];
    assert!(matches!(
        l::fractional_matrix_power(&singular, -1.),
        Err(Error::NumericalFailure(_))
    ));
    assert!(matches!(
        l::fractional_matrix_power(&singular, 0.5),
        Err(Error::InvalidInput(_))
    ));
    let zero_power = l::fractional_matrix_power(&singular, 0.).unwrap();
    for (i, row) in zero_power.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            assert_eq!(value.re, f64::from(i == j));
            assert_eq!(value.im, 0.);
        }
    }
    for exponent in [f64::NAN, f64::INFINITY, 4_294_967_297.] {
        assert!(matches!(
            l::fractional_matrix_power(&[[1.]], exponent),
            Err(Error::InvalidInput(_))
        ));
    }
    let rotation = [[0., -1.], [1., 0.]];
    assert!(matches!(l::signm(&rotation), Err(Error::InvalidInput(_))));
}

#[test]
fn structured_work_limits_and_matrix_equation_overflow_are_explicit() {
    let long = vec![1.; 257];
    for result in [
        l::solve_toeplitz(&long, None, &long),
        l::solve_toeplitz(&[1.], Some(&long), &[1.]),
        l::solve_circulant(&long, &long),
    ] {
        assert!(matches!(result, Err(Error::InvalidInput(_))));
    }
    assert!(matches!(
        l::matmul_toeplitz(&long, None, &[[1.]]),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        l::matmul_toeplitz(&[1.], Some(&long), &[[1.]]),
        Err(Error::InvalidInput(_))
    ));
    assert_eq!(
        l::solve_discrete_lyapunov(&[[f64::MAX]], &[[1.]]),
        Err(Error::NonFiniteEvaluation)
    );
    assert_eq!(
        l::rsf2csf(&identity(2), &identity(3)),
        Err(Error::DimensionMismatch)
    );
    let nontriangular = [[1., 0., 0.], [0., 2., 0.], [1., 0., 3.]];
    assert!(matches!(
        l::rsf2csf(&nontriangular, &identity(3)),
        Err(Error::InvalidInput(_))
    ));
    use l::interpolative as ip;
    assert_eq!(
        ip::reconstruct_matrix_from_id(&[[1.]], &[0, 1], &[[1.], [2.]]),
        Err(Error::DimensionMismatch)
    );
}

#[test]
fn balancing_isolated_columns_preserves_similarity_and_checks_overflow() {
    // The first column is isolated while its row is coupled to the remaining
    // irreducible block. This exercises the lower-end permutation path.
    let a = [[1., 0., 2.], [0., 3., 4.], [0., 5., 6.]];
    for scale in [false, true] {
        let (balanced, t) = l::matrix_balance(&a, true, scale).unwrap();
        matrix_close(
            &l::matmul(&l::matmul(&t, &balanced).unwrap(), &l::inv(&t).unwrap()).unwrap(),
            &a.map(Vec::from),
            2e-12,
            2e-12,
        );
    }
    let huge = [[1., f64::MAX, f64::MAX], [1., 2., 1.], [1., 1., 3.]];
    assert_eq!(
        l::matrix_balance(&huge, false, true),
        Err(Error::NonFiniteEvaluation)
    );
    let overflowing = [[f64::MAX, f64::MAX], [f64::MAX, -f64::MAX]];
    assert!(matches!(
        l::eig(&overflowing),
        Err(Error::NonFiniteEvaluation)
    ));
}

#[test]
fn qz_selection_runs_once_per_eigenvalue_and_reports_inseparable_pairs() {
    let mut calls = 0;
    let result = l::ordqz(&[[2., 0.], [0., -1.]], &identity(2), |alpha, beta| {
        calls += 1;
        alpha.re * beta.re < 0.
    })
    .unwrap();
    assert_eq!(calls, 2);
    assert_eq!(result.selected, 1);
    assert!(result.alpha[0].re * result.beta[0].re < 0.);

    // A stateful callback can select only one of two identical eigenvalues.
    // The documented reordering failure must remain a Result, without a panic.
    calls = 0;
    let result = l::ordqz(&identity(2), &identity(2), |_, _| {
        calls += 1;
        calls == 2
    });
    assert_eq!(calls, 2);
    assert!(matches!(result, Err(Error::NumericalFailure(_))));
}

#[test]
fn triangular_and_reusable_lu_solves_reject_arithmetic_overflow() {
    // Finite, nonsingular inputs can overflow during either substitution.
    // The reciprocal of the smallest positive subnormal exceeds f64::MAX.
    let tiny = f64::from_bits(1);
    for transpose in [false, true] {
        assert_eq!(
            l::solve_triangular(&[[tiny]], &[1.], true, false, transpose),
            Err(Error::NonFiniteEvaluation)
        );
        let factor = l::LuFactor {
            packed: vec![vec![tiny]],
            row_permutation: vec![0],
        };
        assert_eq!(
            l::lu_solve(&factor, &[1.], transpose),
            Err(Error::NonFiniteEvaluation)
        );
        // A unit lower factor can overflow before the upper solve, too.
        let factor = l::LuFactor {
            packed: vec![vec![1., f64::MAX], vec![f64::MAX, 1.]],
            row_permutation: vec![0, 1],
        };
        assert_eq!(
            l::lu_solve(&factor, &[2., 2.], transpose),
            Err(Error::NonFiniteEvaluation)
        );
    }
}

#[test]
fn interpolative_projection_rejects_nonfinite_coefficients() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            l::interpolative::reconstruct_interp_matrix(&[1, 0], &[[value]]),
            Err(Error::InvalidInput("coordinates must be finite"))
        );
    }
}
