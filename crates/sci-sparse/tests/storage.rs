//! Storage and arithmetic must work without solver, RNG, or NPZ features.

use sci_sparse as sparse;

#[test]
fn canonical_storage_arithmetic_and_arrays_work_without_backends() -> sparse::Result<()> {
    let a =
        sparse::construct::from_coo(2, 2, &[(1, 1, 4.0), (0, 0, 1.0), (0, 0, 2.0), (0, 1, 0.0)])?;
    assert_eq!(a.indptr, [0, 2, 3]);
    assert_eq!(a.indices, [0, 1, 1]);
    assert_eq!(a.data, [3.0, 0.0, 4.0]);
    sparse::operations::validate(&a)?;

    let b = sparse::operations::transpose(&a)?;
    assert_eq!(sparse::operations::matvec(&b, &[2.0, 3.0])?, [6.0, 12.0]);
    let product = sparse::operations::matmul(&a, &b)?;
    assert_eq!(
        sparse::construct::to_dense(&product)?,
        [vec![9.0, 0.0], vec![0.0, 16.0]]
    );

    let array = sparse::array::as_array(&a)?;
    let expanded = sparse::array::expand_dims(&array, 1)?;
    assert_eq!(expanded.shape, [2, 1, 2]);
    assert_eq!(expanded.data, a.data);

    let malformed = sparse::construct::from_csr(1, 2, &[0, 1], &[2], &[1.0]);
    assert!(malformed.is_err());
    Ok(())
}

#[test]
fn canonical_conversions_and_filters_preserve_storage_and_signed_zero() -> sparse::Result<()> {
    let a = sparse::construct::from_csr(
        3,
        4,
        &[0, 2, 3, 5],
        &[0, 3, 1, 0, 2],
        &[-0.0, 2.0, 3.0, -4.0, 0.0],
    )?;
    assert!(a.data[0].is_sign_negative());
    let transposed = sparse::operations::transpose(&a)?;
    assert_eq!(sparse::operations::transpose(&transposed)?, a);
    assert!(transposed.data[0].is_sign_negative());

    let lower = sparse::operations::tril(&a, 0)?;
    assert_eq!(lower.indptr, [0, 1, 2, 4]);
    assert_eq!(lower.indices, [0, 1, 0, 2]);
    assert!(lower.data[0].is_sign_negative());
    let upper = sparse::operations::triu(&a, 0)?;
    assert_eq!(upper.indptr, [0, 2, 3, 4]);
    assert_eq!(upper.indices, [0, 3, 1, 2]);
    assert!(upper.data[0].is_sign_negative());
    let cleaned = sparse::operations::eliminate_zeros(&a)?;
    assert_eq!(cleaned.indptr, [0, 1, 2, 3]);
    assert_eq!(cleaned.indices, [3, 1, 0]);
    assert_eq!(cleaned.data, [2.0, 3.0, -4.0]);

    // Both format constructors must still combine unsorted duplicate inputs.
    let unordered = sparse::construct::from_csr(1, 3, &[0, 4], &[2, 0, 2, 2], &[1., 5., 2., -3.])?;
    assert_eq!(unordered.indptr, [0, 2]);
    assert_eq!(unordered.indices, [0, 2]);
    assert_eq!(unordered.data, [5.0, 0.0]);
    let csc = sparse::construct::from_csc(3, 1, &[0, 4], &[2, 0, 2, 2], &[1., 5., 2., -3.])?;
    assert_eq!(csc.indptr, [0, 1, 1, 2]);
    assert_eq!(csc.indices, [0, 0]);
    assert_eq!(csc.data, [5.0, 0.0]);

    let dense = sparse::construct::from_dense(&[vec![-0.0, 1.0], vec![2.0, 0.0]])?;
    assert_eq!(dense.indptr, [0, 1, 2]);
    assert_eq!(dense.indices, [1, 0]);
    assert_eq!(dense.data, [1.0, 2.0]);
    Ok(())
}

#[test]
fn sparse_product_preserves_accumulation_and_wide_fallback() -> sparse::Result<()> {
    let a = sparse::construct::from_coo(
        2,
        3,
        &[
            (0, 0, 1e16),
            (0, 1, 1.),
            (0, 2, -1e16),
            (1, 0, 2.),
            (1, 1, 3.),
        ],
    )?;
    let b = sparse::construct::from_coo(
        3,
        3,
        &[
            (0, 0, 1.),
            (0, 2, 2.),
            (1, 0, 1.),
            (1, 1, 4.),
            (2, 0, 1.),
            (2, 2, 2.),
        ],
    )?;
    let product = sparse::operations::matmul(&a, &b)?;
    sparse::operations::validate(&product)?;
    // Sequential accumulation is intentional: (1e16 + 1) - 1e16 = 0.
    assert_eq!(product.indptr, [0, 1, 4]);
    assert_eq!(product.indices, [1, 0, 1, 2]);
    assert_eq!(product.data, [4.0, 5.0, 12.0, 4.0]);

    let wide = sparse::construct::from_coo(3, 100_000, &[(0, 99_999, 2.), (2, 4, -3.)])?;
    let selector = sparse::construct::from_coo(1, 3, &[(0, 0, 5.), (0, 2, 7.)])?;
    let selected = sparse::operations::matmul(&selector, &wide)?;
    assert_eq!(selected.indptr, [0, 2]);
    assert_eq!(selected.indices, [4, 99_999]);
    assert_eq!(selected.data, [-21., 10.]);
    Ok(())
}

#[cfg(feature = "solver")]
#[test]
fn solver_feature_retains_public_solve() -> sparse::Result<()> {
    let a = sparse::construct::identity(2)?;
    assert_eq!(sparse::linalg::spsolve(&a, &[2.0, 3.0])?, [2.0, 3.0]);
    let diagonal = |x: &[f64]| {
        Ok(x.iter()
            .enumerate()
            .map(|(i, &v)| (i + 2) as f64 * v)
            .collect())
    };
    let op = sparse::linalg::LinearOperator::new(3, 3, diagonal, diagonal)?;
    let options = sparse::linalg::SolverOptions {
        rtol: 1e-12,
        atol: 1e-12,
        ..Default::default()
    };
    let result = sparse::linalg::cg(&op, &[2., 6., 12.], &options)?;
    assert!(result.converged);
    for (actual, expected) in result.x.iter().zip([1., 2., 3.]) {
        assert!((*actual - expected).abs() < 1e-10);
    }
    Ok(())
}

#[cfg(feature = "random")]
#[test]
fn random_feature_retains_seeded_sampling() -> sparse::Result<()> {
    assert_eq!(
        sparse::construct::random(2, 3, 1.0, 42)?,
        sparse::construct::random(2, 3, 1.0, 42)?
    );
    Ok(())
}
