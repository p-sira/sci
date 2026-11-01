//! Storage and arithmetic must work even without solver, graph, RNG, or NPZ features.

use sci_sparse as sparse;

#[test]
fn canonical_storage_arithmetic_and_arrays_work_without_backends() -> sparse::Result<()> {
    let a = sparse::from_coo(2, 2, &[(1, 1, 4.0), (0, 0, 1.0), (0, 0, 2.0), (0, 1, 0.0)])?;
    assert_eq!(a.indptr, [0, 2, 3]);
    assert_eq!(a.indices, [0, 1, 1]);
    assert_eq!(a.data, [3.0, 0.0, 4.0]);
    sparse::validate(&a)?;

    let b = sparse::transpose(&a)?;
    assert_eq!(sparse::matvec(&b, &[2.0, 3.0])?, [6.0, 12.0]);
    let product = sparse::matmul(&a, &b)?;
    assert_eq!(
        sparse::to_dense(&product)?,
        [vec![9.0, 0.0], vec![0.0, 16.0]]
    );

    let array = sparse::as_array(&a)?;
    let expanded = sparse::expand_dims(&array, 1)?;
    assert_eq!(expanded.shape, [2, 1, 2]);
    assert_eq!(expanded.data, a.data);

    let malformed = sparse::from_csr(1, 2, &[0, 1], &[2], &[1.0]);
    assert!(malformed.is_err());
    Ok(())
}

#[cfg(feature = "solver")]
#[test]
fn solver_feature_retains_public_solve() -> sparse::Result<()> {
    let a = sparse::identity(2)?;
    assert_eq!(sparse::linalg::spsolve(&a, &[2.0, 3.0])?, [2.0, 3.0]);
    Ok(())
}

#[cfg(feature = "graph")]
#[test]
fn graph_feature_retains_public_graph_algorithms() -> sparse::Result<()> {
    let a = sparse::identity(2)?;
    let connected = sparse::csgraph::connected_components(&a, false, false)?;
    assert_eq!(connected.count, 2);
    Ok(())
}

#[cfg(feature = "random")]
#[test]
fn random_feature_retains_seeded_sampling() -> sparse::Result<()> {
    assert_eq!(
        sparse::random(2, 3, 1.0, 42)?,
        sparse::random(2, 3, 1.0, 42)?
    );
    Ok(())
}
