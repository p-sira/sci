use sci_core::{
    Error,
    csr::{from_coo, to_dense, validate},
};

#[test]
fn canonical_coo_stable_duplicate_summation_and_signed_zero() {
    let a = from_coo(
        2,
        3,
        &[
            (1, 2, -0.0),
            (0, 1, 1e16),
            (0, 2, 3.0),
            (0, 1, -1e16),
            (0, 1, 1.0),
            (1, 0, -2.0),
            (1, 0, 2.0),
        ],
    )
    .unwrap();
    assert_eq!(a.indptr, [0, 2, 4]);
    assert_eq!(a.indices, [1, 2, 0, 2]);
    assert_eq!(a.data, [1.0, 3.0, 0.0, -0.0]);
    assert_eq!(a.data[3].to_bits(), (-0.0f64).to_bits());
    validate(&a).unwrap();
    let dense = to_dense(&a).unwrap();
    assert_eq!(dense, [vec![0., 1., 3.], vec![0., 0., -0.]]);
    assert_eq!(dense[1][2].to_bits(), (-0.0f64).to_bits());
    assert_eq!(dense[1][1].to_bits(), 0.0f64.to_bits());

    // The duplicate sum is deliberately order-sensitive in binary64.
    assert_eq!(
        from_coo(1, 1, &[(0, 0, 1e16), (0, 0, 1.), (0, 0, -1e16)])
            .unwrap()
            .data,
        [0.]
    );
    assert_eq!(from_coo(0, 0, &[]).unwrap().indptr, [0]);
}

#[test]
fn canonical_coo_and_csr_errors_are_precise() {
    assert_eq!(
        from_coo(1_000_001, 0, &[]),
        Err(Error::InvalidInput("sparse axis exceeds 1,000,000"))
    );
    assert_eq!(
        from_coo(1, 1, &[(1, 0, 1.)]),
        Err(Error::InvalidInput(
            "invalid sparse coordinate or nonfinite value"
        ))
    );
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            from_coo(1, 1, &[(0, 0, value)]),
            Err(Error::InvalidInput(
                "invalid sparse coordinate or nonfinite value"
            ))
        );
    }
    assert_eq!(
        from_coo(1, 1, &[(0, 0, 1e308), (0, 0, 1e308)]),
        Err(Error::NonFiniteEvaluation)
    );

    let mut a = from_coo(1, 2, &[(0, 1, 1.)]).unwrap();
    a.indptr = vec![1, 1];
    assert_eq!(
        validate(&a),
        Err(Error::InvalidInput(
            "invalid CSR storage lengths or pointers"
        ))
    );
    let mut bad_pointers = from_coo(2, 2, &[(0, 0, 1.)]).unwrap();
    bad_pointers.indptr = vec![0, 2, 1];
    assert_eq!(
        validate(&bad_pointers),
        Err(Error::InvalidInput("invalid CSR row pointers"))
    );
    a.indptr = vec![0, 1];
    a.indices = vec![2];
    assert_eq!(
        validate(&a),
        Err(Error::InvalidInput(
            "CSR columns must be in bounds, sorted, and unique"
        ))
    );
    a.indices = vec![1, 1];
    a.data = vec![1., 2.];
    a.indptr = vec![0, 2];
    assert_eq!(
        validate(&a),
        Err(Error::InvalidInput(
            "CSR columns must be in bounds, sorted, and unique"
        ))
    );
    a.indices = vec![0, 1];
    a.data[1] = f64::NAN;
    assert_eq!(
        validate(&a),
        Err(Error::InvalidInput("coordinates must be finite"))
    );

    let empty_rows = from_coo(0, 2, &[]).unwrap();
    assert_eq!(
        to_dense(&empty_rows),
        Err(Error::InvalidInput(
            "shared rows cannot represent shape (0,n)"
        ))
    );
    let oversized = from_coo(1001, 1000, &[]).unwrap();
    assert_eq!(
        to_dense(&oversized),
        Err(Error::InvalidInput("dense output exceeds 1,000,000 cells"))
    );
}
