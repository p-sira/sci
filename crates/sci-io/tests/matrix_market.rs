#![cfg(feature = "matrix-market")]

use sci_io::{
    Error, SparseMatrix,
    matrix_market::{MarketMatrix, mmread, mmwrite},
};

#[test]
fn matrix_market_sparse_duplicates_sorting_and_bitwise_round_trip() {
    // Reading coordinate input must preserve original duplicate order
    // after stable sorting; 1e16 + (-1e16) + 1 differs from other orders.
    let input = b"%%MatrixMarket matrix coordinate real general\n2 3 7\n2 3 -0\n1 2 10000000000000000\n1 3 5e-324\n1 2 -10000000000000000\n1 2 1\n2 1 -2\n2 1 2\n";
    let market = mmread(input.as_slice()).unwrap();
    assert_eq!(
        market,
        MarketMatrix::Sparse(SparseMatrix {
            rows: 2,
            cols: 3,
            indptr: vec![0, 2, 4],
            indices: vec![1, 2, 0, 2],
            data: vec![1.0, f64::from_bits(1), 0.0, -0.0],
        })
    );
    let rows = market.to_matrix().unwrap();
    assert_eq!(rows, [vec![0., 1., f64::from_bits(1)], vec![0., 0., -0.]]);
    assert_eq!(rows[1][2].to_bits(), (-0.0f64).to_bits());
    assert_eq!(rows[0][0].to_bits(), 0.0f64.to_bits());
    let mut written = Vec::new();
    mmwrite(&mut written, &market).unwrap();
    let reloaded = mmread(written.as_slice()).unwrap();
    assert_eq!(reloaded, market);
    let round_trip_rows = reloaded.to_matrix().unwrap();
    assert_eq!(round_trip_rows[0][2].to_bits(), 1);
    assert_eq!(round_trip_rows[1][2].to_bits(), (-0.0f64).to_bits());
}

#[test]
fn matrix_market_symmetry_and_dense_special_values() {
    let skew = b"%%MatrixMarket matrix coordinate real skew-symmetric\n2 2 1\n2 1 -0\n";
    let skew = mmread(skew.as_slice()).unwrap();
    assert_eq!(
        skew,
        MarketMatrix::Sparse(SparseMatrix {
            rows: 2,
            cols: 2,
            indptr: vec![0, 1, 2],
            indices: vec![1, 0],
            data: vec![0.0, -0.0],
        })
    );
    let skew_rows = skew.to_matrix().unwrap();
    assert_eq!(skew_rows[0][1].to_bits(), 0.0f64.to_bits());
    assert_eq!(skew_rows[1][0].to_bits(), (-0.0f64).to_bits());

    let symmetric =
        b"%%MatrixMarket matrix coordinate integer symmetric\n2 2 1\n2 1 9007199254740992\n";
    assert_eq!(
        mmread(symmetric.as_slice()).unwrap(),
        MarketMatrix::Sparse(SparseMatrix {
            rows: 2,
            cols: 2,
            indptr: vec![0, 1, 2],
            indices: vec![1, 0],
            data: vec![9_007_199_254_740_992.0; 2],
        })
    );

    let dense = b"%%MatrixMarket matrix array real general\n2 2\nnan\n-inf\n-0\ninf\n";
    let dense = mmread(dense.as_slice()).unwrap();
    assert!(matches!(
        &dense,
        MarketMatrix::Dense {
            rows: 2,
            cols: 2,
            ..
        }
    ));
    let data = dense.to_matrix().unwrap();
    assert!(data[0][0].is_nan());
    assert_eq!(data[0][1].to_bits(), (-0.0f64).to_bits());
    assert_eq!(data[1][0], f64::NEG_INFINITY);
    assert_eq!(data[1][1], f64::INFINITY);
    let mut written = Vec::new();
    mmwrite(&mut written, &dense).unwrap();
    let round_trip = mmread(written.as_slice()).unwrap();
    assert!(matches!(
        &round_trip,
        MarketMatrix::Dense {
            rows: 2,
            cols: 2,
            ..
        }
    ));
    let round_trip = round_trip.to_matrix().unwrap();
    assert!(round_trip[0][0].is_nan());
    assert_eq!(round_trip[0][1].to_bits(), (-0.0f64).to_bits());
    assert_eq!(round_trip[1][0], f64::NEG_INFINITY);
    assert_eq!(round_trip[1][1], f64::INFINITY);
}

#[test]
fn matrix_market_shared_csr_error_mapping_is_unchanged() {
    let overflow = b"%%MatrixMarket matrix coordinate real general\n1 1 2\n1 1 1e308\n1 1 1e308\n";
    assert!(matches!(
        mmread(overflow.as_slice()),
        Err(Error::Numerical(sci_core::Error::NonFiniteEvaluation))
    ));
    for invalid in [
        "%%MatrixMarket matrix coordinate real general\n1 1 1\n1 1 nan\n",
        "%%MatrixMarket matrix coordinate real general\n1 1 1\n1 1 inf\n",
    ] {
        assert!(matches!(
            mmread(invalid.as_bytes()),
            Err(Error::Numerical(sci_core::Error::InvalidInput(
                "invalid sparse coordinate or nonfinite value"
            )))
        ));
    }
    let too_precise =
        b"%%MatrixMarket matrix coordinate integer general\n1 1 1\n1 1 9007199254740993\n";
    assert!(matches!(
        mmread(too_precise.as_slice()),
        Err(Error::Unsupported(
            "Matrix Market integers outside ±2^53 require typed storage"
        ))
    ));

    let invalid = SparseMatrix {
        rows: 1,
        cols: 2,
        indptr: vec![0, 1],
        indices: vec![1],
        data: vec![f64::NAN],
    };
    let mut written = Vec::new();
    assert!(matches!(
        mmwrite(&mut written, &MarketMatrix::Sparse(invalid)),
        Err(Error::Numerical(sci_core::Error::InvalidInput(
            "coordinates must be finite"
        )))
    ));
    assert!(written.is_empty());

    let empty_rows = b"%%MatrixMarket matrix coordinate real general\n0 2 0\n";
    let matrix = mmread(empty_rows.as_slice()).unwrap();
    assert!(matches!(
        matrix.to_matrix(),
        Err(Error::Unsupported(
            "shared rows cannot represent shape (0,n)"
        ))
    ));
}
