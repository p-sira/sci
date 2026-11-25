use sci_spatial::distance::{HausdorffResult, directed_hausdorff, euclidean};
use sci_spatial::{Error, Result};

// Independent oracle: retain the original exhaustive checked scalar kernel.
// All fixtures passed here have rectangular, finite, positive-dimensional rows.
fn checked_scalar_reference(u: &[Vec<f64>], v: &[Vec<f64>]) -> Result<HausdorffResult> {
    let mut result = HausdorffResult {
        distance: -1.0,
        index_u: 0,
        index_v: 0,
    };
    for (i, a) in u.iter().enumerate() {
        let (mut nearest, mut index) = (f64::INFINITY, 0);
        for (j, b) in v.iter().enumerate() {
            let d = euclidean(a, b, None)?;
            if d < nearest {
                nearest = d;
                index = j;
            }
        }
        if nearest > result.distance {
            result = HausdorffResult {
                distance: nearest,
                index_u: i,
                index_v: index,
            };
        }
    }
    Ok(result)
}

fn assert_matches_scalar(u: &[Vec<f64>], v: &[Vec<f64>]) {
    let actual = directed_hausdorff(u, v);
    let expected = checked_scalar_reference(u, v);
    match (actual, expected) {
        (Ok(actual), Ok(expected)) => {
            assert_eq!(actual.distance.to_bits(), expected.distance.to_bits());
            assert_eq!(
                (actual.index_u, actual.index_v),
                (expected.index_u, expected.index_v)
            );
        }
        (Err(actual), Err(expected)) => assert_eq!(actual, expected),
        (actual, expected) => panic!("Hausdorff {actual:?} != checked oracle {expected:?}"),
    }
}

#[test]
fn exact_scalar_parity_in_one_two_three_dimensions() {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    for dimension in [1, 2, 3] {
        for scale in [1e-300, 1e-30, 1.0, 1e30, 1e300] {
            let mut make_points = |count: usize| {
                (0..count)
                    .map(|_| {
                        (0..dimension)
                            .map(|_| {
                                state = state
                                    .wrapping_mul(6364136223846793005)
                                    .wrapping_add(1442695040888963407);
                                let centered = (state >> 33) as f64 / (1_u64 << 31) as f64 - 0.5;
                                centered * scale
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            };
            let u = make_points(17);
            let v = make_points(19);
            assert_matches_scalar(&u, &v);
            assert_matches_scalar(&v, &u);
        }
    }
}

#[test]
fn duplicates_nearest_ties_maximizer_ties_and_directionality() {
    let one_dimensional = vec![vec![0.0], vec![2.0], vec![4.0]];
    let targets = vec![vec![1.0], vec![3.0]];
    let result = directed_hausdorff(&one_dimensional, &targets).unwrap();
    assert_eq!(
        (result.distance, result.index_u, result.index_v),
        (1.0, 0, 0)
    );
    assert_matches_scalar(&one_dimensional, &targets);

    let duplicates = vec![vec![2.0], vec![2.0], vec![2.0]];
    let targets = vec![vec![0.0], vec![4.0], vec![4.0]];
    let result = directed_hausdorff(&duplicates, &targets).unwrap();
    assert_eq!(
        (result.distance, result.index_u, result.index_v),
        (2.0, 0, 0)
    );
    assert_matches_scalar(&duplicates, &targets);

    let two_dimensions = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
    let origin = vec![vec![0.0, 0.0]];
    let forward = directed_hausdorff(&two_dimensions, &origin).unwrap();
    let reverse = directed_hausdorff(&origin, &two_dimensions).unwrap();
    assert_eq!(
        (forward.distance, forward.index_u, forward.index_v),
        (5.0, 1, 0)
    );
    assert_eq!(
        (reverse.distance, reverse.index_u, reverse.index_v),
        (0.0, 0, 0)
    );
    assert_matches_scalar(&two_dimensions, &origin);
    assert_matches_scalar(&origin, &two_dimensions);

    let three_dimensions = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 1.0, 1.0],
        vec![1.0, 1.0, 1.0],
    ];
    assert_matches_scalar(&three_dimensions, &three_dimensions);
}

#[test]
fn exact_distance_bits_for_extremes_and_rounding_edges() {
    let subnormal = f64::from_bits(1);
    let cases = [
        (vec![vec![subnormal]], vec![vec![0.0]]),
        (vec![vec![-0.0]], vec![vec![0.0]]),
        (vec![vec![f64::MAX]], vec![vec![0.0]]),
        (vec![vec![1e300, 1e300]], vec![vec![0.0, 0.0]]),
        (vec![vec![1e-300, 0.0]], vec![vec![0.0, 1e-300]]),
        (
            vec![vec![1e300, 1e300, 1e300]],
            vec![vec![1e300 - 1e284, 1e300, 1e300 - 2e284]],
        ),
        (vec![vec![1e-100, 1e100, 1e-300]], vec![vec![0.0, 0.0, 0.0]]),
    ];
    for (u, v) in cases {
        assert_matches_scalar(&u, &v);
        assert_matches_scalar(&v, &u);
    }
}

#[test]
fn overflow_in_losing_or_later_pairs_is_not_suppressed() {
    // The first target wins with distance zero; the losing second target
    // overflows, so the entire computation must still return an error.
    let u = vec![vec![f64::MAX]];
    let v = vec![vec![f64::MAX], vec![-f64::MAX]];
    assert_eq!(directed_hausdorff(&u, &v), Err(Error::NonFiniteEvaluation));
    assert_matches_scalar(&u, &v);

    // The first source point has a valid distance, but the second pair
    // overflows. Skipping a row because it cannot improve the witness is invalid.
    let u = vec![vec![0.0], vec![f64::MAX]];
    let v = vec![vec![0.0], vec![-f64::MAX]];
    assert_eq!(directed_hausdorff(&u, &v), Err(Error::NonFiniteEvaluation));
    assert_matches_scalar(&u, &v);

    // Individual differences fit in f64, but hypot overflows in two dimensions.
    let u = vec![vec![f64::MAX, f64::MAX]];
    let v = vec![vec![0.0, 0.0]];
    assert_eq!(directed_hausdorff(&u, &v), Err(Error::NonFiniteEvaluation));
    assert_matches_scalar(&u, &v);
}

#[test]
fn validation_shapes_nonfinite_inputs_and_work_limits_unchanged() {
    assert_eq!(
        directed_hausdorff(&[], &[]),
        Err(Error::InvalidInput("point sets must be nonempty"))
    );
    assert_eq!(
        directed_hausdorff(&[vec![]], &[vec![]]),
        Err(Error::InvalidInput("point sets must be nonempty"))
    );
    assert_eq!(
        directed_hausdorff(&[vec![0.0]], &[vec![0.0, 1.0]]),
        Err(Error::DimensionMismatch)
    );
    assert_eq!(
        directed_hausdorff(&[vec![0.0], vec![1.0, 2.0]], &[vec![0.0]]),
        Err(Error::DimensionMismatch)
    );
    assert_eq!(
        directed_hausdorff(&[vec![0.0]], &[vec![0.0], vec![1.0, 2.0]]),
        Err(Error::DimensionMismatch)
    );
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            directed_hausdorff(&[vec![bad]], &[vec![0.0]]),
            Err(Error::InvalidInput("coordinates must be finite"))
        );
        assert_eq!(
            directed_hausdorff(&[vec![0.0]], &[vec![bad]]),
            Err(Error::InvalidInput("coordinates must be finite"))
        );
        // No empty output is allowed to hide nonfinite coordinates.
        assert_eq!(
            directed_hausdorff(&[vec![bad]], &[]),
            Err(Error::InvalidInput("coordinates must be finite"))
        );
    }
    let over_limit = vec![vec![0.0; 2]; 2500];
    assert_eq!(
        directed_hausdorff(&over_limit, &over_limit),
        Err(Error::InvalidInput("geometry work limit exceeded"))
    );
    assert_eq!(
        directed_hausdorff(&[vec![0.0]], &[]),
        Err(Error::DimensionMismatch)
    );
}
