use sci_spatial::distance::{self as distance, Metric};
use sci_spatial::{Error, Result};

fn scalar(u: &[f64], v: &[f64], metric: Metric<'_>) -> Result<f64> {
    match metric {
        Metric::Euclidean => distance::euclidean(u, v, None),
        Metric::SqEuclidean => distance::sqeuclidean(u, v, None),
        Metric::Cityblock => distance::cityblock(u, v, None),
        Metric::Chebyshev => distance::chebyshev(u, v, None),
        Metric::Minkowski(p) => distance::minkowski(u, v, p, None),
        Metric::Canberra => distance::canberra(u, v, None),
        Metric::Braycurtis => distance::braycurtis(u, v, None),
        Metric::Cosine => distance::cosine(u, v, None),
        Metric::Correlation => distance::correlation(u, v, None, true),
        Metric::Hamming => distance::hamming(u, v, None),
        Metric::Jaccard => distance::jaccard(u, v, None),
        Metric::Dice => distance::dice(u, v, None),
        Metric::Rogerstanimoto => distance::rogerstanimoto(u, v, None),
        Metric::Russellrao => distance::russellrao(u, v, None),
        Metric::Sokalsneath => distance::sokalsneath(u, v, None),
        Metric::Yule => distance::yule(u, v, None),
        Metric::Seuclidean(variance) => distance::seuclidean(u, v, variance),
        Metric::Mahalanobis(inverse_covariance) => distance::mahalanobis(u, v, inverse_covariance),
        Metric::JensenShannon(base) => distance::jensenshannon(u, v, base),
    }
}

fn assert_same(actual: Result<f64>, expected: Result<f64>) {
    match (actual, expected) {
        (Ok(a), Ok(b)) if b.is_nan() => assert!(a.is_nan(), "expected NaN, got {a}"),
        (Ok(a), Ok(b)) => assert_eq!(a.to_bits(), b.to_bits(), "{a} != {b}"),
        (Err(a), Err(b)) => assert_eq!(a, b),
        (actual, expected) => panic!("{actual:?} != {expected:?}"),
    }
}

fn assert_pairwise_same(actual: Result<f64>, expected: Result<f64>, metric: Metric<'_>) {
    if matches!(metric, Metric::Euclidean) || matches!(metric, Metric::Minkowski(p) if p == 2.0) {
        match (actual, expected) {
            (Ok(a), Ok(b)) if a.is_finite() && b.is_finite() => {
                assert!(a >= 0.0 && b >= 0.0);
                assert!(
                    a.to_bits().abs_diff(b.to_bits()) <= 2,
                    "fast Euclidean distance differs by more than 2 ulps: {a} != {b}"
                );
            }
            (actual, expected) => assert_same(actual, expected),
        }
    } else {
        assert_same(actual, expected);
    }
}

#[test]
fn all_pairwise_metrics_match_checked_scalar_kernels() {
    let x = vec![
        vec![1.0, 0.0, 1.0],
        vec![0.0, 1.0, 1.0],
        vec![1.0, 1.0, 0.0],
    ];
    let y = vec![
        vec![0.0, 1.0, 0.0],
        vec![1.0, 0.0, 1.0],
        vec![0.0, 0.0, 1.0],
    ];
    let variance = [0.5, 2.0, 4.0];
    let inverse_covariance = vec![
        vec![2.0, 0.0, 0.0],
        vec![0.0, 3.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let metrics = [
        Metric::Euclidean,
        Metric::SqEuclidean,
        Metric::Cityblock,
        Metric::Chebyshev,
        Metric::Minkowski(1.0),
        Metric::Minkowski(2.0),
        Metric::Minkowski(3.0),
        Metric::Minkowski(f64::INFINITY),
        Metric::Canberra,
        Metric::Braycurtis,
        Metric::Cosine,
        Metric::Correlation,
        Metric::Hamming,
        Metric::Jaccard,
        Metric::Dice,
        Metric::Rogerstanimoto,
        Metric::Russellrao,
        Metric::Sokalsneath,
        Metric::Yule,
        Metric::Seuclidean(&variance),
        Metric::Mahalanobis(&inverse_covariance),
        Metric::JensenShannon(None),
        Metric::JensenShannon(Some(2.0)),
    ];

    for metric in metrics {
        let cross = distance::cdist(&x, &y, metric).unwrap();
        assert_eq!(cross.len(), x.len());
        for (i, row) in cross.iter().enumerate() {
            assert_eq!(row.len(), y.len());
            for (j, &actual) in row.iter().enumerate() {
                assert_pairwise_same(Ok(actual), scalar(&x[i], &y[j], metric), metric);
            }
        }

        let within = distance::pdist(&x, metric).unwrap();
        assert_eq!(within.len(), x.len() * (x.len() - 1) / 2);
        let mut index = 0;
        for i in 0..x.len() {
            for j in i + 1..x.len() {
                assert_pairwise_same(Ok(within[index]), scalar(&x[i], &x[j], metric), metric);
                index += 1;
            }
        }
    }
}

#[test]
fn euclidean_pairwise_preserves_ties_duplicates_and_order() {
    let points = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![-1.0, 0.0],
        vec![0.0, 0.0],
    ];
    assert_eq!(
        distance::pdist(&points, Metric::Euclidean).unwrap(),
        vec![1.0, 1.0, 0.0, 2.0, 1.0, 1.0]
    );
    assert_eq!(
        distance::cdist(&points, &points[..2], Metric::Euclidean).unwrap(),
        vec![
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 2.0],
            vec![0.0, 1.0]
        ]
    );
}

#[test]
fn euclidean_pairwise_keeps_large_and_subnormal_distances() {
    let large = vec![vec![1e300, 1e300], vec![0.0, 0.0]];
    let actual = distance::pdist(&large, Metric::Euclidean).unwrap()[0];
    assert!((actual / 1e300 - 2.0_f64.sqrt()).abs() < 5e-15);
    assert_same(Ok(actual), distance::euclidean(&large[0], &large[1], None));

    let tiny = vec![vec![1e-300, 0.0], vec![0.0, 1e-300]];
    let actual = distance::cdist(&tiny[..1], &tiny[1..], Metric::Euclidean).unwrap()[0][0];
    assert!((actual / 1e-300 - 2.0_f64.sqrt()).abs() < 5e-15);
    assert_same(Ok(actual), distance::euclidean(&tiny[0], &tiny[1], None));

    let subnormal = f64::from_bits(1);
    assert_eq!(
        distance::pdist(&[vec![subnormal], vec![0.0]], Metric::Euclidean),
        Ok(vec![subnormal])
    );
    assert_eq!(
        distance::cdist(&[vec![subnormal]], &[vec![0.0]], Metric::SqEuclidean),
        Ok(vec![vec![0.0]])
    );

    for metric in [
        Metric::Euclidean,
        Metric::Minkowski(2.0),
        Metric::SqEuclidean,
    ] {
        assert_eq!(
            distance::pdist(&[vec![f64::MAX], vec![-f64::MAX]], metric),
            Err(Error::NonFiniteEvaluation)
        );
    }
    assert_eq!(
        distance::cdist(&[vec![1e300]], &[vec![0.0]], Metric::SqEuclidean),
        Err(Error::NonFiniteEvaluation)
    );
}

#[test]
fn moderate_euclidean_fast_path_stays_within_two_ulps() {
    // The pairwise fast path sums squared differences before taking one sqrt.
    // The checked vector API is an independent sequential-hypot reference.
    let mut state = 0x8741_d5a9_8a2c_9e31_u64;
    for dimension in 1..=3 {
        for scale in [1e-149, 1e-12, 1.0, 1e12, 1e149] {
            for _ in 0..100 {
                let mut a = Vec::with_capacity(dimension);
                let mut b = Vec::with_capacity(dimension);
                for _ in 0..dimension {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let x = (state >> 32) as i32 as f64 / 1e9;
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let y = (state >> 32) as i32 as f64 / 1e9;
                    a.push(x * scale);
                    b.push(y * scale);
                }
                let expected = distance::euclidean(&a, &b, None);
                let cross = distance::cdist(&[a.clone()], &[b.clone()], Metric::Euclidean);
                assert_pairwise_same(cross.map(|rows| rows[0][0]), expected, Metric::Euclidean);
                let within = distance::pdist(&[a.clone(), b.clone()], Metric::Minkowski(2.0));
                assert_pairwise_same(
                    within.map(|distances| distances[0]),
                    distance::minkowski(&a, &b, 2.0, None),
                    Metric::Minkowski(2.0),
                );
            }
        }
    }
}

#[test]
fn euclidean_fallback_handles_cancellation_and_extreme_exponents() {
    let below_fast_range = 1e-151;
    let above_fast_range = 1e151;
    let cases = [
        (vec![0.0, 0.0], vec![below_fast_range, 1.0]),
        (vec![0.0, 0.0], vec![above_fast_range, 1.0]),
        (vec![1e300, 1e300], vec![1e300 - 1e284, 1e300]),
        (vec![f64::MAX], vec![0.0]),
        (vec![f64::MIN_POSITIVE], vec![0.0]),
        (vec![f64::from_bits(1)], vec![0.0]),
        (vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]),
        (vec![1e300, 1e-300, 1.0], vec![0.0, 0.0, 0.0]),
        (vec![f64::MAX], vec![-f64::MAX]),
    ];
    for (a, b) in cases {
        let expected = distance::euclidean(&a, &b, None);
        let got =
            distance::cdist(&[a.clone()], &[b.clone()], Metric::Euclidean).map(|rows| rows[0][0]);
        assert_same(got, expected);
        let expected = distance::minkowski(&a, &b, 2.0, None);
        let got = distance::pdist(&[a.clone(), b.clone()], Metric::Minkowski(2.0))
            .map(|distances| distances[0]);
        assert_same(got, expected);
    }
}

#[test]
fn near_equal_large_coordinates_follow_rounded_differences() {
    let start = 2.0_f64.powi(53);
    let a = vec![start + 2.0, start + 4.0, start + 8.0];
    let b = vec![start, start + 2.0, start + 4.0];
    let expected = (4.0_f64 + 4.0 + 16.0).sqrt();
    let actual = distance::cdist(&[a.clone()], &[b.clone()], Metric::Euclidean).unwrap()[0][0];
    assert_eq!(actual.to_bits(), expected.to_bits());
    assert_pairwise_same(
        Ok(actual),
        distance::euclidean(&a, &b, None),
        Metric::Euclidean,
    );
}

#[test]
fn zero_dimensional_and_empty_batches_preserve_metric_semantics() {
    let zero_dimensional = vec![vec![], vec![]];
    for metric in [
        Metric::Euclidean,
        Metric::SqEuclidean,
        Metric::Cityblock,
        Metric::Chebyshev,
        Metric::Minkowski(2.0),
        Metric::Minkowski(3.0),
        Metric::Canberra,
        Metric::Braycurtis,
        Metric::Cosine,
        Metric::Correlation,
        Metric::Hamming,
        Metric::Jaccard,
        Metric::Dice,
        Metric::Rogerstanimoto,
        Metric::Russellrao,
        Metric::Sokalsneath,
        Metric::Yule,
        Metric::Seuclidean(&[]),
        Metric::JensenShannon(None),
    ] {
        let expected = scalar(&[], &[], metric);
        let within = distance::pdist(&zero_dimensional, metric);
        match within {
            Ok(distances) => {
                assert_eq!(distances.len(), 1);
                assert_same(Ok(distances[0]), expected);
            }
            Err(err) => assert_same(Err(err), expected),
        }
    }

    assert_eq!(
        distance::cdist(&[vec![], vec![]], &[vec![]], Metric::Euclidean),
        Ok(vec![vec![0.0], vec![0.0]])
    );
    assert_eq!(
        distance::cdist(&[vec![0.0]], &[], Metric::Euclidean),
        Ok(vec![vec![]])
    );
    assert_eq!(distance::pdist(&[vec![0.0]], Metric::Euclidean), Ok(vec![]));
    assert_eq!(
        distance::pdist(&[], Metric::Minkowski(0.0)),
        Err(Error::InvalidInput("p must be positive"))
    );
    assert_eq!(
        distance::cdist(&[], &[], Metric::JensenShannon(Some(0.5))),
        Err(Error::InvalidInput("logarithm base must exceed one"))
    );
}

#[test]
fn pairwise_still_rejects_nonfinite_and_ragged_inputs() {
    for bad in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        assert_eq!(
            distance::pdist(&[vec![0.0], vec![bad]], Metric::Euclidean),
            Err(Error::InvalidInput("coordinates must be finite"))
        );
        assert_eq!(
            distance::cdist(&[vec![bad]], &[], Metric::Euclidean),
            Err(Error::InvalidInput("coordinates must be finite"))
        );
    }
    assert_eq!(
        distance::pdist(&[vec![0.0], vec![0.0, 1.0]], Metric::Euclidean),
        Err(Error::DimensionMismatch)
    );
    assert_eq!(
        distance::cdist(&[vec![0.0]], &[vec![0.0, 1.0]], Metric::Euclidean),
        Err(Error::DimensionMismatch)
    );
    assert_eq!(
        distance::pdist(&[vec![0.0], vec![0.0]], Metric::Sokalsneath),
        Err(Error::InvalidInput(
            "Sokal-Sneath is undefined for all-false vectors"
        ))
    );
}
