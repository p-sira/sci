use sci_spatial::distance::{self, Metric};

#[test]
fn unit_weight_pairwise_cosine_and_correlation_have_known_distances() {
    let axes = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let cosine = distance::pdist(&axes, Metric::Cosine).unwrap();
    let correlation = distance::pdist(&axes, Metric::Correlation).unwrap();
    for value in cosine {
        assert!((value - 1.0).abs() < 1e-14);
    }
    for value in correlation {
        assert!((value - 1.5).abs() < 1e-14);
    }

    let other = vec![vec![0.0, 0.0, 1.0], vec![0.0, 1.0, 0.0]];
    for metric in [Metric::Cosine, Metric::Correlation] {
        let cross = distance::cdist(&axes, &other, metric).unwrap();
        let similarity = if matches!(metric, Metric::Cosine) {
            1.0
        } else {
            1.5
        };
        assert!((cross[0][0] - similarity).abs() < 1e-14);
        assert!((cross[0][1] - similarity).abs() < 1e-14);
        assert!(cross[1][1].abs() < 1e-14);
        assert!(cross[2][0].abs() < 1e-14);
    }
}

#[test]
fn scalar_weighted_correlation_keeps_centering_and_scale_invariance() {
    let u = [1.0, 2.0, 3.0, 4.0];
    let reversed = [4.0, 3.0, 2.0, 1.0];
    let proportional = [3.0, 5.0, 7.0, 9.0];
    for weights in [&[1.0, 0.25, 0.5, 1.0][..], &[0.5, 0.125, 0.25, 0.5][..]] {
        let opposite = distance::correlation(&u, &reversed, Some(weights), true).unwrap();
        let same = distance::correlation(&u, &proportional, Some(weights), true).unwrap();
        assert!((opposite - 2.0).abs() < 1e-14);
        assert!(same.abs() < 1e-14);
    }
    assert!(
        distance::cosine(&[0.0, 0.0], &[1.0, 2.0], None)
            .unwrap()
            .is_nan()
    );
}

#[test]
fn nearly_parallel_distances_retain_small_nonzero_values() {
    // Independent 150-digit sci-testing references for the cosine cases
    // expanded-0014, -0029, -0044 and -0059, evaluated on binary64 inputs.
    for (n, expected) in [
        (2, 7.999_999_870_760_465e-18),
        (3, 3.316_326_485_564_879_3e-18),
        (8, 2.438_965_752_506_786e-19),
        (32, 4.370_247_269_479_72e-21),
    ] {
        let u: Vec<_> = (1..=n).map(f64::from).collect();
        let mut v = u.clone();
        v[0] = 1.000_000_01;
        let d = distance::cosine(&u, &v, None).unwrap();
        assert!((d / expected - 1.0).abs() < 2e-6, "n={n}: {d:e}");
        assert_eq!(distance::cosine(&u, &u, None).unwrap(), 0.0);
        let pair = vec![u.clone(), v.clone()];
        assert_eq!(distance::pdist(&pair, Metric::Cosine).unwrap(), vec![d]);
        assert_eq!(
            distance::cdist(&[u], &[v], Metric::Cosine).unwrap(),
            vec![vec![d]]
        );
    }

    let u = [1.0, 2.0, 3.0];
    let v = [1.000_000_01, 2.0, 3.0];
    let w = [1.0, 0.25, 0.5];
    // 100-digit mpmath evaluation of the weighted unit-vector identity.
    for (centered, expected) in [
        (false, 6.508_875_640_502_654e-18),
        (true, 1.939_058_168_587_151_4e-18),
    ] {
        let d = distance::correlation(&u, &v, Some(&w), centered).unwrap();
        assert!((d / expected - 1.0).abs() < 2e-6, "{centered}: {d:e}");
    }
    assert!(
        distance::correlation(&[1.0; 3], &v, Some(&w), true)
            .unwrap()
            .is_nan()
    );
}
