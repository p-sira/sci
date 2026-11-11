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
