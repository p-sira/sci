mod common;
use sci::{Result, special, stats};

#[test]
fn power_transforms_match_independent_references() -> Result<()> {
    let inputs = common::lines(include_str!("fixtures/power_stability_inputs.csv"));
    let expected: Vec<_> =
        common::lines(include_str!("fixtures/power_stability_expected.csv")).collect();
    assert_eq!(inputs.clone().count(), expected.len());
    for (input, expected) in inputs.zip(expected) {
        let row = common::fields(input);
        let reference = common::fields(expected);
        assert_eq!(row[0], reference[0]);
        let x = common::vector(row[2]);
        let p = row[3].parse().unwrap();
        let actual = match row[1] {
            "pmean" => vec![stats::descriptive::pmean(&x, p)?],
            "yeojohnson" => stats::transformations::yeojohnson(&x, p)?,
            api => {
                let function = match api {
                    "boxcox" => special::probability::boxcox,
                    "boxcox1p" => special::probability::boxcox1p,
                    _ => panic!("unknown API"),
                };
                let values: Vec<_> = x.iter().map(|&v| function(v, p)).collect();
                if api == "boxcox" && values.iter().all(|v| v.is_finite()) {
                    assert_eq!(stats::transformations::boxcox(&x, p)?, values);
                }
                values
            }
        };
        let expected = common::vector(reference[1]);
        assert_eq!(actual.len(), expected.len());
        let rtol: f64 = reference[2].parse().unwrap();
        let atol: f64 = reference[3].parse().unwrap();
        for (a, b) in actual.into_iter().zip(expected) {
            if b.is_infinite() || b.is_nan() {
                assert!(a == b || (a.is_nan() && b.is_nan()), "{input}: {a} != {b}");
            } else {
                assert!(
                    a.is_finite() && (a - b).abs() <= atol + rtol * b.abs(),
                    "{input}: {a} != {b}"
                );
                if b != 0. {
                    assert_ne!(a, 0., "{input}: lost nonzero value");
                }
            }
        }
    }
    Ok(())
}

#[test]
fn power_means_preserve_limits_bounds_and_scale() -> Result<()> {
    let x = [0.25, 1., 4., 16.];
    let mut previous = 0.;
    for p in [-64., -2., -1e-8, -1e-18, 0., 1e-18, 1e-8, 1.5, 64.] {
        let m = stats::descriptive::pmean(&x, p)?;
        assert!((x[0]..=x[3]).contains(&m));
        assert!(m >= previous - 2e-14);
        previous = m;
        for scale in [1e-200, 1e200] {
            let scaled: Vec<_> = x.iter().map(|v| v * scale).collect();
            assert!((stats::descriptive::pmean(&scaled, p)? / scale - m).abs() < 3e-12 * m);
        }
    }
    for p in [-64., -1e-18, 0., 1e-18, 64.] {
        assert_eq!(stats::descriptive::pmean(&[3.; 4], p)?, 3.);
    }
    Ok(())
}

#[test]
fn yeojohnson_half_power_preserves_small_and_large_values() -> Result<()> {
    // Closed-form values of ((1+x)^1/2-1)/(1/2) for x>=0 and
    // -((1-x)^3/2-1)/(3/2) for x<0, including their small-x limits.
    let xs = [
        -1e200, -8.0, -3.0, -1e-300, -0.0, 0.0, 1e-300, 3.0, 8.0, 1e308,
    ];
    let expected = [
        -(2.0 / 3.0) * 1e300,
        -52.0 / 3.0,
        -14.0 / 3.0,
        -1e-300,
        -0.0,
        0.0,
        1e-300,
        2.0,
        4.0,
        2e154,
    ];
    for (actual, reference) in stats::transformations::yeojohnson(&xs, 0.5)?
        .into_iter()
        .zip(expected)
    {
        assert!(
            (actual - reference).abs() <= 3e-15 * reference.abs(),
            "half-power transformed value {actual} differs from {reference}"
        );
        if reference == 0.0 {
            assert_eq!(actual.to_bits(), reference.to_bits());
        }
    }
    assert!(stats::transformations::yeojohnson(&[-1e250], 0.5).is_err());
    Ok(())
}

#[test]
fn power_transform_domains_remain_explicit() {
    for p in [-1., 0., 1e-20, 1.] {
        let expected = if p > 0. { -1. / p } else { f64::NEG_INFINITY };
        assert_eq!(special::probability::boxcox(0., p), expected);
        assert_eq!(special::probability::boxcox1p(-1., p), expected);
        assert!(special::probability::boxcox(-1., p).is_nan());
        assert!(special::probability::boxcox1p(-2., p).is_nan());
    }
    for value in [f64::NAN, f64::INFINITY, 0., -1.] {
        assert!(stats::descriptive::pmean(&[value], 1.).is_err());
        assert!(stats::transformations::boxcox(&[value], 1.).is_err());
    }
    for p in [f64::NAN, f64::INFINITY, 65.] {
        assert!(stats::descriptive::pmean(&[1.], p).is_err());
    }
    assert!(stats::descriptive::pmean(&[], 1.).is_err());
    assert!(stats::transformations::boxcox(&[100.], 200.).is_err());
    assert!(special::probability::boxcox(2., f64::NAN).is_nan());
    assert!(special::probability::boxcox1p(1., f64::NAN).is_nan());
    assert!(stats::transformations::yeojohnson(&[f64::INFINITY], 1.).is_err());
    assert_eq!(
        stats::transformations::yeojohnson(&[], 1.).unwrap(),
        Vec::<f64>::new()
    );
}
