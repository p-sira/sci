// Preserve original high-precision independent numerical reference literals.
#![allow(clippy::excessive_precision)]
//! Independent public-interface checks for associated Legendre Q values and derivatives.

use sci_special::orthogonal::lqmn;

fn close(actual: f64, expected: f64, message: &str) {
    assert!(
        (actual - expected).abs() <= 2e-11 * (1.0 + expected.abs()),
        "{message}: got {actual}, expected {expected}"
    );
}

#[test]
fn interior_values_and_derivatives_match_scipy() {
    // SciPy order-major reference for lqmn(2, 3, 0.3).
    let values = [
        [
            0.3095196042031118,
            -0.9071441187390664,
            -0.5629746555341357,
            0.3232754180589764,
        ],
        [
            -1.0482848367219182,
            -0.6097483350729811,
            1.5477961718781534,
            2.0754696315180867,
        ],
        [
            0.6593406593406593,
            2.197802197802198,
            2.4043291788151544,
            -5.18471502298606,
        ],
    ];
    let slopes = [
        [
            1.0989010989010988,
            0.6391899338734415,
            -1.6225312573161004,
            -2.175683343797237,
        ],
        [
            -0.34558840771052246,
            -2.1029067834307056,
            -3.0306842948855333,
            4.750837383677293,
        ],
        [
            2.632532302861973,
            1.4491003501992514,
            8.075397015296371,
            18.338340016223324,
        ],
    ];
    let (actual, derivative) = lqmn(2, 3, 0.3).unwrap();
    for order in 0..=2 {
        for degree in 0..=3 {
            close(
                actual[order][degree],
                values[order][degree],
                "interior value",
            );
            close(
                derivative[order][degree],
                slopes[order][degree],
                "interior derivative",
            );
        }
    }
}

#[test]
fn interior_reflection_and_numerical_derivatives() {
    for x in [0.0, 0.1, 0.3, 0.65, 0.9] {
        let (values, slopes) = lqmn(5, 8, x).unwrap();
        let (negative, negative_slopes) = lqmn(5, 8, -x).unwrap();
        for order in 0..=5 {
            for degree in 0..=8 {
                let phase = if (order + degree) % 2 == 0 { -1.0 } else { 1.0 };
                close(
                    negative[order][degree],
                    phase * values[order][degree],
                    "Q parity",
                );
                close(
                    negative_slopes[order][degree],
                    -phase * slopes[order][degree],
                    "Q' parity",
                );
            }
        }
    }
    let h = 1e-6;
    for x in [-0.75, -0.2, 0.0, 0.3, 0.72] {
        let (_, derivatives) = lqmn(4, 6, x).unwrap();
        let (plus, _) = lqmn(4, 6, x + h).unwrap();
        let (minus, _) = lqmn(4, 6, x - h).unwrap();
        for order in 0..=4 {
            for degree in 0..=6 {
                let finite_difference = (plus[order][degree] - minus[order][degree]) / (2.0 * h);
                assert!(
                    (derivatives[order][degree] - finite_difference).abs()
                        <= 3e-6 * (1.0 + finite_difference.abs()),
                    "Q'({order},{degree},{x}): {} vs {finite_difference}",
                    derivatives[order][degree]
                );
            }
        }
    }
    // The exact four inputs of the lqmn CodSpeed benchmark exercise a
    // larger 9x9 table and both signs of the real argument.
    for x in [
        0.34220361683518397,
        0.1614453715565306,
        -0.64194562025659,
        0.7673738538186275,
    ] {
        let (values, derivatives) = lqmn(8, 8, x).unwrap();
        assert!(values.iter().flatten().all(|v| v.is_finite()));
        assert!(derivatives.iter().flatten().all(|v| v.is_finite()));
        let (plus, _) = lqmn(8, 8, x + h).unwrap();
        let (minus, _) = lqmn(8, 8, x - h).unwrap();
        for order in [0, 1, 4, 8] {
            for degree in [0, 1, 4, 8] {
                let finite_difference = (plus[order][degree] - minus[order][degree]) / (2.0 * h);
                assert!(
                    (derivatives[order][degree] - finite_difference).abs()
                        <= 3e-6 * (1.0 + finite_difference.abs()),
                    "benchmark Q'({order},{degree},{x}) mismatch"
                );
            }
        }
    }
    for x in [-1.0 + 2e-12, 1.0 - 2e-12] {
        let (values, derivatives) = lqmn(3, 5, x).unwrap();
        close(values[0][0], x.atanh(), "near-endpoint Q0");
        close(
            derivatives[0][0],
            1.0 / ((1.0 - x) * (1.0 + x)),
            "near-endpoint Q0'",
        );
    }
}

#[test]
fn exterior_singular_nonfinite_and_zero_dimension_cases() {
    let (positive, positive_d) = lqmn(2, 4, 2.0).unwrap();
    let (negative, negative_d) = lqmn(2, 4, -2.0).unwrap();
    close(positive[0][0], 0.5493061443340549, "Q(0,0,2)");
    close(positive[1][0], -0.5773502691896258, "Q(1,0,2)");
    close(positive[2][4], 0.0368160162716521, "Q(2,4,2)");
    close(positive_d[0][0], -1.0 / 3.0, "Q'(0,0,2)");
    for order in 0..=2 {
        for degree in 0..=4 {
            let phase = if (order + degree) % 2 == 0 { -1.0 } else { 1.0 };
            close(
                negative[order][degree],
                phase * positive[order][degree],
                "exterior Q parity",
            );
            close(
                negative_d[order][degree],
                -phase * positive_d[order][degree],
                "exterior Q' parity",
            );
        }
    }
    for x in [-1.0, 1.0] {
        let (v, d) = lqmn(2, 3, x).unwrap();
        assert!(v.iter().flatten().all(|&x| x == 1e300));
        assert!(d.iter().flatten().all(|&x| x == 1e300));
    }
    for x in [f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
        let (v, d) = lqmn(2, 3, x).unwrap();
        assert!(v.iter().flatten().all(|x| x.is_nan()));
        assert!(d.iter().flatten().all(|x| x.is_nan()));
    }
    for (m, n) in [(0, 0), (0, 5), (5, 0), (8, 8)] {
        let (v, d) = lqmn(m, n, 0.3).unwrap();
        assert_eq!(v.len(), m + 1);
        assert_eq!(d.len(), m + 1);
        assert!(v.iter().all(|row| row.len() == n + 1));
        assert!(d.iter().all(|row| row.len() == n + 1));
    }
}
