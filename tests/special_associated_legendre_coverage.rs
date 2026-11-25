//! Boundary and scaled-recurrence checks for associated Legendre tables.
use sci::special as s;

fn close(actual: f64, expected: f64, context: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{context}: expected NaN, got {actual}");
    } else {
        let tolerance = 2e-10 * expected.abs().max(actual.abs()).max(1e-300);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{context}: {actual} vs {expected}, tolerance {tolerance}"
        );
    }
}

#[test]
fn low_order_associated_legendre_matches_independent_rodrigues_polynomials() {
    // P_n^m(x) = (-1)^m (1-x^2)^(m/2) d^m P_n(x)/dx^m.
    // The scalar and vector APIs use a recurrence, rather than these closed
    // forms, so these are independent sign, scale and normalization checks.
    for x in [-0.8_f64, -0.3, 0.0, 0.6, 0.95] {
        let sine = ((1.0 - x) * (1.0 + x)).sqrt();
        let p31 = -0.5 * (15.0 * x * x - 3.0) * sine;
        let p42 = 0.5 * (105.0 * x * x - 15.0) * sine * sine;
        close(
            s::orthogonal::assoc_legendre_p(3, 1, x),
            p31,
            "Rodrigues P_3^1",
        );
        close(
            s::orthogonal::assoc_legendre_p(4, 2, x),
            p42,
            "Rodrigues P_4^2",
        );
        // P_n^{-m} = (-1)^m (n-m)!/(n+m)! P_n^m.
        close(
            s::orthogonal::assoc_legendre_p(3, -1, x),
            -p31 / 12.0,
            "P_3^-1",
        );
        close(
            s::orthogonal::assoc_legendre_p(4, -2, x),
            p42 / 360.0,
            "P_4^-2",
        );
        let theta = x.acos();
        let spherical_norm = ((2.0 * 3.0 + 1.0) / (4.0 * std::f64::consts::PI * 12.0)).sqrt();
        close(
            s::orthogonal::sph_legendre_p(3, 1, theta),
            p31 * spherical_norm,
            "Y_3^1 polar amplitude",
        );
    }
}

#[test]
fn associated_tables_agree_with_scalar_orders_across_scaled_and_polar_paths() {
    // m > 40 takes the logarithmically scaled recurrence rather than the
    // moderate-order direct factorial path. Near the poles the small sine
    // likewise selects the scaled recurrence, including its zero shortcut.
    for &(n, m, x) in &[
        (7, 9, 0.3),
        (64, 48, 0.3),
        (64, 48, 0.0),
        (64, 12, 1.0 - 1e-14),
        (64, 12, -1.0 + 1e-14),
        (32, 8, -1.0),
        (32, 8, 1.0),
    ] {
        let table = s::orthogonal::assoc_legendre_p_all(n, m, x).unwrap();
        assert_eq!(table.len(), n + 1);
        for (degree, row) in table.iter().enumerate() {
            assert_eq!(row.len(), 2 * m + 1);
            for (column, order) in (0..=m as i32).chain(-(m as i32)..0).enumerate() {
                let expected = s::orthogonal::assoc_legendre_p(degree as i32, order, x);
                close(
                    row[column],
                    expected,
                    &format!("n={n} m={m} x={x} k={degree} order={order}"),
                );
                if (order.unsigned_abs() as usize) > degree {
                    assert_eq!(row[column], 0.0);
                }
            }
        }
    }
}

#[test]
fn spherical_tables_and_harmonics_agree_with_scalar_amplitudes_and_phases() {
    for &(n, m, theta, phi) in &[
        (16, 9, 0.6, -2.0),
        (64, 48, 0.7, 0.4),
        (16, 5, std::f64::consts::PI, 0.3),
    ] {
        let amplitude = s::orthogonal::sph_legendre_p_all(n, m, theta).unwrap();
        let harmonics = s::wave::sph_harm_y_all(n, m, theta, phi).unwrap();
        for degree in 0..=n {
            for (column, order) in (0..=m as isize).chain(-(m as isize)..0).enumerate() {
                let scalar = s::orthogonal::sph_legendre_p(degree as i32, order as i32, theta);
                close(
                    amplitude[degree][column],
                    scalar,
                    "spherical table vs scalar",
                );
                let wave = s::wave::sph_harm_y(degree, order, theta, phi);
                if order.unsigned_abs() > degree {
                    // A scalar harmonic outside |order|<=degree is invalid,
                    // while a table represents its structurally zero cells.
                    assert!(wave.re.is_nan() && wave.im.is_nan());
                    assert_eq!(harmonics[degree][column].re, 0.0);
                    assert_eq!(harmonics[degree][column].im, 0.0);
                } else {
                    close(
                        harmonics[degree][column].re,
                        wave.re,
                        "harmonic real component",
                    );
                    close(
                        harmonics[degree][column].im,
                        wave.im,
                        "harmonic imaginary component",
                    );
                    close(wave.re.hypot(wave.im), scalar.abs(), "harmonic magnitude");
                }
            }
        }
    }

    // m*phi overflows even though phi is finite; phase must be reduced via
    // complex exponentiation rather than evaluating sin/cos(infinity).
    let huge_phi = 1e308;
    let wave = s::wave::sph_harm_y(16, -9, 0.7, huge_phi);
    let amplitude = s::orthogonal::sph_legendre_p(16, -9, 0.7);
    assert!(wave.re.is_finite() && wave.im.is_finite());
    close(
        wave.re.hypot(wave.im),
        amplitude.abs(),
        "huge finite azimuth",
    );
}

#[test]
fn associated_legendre_invalid_arguments_propagate_through_tables() {
    for x in [-1.01, 1.01, f64::NAN, f64::INFINITY] {
        assert!(s::orthogonal::assoc_legendre_p(3, 1, x).is_nan());
        let table = s::orthogonal::assoc_legendre_p_all(4, 3, x).unwrap();
        assert!(table.iter().flatten().all(|value| value.is_nan()));
    }
    for theta in [-0.01, std::f64::consts::PI + 0.01, f64::NAN] {
        assert!(s::orthogonal::sph_legendre_p(3, 1, theta).is_nan());
        let table = s::orthogonal::sph_legendre_p_all(4, 3, theta).unwrap();
        assert!(table.iter().flatten().all(|value| value.is_nan()));
    }
    for phi in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let wave = s::wave::sph_harm_y(4, 2, 0.4, phi);
        assert!(wave.re.is_nan() && wave.im.is_nan());
        let table = s::wave::sph_harm_y_all(4, 3, 0.4, phi).unwrap();
        assert!(
            table
                .iter()
                .flatten()
                .all(|z| z.re.is_nan() && z.im.is_nan())
        );
    }
}
