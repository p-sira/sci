// Preserve original high-precision independent numerical reference literals.
#![allow(clippy::excessive_precision)]
// Integration regressions extracted from the sci-special numerical backends.
use sci_special::bessel::spherical_yn;

fn close(actual: f64, expected: f64, relative: f64, absolute: f64) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= relative * expected.abs() + absolute,
        "actual={actual:.17e}, expected={expected:.17e}, relative={relative:e}, absolute={absolute:e}"
    );
}

// Independent elementary expressions obtained by differentiating
// y_3(x) = (6/x²-15/x⁴)cos(x) + (1/x-15/x³)sin(x).
fn elementary_y3_and_derivative(x: f64) -> (f64, f64) {
    let (sin, cos) = x.sin_cos();
    let inverse = 1.0 / x;
    let inverse2 = inverse * inverse;
    let inverse3 = inverse2 * inverse;
    let inverse4 = inverse2 * inverse2;
    let inverse5 = inverse4 * inverse;
    (
        (6.0 * inverse2 - 15.0 * inverse4) * cos + (inverse - 15.0 * inverse3) * sin,
        (inverse - 27.0 * inverse3 + 60.0 * inverse5) * cos
            + (-7.0 * inverse2 + 60.0 * inverse4) * sin,
    )
}

#[test]
fn y3_and_derivative_match_independent_analytic_form_on_exact_benchmark_grid() {
    // The Divan benchmark invokes n=3 for both value and derivative at
    // these 128 evenly spaced points. Check the actual input expression,
    // not only an artificial random test distribution.
    for i in 0..128 {
        let x = 0.5 + (20.0 - 0.5) * i as f64 / 127.0;
        let (expected_y3, expected_jac) = elementary_y3_and_derivative(x);
        close(spherical_yn(3, x, false), expected_y3, 2e-12, 2e-13);
        close(spherical_yn(3, x, true), expected_jac, 2e-12, 2e-13);
    }
}

#[test]
fn negative_arguments_parity_and_y0_closed_form_are_preserved() {
    for x in [0.25_f64, 0.5, 1.0, 3.0, 10.0, 20.0] {
        let (sin, cos) = x.sin_cos();
        close(spherical_yn(0, x, false), -cos / x, 1e-14, 1e-14);
        close(
            spherical_yn(0, x, true),
            sin / x + cos / (x * x),
            2e-14,
            1e-14,
        );
        for n in [1, 2, 3, 4, 8] {
            let positive = spherical_yn(n, x, false);
            let negative = spherical_yn(n, -x, false);
            let jac = spherical_yn(n, x, true);
            let negative_jac = spherical_yn(n, -x, true);
            let odd = n % 2 != 0;
            close(
                negative,
                if odd { positive } else { -positive },
                5e-14,
                1e-14,
            );
            close(negative_jac, if odd { -jac } else { jac }, 5e-14, 1e-14);
        }
    }
}
