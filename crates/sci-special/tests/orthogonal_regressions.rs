// Preserve original high-precision independent numerical reference literals.
#![allow(clippy::excessive_precision)]
// Orthogonal polynomial regressions evaluated through the public Sci API.

fn close(actual: f64, expected: f64, relative: f64) {
    assert!(
        (actual - expected).abs() <= relative * expected.abs().max(f64::MIN_POSITIVE),
        "actual={actual:e}, independent reference={expected:e}"
    );
}

#[test]
fn doubling_matches_independent_three_term_recurrence() {
    for x in [-0.999999_f64, -0.9, -0.3, 0.0, 0.3, 0.9, 0.999999] {
        for second in [false, true] {
            let (mut previous, mut current) = (1.0_f64, if second { 2.0 * x } else { x });
            for n in 1..=4096 {
                if n >= 64 && (n == 64 || n == 65 || n == 127 || n == 1024 || n == 4096) {
                    let actual = if second {
                        sci_special::orthogonal::eval_chebyu(n, x)
                    } else {
                        sci_special::orthogonal::eval_chebyt(n, x)
                    };
                    let tolerance = 2e-8 * (1.0 + current.abs());
                    assert!(
                        (actual - current).abs() <= tolerance,
                        "n={n}, x={x}, second={second}, actual={actual}, reference={current}"
                    );
                }
                (previous, current) = (current, 2.0 * x * current - previous);
            }
        }
    }
}

fn hermite_clenshaw(n: i32, x: f64, scale: f64) -> f64 {
    let (mut previous, mut current) = (0.0, 1.0);
    for k in (2..=n).rev() {
        let next = scale * x * current - scale * k as f64 * previous;
        previous = current;
        current = next;
    }
    scale * x * current - scale * previous
}

#[test]
fn probabilists_hermite_ten_polynomial_matches_clenshaw_and_public_api() {
    let mut xs = vec![
        -3.01,
        -3.0,
        -3.0 + f64::EPSILON,
        -2.999999999,
        -2.000000001,
        -2.0,
        -1.999999999,
        -1.000000001,
        -1.0,
        -0.999999999,
        -0.0,
        0.0,
        0.999999999,
        1.0,
        1.000000001,
        1.999999999,
        2.0,
        2.000000001,
        2.999999999,
        3.0,
        3.01,
    ];
    // Every positive root in the optimized interval and its neighbors.
    for root in [0.484935707515497, 1.4659890943911582, 2.484325841638954] {
        for x in [root, root - 1e-12, root + 1e-12, root - 1e-9, root + 1e-9] {
            xs.extend([x, -x]);
        }
    }
    for i in 0..=20_000 {
        xs.push(-3.0 + i as f64 * (6.0 / 20_000.0));
    }
    for x in xs {
        let expected = hermite_clenshaw(10, x, 1.0);
        let actual = sci_special::orthogonal::eval_hermitenorm(10, x);
        let tolerance = 4e-12 * (1.0 + expected.abs());
        assert!(
            (actual - expected).abs() <= tolerance,
            "He_10({x}): actual={actual}, expected={expected}"
        );
        assert_eq!(actual, sci_special::orthogonal::eval_hermitenorm(10, x));
        assert_eq!(actual, sci_special::orthogonal::eval_hermitenorm(10, -x));
    }
    assert_eq!(
        sci_special::orthogonal::eval_hermitenorm(10, 0.0_f64),
        -945.0
    );
    assert_eq!(
        sci_special::orthogonal::eval_hermitenorm(10, -0.0_f64),
        -945.0
    );
    assert_eq!(
        sci_special::orthogonal::eval_hermitenorm(10, f64::INFINITY),
        f64::INFINITY
    );
    assert_eq!(
        sci_special::orthogonal::eval_hermitenorm(10, f64::NEG_INFINITY),
        f64::INFINITY
    );
    assert!(sci_special::orthogonal::eval_hermitenorm(10, f64::NAN).is_nan());
    assert!(sci_special::orthogonal::eval_hermitenorm(-10, 0.3_f64).is_nan());
    let x = 0.3_f32;
    let expected = hermite_clenshaw(10, f64::from(x), 1.0) as f32;
    assert!(
        (sci_special::orthogonal::eval_hermitenorm(10, x) - expected).abs()
            <= f32::EPSILON * expected.abs()
    );
    // Physicists' H_n and other probabilists' degrees remain on Clenshaw.
    for n in [1, 2, 3, 9, 11, 20] {
        for x in [-2.5, 0.2, 2.5] {
            assert_eq!(
                sci_special::orthogonal::eval_hermite(n, x),
                hermite_clenshaw(n, x, 2.0)
            );
            assert_eq!(
                sci_special::orthogonal::eval_hermitenorm(n, x),
                hermite_clenshaw(n, x, 1.0)
            );
        }
    }
}

#[test]
fn legendre_polynomial_roots_and_public_float_types() {
    // Positive zeros of P_10, as independently tabulated for
    // Gauss-Legendre quadrature. Include both signs and nearby points.
    for root in [
        0.14887433898163122_f64,
        0.4333953941292472,
        0.6794095682990244,
        0.8650633666889845,
        0.9739065285171717,
    ] {
        for x in [root - 1e-12, root, root + 1e-12] {
            for x in [x, -x] {
                let value = sci_special::orthogonal::eval_legendre(10, x);
                assert!(value.abs() < 5e-11, "P_10({x})={value}");
                let shifted = sci_special::orthogonal::eval_sh_legendre(10, (x + 1.0) * 0.5);
                assert!(shifted.abs() < 5e-11, "shifted P_10({x})={shifted}");
                let x32 = x as f32;
                let typed = sci_special::orthogonal::eval_legendre(10, x32);
                let reference = sci_special::orthogonal::eval_legendre(10, f64::from(x32)) as f32;
                assert!(
                    (typed - reference).abs() <= 4.0 * f32::EPSILON * (1.0 + reference.abs()),
                    "f32 P_10({x32})={typed}, f64 reference={reference}"
                );
            }
        }
    }
    for n in [0, 1, 2, 10, 32, 63, 64, 65, 66, 4096] {
        for x in [
            -1.0,
            -1.0 + f64::EPSILON,
            -0.9,
            -0.5,
            0.0,
            0.5,
            0.9,
            1.0 - f64::EPSILON,
            1.0,
        ] {
            let actual = sci_special::orthogonal::eval_legendre(n, x);
            let x32 = x as f32;
            let typed = sci_special::orthogonal::eval_legendre(n, x32);
            let reference = sci_special::orthogonal::eval_legendre(n, f64::from(x32)) as f32;
            assert_eq!(
                typed.to_bits(),
                reference.to_bits(),
                "f32 degree {n}, x={x}"
            );
            if x.abs() == 1.0 {
                assert_eq!(actual, if x < 0.0 && n % 2 == 1 { -1.0 } else { 1.0 });
            }
        }
    }
    assert!(sci_special::orthogonal::eval_legendre(10, f64::NAN).is_nan());
    assert!(sci_special::orthogonal::eval_legendre(-1, 0.5_f32).is_nan());
}

fn laguerre_three_term(n: i32, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let (mut previous, mut value) = (1.0, 1.0 - x);
    for k in 1..n {
        let k = k as f64;
        let next = ((2.0 * k + 1.0 - x) * value - k * previous) / (k + 1.0);
        previous = value;
        value = next;
    }
    value
}

#[test]
fn classical_laguerre_matches_independent_recurrence() {
    for x in [
        -4.0, -1.0, -0.0, 0.0, 1e-10, 0.1, 0.5, 1.0, 2.0, 4.0, 9.5, 10.0,
    ] {
        for n in [0, 1, 2, 3, 10, 16, 32] {
            let expected = laguerre_three_term(n, x);
            let actual = sci_special::orthogonal::eval_laguerre(n, x);
            assert!(
                (actual - expected).abs() <= 1e-11 * (1.0 + expected.abs()),
                "L_{n}({x}): actual={actual}, expected={expected}"
            );
            assert_eq!(actual, sci_special::orthogonal::eval_genlaguerre(n, 0.0, x));
        }
    }
}

#[test]
fn jacobi_constant_endpoints_reflection_and_infinite_arguments() {
    let (alpha, beta) = (2.5, 0.5);
    assert_eq!(
        sci_special::orthogonal::eval_jacobi(0, alpha, beta, 0.3),
        1.0
    );
    assert_eq!(
        sci_special::orthogonal::eval_jacobi(0, alpha, beta, f64::INFINITY),
        1.0
    );
    // P_2^(5/2,1/2)(x)=(42x²+24x-3)/8.
    let x = 0.3_f64;
    close(
        sci_special::orthogonal::eval_jacobi(2, alpha, beta, x),
        (42.0 * x * x + 24.0 * x - 3.0) / 8.0,
        4e-14,
    );
    // P_2^(a,b)(1)=(a+1)(a+2)/2; left endpoint exchanges a,b.
    let at_one = (alpha + 1.0) * (alpha + 2.0) / 2.0;
    let at_minus_one = (beta + 1.0) * (beta + 2.0) / 2.0;
    assert_eq!(
        sci_special::orthogonal::eval_jacobi(2, alpha, beta, 1.0),
        at_one
    );
    assert_eq!(
        sci_special::orthogonal::eval_jacobi(2, alpha, beta, -1.0),
        at_minus_one
    );
    assert_eq!(
        sci_special::orthogonal::eval_jacobi(3, alpha, beta, -1.0),
        -(beta + 1.0) * (beta + 2.0) * (beta + 3.0) / 6.0
    );
    // The leading coefficient is positive for alpha,beta > -1.
    // Thus the high-degree polynomial diverges with degree parity.
    assert_eq!(
        sci_special::orthogonal::eval_jacobi(2, alpha, beta, f64::INFINITY),
        f64::INFINITY
    );
    assert_eq!(
        sci_special::orthogonal::eval_jacobi(3, alpha, beta, f64::NEG_INFINITY),
        f64::NEG_INFINITY
    );
    assert!(sci_special::orthogonal::eval_jacobi(-1, alpha, beta, 0.3).is_nan());
    assert!(sci_special::orthogonal::eval_jacobi(2, -1.0, beta, 0.3).is_nan());
    assert!(sci_special::orthogonal::eval_jacobi(2, alpha, f64::INFINITY, 0.3).is_nan());
}
