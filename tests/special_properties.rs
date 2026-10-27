mod common;
use common::finite_close;

use sci::{linalg, special as s};
use std::collections::BTreeSet;

fn inventory_names(inventory: &str) -> BTreeSet<&str> {
    inventory
        .lines()
        .skip_while(|line| *line != "#### Functions")
        .skip(1)
        .take_while(|line| !line.starts_with("#### "))
        .filter_map(|line| {
            line.strip_prefix("- ")
                .and_then(|line| line.split('`').nth(1))
        })
        .collect()
}

#[test]
fn inventory_has_references_or_intentional_exclusions() {
    let inventory = include_str!("../function_coverages/scipy_function_list/scipy.special.md");
    let mut expected = inventory_names(inventory);
    let unix = inventory.replace("\r\n", "\n");
    let windows = unix.replace('\n', "\r\n");
    assert_eq!(inventory_names(&unix), inventory_names(&windows));
    assert_eq!(expected.len(), 355);
    for name in [
        "psi",
        "assoc_laguerre",
        "geterr",
        "seterr",
        "cbrt",
        "exp2",
        "exp10",
        "expm1",
        "log1p",
        "round",
    ] {
        assert!(expected.remove(name));
    }
    for name in [
        "c_roots", "cg_roots", "h_roots", "he_roots", "j_roots", "js_roots", "l_roots", "la_roots",
        "p_roots", "ps_roots", "s_roots", "t_roots", "ts_roots", "u_roots", "us_roots",
    ] {
        assert!(expected.remove(name));
    }
    let covered: BTreeSet<_> = common::lines(include_str!("fixtures/special_extended.csv"))
        .map(|line| line.split(',').next().unwrap())
        .collect();
    assert!(
        expected.is_subset(&covered),
        "missing: {:?}",
        expected.difference(&covered).collect::<Vec<_>>()
    );
}

#[test]
fn small_sequences_and_native_work_limits_are_safe() {
    for m in 0..=1 {
        for n in 0..=1 {
            let (values, derivatives) = s::lqmn(m, n, 0.3).unwrap();
            assert_eq!(values.len(), m + 1);
            assert_eq!(values[0].len(), n + 1);
            assert_eq!(derivatives.len(), m + 1);
            finite_close(values[0][0], 0.5 * (1.3_f64 / 0.7).ln(), 1e-14, 1e-14);
        }
    }
    assert_eq!(s::lqn(0, 0.3).unwrap().0.len(), 1);
    assert!(s::polygamma(u32::MAX, 1.0).is_nan());
    assert!(s::multigammaln(1.0, u32::MAX).is_nan());
    assert!(s::mathieu_a(u32::MAX, 0.5).is_nan());
    assert!(s::mathieu_even_coef(1, f64::MAX).is_err());
    assert!(s::pro_cv(1, 3, f64::MAX).is_nan());
    assert!(s::pro_cv_seq(1, 3, f64::MAX).is_err());
    assert!(s::chndtr(f64::INFINITY, 3.0, -1.0).is_nan());
    assert!(s::ncfdtr(3.0, 5.0, -1.0, f64::INFINITY).is_nan());
}

#[test]
fn lambert_real_branches_preserve_domains_and_inverse_identity() {
    let branch_point = -1.0 / std::f64::consts::E;
    assert_eq!(s::lambertw(branch_point), -1.0);
    assert_eq!(s::lambertwm1(branch_point), -1.0);
    for x in [f64::NAN, f64::NEG_INFINITY, -0.5] {
        assert!(s::lambertw(x).is_nan());
        assert!(s::lambertwm1(x).is_nan());
    }
    assert_eq!(s::lambertw(f64::INFINITY), f64::INFINITY);
    for x in [0.1, f64::INFINITY] {
        assert!(s::lambertwm1(x).is_nan());
    }
    for x in [0.0_f64, -0.0] {
        assert_eq!(s::lambertw(x).to_bits(), x.to_bits());
        assert_eq!(s::lambertwm1(x), f64::NEG_INFINITY);
    }
    for x in [-0.36, -1e-3, -1e-12, 1e-300, 1e-3, 0.5, 1e100, 1e300] {
        let w: f64 = s::lambertw(x);
        assert!(w >= -1.0);
        // The relative condition number of w * exp(w) is |1 + w|.
        let rtol = 8.0 * f64::EPSILON * (1.0 + w.abs());
        finite_close(w * w.exp(), x, 0.0, rtol);
    }
    for x in [-0.36, -1e-12, -1e-100, -1e-300] {
        let w: f64 = s::lambertwm1(x);
        assert!(w <= -1.0);
        let rtol = 8.0 * f64::EPSILON * (1.0 + w.abs());
        finite_close(w * w.exp(), x, 0.0, rtol);
    }
}

#[test]
fn distribution_inverses_round_trip() {
    for p in [0.01, 0.2, 0.5, 0.9, 0.99] {
        for nc in [0.0, 0.5, 5.0] {
            finite_close(s::chndtr(s::chndtrix(p, 3.0, nc), 3.0, nc), p, 2e-13, 2e-13);
            finite_close(
                s::ncfdtr(3.0, 5.0, nc, s::ncfdtri(3.0, 5.0, nc, p)),
                p,
                2e-13,
                2e-13,
            );
            finite_close(s::nctdtr(5.0, nc, s::nctdtrit(5.0, nc, p)), p, 2e-12, 2e-12);
        }
    }
    let p = s::ncfdtr(3.0, 5.0, 0.5, 0.5);
    finite_close(s::ncfdtrinc(3.0, 5.0, p, 0.5), 0.5, 2e-12, 2e-12);
    finite_close(s::ncfdtridfd(3.0, p, 0.5, 0.5), 5.0, 2e-12, 2e-12);
}

#[test]
fn quadrature_composes_with_polynomial_evaluation_and_linalg() {
    let (nodes, weights) = s::roots_legendre(16).unwrap();
    for degree in 0..=30 {
        let values: Vec<_> = nodes.iter().map(|x| x.powi(degree)).collect();
        let expected = if degree % 2 == 0 {
            2.0 / (degree + 1) as f64
        } else {
            0.0
        };
        finite_close(
            linalg::dot(&weights, &values).unwrap(),
            expected,
            3e-14,
            3e-14,
        );
    }
    let coefficients = s::legendre(7, false).unwrap();
    for x in [-0.9, 0.0, 0.3, 0.9] {
        let value = coefficients.iter().fold(0.0, |v, c| v * x + c);
        finite_close(value, s::eval_legendre(7, x), 2e-13, 2e-13);
    }
}

#[test]
fn elementary_limits_and_elliptic_conventions() {
    assert_eq!(s::kl_div(-1.0, 2.0), f64::INFINITY);
    assert_eq!(s::kl_div(0.0, f64::INFINITY), f64::INFINITY);
    assert_eq!(s::pseudo_huber(1.0, f64::INFINITY), f64::INFINITY);
    assert_eq!(s::pseudo_huber(f64::INFINITY, 2.0), 2.0);
    for m in [-2.0, 0.0, 0.5, 0.99] {
        let (k, e) = s::ellipke(m).unwrap();
        finite_close(k, s::ellipk(m).unwrap(), 1e-14, 1e-14);
        finite_close(e, s::ellipe(m).unwrap(), 1e-14, 1e-14);
        finite_close(
            s::cel3((1.0 - m).sqrt(), 0.8).unwrap(),
            s::ellippi(0.2, m).unwrap(),
            1e-14,
            1e-14,
        );
    }
    assert!(s::ellipkm1(1e-20_f64).unwrap().is_finite());
    finite_close(
        s::logsumexp(&[1000.0, 1000.0]),
        1000.0 + 2.0_f64.ln(),
        1e-14,
        1e-14,
    );
}

#[test]
fn kelvin_zero_sequences_agree_and_remain_ordered_after_exponential_underflow() {
    let zeros = s::keip_zeros(4096).unwrap();
    assert_eq!(zeros.len(), 4096);
    assert!(zeros.iter().all(|x| x.is_finite() && *x > 0.));
    assert!(zeros.windows(2).all(|pair| pair[0] < pair[1]));
    for count in [1, 3, 5] {
        let grouped = s::kelvin_zeros(count).unwrap();
        assert_eq!(grouped[7], zeros[..count]);
        for &x in &grouped[7] {
            // The scalar Kelvin series has cancellation near x=9.4;
            // compare the derivative residual on its local envelope scale.
            let envelope = (-x / std::f64::consts::SQRT_2).exp() / x.sqrt();
            assert!(s::keip(x).abs() <= 2e-9 * envelope);
        }
    }
    assert!(s::keip_zeros(0).is_err());
    assert!(s::keip_zeros(4097).is_err());
}

#[test]
fn rust_kernel_recurrences_and_reflections() {
    for x in [0.0_f64, 0.25, 2.0, 8.0] {
        for v in [-2.5, -0.5, 0.5, 2.5] {
            let previous = s::pbdv(v - 1.0, x).0;
            let (value, slope) = s::pbdv(v, x);
            let next = s::pbdv(v + 1.0, x).0;
            finite_close(next, x * value - v * previous, 2e-11, 2e-11);
            finite_close(slope, 0.5 * x * value - next, 2e-11, 2e-11);
        }
    }
    for x in [0.25_f64, 1.0, 8.0] {
        let (positive, _) = s::modfresnelp(x);
        let (negative, _) = s::modfresnelp(-x);
        let (minus, _) = s::modfresnelm(-x);
        let complete = (std::f64::consts::PI / 2.0).sqrt();
        finite_close(positive.re + negative.re, complete, 1e-14, 1e-14);
        finite_close(positive.im + negative.im, complete, 1e-14, 1e-14);
        finite_close(minus.re, negative.re, 1e-14, 1e-14);
        finite_close(minus.im, -negative.im, 1e-14, 1e-14);
    }
    for n in 0..=8 {
        finite_close(s::pro_cv(0, n, 0.0), (n * (n + 1)) as f64, 0.0, 0.0);
        finite_close(
            s::pro_ang1(0, n, 0.0, 0.3).0,
            s::eval_legendre(n as i32, 0.3),
            2e-13,
            2e-13,
        );
    }
    assert!(s::pbdv(4097.0, 1.0).0.is_nan());
    assert!(s::hyperu(100001.0, 1.0, 1.0).is_nan());
    assert!(s::hyp2f1(100001.0, 1.0, 2.0, 0.5).is_nan());
    assert!(s::lpmv(4097, 0.5, 0.3).is_nan());
}

#[test]
fn oblate_radial_pairs_preserve_the_defining_wronskian() {
    // DLMF 30.4: W(R^(1),R^(2)) = 1/[c*(1+x^2)]. This independent
    // identity checks values, derivatives, and normalization together,
    // including the small-argument convolution and both parity branches.
    for m in 0..=1 {
        for n in [1, 3] {
            for c in [0.01, 1.0, 5.0] {
                for x in [0.0, 0.5, 1.01, 2.0] {
                    let (first, first_prime) = s::obl_rad1(m, n, c, x);
                    let (second, second_prime) = s::obl_rad2(m, n, c, x);
                    finite_close(
                        first * second_prime - first_prime * second,
                        1.0 / (c * (1.0 + x * x)),
                        2e-12,
                        2e-10,
                    );
                }
            }
        }
    }
}

#[test]
fn real_modified_bessel_i_matches_scipy() {
    // Check both public float APIs, including relative comparisons in tiny tails.
    for row in common::rows(include_str!("fixtures/bessel_iv.csv")) {
        let v: f64 = common::number(row[1]);
        let x: f64 = common::number(row[2]);
        let actual = if row[0] == "f32" {
            [
                s::iv(v as f32, x as f32) as f64,
                s::ive(v as f32, x as f32) as f64,
            ]
        } else {
            [s::iv(v, x), s::ive(v, x)]
        };
        let (rtol, atol) = if row[0] == "f32" {
            (3e-6, 2.0 * f32::from_bits(1) as f64)
        } else {
            (5e-12, 2.0 * f64::from_bits(1))
        };
        for (value, reference) in actual.into_iter().zip(&row[3..]) {
            let expected: f64 = common::number(reference);
            if expected.is_nan() {
                assert!(value.is_nan(), "{row:?}: {value}");
            } else if expected.is_infinite() {
                assert_eq!(value, expected, "{row:?}");
            } else {
                assert!(
                    (value - expected).abs() <= atol + rtol * expected.abs(),
                    "{row:?}: {value}, expected {expected}"
                );
            }
        }
    }
}

#[test]
fn real_modified_bessel_i_recurrence_and_scaling() {
    for v in [0.3_f64, 1.0, 3.25, 20.5, 49.0, 50.0, 51.0, 100.25] {
        for x in [0.1, 2.0, 5.0, 20.0, 100.0, 710.0] {
            let i = s::ive(v, x);
            let lower = s::ive(v - 1.0, x);
            let upper = s::ive(v + 1.0, x);
            assert!((lower - upper - 2.0 * v / x * i).abs() <= 2e-12 * lower.abs());
            let unscaled = s::iv(v, x);
            let half = (0.5_f64 * x).exp();
            let restored = (i * half) * half;
            assert!((unscaled - restored).abs() <= 2e-12 * unscaled.abs());
        }
    }
    for v in [0.0, 1.0, 2.0, 3.0, 50.0, 51.0] {
        let sign = if v % 2.0 == 0.0 { 1.0 } else { -1.0 };
        assert_eq!(s::iv(-v, 2.0), s::iv(v, 2.0));
        assert_eq!(s::ive(v, -2.0), sign * s::ive(v, 2.0));
    }
    for v in [0.0, 0.5, 3.25, 51.0] {
        let x = 1e300_f64;
        let expected = (2.0 * core::f64::consts::PI).sqrt().recip() / x.sqrt();
        assert!((s::ive(v, x) / expected - 1.0).abs() < 1e-12);
    }
}

#[test]
fn real_modified_bessel_i_special_values() {
    for kernel in [s::iv::<f64>, s::ive::<f64>] {
        assert!(kernel(f64::INFINITY, 1.0).is_nan());
        assert!(kernel(f64::NAN, 1.0).is_nan());
        assert!(kernel(1.0, f64::NAN).is_nan());
        assert!(kernel(0.3, -1.0).is_nan());
        assert_eq!(kernel(0.0, 0.0), 1.0);
        assert_eq!(kernel(-3.0, 0.0), 0.0);
        assert_eq!(kernel(-3.25, 0.0), f64::INFINITY);
        assert_eq!(kernel(3.0, -1e-300).to_bits(), (-0.0_f64).to_bits());
    }
    assert_eq!(s::iv(0.3, f64::INFINITY), f64::INFINITY);
    assert_eq!(s::iv(3.0, f64::NEG_INFINITY), f64::NEG_INFINITY);
    assert_eq!(s::ive(0.3, f64::INFINITY), 0.0);
    assert!(s::ive(0.3, f64::NEG_INFINITY).is_nan());
}

#[test]
fn kelvin_first_kind_matches_references_across_kernel_switches() {
    // References from mpmath 1.4 at 50 digits at the binary64 value of x:
    // I0(z) and exp(iπ/4) I1(z) at z = x exp(iπ/4). The points straddle the
    // binary64 series, double-double series and large-argument expansion
    // used for ber, bei, ber′ and bei′.
    let references: [(f64, [f64; 4]); 8] = [
        (
            0.5_f64,
            [
                0.99902346399083826,
                0.062493218382199459,
                -0.0078120761475077336,
                0.24991862111621022,
            ],
        ),
        (
            3.99_f64,
            [
                -2.5321455801442405,
                2.297480090877188,
                -3.1195304333278595,
                -0.4668586846429602,
            ],
        ),
        (
            4.0_f64,
            [
                -2.5634165572585798,
                2.2926903226992998,
                -3.1346539628386434,
                -0.49113744063072261,
            ],
        ),
        (
            12.96_f64,
            [
                -841.208251812595,
                653.6560322797799,
                -1024.521731612371,
                -158.67252147619254,
            ],
        ),
        (
            19.99_f64,
            [
                47971.8032028607,
                113658.75032958809,
                -47685.159277070146,
                111433.5463421547,
            ],
        ),
        (
            20.0_f64,
            [
                47489.37026506176,
                114775.19736006622,
                -48803.197847170976,
                111855.02522349712,
            ],
        ),
        (
            32.145_f64,
            [
                -494229336.98237234,
                -176657800.23154956,
                -216811482.771475,
                -471670130.6416777,
            ],
        ),
        (
            300.0_f64,
            [
                -9.68152922933538e+89,
                -2.9365929560917256e+90,
                1.3935147324614752e+90,
                -2.7561760396340746e+90,
            ],
        ),
    ];
    for (x, [ber, bei, berp, beip]) in references {
        let value_scale = ber.hypot(bei);
        let derivative_scale = berp.hypot(beip);
        for sign in [1.0, -1.0] {
            let got = [
                (s::ber(sign * x), ber, value_scale),
                (s::bei(sign * x), bei, value_scale),
                (sign * s::berp(sign * x), berp, derivative_scale),
                (sign * s::beip(sign * x), beip, derivative_scale),
            ];
            for (index, (value, expected, scale)) in got.into_iter().enumerate() {
                assert!(
                    (value - expected).abs() <= 2e-15 * scale,
                    "component {index} at x = {}: {value}, expected {expected}",
                    sign * x
                );
            }
        }
        let [be, _, bep, _] = s::kelvin(x);
        assert!((be.re - ber).abs() <= 2e-15 * value_scale);
        assert!((be.im - bei).abs() <= 2e-15 * value_scale);
        assert!((bep.re - berp).abs() <= 2e-15 * derivative_scale);
        assert!((bep.im - beip).abs() <= 2e-15 * derivative_scale);
    }
}

#[test]
fn real_modified_bessel_i_large_arguments_beyond_cf1() {
    // References from mpmath 1.3 at 50 digits: besseli(v, x) * exp(-x).
    // CF1 needs O(x) terms here, so these exercise the large-x expansion.
    for (v, x, expected) in [
        (40.0_f64, 5e4_f64, 0.001_755_809_395_200_794_5_f64),
        (-20.7, 1e5, 0.001_258_867_871_776_090_9),
        (12.5, 3e4, 0.002_297_313_443_476_216_7),
        (3.25, 2e3, 0.008_897_645_994_093_099),
        (0.0, 1500.0, 0.010_301_504_096_519_597),
    ] {
        let value = s::ive(v, x);
        assert!(
            (value - expected).abs() <= 1e-13 * expected,
            "ive({v}, {x}) = {value}, expected {expected}"
        );
    }
    for v in [0.3_f64, 7.0, 25.5, 49.0] {
        for x in [1e3_f64, 1.5e3, 4e4, 1e5, 1e7] {
            let i = s::ive(v, x);
            let lower = s::ive(v - 1.0, x);
            let upper = s::ive(v + 1.0, x);
            assert!((lower - upper - 2.0 * v / x * i).abs() <= 1e-11 * lower.abs());
        }
    }
}
