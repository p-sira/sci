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
            let (values, derivatives) = s::orthogonal::lqmn(m, n, 0.3).unwrap();
            assert_eq!(values.len(), m + 1);
            assert_eq!(values[0].len(), n + 1);
            assert_eq!(derivatives.len(), m + 1);
            finite_close(values[0][0], 0.5 * (1.3_f64 / 0.7).ln(), 1e-14, 1e-14);
        }
    }
    assert_eq!(s::orthogonal::lqn(0, 0.3).unwrap().0.len(), 1);
    assert!(s::gamma::polygamma(u32::MAX, 1.0).is_nan());
    assert!(s::gamma::multigammaln(1.0, u32::MAX).is_nan());
    assert!(s::wave::mathieu_a(u32::MAX, 0.5).is_nan());
    assert!(s::wave::mathieu_even_coef(1, f64::MAX).is_err());
    assert!(s::wave::pro_cv(1, 3, f64::MAX).is_nan());
    assert!(s::wave::pro_cv_seq(1, 3, f64::MAX).is_err());
    assert!(s::probability::chndtr(f64::INFINITY, 3.0, -1.0).is_nan());
    assert!(s::probability::ncfdtr(3.0, 5.0, -1.0, f64::INFINITY).is_nan());
}

#[test]
fn lambert_real_branches_preserve_domains_and_inverse_identity() {
    let branch_point = -1.0 / std::f64::consts::E;
    assert_eq!(s::elementary::lambertw(branch_point), -1.0);
    assert_eq!(s::elementary::lambertwm1(branch_point), -1.0);
    // lambert_w 2.1 corrects the rounded branch-point offset internally.
    // The first representable input above -1/e must produce distinct roots
    // on either side of -1 rather than losing the square-root separation.
    let nearest_interior = f64::from_bits(branch_point.to_bits() - 1);
    let upper = s::elementary::lambertw(nearest_interior);
    let lower = s::elementary::lambertwm1(nearest_interior);
    assert!(upper > -1.0);
    assert!(lower < -1.0);
    finite_close(
        upper * upper.exp(),
        nearest_interior,
        0.0,
        8.0 * f64::EPSILON,
    );
    finite_close(
        lower * lower.exp(),
        nearest_interior,
        0.0,
        8.0 * f64::EPSILON,
    );
    for x in [f64::NAN, f64::NEG_INFINITY, -0.5] {
        assert!(s::elementary::lambertw(x).is_nan());
        assert!(s::elementary::lambertwm1(x).is_nan());
    }
    assert_eq!(s::elementary::lambertw(f64::INFINITY), f64::INFINITY);
    for x in [0.1, f64::INFINITY] {
        assert!(s::elementary::lambertwm1(x).is_nan());
    }
    for x in [0.0_f64, -0.0] {
        assert_eq!(s::elementary::lambertw(x).to_bits(), x.to_bits());
        assert_eq!(s::elementary::lambertwm1(x), f64::NEG_INFINITY);
    }
    for x in [-0.36, -1e-3, -1e-12, 1e-300, 1e-3, 0.5, 1e100, 1e300] {
        let w: f64 = s::elementary::lambertw(x);
        assert!(w >= -1.0);
        // The relative condition number of w * exp(w) is |1 + w|.
        let rtol = 8.0 * f64::EPSILON * (1.0 + w.abs());
        finite_close(w * w.exp(), x, 0.0, rtol);
    }
    for x in [-0.36, -1e-12, -1e-100, -1e-300] {
        let w: f64 = s::elementary::lambertwm1(x);
        assert!(w <= -1.0);
        let rtol = 8.0 * f64::EPSILON * (1.0 + w.abs());
        finite_close(w * w.exp(), x, 0.0, rtol);
    }
}

#[test]
fn distribution_inverses_round_trip() {
    for p in [0.01, 0.2, 0.5, 0.9, 0.99] {
        for nc in [0.0, 0.5, 5.0] {
            finite_close(
                s::probability::chndtr(s::probability::chndtrix(p, 3.0, nc), 3.0, nc),
                p,
                2e-13,
                2e-13,
            );
            finite_close(
                s::probability::ncfdtr(3.0, 5.0, nc, s::probability::ncfdtri(3.0, 5.0, nc, p)),
                p,
                2e-13,
                2e-13,
            );
            finite_close(
                s::probability::nctdtr(5.0, nc, s::probability::nctdtrit(5.0, nc, p)),
                p,
                2e-12,
                2e-12,
            );
        }
    }
    let p = s::probability::ncfdtr(3.0, 5.0, 0.5, 0.5);
    finite_close(
        s::probability::ncfdtrinc(3.0, 5.0, p, 0.5),
        0.5,
        2e-12,
        2e-12,
    );
    finite_close(
        s::probability::ncfdtridfd(3.0, p, 0.5, 0.5),
        5.0,
        2e-12,
        2e-12,
    );
}

#[test]
fn quadrature_composes_with_polynomial_evaluation_and_linalg() {
    let (nodes, weights) = s::orthogonal::roots_legendre(16).unwrap();
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
    let coefficients = s::orthogonal::legendre(7, false).unwrap();
    for x in [-0.9, 0.0, 0.3, 0.9] {
        let value = coefficients.iter().fold(0.0, |v, c| v * x + c);
        finite_close(value, s::orthogonal::eval_legendre(7, x), 2e-13, 2e-13);
    }
}

#[test]
fn elementary_limits_and_elliptic_conventions() {
    assert_eq!(s::elementary::kl_div(-1.0, 2.0), f64::INFINITY);
    assert_eq!(s::elementary::kl_div(0.0, f64::INFINITY), f64::INFINITY);
    assert_eq!(
        s::elementary::pseudo_huber(1.0, f64::INFINITY),
        f64::INFINITY
    );
    assert_eq!(s::elementary::pseudo_huber(f64::INFINITY, 2.0), 2.0);
    for m in [-2.0, 0.0, 0.5, 0.99] {
        let (k, e) = s::elliptic::ellipke(m).unwrap();
        finite_close(k, s::elliptic::ellipk(m).unwrap(), 1e-14, 1e-14);
        finite_close(e, s::elliptic::ellipe(m).unwrap(), 1e-14, 1e-14);
        finite_close(
            s::elliptic::cel3((1.0 - m).sqrt(), 0.8).unwrap(),
            s::elliptic::ellippi(0.2, m).unwrap(),
            1e-14,
            1e-14,
        );
    }
    assert!(s::elliptic::ellipkm1(1e-20_f64).unwrap().is_finite());
    finite_close(
        s::elementary::logsumexp(&[1000.0, 1000.0]),
        1000.0 + 2.0_f64.ln(),
        1e-14,
        1e-14,
    );
}

#[test]
fn kelvin_zero_sequences_agree_and_remain_ordered_after_exponential_underflow() {
    let zeros = s::bessel::keip_zeros(4096).unwrap();
    assert_eq!(zeros.len(), 4096);
    assert!(zeros.iter().all(|x| x.is_finite() && *x > 0.));
    assert!(zeros.windows(2).all(|pair| pair[0] < pair[1]));
    for count in [1, 3, 5] {
        let grouped = s::bessel::kelvin_zeros(count).unwrap();
        assert_eq!(grouped[7], zeros[..count]);
        for &x in &grouped[7] {
            // The scalar Kelvin series has cancellation near x=9.4;
            // compare the derivative residual on its local envelope scale.
            let envelope = (-x / std::f64::consts::SQRT_2).exp() / x.sqrt();
            assert!(s::bessel::keip(x).abs() <= 2e-9 * envelope);
        }
    }
    assert!(s::bessel::keip_zeros(0).is_err());
    assert!(s::bessel::keip_zeros(4097).is_err());
}

#[test]
fn rust_kernel_recurrences_and_reflections() {
    for x in [0.0_f64, 0.25, 2.0, 8.0] {
        for v in [-2.5, -0.5, 0.5, 2.5] {
            let previous = s::wave::pbdv(v - 1.0, x).0;
            let (value, slope) = s::wave::pbdv(v, x);
            let next = s::wave::pbdv(v + 1.0, x).0;
            finite_close(next, x * value - v * previous, 2e-11, 2e-11);
            finite_close(slope, 0.5 * x * value - next, 2e-11, 2e-11);
        }
    }
    for x in [0.25_f64, 1.0, 8.0] {
        let (positive, _) = s::integrals::modfresnelp(x);
        let (negative, _) = s::integrals::modfresnelp(-x);
        let (minus, _) = s::integrals::modfresnelm(-x);
        let complete = (std::f64::consts::PI / 2.0).sqrt();
        finite_close(positive.re + negative.re, complete, 1e-14, 1e-14);
        finite_close(positive.im + negative.im, complete, 1e-14, 1e-14);
        finite_close(minus.re, negative.re, 1e-14, 1e-14);
        finite_close(minus.im, -negative.im, 1e-14, 1e-14);
    }
    for n in 0..=8 {
        finite_close(s::wave::pro_cv(0, n, 0.0), (n * (n + 1)) as f64, 0.0, 0.0);
        finite_close(
            s::wave::pro_ang1(0, n, 0.0, 0.3).0,
            s::orthogonal::eval_legendre(n as i32, 0.3),
            2e-13,
            2e-13,
        );
    }
    assert!(s::wave::pbdv(4097.0, 1.0).0.is_nan());
    assert!(s::hypergeometric::hyperu(100001.0, 1.0, 1.0).is_nan());
    assert!(s::hypergeometric::hyp2f1(100001.0, 1.0, 2.0, 0.5).is_nan());
    assert!(s::orthogonal::lpmv(4097, 0.5, 0.3).is_nan());
}

#[test]
fn parabolic_cylinder_sequence_reuses_the_shared_recurrence() {
    // The sequence must agree with the independently exposed scalar kernel,
    // including the negative-order convention. The identities below also
    // check the normalization and derivatives without using either sequence.
    for v in [-3.5, -0.5, 0.0, 0.5, 2.5, 16.5] {
        for x in [-2.0, 0.0, 1.5, 5.0] {
            let (values, derivatives) = s::wave::pbdv_seq(v, x).unwrap();
            let n = v.trunc().max(1.0) as usize;
            assert_eq!(values.len(), n + 1);
            assert_eq!(derivatives.len(), n + 1);
            let base = v - v.trunc();
            for i in 0..=n {
                let order = base + i as f64;
                let (expected, expected_derivative) = s::wave::pbdv(order, x);
                finite_close(values[i], expected, 2e-12, 2e-11);
                finite_close(derivatives[i], expected_derivative, 2e-12, 2e-11);
                if i < n {
                    finite_close(
                        derivatives[i],
                        0.5 * x * values[i] - values[i + 1],
                        2e-12,
                        2e-11,
                    );
                }
            }
        }
    }
    for x in [-2.0_f64, 0.0, 1.5, 5.0] {
        let (values, derivatives) = s::wave::pbdv_seq(16.0, x).unwrap();
        let d0 = (-0.25 * x * x).exp();
        finite_close(values[0], d0, 2e-14, 2e-14);
        finite_close(values[1], x * d0, 2e-14, 2e-14);
        finite_close(derivatives[0], -0.5 * x * d0, 2e-14, 2e-14);
        for k in 1..16 {
            finite_close(
                values[k + 1],
                x * values[k] - k as f64 * values[k - 1],
                2e-12,
                2e-11,
            );
        }
    }
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
                    let (first, first_prime) = s::wave::obl_rad1(m, n, c, x);
                    let (second, second_prime) = s::wave::obl_rad2(m, n, c, x);
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
                s::bessel::iv(v as f32, x as f32) as f64,
                s::bessel::ive(v as f32, x as f32) as f64,
            ]
        } else {
            [s::bessel::iv(v, x), s::bessel::ive(v, x)]
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
            let i = s::bessel::ive(v, x);
            let lower = s::bessel::ive(v - 1.0, x);
            let upper = s::bessel::ive(v + 1.0, x);
            assert!((lower - upper - 2.0 * v / x * i).abs() <= 2e-12 * lower.abs());
            let unscaled = s::bessel::iv(v, x);
            let half = (0.5_f64 * x).exp();
            let restored = (i * half) * half;
            assert!((unscaled - restored).abs() <= 2e-12 * unscaled.abs());
        }
    }
    for v in [0.0, 1.0, 2.0, 3.0, 50.0, 51.0] {
        let sign = if v % 2.0 == 0.0 { 1.0 } else { -1.0 };
        assert_eq!(s::bessel::iv(-v, 2.0), s::bessel::iv(v, 2.0));
        assert_eq!(s::bessel::ive(v, -2.0), sign * s::bessel::ive(v, 2.0));
    }
    for v in [0.0, 0.5, 3.25, 51.0] {
        let x = 1e300_f64;
        let expected = (2.0 * core::f64::consts::PI).sqrt().recip() / x.sqrt();
        assert!((s::bessel::ive(v, x) / expected - 1.0).abs() < 1e-12);
    }
}

#[test]
fn real_modified_bessel_i_special_values() {
    for kernel in [s::bessel::iv::<f64>, s::bessel::ive::<f64>] {
        assert!(kernel(f64::INFINITY, 1.0).is_nan());
        assert!(kernel(f64::NAN, 1.0).is_nan());
        assert!(kernel(1.0, f64::NAN).is_nan());
        assert!(kernel(0.3, -1.0).is_nan());
        assert_eq!(kernel(0.0, 0.0), 1.0);
        assert_eq!(kernel(-3.0, 0.0), 0.0);
        assert_eq!(kernel(-3.25, 0.0), f64::INFINITY);
        assert_eq!(kernel(3.0, -1e-300).to_bits(), (-0.0_f64).to_bits());
    }
    assert_eq!(s::bessel::iv(0.3, f64::INFINITY), f64::INFINITY);
    assert_eq!(s::bessel::iv(3.0, f64::NEG_INFINITY), f64::NEG_INFINITY);
    assert_eq!(s::bessel::ive(0.3, f64::INFINITY), 0.0);
    assert!(s::bessel::ive(0.3, f64::NEG_INFINITY).is_nan());
}

#[test]
fn modified_fresnel_matches_high_precision_around_real_kernel_switch() {
    // mpmath 1.4, 70 digits: erfc((1-i)x/sqrt(2)) times
    // sqrt(pi)*(1+i)/(2sqrt(2)), with K = exp(-i(x²+pi/4))*F/sqrt(pi).
    let references = [
        (
            2.5,
            [
                0.021_349_229_542_882_17,
                0.196139324890222,
                0.08440398319766853,
                0.07257157385142538,
            ],
        ),
        (
            3.0,
            [
                -0.0762064890725186,
                -0.1469054582360189,
                0.06947487853046293,
                0.06238042139537748,
            ],
        ),
        (
            3.0000001,
            [
                -0.07620639795948005,
                -0.146_905_499_447_840_1,
                0.06947487606448781,
                0.06238041960467845,
            ],
        ),
        (
            -3.0,
            [
                1.3295206263880189,
                1.400_219_595_551_519,
                -0.9806051404151399,
                -0.47449890663713405,
            ],
        ),
        (
            -2.5,
            [
                1.231964907772618,
                1.0571748124252783,
                0.9150454350268309,
                -0.03939235730386856,
            ],
        ),
    ];
    for (x, expected) in references {
        let (f, k) = s::integrals::modfresnelp(x);
        let (f_minus, k_minus) = s::integrals::modfresnelm(x);
        for (got, reference) in [f.re, f.im, k.re, k.im].into_iter().zip(expected) {
            finite_close(got, reference, 1e-14, 1e-14);
        }
        finite_close(f_minus.re, f.re, 1e-14, 1e-14);
        finite_close(f_minus.im, -f.im, 1e-14, 1e-14);
        finite_close(k_minus.re, k.re, 1e-14, 1e-14);
        finite_close(k_minus.im, -k.im, 1e-14, 1e-14);
    }
}

#[test]
fn kelvin_first_kind_matches_references_across_kernel_switches() {
    // References from mpmath 1.4 at 50 digits at the binary64 value of x:
    // I0(z) and exp(iπ/4) I1(z) at z = x exp(iπ/4). The points straddle the
    // binary64 series, double-double series and large-argument expansion
    // used for ber, bei, ber′ and bei′.
    let references: [(f64, [f64; 4]); 10] = [
        (
            0.5_f64,
            [
                0.999_023_463_990_838_3,
                0.062493218382199459,
                -0.007_812_076_147_507_734,
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
                -2.563_416_557_258_58,
                2.2926903226992998,
                -3.1346539628386434,
                -0.491_137_440_630_722_6,
            ],
        ),
        (
            6.0,
            [
                -8.858315966045036,
                -7.334746540847962,
                -0.29307996650807906,
                -10.846223328651287,
            ],
        ),
        (
            7.249,
            [
                0.18206539846508056,
                -25.239898988663878,
                18.016716311828627,
                -15.93668055700112,
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
                -48_803.197_847_170_97,
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
                (s::bessel::ber(sign * x), ber, value_scale),
                (s::bessel::bei(sign * x), bei, value_scale),
                (sign * s::bessel::berp(sign * x), berp, derivative_scale),
                (sign * s::bessel::beip(sign * x), beip, derivative_scale),
            ];
            for (index, (value, expected, scale)) in got.into_iter().enumerate() {
                assert!(
                    (value - expected).abs() <= 2e-15 * scale,
                    "component {index} at x = {}: {value}, expected {expected}",
                    sign * x
                );
            }
        }
        let [be, _, bep, _] = s::bessel::kelvin(x);
        assert!((be.re - ber).abs() <= 2e-15 * value_scale);
        assert!((be.im - bei).abs() <= 2e-15 * value_scale);
        assert!((bep.re - berp).abs() <= 2e-15 * derivative_scale);
        assert!((bep.im - beip).abs() <= 2e-15 * derivative_scale);
    }
}

#[test]
fn kelvin_second_kind_matches_references_across_kernel_switches() {
    // References from mpmath 1.4 at 50 digits at the binary64 value of x:
    // K0(z) and -exp(iπ/4) K1(z) at z = x exp(iπ/4). The points straddle the
    // power series and the continued fraction used for ker, kei, ker′ and
    // kei′, and include large arguments where the phase of exp(-z) matters.
    let references: [(f64, [f64; 4]); 9] = [
        (
            0.5_f64,
            [
                0.855_905_872_118_634_2,
                -0.6715816950943676,
                -1.8197997533173526,
                0.33320379160332984,
            ],
        ),
        (
            1.99_f64,
            [
                -0.040_585_656_714_261_09,
                -0.20460572926266777,
                -0.10917676934115904,
                0.2213247824648346,
            ],
        ),
        (
            2.0_f64,
            [
                -0.041664513991509532,
                -0.202_400_067_764_704_3,
                -0.10660096588105265,
                0.21980790991960552,
            ],
        ),
        (
            2.4_f64,
            [
                -0.06737349336152855,
                -0.1262414877404732,
                -0.029712257069996982,
                0.16210691534821758,
            ],
        ),
        (
            8.0_f64,
            [
                0.0014858340685189625,
                3.695_839_561_259_596e-4,
                -8.797_240_991_414_724e-4,
                -0.00133631291485889,
            ],
        ),
        (
            19.99_f64,
            [
                -7.639_262_950_051_707e-8,
                -1.8780471626416795e-7,
                -7.692507386882865e-8,
                1.9148756853049503e-7,
            ],
        ),
        (
            20.0_f64,
            [
                -7.715_233_109_860_961e-8,
                -1.8589415111194372e-7,
                -7.501_859_210_700_241e-8,
                1.9062427567453116e-7,
            ],
        ),
        (
            45.0_f64,
            [
                1.9809218748338303e-15,
                -2.0178948281079617e-15,
                -2.8495996265400866e-15,
                4.839_305_146_504_315e-17,
            ],
        ),
        (
            300.0_f64,
            [
                2.4263778900964785e-94,
                4.813_129_265_455_171e-94,
                1.6836512069978793e-94,
                -5.127_124_129_126_373e-94,
            ],
        ),
    ];
    for (x, [ker, kei, kerp, keip]) in references {
        let value_scale = ker.hypot(kei);
        let derivative_scale = kerp.hypot(keip);
        let got = [
            (s::bessel::ker(x), ker, value_scale),
            (s::bessel::kei(x), kei, value_scale),
            (s::bessel::kerp(x), kerp, derivative_scale),
            (s::bessel::keip(x), keip, derivative_scale),
        ];
        for (index, (value, expected, scale)) in got.into_iter().enumerate() {
            assert!(
                (value - expected).abs() <= 2e-15 * scale,
                "component {index} at x = {x}: {value}, expected {expected}"
            );
        }
        let [_, ke, _, kep] = s::bessel::kelvin(x);
        assert!((ke.re - ker).abs() <= 2e-15 * value_scale);
        assert!((ke.im - kei).abs() <= 2e-15 * value_scale);
        assert!((kep.re - kerp).abs() <= 2e-15 * derivative_scale);
        assert!((kep.im - keip).abs() <= 2e-15 * derivative_scale);
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
        let value = s::bessel::ive(v, x);
        assert!(
            (value - expected).abs() <= 1e-13 * expected,
            "ive({v}, {x}) = {value}, expected {expected}"
        );
    }
    for v in [0.3_f64, 7.0, 25.5, 49.0] {
        for x in [1e3_f64, 1.5e3, 4e4, 1e5, 1e7] {
            let i = s::bessel::ive(v, x);
            let lower = s::bessel::ive(v - 1.0, x);
            let upper = s::bessel::ive(v + 1.0, x);
            assert!((lower - upper - 2.0 * v / x * i).abs() <= 1e-11 * lower.abs());
        }
    }
}

#[test]
fn odd_modified_mathieu_functions_of_order_zero_are_undefined() {
    // Odd Mathieu functions start at order one; order zero used to panic.
    for f in [s::wave::mathieu_modsem1, s::wave::mathieu_modsem2] {
        let (value, derivative) = f(0, 1.0, 0.5);
        assert!(value.is_nan() && derivative.is_nan());
    }
}

#[test]
fn modified_mathieu_pairs_preserve_wronskians_and_first_kind_parity() {
    // DLMF 28.22: W(Mc^(1), Mc^(2)) = W(Ms^(1), Ms^(2)) = 2/pi.
    // The first-kind solutions inherit even/odd parity in the real coordinate.
    for (first, second, odd) in [
        (
            s::wave::mathieu_modcem1 as fn(u32, f64, f64) -> (f64, f64),
            s::wave::mathieu_modcem2 as fn(u32, f64, f64) -> (f64, f64),
            false,
        ),
        (
            s::wave::mathieu_modsem1 as fn(u32, f64, f64) -> (f64, f64),
            s::wave::mathieu_modsem2 as fn(u32, f64, f64) -> (f64, f64),
            true,
        ),
    ] {
        for m in 0..=6 {
            if odd && m == 0 {
                continue;
            }
            for q in [0.5, 1.0, 10.0, 100.0] {
                for x in [0.0, 0.3, 2.0, 7.0] {
                    let (f, d) = first(m, q, x);
                    let (g, e) = second(m, q, x);
                    // At m=6, q<=1, cancellation in the Bessel-product sum
                    // divided by the small first Fourier coefficient limits
                    // the Wronskian residual to about 7e-11 (SciPy: 4e-10).
                    finite_close(f * e - d * g, std::f64::consts::FRAC_2_PI, 2e-10, 2e-11);
                    let (negative_f, negative_d) = first(m, q, -x);
                    let sign = if odd { -1.0 } else { 1.0 };
                    finite_close(negative_f, sign * f, 2e-12, 2e-11);
                    finite_close(negative_d, -sign * d, 2e-12, 2e-11);
                }
            }
        }
    }
}

#[test]
fn mathieu_zero_coupling_scalar_and_fourier_paths_agree() {
    // q=0 bypasses the eigenvalue search, but still constructs the public
    // coefficient vector (and the shorter vector used for angular functions).
    // The ce_0 convention is cos(0*x)/sqrt(2), unlike the unit coefficients
    // for all positive orders.
    for m in [0_u32, 1, 2, 3, 8, 32, 128] {
        for odd in [false, true] {
            if odd && m == 0 {
                continue;
            }
            let lambda = if odd {
                s::wave::mathieu_b(m, 0.0)
            } else {
                s::wave::mathieu_a(m, 0.0)
            };
            assert_eq!(lambda, f64::from(m).powi(2), "order={m}, odd={odd}");

            let coefficients = if odd {
                s::wave::mathieu_odd_coef(m, 0.0).unwrap()
            } else {
                s::wave::mathieu_even_coef(m, 0.0).unwrap()
            };
            let harmonic = if odd && m % 2 == 0 {
                (m / 2 - 1) as usize
            } else {
                (m / 2) as usize
            };
            assert!(coefficients.len() > harmonic);
            let amplitude = if !odd && m == 0 {
                std::f64::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            for (j, &coefficient) in coefficients.iter().enumerate() {
                if j == harmonic && !odd && m == 0 {
                    // The implementation computes 1/sqrt(2), which may differ
                    // by one ULP from the independently rounded FRAC_1_SQRT_2.
                    assert!(
                        coefficient.is_finite()
                            && coefficient.to_bits().abs_diff(amplitude.to_bits()) <= 1,
                        "ce_0 coefficient={coefficient}, expected {amplitude} within one ULP"
                    );
                } else {
                    // Zero Fourier harmonics remain exactly zero; all other
                    // q=0 basis coefficients are exactly one.
                    assert_eq!(
                        coefficient,
                        if j == harmonic { amplitude } else { 0.0 },
                        "m={m}, odd={odd}, coefficient={j}"
                    );
                }
            }

            let radians = 23.0_f64.to_radians();
            let frequency = f64::from(m);
            let (value, derivative) = if odd {
                s::wave::mathieu_sem(m, 0.0, 23.0)
            } else {
                s::wave::mathieu_cem(m, 0.0, 23.0)
            };
            let (expected, expected_derivative) = if odd {
                (
                    amplitude * (frequency * radians).sin(),
                    amplitude * frequency * (frequency * radians).cos(),
                )
            } else {
                (
                    amplitude * (frequency * radians).cos(),
                    -amplitude * frequency * (frequency * radians).sin(),
                )
            };
            finite_close(value, expected, 3e-12, 3e-12);
            finite_close(derivative, expected_derivative, 4e-10, 4e-12);
        }
    }
}

#[test]
fn mathieu_fourier_coefficients_satisfy_eigenvalue_recurrences_and_norms() {
    // These are the four tridiagonal Fourier recurrences in DLMF 28.4.
    // The even-even cosine basis has an exceptional 2*q coupling at row 1;
    // the odd-harmonic recurrences have opposite +/-q diagonal shifts.
    // Testing the recurrence ties the vector-producing const specialization
    // to the independently computed scalar eigenvalue.
    for (m, q) in [
        (0, 1e-8),
        (0, 0.5),
        (1, 5.0),
        (2, 0.5),
        (2, 1.0),
        (3, 5.0),
        (3, 1.0001),
        (4, 20.0),
        (7, 50.0),
        (30, 50.0),
        (128, 100.0),
    ] {
        for odd in [false, true] {
            if odd && m == 0 {
                continue;
            }
            let lambda = if odd {
                s::wave::mathieu_b(m, q)
            } else {
                s::wave::mathieu_a(m, q)
            };
            let coefficients = if odd {
                s::wave::mathieu_odd_coef(m, q).unwrap()
            } else {
                s::wave::mathieu_even_coef(m, q).unwrap()
            };
            // Finite guards are prerequisites, not the numerical test: below
            // we also require weighted unit norm and every Fourier row's
            // eigenvalue residual to be small. In particular, an infinite
            // eigenvalue must not produce an infinite residual tolerance.
            assert!(lambda.is_finite());
            assert!(coefficients.len() > (m / 2) as usize + 2);
            assert!(coefficients.iter().all(|c| c.is_finite()));

            let even_harmonics = m % 2 == 0;
            let norm = coefficients
                .iter()
                .enumerate()
                .map(|(j, c)| {
                    let weight = if !odd && even_harmonics && j == 0 {
                        2.0
                    } else {
                        1.0
                    };
                    weight * c * c
                })
                .sum::<f64>();
            finite_close(norm, 1.0, 3e-12, 3e-12);

            // The last row of a published coefficient vector is padded or
            // truncated, so inspect interior recurrence rows only.
            let first_harmonic = if even_harmonics {
                if odd { 2 } else { 0 }
            } else {
                1
            };
            for j in 0..coefficients.len() - 2 {
                let frequency = (2 * j + first_harmonic) as f64;
                let diagonal = frequency * frequency
                    + if j == 0 && !even_harmonics {
                        if odd { -q } else { q }
                    } else {
                        0.0
                    };
                let left = if j == 0 {
                    0.0
                } else if j == 1 && !odd && even_harmonics {
                    2.0 * q * coefficients[j - 1]
                } else {
                    q * coefficients[j - 1]
                };
                let right = q * coefficients[j + 1];
                let residual = (lambda - diagonal) * coefficients[j] - left - right;
                let tolerance = 8e-13 * (1.0 + lambda.abs() + diagonal.abs() + q);
                assert!(
                    residual.abs() <= tolerance,
                    "m={m}, q={q}, odd={odd}, row={j}: residual={residual:e}, tolerance={tolerance:e}"
                );
            }
        }
    }
}

#[test]
fn mathieu_angular_short_vectors_match_full_fourier_coefficients() {
    // Angular ce/se use vector_margin=16, while the public coefficient APIs
    // use vector_margin=48. Both must describe the same Fourier solution;
    // derivatives are with respect to radians, despite degree inputs.
    for (m, q) in [(0, 0.5), (1, 5.0), (2, 20.0), (5, 0.5), (12, 20.0)] {
        for odd in [false, true] {
            if odd && m == 0 {
                continue;
            }
            let coefficients = if odd {
                s::wave::mathieu_odd_coef(m, q).unwrap()
            } else {
                s::wave::mathieu_even_coef(m, q).unwrap()
            };
            let first_harmonic = if m % 2 == 1 {
                1
            } else if odd {
                2
            } else {
                0
            };
            for degrees in [0.0_f64, 17.0, 65.0] {
                let x = degrees.to_radians();
                let (mut value, mut derivative) = (0.0, 0.0);
                for (j, &coefficient) in coefficients.iter().enumerate() {
                    let frequency = (first_harmonic + 2 * j) as f64;
                    let (sin, cos) = (frequency * x).sin_cos();
                    if odd {
                        value += coefficient * sin;
                        derivative += coefficient * frequency * cos;
                    } else {
                        value += coefficient * cos;
                        derivative -= coefficient * frequency * sin;
                    }
                }
                let (angular, angular_derivative) = if odd {
                    s::wave::mathieu_sem(m, q, degrees)
                } else {
                    s::wave::mathieu_cem(m, q, degrees)
                };
                finite_close(angular, value, 5e-11, 5e-11);
                finite_close(angular_derivative, derivative, 1e-9, 5e-11);
            }
        }
    }
}

#[test]
fn mathieu_characteristic_values_respect_negative_coupling_symmetry() {
    // Translation by pi/2 changes q into -q. Even-order eigenvalues stay
    // in their family; odd-order a_m and b_m exchange families. This probes
    // both signs without sharing the scalar solver's starting estimate.
    for q in [0.1, 2.9, 9.5, 75.0] {
        for m in [0_u32, 2, 4, 12] {
            finite_close(
                s::wave::mathieu_a(m, -q),
                s::wave::mathieu_a(m, q),
                3e-11,
                3e-12,
            );
            if m > 0 {
                finite_close(
                    s::wave::mathieu_b(m, -q),
                    s::wave::mathieu_b(m, q),
                    3e-11,
                    3e-12,
                );
            }
        }
        for m in [1_u32, 3, 5, 13] {
            finite_close(
                s::wave::mathieu_a(m, -q),
                s::wave::mathieu_b(m, q),
                3e-11,
                3e-12,
            );
            finite_close(
                s::wave::mathieu_b(m, -q),
                s::wave::mathieu_a(m, q),
                3e-11,
                3e-12,
            );
        }
    }
}

#[test]
fn cylinder_sequences_match_hermite_polynomials_and_derivatives() {
    for x in [-5.0, -1.5, 0.0, 0.3, 1.5, 5.0] {
        let (values, slopes) = s::wave::pbdn_seq(32, x).unwrap();
        assert_eq!(values.len(), 33);
        assert_eq!(slopes.len(), 33);
        for n in 0..=32 {
            // D_n(x) = exp(-x^2/4) He_n(x), D_n' = n D_(n-1) - x D_n/2.
            let expected = (-0.25 * x * x).exp() * s::orthogonal::eval_hermitenorm(n as i32, x);
            finite_close(values[n], expected, 2e-12, 2e-12);
            let derivative = if n == 0 {
                0.0
            } else {
                n as f64 * values[n - 1]
            } - 0.5 * x * values[n];
            finite_close(slopes[n], derivative, 2e-12, 2e-12);
        }
    }
}

#[test]
fn lambda_sequences_preserve_half_order_identities() {
    for v in [16.5, 32.5, 128.5] {
        for x in [0.1_f64, 0.7, 1.2, 3.0, 7.5, 16.0, 32.0, 50.0] {
            let (values, slopes) = s::wave::lmbda(v, x).unwrap();
            assert_eq!(values.len(), v.floor() as usize + 1);
            assert_eq!(slopes.len(), values.len());
            // Lambda_(1/2) = sin(x)/x from J_(1/2)'s elementary form.
            finite_close(values[0], x.sin() / x, 5e-15, 1e-13);
            finite_close(slopes[0], (x * x.cos() - x.sin()) / (x * x), 5e-15, 1e-13);
        }
    }
}

#[test]
fn lambda_sequences_match_high_precision_series_across_cancellation() {
    // Independent 180-digit sums of the defining series
    // 0F1(; nu+1; -x²/4), differentiated by
    // -x 0F1(; nu+2; -x²/4)/(2(nu+1)). Covers the binary64 shortcut,
    // oscillatory cancellation, a zero of sin(x)/x, and the x=50 cutoff.
    type LambdaCase = (f64, f64, &'static [(usize, f64, f64)]);
    let cases: &[LambdaCase] = &[
        (
            16.5,
            7.5,
            &[
                (0, 0.125_066_663_569_965_18, 0.029_542_487_235_341_416),
                (1, -0.011_816_994_894_136_568, 0.054_753_463_385_640_7),
                (2, -0.036_502_308_923_760_47, 0.016_456_876_019_749_267),
                (7, 0.158_655_986_316_906_53, -0.088_360_928_936_590_45),
                (16, 0.439_525_114_547_865_3, -0.098_649_571_333_913_61),
            ],
        ),
        (
            32.5,
            7.5,
            &[
                (0, 0.125_066_663_569_965_18, 0.029_542_487_235_341_416),
                (1, -0.011_816_994_894_136_568, 0.054_753_463_385_640_7),
                (32, 0.655_490_757_188_659, -0.074_290_370_564_872_5),
            ],
        ),
        (
            16.5,
            std::f64::consts::PI,
            &[
                (0, 3.898_171_832_519_375_5e-17, -std::f64::consts::FRAC_1_PI),
                (1, 0.303_963_550_927_013_3, -0.290_263_809_898_795_4),
                (16, 0.868_023_163_924_265_3, -0.078_516_120_091_902_83),
            ],
        ),
        (
            16.5,
            30.0,
            &[
                (0, -0.032_934_387_469_762_06, 0.006_239_527_911_911_537),
                (
                    1,
                    -0.000_623_952_791_191_153_7,
                    -0.003_231_043_467_857_090_6,
                ),
                (16, -5.241_700_318_778_627e-8, -4.161_122_142_277_8e-7),
            ],
        ),
        (
            16.5,
            49.0,
            &[
                (0, -0.019_464_339_852_234_117, 0.006_531_773_134_609_616),
                (
                    1,
                    -0.000_399_904_477_629_160_2,
                    -0.001_167_210_329_057_446_3,
                ),
                (16, 1.205_148_237_885_124_6e-10, -3.807_267_823_075_033e-11),
            ],
        ),
        (
            16.5,
            50.1,
            &[
                (0, -0.003_287_988_357_826_99, 0.019_753_033_615_703_57),
                (
                    1,
                    -0.001_182_816_384_173_866_5,
                    -0.000_126_058_202_015_157_1,
                ),
                (16, 4.439_996_722_513_546e-11, -8.085_162_756_787_413e-11),
            ],
        ),
    ];
    for &(v, x, reference) in cases {
        let (values, derivatives) = s::wave::lmbda(v, x).unwrap();
        for &(index, value, derivative) in reference {
            finite_close(values[index], value, 2e-13, 5e-13);
            finite_close(derivatives[index], derivative, 2e-13, 5e-13);
        }
    }
}
