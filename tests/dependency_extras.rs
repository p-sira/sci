mod common;

use sci::{Error, complex as c, special as s};

#[test]
fn independent_dependency_references() {
    for row in common::rows(include_str!("fixtures/dependency_extras.csv")) {
        let args: Vec<_> = row[1].split(';').collect();
        let expected = common::vector(row[2]);
        let actual = match row[0] {
            "harmonic" => vec![s::harmonic(common::number(args[0]))],
            "gen_harmonic" => {
                vec![s::gen_harmonic(common::number(args[0]), common::number(args[1])).unwrap()]
            }
            "ln_factorial" => vec![s::ln_factorial(common::number(args[0]))],
            "ln_binomial" => {
                vec![s::ln_binomial(common::number(args[0]), common::number(args[1])).unwrap()]
            }
            "ln_multinomial" | "multinomial" => {
                let counts: Vec<u64> = if args[1].is_empty() {
                    vec![]
                } else {
                    args[1].split('/').map(common::number).collect()
                };
                let function = if row[0] == "multinomial" {
                    s::multinomial
                } else {
                    s::ln_multinomial
                };
                vec![function(common::number(args[0]), &counts).unwrap()]
            }
            "cbrt" | "powu" => {
                let z = c::Complex {
                    re: common::number(args[0]),
                    im: common::number(args[1]),
                };
                let value = if row[0] == "cbrt" {
                    c::cbrt(z)
                } else {
                    c::powu(z, common::number(args[2]))
                };
                vec![value.re, value.im]
            }
            name => panic!("unknown fixture {name}"),
        };
        assert_eq!(actual.len(), expected.len(), "{row:?}");
        for (&actual, expected) in actual.iter().zip(expected) {
            // Relative comparisons preserve tiny roots; an absolute tolerance is
            // needed only for cancellation near the axes in complex powers.
            let atol = if row[0] == "powu" { 2e-14 } else { 0.0 };
            assert!(
                approx::abs_diff_eq!(actual, expected, epsilon = atol + 3e-13 * expected.abs()),
                "{row:?}: {actual} != {expected}"
            );
        }
    }
}

#[test]
fn combinatorial_domains_limits_and_overflow() {
    for m in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(s::gen_harmonic(0, m), Err(Error::InvalidInput(_))));
    }
    assert!(s::gen_harmonic(1_000_001, 1.0).is_err());
    assert!(matches!(
        s::gen_harmonic(10, -1000.0),
        Err(Error::NonFiniteEvaluation)
    ));
    assert!(s::ln_binomial(3, 4).is_err());
    assert!(s::ln_binomial(2_000_002, 1_000_001).is_err());
    for (n, counts) in [
        (1, vec![]),
        (2, vec![1]),
        (1, vec![u64::MAX, 2]),
        (u64::MAX, vec![u64::MAX, 1]),
    ] {
        assert!(s::multinomial(n, &counts).is_err());
        assert!(s::ln_multinomial(n, &counts).is_err());
    }
    assert!(s::ln_multinomial(2_000_002, &[1_000_001, 1_000_001]).is_err());
    assert!(s::ln_multinomial(0, &vec![0; 1_000_001]).is_err());
    // The aggregate limit applies across individually permitted factors.
    assert!(s::ln_multinomial(2_000_000, &[500_001, 500_001, 999_998]).is_err());
    assert!(matches!(
        s::multinomial(2000, &[1000, 1000]),
        Err(Error::NonFiniteEvaluation)
    ));
    assert!(s::ln_multinomial(2000, &[1000, 1000]).unwrap().is_finite());
    assert_eq!(s::harmonic(0).to_bits(), 0.0_f64.to_bits());
    assert_eq!(
        s::gen_harmonic(0, 2.0).unwrap().to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(s::multinomial(4, &[2, 1, 1]).unwrap(), 12.0);
}

#[test]
fn identities_recurrences_and_shared_complex_operations() {
    for n in 1..50 {
        assert!((s::harmonic(n) - s::harmonic(n - 1) - 1.0 / n as f64).abs() < 2e-13);
        assert!((s::gen_harmonic(n, 1.0).unwrap() - s::harmonic(n)).abs() < 2e-13);
        for k in 0..=n {
            assert_eq!(
                s::ln_binomial(n, k).unwrap(),
                s::ln_binomial(n, n - k).unwrap()
            );
        }
    }
    let a = s::ln_multinomial(30, &[5, 10, 15]).unwrap();
    let b = s::ln_multinomial(30, &[15, 5, 10]).unwrap();
    assert!((a - b).abs() < 2e-13);
    let z = c::Complex {
        re: 3.0_f64,
        im: -4.0,
    };
    let restored = c::powu(c::cbrt(z), 3);
    assert!((restored.re - z.re).abs() < 2e-14);
    assert!((restored.im - z.im).abs() < 2e-14);
    for n in 0..20 {
        assert_eq!(c::powu(z, n), c::powi(z, n as i32));
    }
    let upper = c::cbrt(c::Complex { re: -8.0, im: 0.0 });
    let lower = c::cbrt(c::Complex { re: -8.0, im: -0.0 });
    assert_eq!(upper, c::conjugate(lower));
    let zero = c::cbrt(c::Complex {
        re: 0.0_f64,
        im: -0.0,
    });
    assert_eq!(zero.im.to_bits(), (-0.0_f64).to_bits());
    assert!(
        c::cbrt(c::Complex {
            re: f64::NAN,
            im: 1.0
        })
        .re
        .is_nan()
    );
    assert_eq!(
        c::cbrt(c::Complex {
            re: f64::INFINITY,
            im: 0.0
        })
        .re,
        f64::INFINITY
    );
    assert_eq!(
        c::powu(
            c::Complex {
                re: f64::NAN,
                im: f64::NAN
            },
            0
        ),
        c::Complex { re: 1.0, im: 0.0 }
    );
}
