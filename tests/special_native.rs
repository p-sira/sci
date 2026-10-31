//! Native scalar and shared-complex contracts, independently referenced.
mod common;
use sci::{Float, complex::Complex, special as s};

fn evaluate<T: Float>(name: &str, args: &[&str]) -> Vec<f64> {
    let real = |i: usize| T::from(args[i].parse::<f64>().unwrap()).unwrap();
    let scalar = match name {
        "log1pmx" => Some(s::elementary::log1pmx(real(0))),
        "digammainv" => Some(s::gamma::digammainv(real(0))),
        "gamma_ratio" => Some(s::gamma::gamma_ratio(real(0), real(1))),
        "scaled_exp1" => Some(s::integrals::scaled_exp1(real(0))),
        "iv_ratio" => Some(s::bessel::iv_ratio(real(0), real(1))),
        "iv_ratio_c" => Some(s::bessel::iv_ratio_c(real(0), real(1))),
        "iv_ratioinv" => Some(s::bessel::iv_ratioinv(real(0), real(1))),
        "normalized_gen_harmonic" => Some(
            s::combinatorics::normalized_gen_harmonic(
                args[0].parse().unwrap(),
                args[1].parse().unwrap(),
                args[2].parse().unwrap(),
                real(3),
            )
            .unwrap(),
        ),
        _ => None,
    };
    if let Some(value) = scalar {
        return vec![value.to_f64().unwrap()];
    }
    let unary = Complex {
        re: real(0),
        im: real(1),
    };
    let values = match name {
        "complex.loggamma" => vec![s::complex::loggamma(unary).unwrap()],
        "complex.rgamma" => vec![s::complex::rgamma(unary).unwrap()],
        "complex.digamma" => vec![s::complex::digamma(unary).unwrap()],
        "complex.gamma" => vec![s::complex::gamma(unary).unwrap()],
        "complex.erfcx" => vec![s::complex::erfcx(unary).unwrap()],
        "complex.erfi" => vec![s::complex::erfi(unary).unwrap()],
        "complex.dawsn" => vec![s::complex::dawsn(unary).unwrap()],
        "complex.erf" => vec![s::complex::erf(unary).unwrap()],
        "complex.erfc" => vec![s::complex::erfc(unary).unwrap()],
        "complex.wofz" => vec![s::complex::wofz(unary).unwrap()],
        "complex.fresnel" => {
            let (a, b) = s::complex::fresnel(unary).unwrap();
            vec![a, b]
        }
        "complex.airy" => s::complex::airy(unary).unwrap().to_vec(),
        "complex.airye" => s::complex::airye(unary).unwrap().to_vec(),
        _ => {
            let z = Complex {
                re: real(1),
                im: real(2),
            };
            vec![
                match name {
                    "complex.jv" => s::complex::jv(real(0), z),
                    "complex.jve" => s::complex::jve(real(0), z),
                    "complex.yv" => s::complex::yv(real(0), z),
                    "complex.yve" => s::complex::yve(real(0), z),
                    "complex.iv" => s::complex::iv(real(0), z),
                    "complex.ive" => s::complex::ive(real(0), z),
                    "complex.kv" => s::complex::kv(real(0), z),
                    "complex.kve" => s::complex::kve(real(0), z),
                    "complex.hankel1" => s::complex::hankel1(real(0), z),
                    "complex.hankel1e" => s::complex::hankel1e(real(0), z),
                    "complex.hankel2" => s::complex::hankel2(real(0), z),
                    "complex.hankel2e" => s::complex::hankel2e(real(0), z),
                    _ => panic!("unknown native API {name}"),
                }
                .unwrap(),
            ]
        }
    };
    values
        .into_iter()
        .flat_map(|z| [z.re.to_f64().unwrap(), z.im.to_f64().unwrap()])
        .collect()
}

#[test]
fn native_apis_match_independent_references() {
    for row in common::rows(include_str!("fixtures/special_native.csv")) {
        let args: Vec<_> = row[2].split(';').collect();
        let expected = common::vector(row[3]);
        let actual = if row[0] == "f32" {
            evaluate::<f32>(row[1], &args)
        } else {
            evaluate::<f64>(row[1], &args)
        };
        assert_eq!(actual.len(), expected.len(), "{row:?}");
        // Componentwise error: small Bessel complements and log remainders use
        // relative comparisons, not a large absolute floor that could hide zero.
        // Digamma inversion references in log x introduce up to ~1e-13 relative
        // error; complex near-zero components inherit binary64 backend rounding.
        let rtol = if row[0] == "f32" { 3e-6 } else { 3e-12 };
        for (got, want) in actual.into_iter().zip(expected) {
            if want.is_infinite() {
                assert_eq!(got, want, "{row:?}");
            } else {
                assert!(
                    (got - want).abs() <= rtol * want.abs().max(1e-300),
                    "{row:?}: got {got:e}, want {want:e}"
                );
            }
        }
    }
}

#[test]
fn scalar_domains_and_limits() {
    for x in [f64::NAN, -2.0, f64::NEG_INFINITY] {
        assert!(s::elementary::log1pmx(x).is_nan());
    }
    assert_eq!(s::elementary::log1pmx(-1.0), f64::NEG_INFINITY);
    assert_eq!(s::elementary::log1pmx(f64::INFINITY), f64::NEG_INFINITY);
    for zero in [0.0_f64, -0.0] {
        assert!(s::elementary::log1pmx(zero).is_sign_negative());
    }
    assert!(s::gamma::digammainv(f64::NAN).is_nan());
    assert_eq!(s::gamma::digammainv(f64::NEG_INFINITY), 0.0);
    assert_eq!(s::gamma::digammainv(f64::INFINITY), f64::INFINITY);
    assert_eq!(s::gamma::digammainv(1000.0), f64::INFINITY);
    for (a, b) in [
        (0.0, 1.0),
        (-1.0, 1.0),
        (1.0, 0.0),
        (f64::INFINITY, 1.0),
        (1.0, f64::NAN),
    ] {
        assert!(s::gamma::gamma_ratio(a, b).is_nan());
    }
    assert_eq!(s::integrals::scaled_exp1(f64::INFINITY), 1.0);
    assert!(s::integrals::scaled_exp1(-1.0).is_nan());
    assert!(s::integrals::scaled_exp1(f64::NAN).is_nan());
    for (v, x) in [
        (0.49, 1.0),
        (1.0, -1.0),
        (f64::NAN, 1.0),
        (1.0, f64::NAN),
        (f64::INFINITY, f64::INFINITY),
    ] {
        assert!(s::bessel::iv_ratio(v, x).is_nan());
        assert!(s::bessel::iv_ratio_c(v, x).is_nan());
    }
    assert!(s::bessel::iv_ratio(1.0, -0.0_f64).is_sign_negative());
    assert_eq!(s::bessel::iv_ratio_c(1.0, 0.0), 1.0);
    assert_eq!(s::bessel::iv_ratio(1.0, f64::INFINITY), 1.0);
    assert_eq!(s::bessel::iv_ratio_c(1.0, f64::INFINITY), 0.0);
    assert_eq!(s::bessel::iv_ratio(f64::INFINITY, 1.0), 0.0);
    assert_eq!(s::bessel::iv_ratio_c(f64::INFINITY, 1.0), 1.0);
    assert_eq!(s::bessel::iv_ratioinv(1.0, 0.0), 0.0);
    assert_eq!(s::bessel::iv_ratioinv(1.0, 1.0), f64::INFINITY);
    for (v, r) in [
        (0.0, 0.5),
        (f64::INFINITY, 0.5),
        (1.0, -0.1),
        (1.0, 1.1),
        (1.0, f64::NAN),
    ] {
        assert!(s::bessel::iv_ratioinv(v, r).is_nan());
    }
    for (j, k, n, a) in [
        (0, 1, 1, 1.0),
        (2, 1, 2, 1.0),
        (1, 3, 2, 1.0),
        (1, 1, 1_000_001, 1.0),
        (1, 1, 1, f64::NAN),
    ] {
        assert!(s::combinatorics::normalized_gen_harmonic(j, k, n, a).is_err());
    }
}

#[test]
fn identities_and_shared_types() {
    for v in [0.51, 1.0, 10.0, 1000.0] {
        for x in [1e-6, 0.1, 1.0, 100.0, 1000.0] {
            assert!((s::bessel::iv_ratio(v, x) + s::bessel::iv_ratio_c(v, x) - 1.0).abs() < 3e-15);
            let r = s::bessel::iv_ratio(v, x);
            assert!((s::bessel::iv_ratioinv(v, r) / x - 1.0).abs() < 2e-10);
        }
    }
    for a in [-1000.0, -2.0, 0.0, 2.0, 1000.0] {
        let left = s::combinatorics::normalized_gen_harmonic(1, 5, 10, a).unwrap();
        let right = s::combinatorics::normalized_gen_harmonic(6, 10, 10, a).unwrap();
        assert!((left + right - 1.0_f64).abs() < 3e-15);
    }
    let z = Complex {
        re: 1.3_f64,
        im: 0.7,
    };
    let conjugate = sci::complex::conjugate(z);
    let gamma = s::complex::gamma(z).unwrap();
    let next = s::complex::gamma(Complex {
        re: z.re + 1.0,
        im: z.im,
    })
    .unwrap();
    let product = sci::complex::multiply(z, gamma);
    assert!((product.re - next.re).abs() < 1e-13 && (product.im - next.im).abs() < 1e-13);
    for f in [s::complex::gamma, s::complex::erf, s::complex::erfc] {
        let a = f(z).unwrap();
        let b = f(conjugate).unwrap();
        assert!((a.re - b.re).abs() < 1e-13 && (a.im + b.im).abs() < 1e-13);
    }
}

#[test]
fn complex_domains_cuts_and_scaling() {
    let nan = Complex {
        re: f64::NAN,
        im: 0.0,
    };
    for f in [
        s::complex::gamma,
        s::complex::erf,
        s::complex::erfc,
        s::complex::wofz,
    ] {
        assert!(f(nan).is_err());
        assert!(
            f(Complex {
                re: 0.0,
                im: f64::INFINITY
            })
            .is_err()
        );
    }
    assert!(s::complex::gamma(Complex { re: -2.0, im: 0.0 }).is_err());
    assert!(
        s::complex::gamma(Complex {
            re: 1000.0,
            im: 1.0
        })
        .is_err()
    );
    assert!(s::complex::fresnel(nan).is_err());
    assert!(s::complex::airy(nan).is_err());
    assert!(s::complex::airye(nan).is_err());
    type Bessel = fn(f64, Complex) -> sci::Result<Complex>;
    let kinds: [Bessel; 12] = [
        s::complex::jv,
        s::complex::jve,
        s::complex::yv,
        s::complex::yve,
        s::complex::iv,
        s::complex::ive,
        s::complex::kv,
        s::complex::kve,
        s::complex::hankel1,
        s::complex::hankel1e,
        s::complex::hankel2,
        s::complex::hankel2e,
    ];
    for (i, f) in kinds.iter().enumerate() {
        assert!(f(f64::NAN, Complex { re: 1.0, im: 0.0 }).is_err());
        assert!(f(1.0, nan).is_err());
        let upper = Complex { re: -2.0, im: 0.0 };
        let lower = Complex { re: -2.0, im: -0.0 };
        let reflected = if i >= 8 {
            kinds[if i < 10 { i + 2 } else { i - 2 }]
        } else {
            *f
        };
        let a = reflected(0.3, upper).unwrap();
        let b = f(0.3, lower).unwrap();
        assert!((a.re - b.re).abs() < 1e-14 && (a.im + b.im).abs() < 1e-14);
        // Independent analytic continuation for I and J at the cut, rather
        // than testing only the wrapper's own reflection construction.
        if i == 0 || i == 4 {
            let magnitude = if i == 0 {
                s::bessel::jv(0.3, 2.0)
            } else {
                s::bessel::iv(0.3, 2.0)
            };
            assert!((b.re - magnitude * (0.3 * std::f64::consts::PI).cos()).abs() < 1e-14);
            assert!((b.im + magnitude * (0.3 * std::f64::consts::PI).sin()).abs() < 1e-14);
        }
    }
    let z = Complex { re: 1.3, im: 0.7 };
    let j = s::complex::jv(0.3, z).unwrap();
    let y = s::complex::yv(0.3, z).unwrap();
    let h = s::complex::hankel1(0.3, z).unwrap();
    assert!((h.re - (j.re - y.im)).abs() < 1e-13 && (h.im - (j.im + y.re)).abs() < 1e-13);
    let k = s::complex::kv(0.3, z).unwrap();
    let ke = s::complex::kve(0.3, z).unwrap();
    let scaled = sci::complex::multiply(sci::complex::exp(z), k);
    assert!((ke.re - scaled.re).abs() < 1e-13 && (ke.im - scaled.im).abs() < 1e-13);
}

#[test]
fn normalized_harmonics_retain_small_probabilities_at_work_limit() {
    // For a=-1, sum i = n(n+1)/2. This also checks the f32 logarithm
    // far from the maximum weight, where ln1p(i/n-1) would lose digits.
    let n = 1_000_000;
    let expected = 2.0_f64 / (n as f64 * (n + 1) as f64);
    let f = s::combinatorics::normalized_gen_harmonic(1, 1, n, -1.0_f32).unwrap() as f64;
    assert!((f / expected - 1.0).abs() < 3e-6);
    let d = s::combinatorics::normalized_gen_harmonic(1, 1, n, -1.0_f64).unwrap();
    assert!((d / expected - 1.0).abs() < 5e-15);
}

#[test]
fn complex_gamma_family_contracts() {
    use s::complex::{digamma, gamma, loggamma, rgamma};
    let close = |a: Complex, b: Complex| {
        let error = (a.re - b.re).hypot(a.im - b.im);
        assert!(error <= 2e-12 * b.re.hypot(b.im).max(1.0), "{a:?} != {b:?}");
    };
    for z in [
        Complex { re: 0.3, im: 0.7 },
        Complex { re: -4.7, im: 0.3 },
        Complex {
            re: -20.1,
            im: -7.1,
        },
    ] {
        let shifted = Complex {
            re: z.re + 1.0,
            im: z.im,
        };
        let inverse = Complex {
            re: z.re / (z.re * z.re + z.im * z.im),
            im: -z.im / (z.re * z.re + z.im * z.im),
        };
        close(
            digamma(shifted).unwrap(),
            sci::complex::add(digamma(z).unwrap(), inverse),
        );
        let log_z = Complex {
            re: z.re.hypot(z.im).ln(),
            im: z.im.atan2(z.re),
        };
        close(
            loggamma(shifted).unwrap(),
            sci::complex::add(loggamma(z).unwrap(), log_z),
        );
        close(
            sci::complex::multiply(gamma(z).unwrap(), rgamma(z).unwrap()),
            Complex { re: 1.0, im: 0.0 },
        );
        for f in [loggamma, rgamma, digamma] {
            let a = f(z).unwrap();
            close(
                f(Complex {
                    re: z.re,
                    im: -z.im,
                })
                .unwrap(),
                Complex {
                    re: a.re,
                    im: -a.im,
                },
            );
        }
    }
    for x in [0.0, -0.0, -1.0, -20.0] {
        for im in [0.0, -0.0] {
            let z = Complex { re: x, im };
            assert!(loggamma(z).is_err());
            assert!(digamma(z).is_err());
            assert_eq!(rgamma(z).unwrap(), Complex { re: 0.0, im: 0.0 });
        }
    }
    for f in [loggamma, rgamma, digamma] {
        for z in [
            Complex {
                re: f64::NAN,
                im: 0.0,
            },
            Complex {
                re: 1.0,
                im: f64::INFINITY,
            },
        ] {
            assert!(matches!(f(z), Err(sci::Error::InvalidInput(_))));
        }
    }
    let upper = loggamma(Complex {
        re: -2.25_f64,
        im: 0.0,
    })
    .unwrap();
    let lower = loggamma(Complex {
        re: -2.25_f64,
        im: -0.0,
    })
    .unwrap();
    assert!((upper.im + 3.0 * std::f64::consts::PI).abs() < 1e-14);
    close(
        lower,
        Complex {
            re: upper.re,
            im: -upper.im,
        },
    );
    assert!(upper.im.abs() > std::f64::consts::PI); // Unwrapped log-gamma phase.

    let z = Complex {
        re: 172.0_f64,
        im: 0.0,
    };
    assert!(gamma(z).is_err());
    assert!(loggamma(z).unwrap().re.is_finite());
    assert!(rgamma(z).unwrap().re > 0.0); // Subnormal reciprocal survives gamma overflow.
    assert_eq!(
        rgamma(Complex {
            re: 1000.0_f64,
            im: 0.0
        })
        .unwrap()
        .re,
        0.0
    );
    assert!(
        rgamma(Complex {
            re: -170.5_f32,
            im: 0.5
        })
        .is_err()
    );

    // A scaled cotangent must retain poles approached closer than sqrt(MIN_POSITIVE).
    for y in [1e-20_f64, 1e-200, 1e-300] {
        let value = digamma(Complex { re: -100.0, im: y }).unwrap();
        assert!((value.im * y - 1.0).abs() < 2e-15);
    }
    let huge = digamma(Complex {
        re: -1e200_f64,
        im: 1e200,
    })
    .unwrap();
    close(
        huge,
        Complex {
            re: 200.0 * 10.0_f64.ln() + 0.5 * 2.0_f64.ln(),
            im: 0.75 * std::f64::consts::PI,
        },
    );
}

#[test]
fn complex_error_family_identities_and_domains() {
    use s::complex::{dawsn, erfcx, erfi};
    use sci::complex::{conjugate, exp, multiply};
    let close = |a: Complex, b: Complex| {
        for (a, b) in [(a.re, b.re), (a.im, b.im)] {
            assert!(
                (a - b).abs() <= 3e-13 * b.abs().max(1e-300),
                "{a:e} vs {b:e}"
            );
        }
    };
    for z in [
        Complex { re: 0.1, im: 0.2 },
        Complex { re: 1.3, im: -0.7 },
        Complex { re: -2.0, im: 1.0 },
    ] {
        for f in [erfcx, erfi, dawsn] {
            close(f(conjugate(z)).unwrap(), conjugate(f(z).unwrap()));
        }
        let iz = Complex {
            re: -z.im,
            im: z.re,
        };
        let z2 = multiply(z, z);
        close(erfcx(z).unwrap(), s::complex::wofz(iz).unwrap());
        close(
            erfcx(z).unwrap(),
            multiply(exp(z2), s::complex::erfc(z).unwrap()),
        );
        let erf_iz = s::complex::erf(iz).unwrap();
        close(
            erfi(z).unwrap(),
            Complex {
                re: erf_iz.im,
                im: -erf_iz.re,
            },
        );
        let scaled = multiply(
            exp(Complex {
                re: -z2.re,
                im: -z2.im,
            }),
            erfi(z).unwrap(),
        );
        let factor = std::f64::consts::PI.sqrt() / 2.0;
        close(
            dawsn(z).unwrap(),
            Complex {
                re: factor * scaled.re,
                im: factor * scaled.im,
            },
        );
        let negative = Complex {
            re: -z.re,
            im: -z.im,
        };
        for f in [erfi, dawsn] {
            let value = f(z).unwrap();
            close(
                f(negative).unwrap(),
                Complex {
                    re: -value.re,
                    im: -value.im,
                },
            );
        }
    }
    for f in [erfcx, erfi, dawsn] {
        for z in [
            Complex {
                re: f64::NAN,
                im: 0.0,
            },
            Complex {
                re: 0.0,
                im: f64::NAN,
            },
            Complex {
                re: f64::INFINITY,
                im: 0.0,
            },
            Complex {
                re: 0.0,
                im: f64::NEG_INFINITY,
            },
        ] {
            assert!(matches!(f(z), Err(sci::Error::InvalidInput(_))));
        }
    }
    // A finite binary64 backend result may overflow the requested precision.
    for (f, z) in [
        (erfcx::<f32> as fn(_) -> _, Complex { re: -10.0, im: 0.0 }),
        (erfi::<f32>, Complex { re: 10.0, im: 0.0 }),
        (dawsn::<f32>, Complex { re: 0.0, im: 10.0 }),
    ] {
        assert!(matches!(f(z), Err(sci::Error::NonFiniteEvaluation)));
    }
    for (f, z) in [
        (erfcx::<f64> as fn(_) -> _, Complex { re: -30.0, im: 0.0 }),
        (erfi::<f64>, Complex { re: 30.0, im: 0.0 }),
        (dawsn::<f64>, Complex { re: 0.0, im: 30.0 }),
    ] {
        assert!(f(z).is_err());
    }
    // Scaling and dedicated real-axis paths retain values after naive products overflow.
    for x in [30.0_f64, 100.0, 1e30] {
        let z = Complex { re: x, im: 0.0 };
        let scaled = erfcx(z).unwrap();
        let dawson = dawsn(z).unwrap();
        // The separate real erfcx path uses a rational approximation at these inputs.
        assert!(
            (scaled.re / s::elementary::erfcx(x) - 1.0).abs() < 2e-14,
            "erfcx({x}): complex={}, real={}",
            scaled.re,
            s::elementary::erfcx(x)
        );
        assert!((dawson.re / s::elementary::dawsn(x) - 1.0).abs() < 2e-15);
    }
    // Erfi agrees with the real facade and preserves the input imaginary zero.
    for y in [0.0_f64, -0.0] {
        let value = erfi(Complex { re: 1.0, im: y }).unwrap();
        assert_eq!(value.im.to_bits(), y.to_bits());
        assert!((value.re / s::elementary::erfi(1.0) - 1.0).abs() < 2e-15);
    }
    for (f, slope) in [
        (erfi::<f64> as fn(_) -> _, 2.0 / std::f64::consts::PI.sqrt()),
        (dawsn::<f64>, 1.0),
    ] {
        let z = Complex {
            re: 1e-200,
            im: -2e-200,
        };
        let value = f(z).unwrap();
        close(
            value,
            Complex {
                re: slope * z.re,
                im: slope * z.im,
            },
        );
    }
}
