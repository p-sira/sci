//! Native scalar and shared-complex contracts, independently referenced.
mod common;
use sci::{Float, complex::Complex, special as s};

fn evaluate<T: Float>(name: &str, args: &[&str]) -> Vec<f64> {
    let real = |i: usize| T::from(args[i].parse::<f64>().unwrap()).unwrap();
    let scalar = match name {
        "log1pmx" => Some(s::log1pmx(real(0))),
        "digammainv" => Some(s::digammainv(real(0))),
        "gamma_ratio" => Some(s::gamma_ratio(real(0), real(1))),
        "scaled_exp1" => Some(s::scaled_exp1(real(0))),
        "iv_ratio" => Some(s::iv_ratio(real(0), real(1))),
        "iv_ratio_c" => Some(s::iv_ratio_c(real(0), real(1))),
        "iv_ratioinv" => Some(s::iv_ratioinv(real(0), real(1))),
        "normalized_gen_harmonic" => Some(
            s::normalized_gen_harmonic(
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
        "complex.gamma" => vec![s::complex::gamma(unary).unwrap()],
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
        assert!(s::log1pmx(x).is_nan());
    }
    assert_eq!(s::log1pmx(-1.0), f64::NEG_INFINITY);
    assert_eq!(s::log1pmx(f64::INFINITY), f64::NEG_INFINITY);
    for zero in [0.0_f64, -0.0] {
        assert!(s::log1pmx(zero).is_sign_negative());
    }
    assert!(s::digammainv(f64::NAN).is_nan());
    assert_eq!(s::digammainv(f64::NEG_INFINITY), 0.0);
    assert_eq!(s::digammainv(f64::INFINITY), f64::INFINITY);
    assert_eq!(s::digammainv(1000.0), f64::INFINITY);
    for (a, b) in [
        (0.0, 1.0),
        (-1.0, 1.0),
        (1.0, 0.0),
        (f64::INFINITY, 1.0),
        (1.0, f64::NAN),
    ] {
        assert!(s::gamma_ratio(a, b).is_nan());
    }
    assert_eq!(s::scaled_exp1(f64::INFINITY), 1.0);
    assert!(s::scaled_exp1(-1.0).is_nan());
    assert!(s::scaled_exp1(f64::NAN).is_nan());
    for (v, x) in [
        (0.49, 1.0),
        (1.0, -1.0),
        (f64::NAN, 1.0),
        (1.0, f64::NAN),
        (f64::INFINITY, f64::INFINITY),
    ] {
        assert!(s::iv_ratio(v, x).is_nan());
        assert!(s::iv_ratio_c(v, x).is_nan());
    }
    assert!(s::iv_ratio(1.0, -0.0_f64).is_sign_negative());
    assert_eq!(s::iv_ratio_c(1.0, 0.0), 1.0);
    assert_eq!(s::iv_ratio(1.0, f64::INFINITY), 1.0);
    assert_eq!(s::iv_ratio_c(1.0, f64::INFINITY), 0.0);
    assert_eq!(s::iv_ratio(f64::INFINITY, 1.0), 0.0);
    assert_eq!(s::iv_ratio_c(f64::INFINITY, 1.0), 1.0);
    assert_eq!(s::iv_ratioinv(1.0, 0.0), 0.0);
    assert_eq!(s::iv_ratioinv(1.0, 1.0), f64::INFINITY);
    for (v, r) in [
        (0.0, 0.5),
        (f64::INFINITY, 0.5),
        (1.0, -0.1),
        (1.0, 1.1),
        (1.0, f64::NAN),
    ] {
        assert!(s::iv_ratioinv(v, r).is_nan());
    }
    for (j, k, n, a) in [
        (0, 1, 1, 1.0),
        (2, 1, 2, 1.0),
        (1, 3, 2, 1.0),
        (1, 1, 1_000_001, 1.0),
        (1, 1, 1, f64::NAN),
    ] {
        assert!(s::normalized_gen_harmonic(j, k, n, a).is_err());
    }
}

#[test]
fn identities_and_shared_types() {
    for v in [0.51, 1.0, 10.0, 1000.0] {
        for x in [1e-6, 0.1, 1.0, 100.0, 1000.0] {
            assert!((s::iv_ratio(v, x) + s::iv_ratio_c(v, x) - 1.0).abs() < 3e-15);
            let r = s::iv_ratio(v, x);
            assert!((s::iv_ratioinv(v, r) / x - 1.0).abs() < 2e-10);
        }
    }
    for a in [-1000.0, -2.0, 0.0, 2.0, 1000.0] {
        let left = s::normalized_gen_harmonic(1, 5, 10, a).unwrap();
        let right = s::normalized_gen_harmonic(6, 10, 10, a).unwrap();
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
                s::jv(0.3, 2.0)
            } else {
                s::iv(0.3, 2.0)
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
    let f = s::normalized_gen_harmonic(1, 1, n, -1.0_f32).unwrap() as f64;
    assert!((f / expected - 1.0).abs() < 3e-6);
    let d = s::normalized_gen_harmonic(1, 1, n, -1.0_f64).unwrap();
    assert!((d / expected - 1.0).abs() < 5e-15);
}
