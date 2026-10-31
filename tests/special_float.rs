//! Both public scalar precisions, compared with independent SciPy fixtures.
mod common;
use sci::{Float, special as s};

fn evaluate<T: Float>(name: &str, args: &[&str]) -> T {
    let real = |i: usize| T::from(args[i].parse::<f64>().unwrap()).unwrap();
    match name {
        "agm" => s::elementary::agm(real(0), real(1)),
        "bdtr" => s::probability::bdtr(real(0), args[1].parse::<i32>().unwrap(), real(2)),
        "bdtrc" => s::probability::bdtrc(real(0), args[1].parse::<i32>().unwrap(), real(2)),
        "bdtri" => s::probability::bdtri(real(0), args[1].parse::<i32>().unwrap(), real(2)),
        "bei" => s::bessel::bei(real(0)),
        "beip" => s::bessel::beip(real(0)),
        "ber" => s::bessel::ber(real(0)),
        "berp" => s::bessel::berp(real(0)),
        "betainc" => s::beta::betainc(real(0), real(1), real(2)),
        "betaincinv" => s::beta::betaincinv(real(0), real(1), real(2)),
        "binom" => s::combinatorics::binom(real(0), real(1)),
        "boxcox" => s::probability::boxcox(real(0), real(1)),
        "boxcox1p" => s::probability::boxcox1p(real(0), real(1)),
        "chdtr" => s::probability::chdtr(real(0), real(1)),
        "chdtrc" => s::probability::chdtrc(real(0), real(1)),
        "chdtri" => s::probability::chdtri(real(0), real(1)),
        "comb" => s::combinatorics::comb(real(0), real(1)),
        "dawsn" => s::elementary::dawsn(real(0)),
        "digamma" => s::gamma::digamma(real(0)),
        "erf" => s::elementary::erf(real(0)),
        "erfc" => s::elementary::erfc(real(0)),
        "erfcinv" => s::elementary::erfcinv(real(0)),
        "erfcx" => s::elementary::erfcx(real(0)),
        "erfi" => s::elementary::erfi(real(0)),
        "erfinv" => s::elementary::erfinv(real(0)),
        "eval_chebyt" => s::orthogonal::eval_chebyt(args[0].parse::<i32>().unwrap(), real(1)),
        "eval_chebyu" => s::orthogonal::eval_chebyu(args[0].parse::<i32>().unwrap(), real(1)),
        "eval_gegenbauer" => {
            s::orthogonal::eval_gegenbauer(args[0].parse::<i32>().unwrap(), real(1), real(2))
        }
        "eval_genlaguerre" => {
            s::orthogonal::eval_genlaguerre(args[0].parse::<i32>().unwrap(), real(1), real(2))
        }
        "eval_hermite" => s::orthogonal::eval_hermite(args[0].parse::<i32>().unwrap(), real(1)),
        "eval_hermitenorm" => {
            s::orthogonal::eval_hermitenorm(args[0].parse::<i32>().unwrap(), real(1))
        }
        "eval_jacobi" => {
            s::orthogonal::eval_jacobi(args[0].parse::<i32>().unwrap(), real(1), real(2), real(3))
        }
        "eval_laguerre" => s::orthogonal::eval_laguerre(args[0].parse::<i32>().unwrap(), real(1)),
        "eval_legendre" => s::orthogonal::eval_legendre(args[0].parse::<i32>().unwrap(), real(1)),
        "exp1" => s::integrals::exp1(real(0)),
        "expi" => s::integrals::expi(real(0)),
        "expn" => s::integrals::expn(args[0].parse::<u32>().unwrap(), real(1)),
        "fdtr" => s::probability::fdtr(real(0), real(1), real(2)),
        "fdtrc" => s::probability::fdtrc(real(0), real(1), real(2)),
        "fdtri" => s::probability::fdtri(real(0), real(1), real(2)),
        "gamma" => s::gamma::gamma(real(0)),
        "gammainc" => s::gamma::gammainc(real(0), real(1)),
        "gammaincc" => s::gamma::gammaincc(real(0), real(1)),
        "gammainccinv" => s::gamma::gammainccinv(real(0), real(1)),
        "gammaincinv" => s::gamma::gammaincinv(real(0), real(1)),
        "gammaln" => s::gamma::gammaln(real(0)),
        "gammasgn" => s::gamma::gammasgn(real(0)),
        "gdtr" => s::probability::gdtr(real(0), real(1), real(2)),
        "gdtrc" => s::probability::gdtrc(real(0), real(1), real(2)),
        "hyp0f1" => s::hypergeometric::hyp0f1(real(0), real(1)),
        "i0" => s::bessel::i0(real(0)),
        "i0e" => s::bessel::i0e(real(0)),
        "i1" => s::bessel::i1(real(0)),
        "i1e" => s::bessel::i1e(real(0)),
        "iv" => s::bessel::iv(real(0), real(1)),
        "ive" => s::bessel::ive(real(0), real(1)),
        "j0" => s::bessel::j0(real(0)),
        "j1" => s::bessel::j1(real(0)),
        "jv" => s::bessel::jv(real(0), real(1)),
        "jve" => s::bessel::jve(real(0), real(1)),
        "k0" => s::bessel::k0(real(0)),
        "k0e" => s::bessel::k0e(real(0)),
        "k1" => s::bessel::k1(real(0)),
        "k1e" => s::bessel::k1e(real(0)),
        "kei" => s::bessel::kei(real(0)),
        "keip" => s::bessel::keip(real(0)),
        "ker" => s::bessel::ker(real(0)),
        "kerp" => s::bessel::kerp(real(0)),
        "kolmogi" => s::probability::kolmogi(real(0)),
        "kolmogorov" => s::probability::kolmogorov(real(0)),
        "kv" => s::bessel::kv(real(0), real(1)),
        "kve" => s::bessel::kve(real(0), real(1)),
        "lambertw" => s::elementary::lambertw(real(0)),
        "lambertwm1" => s::elementary::lambertwm1(real(0)),
        "legendre_p" => s::orthogonal::legendre_p(args[0].parse::<i32>().unwrap(), real(1)),
        "log_ndtr" => s::probability::log_ndtr(real(0)),
        "loggamma" => s::gamma::loggamma(real(0)),
        "modstruve" => s::bessel::modstruve(real(0), real(1)),
        "nbdtr" => s::probability::nbdtr(
            args[0].parse::<i32>().unwrap(),
            args[1].parse::<i32>().unwrap(),
            real(2),
        ),
        "nbdtrc" => s::probability::nbdtrc(
            args[0].parse::<i32>().unwrap(),
            args[1].parse::<i32>().unwrap(),
            real(2),
        ),
        "nbdtri" => s::probability::nbdtri(
            args[0].parse::<i32>().unwrap(),
            args[1].parse::<i32>().unwrap(),
            real(2),
        ),
        "ndtr" => s::probability::ndtr(real(0)),
        "ndtri" => s::probability::ndtri(real(0)),
        "ndtri_exp" => s::probability::ndtri_exp(real(0)),
        "pdtr" => s::probability::pdtr(real(0), real(1)),
        "pdtrc" => s::probability::pdtrc(real(0), real(1)),
        "pdtri" => s::probability::pdtri(args[0].parse::<i32>().unwrap(), real(1)),
        "poch" => s::combinatorics::poch(real(0), real(1)),
        "rgamma" => s::gamma::rgamma(real(0)),
        "spence" => s::integrals::spence(real(0)),
        "spherical_in" => s::bessel::spherical_in(
            args[0].parse::<i64>().unwrap(),
            real(1),
            args[2].parse::<bool>().unwrap(),
        ),
        "spherical_jn" => s::bessel::spherical_jn(
            args[0].parse::<i64>().unwrap(),
            real(1),
            args[2].parse::<bool>().unwrap(),
        ),
        "spherical_kn" => s::bessel::spherical_kn(
            args[0].parse::<i64>().unwrap(),
            real(1),
            args[2].parse::<bool>().unwrap(),
        ),
        "spherical_yn" => s::bessel::spherical_yn(
            args[0].parse::<i64>().unwrap(),
            real(1),
            args[2].parse::<bool>().unwrap(),
        ),
        "struve" => s::bessel::struve(real(0), real(1)),
        "tklmbda" => s::probability::tklmbda(real(0), real(1)),
        "voigt_profile" => s::probability::voigt_profile(real(0), real(1), real(2)),
        "wrightomega" => s::elementary::wrightomega(real(0)),
        "y0" => s::bessel::y0(real(0)),
        "y1" => s::bessel::y1(real(0)),
        "yv" => s::bessel::yv(real(0), real(1)),
        "yve" => s::bessel::yve(real(0), real(1)),
        "zeta" => s::zeta::zeta(real(0), real(1)),
        "zetac" => s::zeta::zetac(real(0)),
        _ => panic!("unknown generic special function {name}"),
    }
}

fn verify<T: Float>(precision: &str, rtol: f64, atol: f64) {
    let mut covered = std::collections::BTreeSet::new();
    let mut failures = Vec::new();
    for row in common::lines(include_str!("fixtures/special_float.csv")) {
        let fields: Vec<_> = common::fields(row);
        if fields[0] != precision {
            continue;
        }
        let args: Vec<_> = fields[2].split(';').collect();
        let actual = evaluate::<T>(fields[1], &args).to_f64().unwrap();
        let expected: f64 = fields[3].parse().unwrap();
        covered.insert(fields[1]);
        let correct = if expected.is_nan() {
            actual.is_nan()
        } else if expected.is_infinite() {
            actual == expected
        } else {
            (actual - expected).abs() <= atol + rtol * expected.abs()
        };
        if !correct {
            failures.push(format!("{precision} {row}: {actual}"));
        }
        if correct
            && expected == 0.0
            && actual == 0.0
            && actual.is_sign_negative() != expected.is_sign_negative()
        {
            failures.push(format!("{precision} {row}: signed zero {actual}"));
        }
    }
    assert_eq!(covered.len(), 104);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn scalar_f64_matches_scipy() {
    verify::<f64>("f64", 2e-11, 2e-12);
}

#[test]
fn scalar_f32_matches_scipy() {
    // Inputs and SciPy results are rounded to f32 by the reference generator.
    // This covers input quantization separately from the binary64 kernels.
    verify::<f32>("f32", 4e-6, 4e-7);
}

fn nonfinite_contracts<T: Float + std::fmt::Debug>() {
    let inf = T::infinity();
    let nan = T::nan();
    let one = T::one();
    let zero = T::zero();
    assert!(s::bessel::jv(inf, one).is_nan());
    assert!(s::bessel::iv(inf, one).is_nan());
    assert!(s::bessel::struve(inf, one).is_nan());
    assert!(s::bessel::modstruve(inf, one).is_nan());
    assert!(s::hypergeometric::hyp0f1(inf, one).is_nan());
    assert!(s::beta::betainc(inf, one, one).is_nan());
    assert!(s::gamma::gammainc(inf, one).is_nan());
    assert!(s::combinatorics::comb(nan, one).is_nan());
    assert!(s::elementary::lambertwm1(inf).is_nan());
    assert!(s::elementary::lambertwm1(-inf).is_nan());
    assert!(s::probability::fdtrc(inf, one, one).is_nan());
    assert_eq!(s::bessel::i0(inf), inf);
    assert_eq!(s::bessel::i0(-inf), inf);
    assert_eq!(s::bessel::i1(inf), inf);
    assert_eq!(s::bessel::i1(-inf), -inf);
    assert_eq!(s::bessel::i0e(-inf), zero);
    assert!(!s::bessel::i0e(-inf).is_sign_negative());
    for n in 0..=3 {
        let expected = if n == 0 {
            one
        } else if n % 2 == 0 {
            inf
        } else {
            -inf
        };
        assert_eq!(s::orthogonal::eval_legendre(n, -inf), expected);
        assert_eq!(s::orthogonal::eval_chebyt(n, -inf), expected);
        assert_eq!(s::orthogonal::eval_chebyu(n, -inf), expected);
        assert_eq!(s::orthogonal::eval_hermite(n, -inf), expected);
        assert_eq!(s::orthogonal::eval_hermitenorm(n, -inf), expected);
        assert_eq!(s::orthogonal::eval_laguerre(n, inf), expected);
    }
}

#[test]
fn nonfinite_values_follow_sci_domains_and_limits() {
    nonfinite_contracts::<f32>();
    nonfinite_contracts::<f64>();
}
