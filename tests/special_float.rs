//! Both public scalar precisions, compared with independent SciPy fixtures.
mod common;
use sci::{Float, special as s};

fn evaluate<T: Float>(name: &str, args: &[&str]) -> T {
    let real = |i: usize| T::from(args[i].parse::<f64>().unwrap()).unwrap();
    match name {
        "agm" => s::agm(real(0), real(1)),
        "bdtr" => s::bdtr(real(0), args[1].parse::<i32>().unwrap(), real(2)),
        "bdtrc" => s::bdtrc(real(0), args[1].parse::<i32>().unwrap(), real(2)),
        "bdtri" => s::bdtri(real(0), args[1].parse::<i32>().unwrap(), real(2)),
        "bei" => s::bei(real(0)),
        "beip" => s::beip(real(0)),
        "ber" => s::ber(real(0)),
        "berp" => s::berp(real(0)),
        "betainc" => s::betainc(real(0), real(1), real(2)),
        "betaincinv" => s::betaincinv(real(0), real(1), real(2)),
        "binom" => s::binom(real(0), real(1)),
        "boxcox" => s::boxcox(real(0), real(1)),
        "boxcox1p" => s::boxcox1p(real(0), real(1)),
        "chdtr" => s::chdtr(real(0), real(1)),
        "chdtrc" => s::chdtrc(real(0), real(1)),
        "chdtri" => s::chdtri(real(0), real(1)),
        "comb" => s::comb(real(0), real(1)),
        "dawsn" => s::dawsn(real(0)),
        "digamma" => s::digamma(real(0)),
        "erf" => s::erf(real(0)),
        "erfc" => s::erfc(real(0)),
        "erfcinv" => s::erfcinv(real(0)),
        "erfcx" => s::erfcx(real(0)),
        "erfi" => s::erfi(real(0)),
        "erfinv" => s::erfinv(real(0)),
        "eval_chebyt" => s::eval_chebyt(args[0].parse::<i32>().unwrap(), real(1)),
        "eval_chebyu" => s::eval_chebyu(args[0].parse::<i32>().unwrap(), real(1)),
        "eval_gegenbauer" => s::eval_gegenbauer(args[0].parse::<i32>().unwrap(), real(1), real(2)),
        "eval_genlaguerre" => {
            s::eval_genlaguerre(args[0].parse::<i32>().unwrap(), real(1), real(2))
        }
        "eval_hermite" => s::eval_hermite(args[0].parse::<i32>().unwrap(), real(1)),
        "eval_hermitenorm" => s::eval_hermitenorm(args[0].parse::<i32>().unwrap(), real(1)),
        "eval_jacobi" => s::eval_jacobi(args[0].parse::<i32>().unwrap(), real(1), real(2), real(3)),
        "eval_laguerre" => s::eval_laguerre(args[0].parse::<i32>().unwrap(), real(1)),
        "eval_legendre" => s::eval_legendre(args[0].parse::<i32>().unwrap(), real(1)),
        "exp1" => s::exp1(real(0)),
        "expi" => s::expi(real(0)),
        "expn" => s::expn(args[0].parse::<u32>().unwrap(), real(1)),
        "fdtr" => s::fdtr(real(0), real(1), real(2)),
        "fdtrc" => s::fdtrc(real(0), real(1), real(2)),
        "fdtri" => s::fdtri(real(0), real(1), real(2)),
        "gamma" => s::gamma(real(0)),
        "gammainc" => s::gammainc(real(0), real(1)),
        "gammaincc" => s::gammaincc(real(0), real(1)),
        "gammainccinv" => s::gammainccinv(real(0), real(1)),
        "gammaincinv" => s::gammaincinv(real(0), real(1)),
        "gammaln" => s::gammaln(real(0)),
        "gammasgn" => s::gammasgn(real(0)),
        "gdtr" => s::gdtr(real(0), real(1), real(2)),
        "gdtrc" => s::gdtrc(real(0), real(1), real(2)),
        "hyp0f1" => s::hyp0f1(real(0), real(1)),
        "i0" => s::i0(real(0)),
        "i0e" => s::i0e(real(0)),
        "i1" => s::i1(real(0)),
        "i1e" => s::i1e(real(0)),
        "iv" => s::iv(real(0), real(1)),
        "ive" => s::ive(real(0), real(1)),
        "j0" => s::j0(real(0)),
        "j1" => s::j1(real(0)),
        "jv" => s::jv(real(0), real(1)),
        "jve" => s::jve(real(0), real(1)),
        "k0" => s::k0(real(0)),
        "k0e" => s::k0e(real(0)),
        "k1" => s::k1(real(0)),
        "k1e" => s::k1e(real(0)),
        "kei" => s::kei(real(0)),
        "keip" => s::keip(real(0)),
        "ker" => s::ker(real(0)),
        "kerp" => s::kerp(real(0)),
        "kolmogi" => s::kolmogi(real(0)),
        "kolmogorov" => s::kolmogorov(real(0)),
        "kv" => s::kv(real(0), real(1)),
        "kve" => s::kve(real(0), real(1)),
        "lambertw" => s::lambertw(real(0)),
        "lambertwm1" => s::lambertwm1(real(0)),
        "legendre_p" => s::legendre_p(args[0].parse::<i32>().unwrap(), real(1)),
        "log_ndtr" => s::log_ndtr(real(0)),
        "loggamma" => s::loggamma(real(0)),
        "modstruve" => s::modstruve(real(0), real(1)),
        "nbdtr" => s::nbdtr(
            args[0].parse::<i32>().unwrap(),
            args[1].parse::<i32>().unwrap(),
            real(2),
        ),
        "nbdtrc" => s::nbdtrc(
            args[0].parse::<i32>().unwrap(),
            args[1].parse::<i32>().unwrap(),
            real(2),
        ),
        "nbdtri" => s::nbdtri(
            args[0].parse::<i32>().unwrap(),
            args[1].parse::<i32>().unwrap(),
            real(2),
        ),
        "ndtr" => s::ndtr(real(0)),
        "ndtri" => s::ndtri(real(0)),
        "ndtri_exp" => s::ndtri_exp(real(0)),
        "pdtr" => s::pdtr(real(0), real(1)),
        "pdtrc" => s::pdtrc(real(0), real(1)),
        "pdtri" => s::pdtri(args[0].parse::<i32>().unwrap(), real(1)),
        "poch" => s::poch(real(0), real(1)),
        "rgamma" => s::rgamma(real(0)),
        "spence" => s::spence(real(0)),
        "spherical_in" => s::spherical_in(
            args[0].parse::<i64>().unwrap(),
            real(1),
            args[2].parse::<bool>().unwrap(),
        ),
        "spherical_jn" => s::spherical_jn(
            args[0].parse::<i64>().unwrap(),
            real(1),
            args[2].parse::<bool>().unwrap(),
        ),
        "spherical_kn" => s::spherical_kn(
            args[0].parse::<i64>().unwrap(),
            real(1),
            args[2].parse::<bool>().unwrap(),
        ),
        "spherical_yn" => s::spherical_yn(
            args[0].parse::<i64>().unwrap(),
            real(1),
            args[2].parse::<bool>().unwrap(),
        ),
        "struve" => s::struve(real(0), real(1)),
        "tklmbda" => s::tklmbda(real(0), real(1)),
        "voigt_profile" => s::voigt_profile(real(0), real(1), real(2)),
        "wrightomega" => s::wrightomega(real(0)),
        "y0" => s::y0(real(0)),
        "y1" => s::y1(real(0)),
        "yv" => s::yv(real(0), real(1)),
        "yve" => s::yve(real(0), real(1)),
        "zeta" => s::zeta(real(0), real(1)),
        "zetac" => s::zetac(real(0)),
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
    assert!(s::jv(inf, one).is_nan());
    assert!(s::iv(inf, one).is_nan());
    assert!(s::struve(inf, one).is_nan());
    assert!(s::modstruve(inf, one).is_nan());
    assert!(s::hyp0f1(inf, one).is_nan());
    assert!(s::betainc(inf, one, one).is_nan());
    assert!(s::gammainc(inf, one).is_nan());
    assert!(s::comb(nan, one).is_nan());
    assert!(s::lambertwm1(inf).is_nan());
    assert!(s::lambertwm1(-inf).is_nan());
    assert!(s::fdtrc(inf, one, one).is_nan());
    assert_eq!(s::i0(inf), inf);
    assert_eq!(s::i0(-inf), inf);
    assert_eq!(s::i1(inf), inf);
    assert_eq!(s::i1(-inf), -inf);
    assert_eq!(s::i0e(-inf), zero);
    assert!(!s::i0e(-inf).is_sign_negative());
    for n in 0..=3 {
        let expected = if n == 0 {
            one
        } else if n % 2 == 0 {
            inf
        } else {
            -inf
        };
        assert_eq!(s::eval_legendre(n, -inf), expected);
        assert_eq!(s::eval_chebyt(n, -inf), expected);
        assert_eq!(s::eval_chebyu(n, -inf), expected);
        assert_eq!(s::eval_hermite(n, -inf), expected);
        assert_eq!(s::eval_hermitenorm(n, -inf), expected);
        assert_eq!(s::eval_laguerre(n, inf), expected);
    }
}

#[test]
fn nonfinite_values_follow_sci_domains_and_limits() {
    nonfinite_contracts::<f32>();
    nonfinite_contracts::<f64>();
}
