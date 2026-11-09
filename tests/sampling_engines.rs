mod common;

use common::{fields, lines, number, vector};
use sci::stats::{
    distributions::{Distribution, cdf, pdf},
    sampling::{
        DiscreteAliasUrn, DiscreteGuideTable, FastGeneratorInversion, NumericalInverseHermite,
        NumericalInversePolynomial, RatioUniforms, SimpleRatioUniforms,
        TransformedDensityRejection,
    },
};

#[test]
fn public_inverse_engines_match_independent_scipy_quantile_fixtures() {
    let cases: Vec<_> = lines(include_str!("scipy/sampling_quantile_cases.csv")).collect();
    let reference: Vec<_> = lines(include_str!("scipy/sampling_quantile_expected.csv")).collect();
    assert_eq!(cases.len(), 4);
    assert_eq!(cases.len(), reference.len());
    for (case, expected) in cases.iter().zip(reference) {
        let c = fields(case);
        let e = fields(expected);
        assert_eq!(c.len(), 4);
        assert_eq!(e.len(), 2);
        assert_eq!(c[0], e[0]);
        let distribution = match c[0] {
            "uniform" => Distribution::Uniform {
                low: -2.0,
                high: 2.0,
            },
            "exponential" => Distribution::Exponential { scale: 1.0 },
            "normal" => Distribution::Normal {
                mean: 0.0,
                std_dev: 1.0,
            },
            "beta" => Distribution::Beta { a: 2.0, b: 5.0 },
            other => panic!("unexpected sampling fixture family {other}"),
        };
        let (lower, upper) = (number::<f64>(c[1]), number::<f64>(c[2]));
        let probabilities: Vec<f64> = vector(c[3]);
        let references: Vec<f64> = vector(e[1]);
        assert_eq!(probabilities.len(), references.len());
        let lower_mass = cdf(distribution, lower).unwrap();
        let mass = cdf(distribution, upper).unwrap() - lower_mass;
        let exact = FastGeneratorInversion::new(distribution, 37).unwrap();
        let cubic =
            NumericalInverseHermite::new(distribution, (lower, upper), 1e-4, 1024, 37).unwrap();
        let linear =
            NumericalInversePolynomial::new(distribution, (lower, upper), 1e-4, 1024, 37).unwrap();
        for (&p, &value) in probabilities.iter().zip(&references) {
            let native = exact.ppf(lower_mass + p * mass).unwrap();
            assert!((native - value).abs() < 2e-7, "{}: native ppf({p})", c[0]);
            for actual in [cubic.ppf(p).unwrap(), linear.ppf(p).unwrap()] {
                let conditional_density = pdf(distribution, value).unwrap() / mass;
                let x_tolerance = 3e-4 / conditional_density.max(1e-8) + 1e-6;
                assert!((actual - value).abs() < x_tolerance, "{}: ppf({p})", c[0]);
                let cdf_actual = (cdf(distribution, actual).unwrap() - lower_mass) / mass;
                assert!((cdf_actual - p).abs() < 3e-4, "{}: CDF(PPF({p}))", c[0]);
            }
        }
    }
}

#[test]
fn public_discrete_samplers_preserve_mass_and_inverse_cdf() {
    let weights = [1.0, 3.0, 6.0];
    let mut alias = DiscreteAliasUrn::new(&weights, 147).unwrap();
    let mut guide = DiscreteGuideTable::new(&weights, 147).unwrap();
    assert_eq!(
        guide.ppf(&[0.0, 0.05, 0.35, 0.75, 1.0]).unwrap(),
        [0, 0, 1, 2, 2]
    );

    for draws in [alias.rvs(8_000).unwrap(), guide.rvs(8_000).unwrap()] {
        let counts: Vec<_> = (0..3)
            .map(|k| draws.iter().filter(|&&v| v == k).count())
            .collect();
        for (count, weight) in counts.iter().zip(weights) {
            assert!((*count as f64 / 8_000.0 - weight / 10.0).abs() < 0.025);
        }
    }

    alias.reset();
    guide.reset();
    let a = alias.rvs(10).unwrap();
    alias.reset();
    assert_eq!(alias.rvs(10).unwrap(), a);
    let b = guide.rvs(10).unwrap();
    guide.reset();
    assert_eq!(guide.rvs(10).unwrap(), b);
    assert!(DiscreteAliasUrn::new(&[0.0, 0.0], 0).is_err());
}

#[test]
fn public_inverse_samplers_agree_with_analytic_conditional_quantiles() {
    let law = Distribution::Exponential { scale: 1.0 };
    let domain = (0.0, 4.0);
    let mut fast = FastGeneratorInversion::new(law, 10).unwrap();
    let mut hermite = NumericalInverseHermite::new(law, domain, 1e-5, 512, 10).unwrap();
    let mut polynomial = NumericalInversePolynomial::new(law, domain, 1e-5, 512, 10).unwrap();

    for probability in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
        let expected = -(1.0 - probability * (1.0 - (-4.0_f64).exp())).ln();
        let hermite_x = hermite.ppf(probability).unwrap();
        let polynomial_x = polynomial.ppf(probability).unwrap();
        assert!((hermite_x - expected).abs() < 0.005);
        assert!((polynomial_x - expected).abs() < 0.005);
        assert!((hermite.cdf(hermite_x).unwrap() - probability).abs() < 5e-5);
        assert!((polynomial.cdf(polynomial_x).unwrap() - probability).abs() < 5e-5);
    }

    let raw = fast.ppf(0.5).unwrap();
    assert!((raw - std::f64::consts::LN_2).abs() < 1e-10);
    assert!(
        fast.rvs(64)
            .unwrap()
            .iter()
            .all(|x| x.is_finite() && *x >= 0.0)
    );
    assert!(
        hermite
            .rvs(64)
            .unwrap()
            .iter()
            .all(|x| (0.0..=4.0).contains(x))
    );
    assert!(
        polynomial
            .rvs(64)
            .unwrap()
            .iter()
            .all(|x| (0.0..=4.0).contains(x))
    );
    assert!(hermite.u_error(64).unwrap().max_error < 5e-5);
    assert!(polynomial.u_error(64).unwrap().max_error < 5e-5);
}

#[test]
fn public_rejection_samplers_follow_known_density_moments() {
    let normal_density = |x: f64| Ok((-0.5 * x * x).exp());
    let vmax = 2.0_f64.sqrt() * (-0.5_f64).exp();
    let mut ratio = RatioUniforms::new(normal_density, 1.0, -vmax, vmax, 0.0, 8, 100_000).unwrap();
    let normal = ratio.random(2_000).unwrap();
    let normal_mean = normal.iter().sum::<f64>() / normal.len() as f64;
    assert!(normal_mean.abs() < 0.1);

    let mut simple =
        SimpleRatioUniforms::new(|x: f64| Ok(2.0 * x), (0.0, 1.0), 2.0, 0.5, 12, 50_000).unwrap();
    let beta = simple.random(2_000).unwrap();
    assert!(beta.iter().all(|x| (0.0..=1.0).contains(x)));
    let mean = beta.iter().sum::<f64>() / beta.len() as f64;
    assert!((mean - 2.0 / 3.0).abs() < 0.04);

    let mut tdr = TransformedDensityRejection::new(
        |x: f64| Ok(-0.5 * x * x),
        |x: f64| Ok(-x),
        (-5.0, 5.0),
        31,
        50_000,
    )
    .unwrap();
    let samples = tdr.random(2_000).unwrap();
    assert!(samples.iter().all(|x| (-5.0..=5.0).contains(x)));
    assert!((samples.iter().sum::<f64>() / samples.len() as f64).abs() < 0.1);
}
