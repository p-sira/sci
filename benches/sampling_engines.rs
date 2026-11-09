//! CodSpeed benchmarks for the public QMC and distribution sampling engines.
//!
//! Seeds and batch sizes are fixed. Stateful generators reset between iterations
//! to measure a repeatable batch rather than changing work as streams advance.
//! Construction is excluded from draw measurements except where named explicitly.

use divan::{Bencher, black_box};
use sci::stats::{
    distributions::Distribution,
    qmc::{
        Halton, LatinHypercube, MultinomialQMC, MultivariateNormalQMC, PoissonDisk, QmcEngine,
        Sobol,
    },
    sampling::{
        DiscreteAliasUrn, DiscreteGuideTable, FastGeneratorInversion, NumericalInverseHermite,
        NumericalInversePolynomial, RatioUniforms, SimpleRatioUniforms,
        TransformedDensityRejection,
    },
};

fn main() {
    divan::main();
}

mod qmc_benches {
    use super::*;

    #[divan::bench(args = [128, 1024])]
    fn sobol(bencher: Bencher, n: usize) {
        let mut engine = Sobol::new(4, 30).unwrap();
        bencher.bench_local(|| {
            engine.reset();
            black_box(engine.random(black_box(n)).unwrap())
        });
    }

    #[divan::bench(args = [128, 1024])]
    fn sobol_scrambled(bencher: Bencher, n: usize) {
        let mut engine = Sobol::scrambled(4, 30, 42).unwrap();
        bencher.bench_local(|| {
            engine.reset();
            black_box(engine.random(black_box(n)).unwrap())
        });
    }

    #[divan::bench(args = [128, 1024])]
    fn halton(bencher: Bencher, n: usize) {
        let mut engine = Halton::new(4).unwrap();
        bencher.bench_local(|| {
            engine.reset();
            black_box(engine.random(black_box(n)).unwrap())
        });
    }

    #[divan::bench(args = [128, 1024])]
    fn halton_scrambled(bencher: Bencher, n: usize) {
        let mut engine = Halton::scrambled(4, 42).unwrap();
        bencher.bench_local(|| {
            engine.reset();
            black_box(engine.random(black_box(n)).unwrap())
        });
    }

    #[divan::bench(args = [128, 1024])]
    fn latin_hypercube(bencher: Bencher, n: usize) {
        let mut engine = LatinHypercube::new(4, 42).unwrap();
        bencher.bench_local(|| {
            engine.reset();
            black_box(engine.random(black_box(n)).unwrap())
        });
    }

    #[divan::bench]
    fn poisson_disk_2d(bencher: Bencher) {
        // Includes grid construction; the generator cannot be reused after it
        // fills its finite domain, so compare identical fresh sample designs.
        bencher.bench_local(|| {
            let mut engine = PoissonDisk::new(2, 0.06, 42).unwrap();
            black_box(engine.random(black_box(64)).unwrap())
        });
    }

    #[divan::bench]
    fn multivariate_normal(bencher: Bencher) {
        let mean = [0.; 4];
        let covariance = vec![
            vec![1., 0.2, 0., 0.],
            vec![0.2, 1., 0., 0.],
            vec![0., 0., 1., 0.2],
            vec![0., 0., 0.2, 1.],
        ];
        let mut engine = MultivariateNormalQMC::new(&mean, Some(&covariance)).unwrap();
        bencher.bench_local(|| {
            engine.reset();
            black_box(engine.random(black_box(256)).unwrap())
        });
    }

    #[divan::bench]
    fn multinomial(bencher: Bencher) {
        let mut engine = MultinomialQMC::new(&[0.2, 0.3, 0.5], 256).unwrap();
        bencher.bench_local(|| {
            engine.reset();
            black_box(engine.random(black_box(8)).unwrap())
        });
    }

    #[divan::bench]
    fn integer_mapping(bencher: Bencher) {
        let mut engine = Sobol::new(4, 30).unwrap();
        let lower = [i64::MIN, -9, 0, 100];
        let upper = [i64::MAX, 9, 99, 200];
        bencher.bench_local(|| {
            engine.reset();
            black_box(
                QmcEngine::integers(&mut engine, black_box(512), &lower, &upper, true).unwrap(),
            )
        });
    }
}

mod sampling_benches {
    use super::*;

    fn weights() -> Vec<f64> {
        (1..=256).map(|v| v as f64).collect()
    }

    #[divan::bench]
    fn alias_construct(bencher: Bencher) {
        let weights = weights();
        bencher.bench(|| DiscreteAliasUrn::new(black_box(&weights), 42).unwrap());
    }

    #[divan::bench]
    fn alias_draw(bencher: Bencher) {
        let mut engine = DiscreteAliasUrn::new(&weights(), 42).unwrap();
        bencher.bench_local(|| {
            engine.reset();
            black_box(engine.rvs(black_box(1024)).unwrap())
        });
    }

    #[divan::bench]
    fn guide_construct(bencher: Bencher) {
        let weights = weights();
        bencher.bench(|| DiscreteGuideTable::new(black_box(&weights), 42).unwrap());
    }

    #[divan::bench]
    fn guide_draw(bencher: Bencher) {
        let mut engine = DiscreteGuideTable::new(&weights(), 42).unwrap();
        bencher.bench_local(|| {
            engine.reset();
            black_box(engine.rvs(black_box(1024)).unwrap())
        });
    }

    #[divan::bench]
    fn native_inverse_draw(bencher: Bencher) {
        let initial = FastGeneratorInversion::new(
            Distribution::Normal {
                mean: 0.,
                std_dev: 1.,
            },
            42,
        )
        .unwrap();
        bencher.bench_local(|| {
            let mut engine = initial.clone();
            black_box(engine.rvs(black_box(512)).unwrap())
        });
    }

    #[divan::bench]
    fn hermite_construct(bencher: Bencher) {
        let law = Distribution::Exponential { scale: 1. };
        bencher.bench(|| {
            NumericalInverseHermite::new(black_box(law), (0., 6.), 1e-5, 512, 42).unwrap()
        });
    }

    #[divan::bench]
    fn polynomial_construct(bencher: Bencher) {
        let law = Distribution::Exponential { scale: 1. };
        bencher.bench(|| {
            NumericalInversePolynomial::new(black_box(law), (0., 6.), 1e-5, 512, 42).unwrap()
        });
    }

    #[divan::bench]
    fn hermite_quantiles(bencher: Bencher) {
        let engine = NumericalInverseHermite::new(
            Distribution::Exponential { scale: 1. },
            (0., 6.),
            1e-5,
            512,
            42,
        )
        .unwrap();
        let probabilities: Vec<f64> = (0..512).map(|i| (i as f64 + 0.5) / 512.).collect();
        bencher.bench(|| engine.qrvs(black_box(&probabilities)).unwrap());
    }

    #[divan::bench]
    fn polynomial_quantiles(bencher: Bencher) {
        let engine = NumericalInversePolynomial::new(
            Distribution::Exponential { scale: 1. },
            (0., 6.),
            1e-5,
            512,
            42,
        )
        .unwrap();
        let probabilities: Vec<f64> = (0..512).map(|i| (i as f64 + 0.5) / 512.).collect();
        bencher.bench(|| engine.qrvs(black_box(&probabilities)).unwrap());
    }

    #[divan::bench]
    fn ratio_uniforms_normal(bencher: Bencher) {
        let pdf = |x: f64| Ok((-0.5 * x * x).exp());
        let vmax = (2.0_f64 / std::f64::consts::E).sqrt();
        let mut engine = RatioUniforms::new(pdf, 1., -vmax, vmax, 0., 42, 100_000).unwrap();
        bencher.bench_local(|| {
            engine.reset();
            black_box(engine.random(black_box(256)).unwrap())
        });
    }

    #[divan::bench]
    fn simple_ratio_uniforms(bencher: Bencher) {
        let mut engine =
            SimpleRatioUniforms::new(|x: f64| Ok(2. * x), (0., 1.), 2., 0.5, 42, 100_000).unwrap();
        bencher.bench_local(|| {
            engine.reset();
            black_box(engine.random(black_box(256)).unwrap())
        });
    }

    #[divan::bench]
    fn transformed_density_rejection(bencher: Bencher) {
        let mut engine = TransformedDensityRejection::new(
            |x: f64| Ok(-0.5 * x * x),
            |x: f64| Ok(-x),
            (-4., 4.),
            42,
            100_000,
        )
        .unwrap();
        bencher.bench_local(|| {
            engine.reset();
            black_box(engine.random(black_box(256)).unwrap())
        });
    }
}
