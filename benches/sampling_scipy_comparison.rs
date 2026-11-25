//! Standalone Sci 0.3.0 sampling benchmark paired with
//! tests/scipy/benchmark_sampling_engines.py (SciPy 1.18.1 / NumPy 2.5.4).
//!
//! Run only when Cargo is exclusively owned by the prime agent:
//!   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
//!     cargo bench --bench sampling_scipy_comparison \
//!     --no-default-features --features stats -- --out target/sampling_rust.csv
//!
//! Timing covers either construction alone or one draw into a newly allocated
//! output. Draw timing excludes construction, reset and result destruction.
//! Seed and input distributions are fixed. Engines are created afresh before
//! each timed draw so finite or adaptive generators start from the same state.
//! Most Rust QMC engines return nested vectors; SciPy returns NumPy arrays.
//! Separate Sobol flat cases compare contiguous row-major f64 outputs.
//! Identical scramble seeds do not imply identical Sci/SciPy RNG streams.
//! Algorithmic and output-format differences are annotated in the CSV.

use sci::stats::{
    distributions::Distribution,
    qmc::{Halton, LatinHypercube, MultinomialQMC, MultivariateNormalQMC, PoissonDisk, Sobol},
    sampling::{
        DiscreteAliasUrn, DiscreteGuideTable, FastGeneratorInversion, NumericalInverseHermite,
        NumericalInversePolynomial, RatioUniforms, SimpleRatioUniforms,
        TransformedDensityRejection,
    },
};
use std::{
    env,
    error::Error,
    fs::{self, File},
    hint::black_box,
    io::{self, BufWriter, Write},
    path::Path,
    time::Instant,
};

const SEED: u64 = 42;
const STD_SIZES: &[usize] = &[128, 1024, 8192];
const SHORT_SIZES: &[usize] = &[128, 1024];
const POISSON_SIZES: &[usize] = &[64, 128];
const MULTINOMIAL_SIZES: &[usize] = &[8, 32];
const TRIALS: usize = 256;
const COV: [[f64; 4]; 4] = [
    [1.0, 0.2, 0.0, 0.0],
    [0.2, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.2],
    [0.0, 0.0, 0.2, 1.0],
];

#[derive(Clone, Copy)]
enum Engine {
    Sobol,
    SobolScrambled,
    SobolFlat,
    SobolScrambledFlat,
    Halton,
    HaltonScrambled,
    LatinHypercube,
    LatinHypercubeCentered,
    PoissonDisk,
    MultivariateNormal,
    Multinomial,
    Alias,
    Guide,
    FastInversion,
    Hermite,
    Polynomial,
    RatioUniforms,
    SimpleRatioUniforms,
    TransformedRejection,
}

struct Spec {
    engine: Engine,
    key: &'static str,
    family: &'static str,
    d: usize,
    sizes: &'static [usize],
    parameter: &'static str,
    comparability: &'static str,
    difference: &'static str,
    layout: &'static str,
}

const SPECS: &[Spec] = &[
    Spec {
        engine: Engine::Sobol,
        key: "sobol_plain",
        family: "qmc",
        d: 4,
        sizes: STD_SIZES,
        parameter: "bits=30;scramble=false",
        comparability: "matched",
        difference: "different_output_container",
        layout: "vec_vec_f64",
    },
    Spec {
        engine: Engine::SobolScrambled,
        key: "sobol_scrambled",
        family: "qmc",
        d: 4,
        sizes: STD_SIZES,
        parameter: "bits=30;scramble=true;seed=42",
        comparability: "adapted",
        difference: "scramble_rng_differs",
        layout: "vec_vec_f64",
    },
    Spec {
        engine: Engine::SobolFlat,
        key: "sobol_plain_flat",
        family: "qmc",
        d: 4,
        sizes: STD_SIZES,
        parameter: "bits=30;scramble=false",
        comparability: "matched",
        difference: "contiguous_row_major;vec_vs_ndarray",
        layout: "vec_f64_row_major",
    },
    Spec {
        engine: Engine::SobolScrambledFlat,
        key: "sobol_scrambled_flat",
        family: "qmc",
        d: 4,
        sizes: STD_SIZES,
        parameter: "bits=30;scramble=true;seed=42",
        comparability: "adapted",
        difference: "contiguous_row_major;scramble_rng_differs",
        layout: "vec_f64_row_major",
    },
    Spec {
        engine: Engine::Halton,
        key: "halton_plain",
        family: "qmc",
        d: 4,
        sizes: STD_SIZES,
        parameter: "scramble=false",
        comparability: "matched",
        difference: "different_output_container",
        layout: "vec_vec_f64",
    },
    Spec {
        engine: Engine::HaltonScrambled,
        key: "halton_scrambled",
        family: "qmc",
        d: 4,
        sizes: STD_SIZES,
        parameter: "scramble=true;seed=42",
        comparability: "adapted",
        difference: "scramble_rng_differs",
        layout: "vec_vec_f64",
    },
    Spec {
        engine: Engine::LatinHypercube,
        key: "latin_hypercube",
        family: "qmc",
        d: 4,
        sizes: STD_SIZES,
        parameter: "strength=1;scramble=true;seed=42",
        comparability: "matched",
        difference: "rng_and_output_layout_differ",
        layout: "vec_vec_f64",
    },
    Spec {
        engine: Engine::LatinHypercubeCentered,
        key: "latin_hypercube_centered",
        family: "qmc",
        d: 4,
        sizes: STD_SIZES,
        parameter: "strength=1;scramble=false;seed=42",
        comparability: "matched",
        difference: "rng_and_output_layout_differ",
        layout: "vec_vec_f64",
    },
    Spec {
        engine: Engine::PoissonDisk,
        key: "poisson_disk",
        family: "qmc",
        d: 2,
        sizes: POISSON_SIZES,
        parameter: "radius=0.06;ncandidates=30;hypersphere=volume;seed=42",
        comparability: "adapted",
        difference: "independent_candidate_schedules",
        layout: "vec_vec_f64",
    },
    Spec {
        engine: Engine::MultivariateNormal,
        key: "multivariate_normal_qmc",
        family: "qmc",
        d: 4,
        sizes: SHORT_SIZES,
        parameter: "bits=30;unscrambled_sobol;cov_offdiag=0.2",
        comparability: "matched",
        difference: "inverse_normal_backend_and_output_layout_differ",
        layout: "vec_vec_f64",
    },
    Spec {
        engine: Engine::Multinomial,
        key: "multinomial_qmc",
        family: "qmc",
        d: 3,
        sizes: MULTINOMIAL_SIZES,
        parameter: "p=0.2:0.3:0.5;trials=256;unscrambled_sobol",
        comparability: "adapted",
        difference: "tie_conventions_differ",
        layout: "vec_vec_usize",
    },
    Spec {
        engine: Engine::Alias,
        key: "discrete_alias_urn",
        family: "sampling",
        d: 1,
        sizes: STD_SIZES,
        parameter: "weights=1..256;seed=42",
        comparability: "matched",
        difference: "different_rng_and_memory_layout",
        layout: "vec_usize",
    },
    Spec {
        engine: Engine::Guide,
        key: "discrete_guide_table",
        family: "sampling",
        d: 1,
        sizes: STD_SIZES,
        parameter: "weights=1..256;seed=42",
        comparability: "matched",
        difference: "different_rng_and_memory_layout",
        layout: "vec_usize",
    },
    Spec {
        engine: Engine::FastInversion,
        key: "fast_generator_inversion",
        family: "sampling",
        d: 1,
        sizes: STD_SIZES,
        parameter: "normal=0:1;seed=42",
        comparability: "adapted",
        difference: "native_ppf_vs_precomputed_unuran",
        layout: "vec_f64",
    },
    Spec {
        engine: Engine::Hermite,
        key: "numerical_inverse_hermite",
        family: "sampling",
        d: 1,
        sizes: SHORT_SIZES,
        parameter: "expon_domain=0:6;u_resolution=1e-5;seed=42",
        comparability: "adapted",
        difference: "different_knot_selection",
        layout: "vec_f64",
    },
    Spec {
        engine: Engine::Polynomial,
        key: "numerical_inverse_polynomial",
        family: "sampling",
        d: 1,
        sizes: SHORT_SIZES,
        parameter: "expon_domain=0:6;u_resolution=1e-5;seed=42",
        comparability: "adapted",
        difference: "sci_linear_vs_scipy_order3",
        layout: "vec_f64",
    },
    Spec {
        engine: Engine::RatioUniforms,
        key: "ratio_uniforms",
        family: "sampling",
        d: 1,
        sizes: SHORT_SIZES,
        parameter: "unnormalized_gaussian;umax=1;vmax=sqrt(2/e);seed=42",
        comparability: "adapted",
        difference: "python_pdf_callback_and_rng",
        layout: "vec_f64",
    },
    Spec {
        engine: Engine::SimpleRatioUniforms,
        key: "simple_ratio_uniforms",
        family: "sampling",
        d: 1,
        sizes: SHORT_SIZES,
        parameter: "density=2x;support=0:1;seed=42",
        comparability: "adapted",
        difference: "certified_rectangle_vs_auto_envelope",
        layout: "vec_f64",
    },
    Spec {
        engine: Engine::TransformedRejection,
        key: "transformed_density_rejection",
        family: "sampling",
        d: 1,
        sizes: SHORT_SIZES,
        parameter: "truncated_gaussian=-4:4;transform_c=0;seed=42",
        comparability: "adapted",
        difference: "different_adaptive_envelopes_and_callbacks",
        layout: "vec_f64",
    },
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Construct,
    Draw,
}

impl Phase {
    fn as_str(self) -> &'static str {
        match self {
            Self::Construct => "construct_only",
            Self::Draw => "draw_random_only",
        }
    }
}

#[derive(Clone, Copy)]
struct Options {
    repeats: usize,
    warmups: usize,
    quick: bool,
}

fn parse_args() -> Result<(Options, Option<String>), Box<dyn Error>> {
    let mut options = Options {
        repeats: 9,
        warmups: 2,
        quick: false,
    };
    let mut out = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bench" => {}
            "--quick" => options.quick = true,
            "--repeat" => options.repeats = args.next().ok_or("missing --repeat value")?.parse()?,
            "--warmup" => options.warmups = args.next().ok_or("missing --warmup value")?.parse()?,
            "--out" => out = Some(args.next().ok_or("missing --out value")?),
            "--help" | "-h" => {
                println!(
                    "Usage: sampling_scipy_comparison [--quick] [--repeat N] [--warmup N] [--out FILE]"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unrecognized option: {arg}").into()),
        }
    }
    if options.repeats == 0 || options.repeats > 100 || options.warmups > 100 {
        return Err("repeat must be 1..=100 and warmup 0..=100".into());
    }
    Ok((options, out))
}

/// Create inputs outside the measured section. The timed section excludes
/// engine destruction and captures the complete generation and allocation.
fn measure<P, T>(
    phase: Phase,
    options: Options,
    mut create: impl FnMut() -> P,
    mut draw: impl FnMut(&mut P) -> T,
) -> (u128, u128) {
    let mut timings = Vec::with_capacity(options.repeats);
    for iteration in 0..options.warmups + options.repeats {
        let elapsed = if phase == Phase::Construct {
            let start = Instant::now();
            let prepared = create();
            let elapsed = start.elapsed().as_nanos();
            black_box(prepared);
            elapsed
        } else {
            let mut prepared = create();
            let start = Instant::now();
            let generated = draw(&mut prepared);
            let elapsed = start.elapsed().as_nanos();
            black_box(generated);
            elapsed
        };
        if iteration >= options.warmups {
            timings.push(elapsed);
        }
    }
    timings.sort_unstable();
    let middle = timings.len() / 2;
    let median = if timings.len() % 2 == 0 {
        (timings[middle - 1] + timings[middle]) / 2
    } else {
        timings[middle]
    };
    (median, timings[0])
}

fn benchmark(spec: &Spec, phase: Phase, n: usize, options: Options) -> (u128, u128) {
    let (repeats, d) = (options, spec.d);
    let weights: Vec<f64> = (1..=256).map(|i| i as f64).collect();
    let cov: Vec<Vec<f64>> = COV.iter().map(|row| row.to_vec()).collect();
    match spec.engine {
        Engine::Sobol => measure(
            phase,
            repeats,
            || Sobol::new(d, 30).unwrap(),
            |s| s.random(n).unwrap(),
        ),
        Engine::SobolScrambled => measure(
            phase,
            repeats,
            || Sobol::scrambled(d, 30, SEED).unwrap(),
            |s| s.random(n).unwrap(),
        ),
        Engine::SobolFlat => measure(
            phase,
            repeats,
            || Sobol::new(d, 30).unwrap(),
            |s| s.random_flat(n).unwrap(),
        ),
        Engine::SobolScrambledFlat => measure(
            phase,
            repeats,
            || Sobol::scrambled(d, 30, SEED).unwrap(),
            |s| s.random_flat(n).unwrap(),
        ),
        Engine::Halton => measure(
            phase,
            repeats,
            || Halton::new(d).unwrap(),
            |s| s.random(n).unwrap(),
        ),
        Engine::HaltonScrambled => measure(
            phase,
            repeats,
            || Halton::scrambled(d, SEED).unwrap(),
            |s| s.random(n).unwrap(),
        ),
        Engine::LatinHypercube => measure(
            phase,
            repeats,
            || LatinHypercube::new(d, SEED).unwrap(),
            |s| s.random(n).unwrap(),
        ),
        Engine::LatinHypercubeCentered => measure(
            phase,
            repeats,
            || LatinHypercube::centered(d, SEED).unwrap(),
            |s| s.random(n).unwrap(),
        ),
        Engine::PoissonDisk => measure(
            phase,
            repeats,
            || PoissonDisk::new(d, 0.06, SEED).unwrap(),
            |s| {
                let samples = s.random(n).unwrap();
                assert_eq!(samples.len(), n, "PoissonDisk exhausted before requested n");
                samples
            },
        ),
        Engine::MultivariateNormal => measure(
            phase,
            repeats,
            || MultivariateNormalQMC::new(&[0.; 4], Some(&cov)).unwrap(),
            |s| s.random(n).unwrap(),
        ),
        Engine::Multinomial => measure(
            phase,
            repeats,
            || MultinomialQMC::new(&[0.2, 0.3, 0.5], TRIALS).unwrap(),
            |s| s.random(n).unwrap(),
        ),
        Engine::Alias => measure(
            phase,
            repeats,
            || DiscreteAliasUrn::new(&weights, SEED).unwrap(),
            |s| s.rvs(n).unwrap(),
        ),
        Engine::Guide => measure(
            phase,
            repeats,
            || DiscreteGuideTable::new(&weights, SEED).unwrap(),
            |s| s.rvs(n).unwrap(),
        ),
        Engine::FastInversion => measure(
            phase,
            repeats,
            || {
                FastGeneratorInversion::new(
                    Distribution::Normal {
                        mean: 0.,
                        std_dev: 1.,
                    },
                    SEED,
                )
                .unwrap()
            },
            |s| s.rvs(n).unwrap(),
        ),
        Engine::Hermite => measure(
            phase,
            repeats,
            || {
                NumericalInverseHermite::new(
                    Distribution::Exponential { scale: 1. },
                    (0., 6.),
                    1e-5,
                    512,
                    SEED,
                )
                .unwrap()
            },
            |s| s.rvs(n).unwrap(),
        ),
        Engine::Polynomial => measure(
            phase,
            repeats,
            || {
                NumericalInversePolynomial::new(
                    Distribution::Exponential { scale: 1. },
                    (0., 6.),
                    1e-5,
                    512,
                    SEED,
                )
                .unwrap()
            },
            |s| s.rvs(n).unwrap(),
        ),
        Engine::RatioUniforms => {
            let vmax = (2.0_f64 / std::f64::consts::E).sqrt();
            measure(
                phase,
                repeats,
                || {
                    RatioUniforms::new(
                        |x: f64| Ok((-0.5 * x * x).exp()),
                        1.,
                        -vmax,
                        vmax,
                        0.,
                        SEED,
                        100_000,
                    )
                    .unwrap()
                },
                |s| s.random(n).unwrap(),
            )
        }
        Engine::SimpleRatioUniforms => measure(
            phase,
            repeats,
            || {
                SimpleRatioUniforms::new(|x: f64| Ok(2. * x), (0., 1.), 2., 0.5, SEED, 100_000)
                    .unwrap()
            },
            |s| s.random(n).unwrap(),
        ),
        Engine::TransformedRejection => measure(
            phase,
            repeats,
            || {
                TransformedDensityRejection::new(
                    |x: f64| Ok(-0.5 * x * x),
                    |x: f64| Ok(-x),
                    (-4., 4.),
                    SEED,
                    100_000,
                )
                .unwrap()
            },
            |s| s.random(n).unwrap(),
        ),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let (options, out) = parse_args()?;
    let output: Box<dyn Write> = if let Some(path) = out {
        if let Some(parent) = Path::new(&path).parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        Box::new(BufWriter::new(File::create(path)?))
    } else {
        Box::new(BufWriter::new(io::stdout()))
    };
    let mut output = output;
    writeln!(
        output,
        "implementation,case,family,phase,n,d,parameters,repetitions,median_ns,min_ns,comparability,difference,output_layout"
    )?;
    for spec in SPECS {
        for phase in [Phase::Construct, Phase::Draw] {
            let sizes = if phase == Phase::Construct {
                &[][..]
            } else if options.quick {
                &spec.sizes[..1]
            } else {
                spec.sizes
            };
            let once = [0_usize];
            for &n in if phase == Phase::Construct {
                &once[..]
            } else {
                sizes
            } {
                let (median, minimum) = benchmark(spec, phase, n, options);
                writeln!(
                    output,
                    "sci_rust_0.3.0,{},{},{},{},{},{},{},{},{},{},{},{}",
                    spec.key,
                    spec.family,
                    phase.as_str(),
                    n,
                    spec.d,
                    spec.parameter,
                    options.repeats,
                    median,
                    minimum,
                    spec.comparability,
                    spec.difference,
                    spec.layout
                )?;
            }
        }
    }
    output.flush()?;
    Ok(())
}
