//! Paired non-sampling stats benchmark against SciPy 1.18.1 / NumPy 2.5.4.
//!
//! Run only with exclusive Cargo ownership (the prime agent serializes builds):
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
//!   cargo bench --bench stats_scipy_comparison --no-default-features --features stats \
//!   -- --repeat 13 --warmup 3 --batch 4 --out target/stats_rust.csv
//!
//! Both runners build identical deterministic numerical inputs. `prepare_only`
//! measures input creation; `compute_only` prepares inputs outside the timer
//! and measures one public API operation, including output allocation/drop.
//! A block of consecutive calls amortizes timer overhead. Rust uses Vec<f64>,
//! whereas SciPy uses float64 ndarray inputs; preparation times and the
//! explicitly adapted vectorized distribution cases are not kernel comparisons.

use sci::stats::{self, inference::Alternative, ranking::RankMethod};
use std::{
    env,
    error::Error,
    fs::{self, File},
    hint::black_box,
    io::{self, BufWriter, Write},
    path::Path,
    time::Instant,
};

const SIZES: &[usize] = &[128, 2048];

#[derive(Clone, Copy)]
enum Operation {
    Mean,
    Variance,
    Quantile,
    RankData,
    ZScore,
    Pearson,
    Spearman,
    TTestOne,
    TTestWelch,
    MannWhitney,
    JarqueBera,
    BoxCox,
    YeoJohnson,
    Wasserstein,
    NormalCdf,
    NormalPpf,
}

#[derive(Clone, Copy)]
enum InputProfile {
    Sample,
    Paired,
    Tied,
    Positive,
    NormalGrid,
    Probabilities,
}

impl InputProfile {
    fn as_str(self) -> &'static str {
        match self {
            Self::Sample => "sample_formula_v1",
            Self::Paired => "paired_formula_v1",
            Self::Tied => "integer_ties_31_v1",
            Self::Positive => "positive_formula_v1",
            Self::NormalGrid => "normal_grid_-4_to_4_v1",
            Self::Probabilities => "probability_grid_0.01_to_0.99_v1",
        }
    }
}

struct Spec {
    key: &'static str,
    family: &'static str,
    operation: Operation,
    input: InputProfile,
    parameters: &'static str,
    comparability: &'static str,
    difference: &'static str,
    output: &'static str,
}

const SPECS: &[Spec] = &[
    Spec {
        key: "mean",
        family: "descriptive",
        operation: Operation::Mean,
        input: InputProfile::Sample,
        parameters: "arithmetic_mean",
        comparability: "matched",
        difference: "slice_vs_ndarray;scipy_tmean_dispatch",
        output: "f64_vs_numpy_scalar",
    },
    Spec {
        key: "variance_ddof1",
        family: "descriptive",
        operation: Operation::Variance,
        input: InputProfile::Sample,
        parameters: "ddof=1",
        comparability: "matched",
        difference: "slice_vs_ndarray;scipy_tvar_dispatch",
        output: "f64_vs_numpy_scalar",
    },
    Spec {
        key: "quantile_linear_p30",
        family: "descriptive",
        operation: Operation::Quantile,
        input: InputProfile::Sample,
        parameters: "probability=0.3;linear_interpolation",
        comparability: "matched",
        difference: "sort_vs_scipy_percentile_backend",
        output: "f64_vs_numpy_scalar",
    },
    Spec {
        key: "rankdata_average_ties",
        family: "ranking",
        operation: Operation::RankData,
        input: InputProfile::Tied,
        parameters: "method=average;31_distinct_values",
        comparability: "matched",
        difference: "sort_backend_and_output_container",
        output: "vec_f64_vs_ndarray",
    },
    Spec {
        key: "zscore_ddof1",
        family: "descriptive",
        operation: Operation::ZScore,
        input: InputProfile::Sample,
        parameters: "ddof=1;axis=none",
        comparability: "matched",
        difference: "native_vector_vs_numpy_dispatch",
        output: "vec_f64_vs_ndarray",
    },
    Spec {
        key: "pearsonr_two_sided",
        family: "association",
        operation: Operation::Pearson,
        input: InputProfile::Paired,
        parameters: "alternative=two-sided",
        comparability: "matched",
        difference: "beta_tail_backend_and_result_container",
        output: "rust_result_vs_scipy_result",
    },
    Spec {
        key: "spearmanr_two_sided",
        family: "association",
        operation: Operation::Spearman,
        input: InputProfile::Paired,
        parameters: "alternative=two-sided",
        comparability: "matched",
        difference: "rank_and_probability_backend",
        output: "rust_result_vs_scipy_result",
    },
    Spec {
        key: "ttest_1samp_two_sided",
        family: "hypothesis",
        operation: Operation::TTestOne,
        input: InputProfile::Sample,
        parameters: "popmean=0.1;alternative=two-sided",
        comparability: "matched",
        difference: "tail_probability_backend",
        output: "rust_result_vs_scipy_result",
    },
    Spec {
        key: "ttest_ind_welch_two_sided",
        family: "hypothesis",
        operation: Operation::TTestWelch,
        input: InputProfile::Paired,
        parameters: "equal_var=false;alternative=two-sided",
        comparability: "matched",
        difference: "tail_probability_backend",
        output: "rust_result_vs_scipy_result",
    },
    Spec {
        key: "mannwhitneyu_asymptotic",
        family: "hypothesis",
        operation: Operation::MannWhitney,
        input: InputProfile::Paired,
        parameters: "method=asymptotic;use_continuity=true;alternative=two-sided",
        comparability: "matched",
        difference: "rank_and_normal_tail_backend",
        output: "rust_result_vs_scipy_result",
    },
    Spec {
        key: "jarque_bera",
        family: "hypothesis",
        operation: Operation::JarqueBera,
        input: InputProfile::Sample,
        parameters: "asymptotic_chi2_2",
        comparability: "matched",
        difference: "moment_and_tail_backend",
        output: "rust_result_vs_scipy_result",
    },
    Spec {
        key: "boxcox_lambda05",
        family: "transform",
        operation: Operation::BoxCox,
        input: InputProfile::Positive,
        parameters: "lambda=0.5;fixed_no_fit",
        comparability: "matched",
        difference: "native_vec_vs_ndarray",
        output: "vec_f64_vs_ndarray",
    },
    Spec {
        key: "yeojohnson_lambda05",
        family: "transform",
        operation: Operation::YeoJohnson,
        input: InputProfile::Sample,
        parameters: "lambda=0.5;fixed_no_fit",
        comparability: "matched",
        difference: "native_vec_vs_ndarray",
        output: "vec_f64_vs_ndarray",
    },
    Spec {
        key: "wasserstein_distance",
        family: "distance",
        operation: Operation::Wasserstein,
        input: InputProfile::Paired,
        parameters: "uniform_weights;one_dimensional",
        comparability: "matched",
        difference: "sort_backend_and_memory_layout",
        output: "f64_vs_numpy_scalar",
    },
    Spec {
        key: "normal_cdf_batch",
        family: "distribution",
        operation: Operation::NormalCdf,
        input: InputProfile::NormalGrid,
        parameters: "normal_loc=0.5;scale=1.5;cdf",
        comparability: "adapted",
        difference: "rust_scalar_loop_vs_scipy_vectorized_ndarray",
        output: "vec_f64_vs_ndarray",
    },
    Spec {
        key: "normal_ppf_batch",
        family: "distribution",
        operation: Operation::NormalPpf,
        input: InputProfile::Probabilities,
        parameters: "normal_loc=0.5;scale=1.5;ppf",
        comparability: "adapted",
        difference: "rust_scalar_loop_vs_scipy_vectorized_ndarray",
        output: "vec_f64_vs_ndarray",
    },
];

struct Inputs {
    x: Vec<f64>,
    y: Vec<f64>,
}

fn sample_x(i: usize, n: usize) -> f64 {
    ((i * 37 % 257) as f64 - 128.0) / 19.0 + (i as f64 / n as f64) * 0.03
}

fn sample_y(i: usize, n: usize) -> f64 {
    (((i * 59 + 13) % 263) as f64 - 131.0) / 23.0 - (i as f64 / n as f64) * 0.02
}

fn prepare(profile: InputProfile, n: usize) -> Inputs {
    let x = (0..n)
        .map(|i| match profile {
            InputProfile::Sample | InputProfile::Paired => sample_x(i, n),
            InputProfile::Tied => ((i * 17 + 3) % 31) as f64,
            InputProfile::Positive => sample_x(i, n).abs() + 0.75,
            InputProfile::NormalGrid => -4.0 + 8.0 * (i as f64 / (n - 1) as f64),
            InputProfile::Probabilities => 0.01 + 0.98 * (i as f64 / (n - 1) as f64),
        })
        .collect();
    let y = if matches!(profile, InputProfile::Paired) {
        (0..n).map(|i| sample_y(i, n)).collect()
    } else {
        Vec::new()
    };
    Inputs { x, y }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Prepare,
    Compute,
}

impl Phase {
    fn as_str(self) -> &'static str {
        match self {
            Self::Prepare => "prepare_only",
            Self::Compute => "compute_only",
        }
    }
}

struct Options {
    repeats: usize,
    warmups: usize,
    batch: usize,
    quick: bool,
}

fn parse_args() -> Result<(Options, Option<String>), Box<dyn Error>> {
    let mut options = Options {
        repeats: 13,
        warmups: 3,
        batch: 4,
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
            "--batch" => options.batch = args.next().ok_or("missing --batch value")?.parse()?,
            "--out" => out = Some(args.next().ok_or("missing --out value")?),
            "--help" | "-h" => {
                println!(
                    "Usage: stats_scipy_comparison [--quick] [--repeat N] [--warmup N] [--batch N] [--out FILE]"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unrecognized option: {arg}").into()),
        }
    }
    if !(1..=100).contains(&options.repeats)
        || options.warmups > 100
        || !(1..=64).contains(&options.batch)
    {
        return Err("repeat must be 1..=100; warmup 0..=100; batch 1..=64".into());
    }
    Ok((options, out))
}

struct Timing {
    median: u128,
    p25: u128,
    p75: u128,
    minimum: u128,
}

fn measure<T>(
    spec: &Spec,
    phase: Phase,
    n: usize,
    options: &Options,
    mut operation: impl FnMut(&Inputs) -> T,
) -> Timing {
    let mut times = Vec::with_capacity(options.repeats);
    for iteration in 0..options.warmups + options.repeats {
        let elapsed = if phase == Phase::Prepare {
            let start = Instant::now();
            for _ in 0..options.batch {
                black_box(prepare(spec.input, black_box(n)));
            }
            start.elapsed().as_nanos()
        } else {
            let input = prepare(spec.input, n);
            let start = Instant::now();
            for _ in 0..options.batch {
                black_box(operation(black_box(&input)));
            }
            start.elapsed().as_nanos()
        };
        if iteration >= options.warmups {
            times.push(elapsed / options.batch as u128);
        }
    }
    times.sort_unstable();
    let midpoint = times.len() / 2;
    let median = if times.len() % 2 == 0 {
        (times[midpoint - 1] + times[midpoint]) / 2
    } else {
        times[midpoint]
    };
    Timing {
        median,
        p25: times[times.len() / 4],
        p75: times[3 * times.len() / 4],
        minimum: times[0],
    }
}

fn benchmark(spec: &Spec, phase: Phase, n: usize, opts: &Options) -> Timing {
    match spec.operation {
        Operation::Mean => measure(spec, phase, n, opts, |i| {
            stats::descriptive::mean(black_box(&i.x)).unwrap()
        }),
        Operation::Variance => measure(spec, phase, n, opts, |i| {
            stats::descriptive::variance(black_box(&i.x), 1).unwrap()
        }),
        Operation::Quantile => measure(spec, phase, n, opts, |i| {
            stats::descriptive::quantile(black_box(&i.x), 0.3).unwrap()
        }),
        Operation::RankData => measure(spec, phase, n, opts, |i| {
            stats::ranking::rankdata(black_box(&i.x), RankMethod::Average).unwrap()
        }),
        Operation::ZScore => measure(spec, phase, n, opts, |i| {
            stats::descriptive::zscore(black_box(&i.x), 1).unwrap()
        }),
        Operation::Pearson => measure(spec, phase, n, opts, |i| {
            stats::association::pearsonr(black_box(&i.x), black_box(&i.y), Alternative::TwoSided)
                .unwrap()
        }),
        Operation::Spearman => measure(spec, phase, n, opts, |i| {
            stats::association::spearmanr(black_box(&i.x), black_box(&i.y), Alternative::TwoSided)
                .unwrap()
        }),
        Operation::TTestOne => measure(spec, phase, n, opts, |i| {
            stats::inference::ttest_1samp(black_box(&i.x), 0.1, Alternative::TwoSided).unwrap()
        }),
        Operation::TTestWelch => measure(spec, phase, n, opts, |i| {
            stats::inference::ttest_ind(
                black_box(&i.x),
                black_box(&i.y),
                false,
                Alternative::TwoSided,
            )
            .unwrap()
        }),
        Operation::MannWhitney => measure(spec, phase, n, opts, |i| {
            stats::inference::mannwhitneyu(
                black_box(&i.x),
                black_box(&i.y),
                Alternative::TwoSided,
                true,
            )
            .unwrap()
        }),
        Operation::JarqueBera => measure(spec, phase, n, opts, |i| {
            stats::normality::jarque_bera(black_box(&i.x)).unwrap()
        }),
        Operation::BoxCox => measure(spec, phase, n, opts, |i| {
            stats::transformations::boxcox(black_box(&i.x), 0.5).unwrap()
        }),
        Operation::YeoJohnson => measure(spec, phase, n, opts, |i| {
            stats::transformations::yeojohnson(black_box(&i.x), 0.5).unwrap()
        }),
        Operation::Wasserstein => measure(spec, phase, n, opts, |i| {
            stats::distances::wasserstein_distance(black_box(&i.x), black_box(&i.y)).unwrap()
        }),
        Operation::NormalCdf => measure(spec, phase, n, opts, |i| {
            i.x.iter()
                .map(|&x| stats::distributions::normal_cdf(black_box(x), 0.5, 1.5).unwrap())
                .collect::<Vec<_>>()
        }),
        Operation::NormalPpf => measure(spec, phase, n, opts, |i| {
            i.x.iter()
                .map(|&p| stats::distributions::normal_ppf(black_box(p), 0.5, 1.5).unwrap())
                .collect::<Vec<_>>()
        }),
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
        "implementation,case,family,phase,n,parameters,input_profile,repetitions,batch_calls,median_ns,p25_ns,p75_ns,min_ns,comparability,difference,output_layout"
    )?;
    for spec in SPECS {
        let sizes = if options.quick { &SIZES[..1] } else { SIZES };
        for &n in sizes {
            for phase in [Phase::Prepare, Phase::Compute] {
                let t = benchmark(spec, phase, n, &options);
                let (comparability, difference, output_layout) = if phase == Phase::Prepare {
                    (
                        "adapted",
                        "rust_vec_preparation_vs_numpy_ndarray_preparation",
                        "vec_f64_vs_numpy_ndarray",
                    )
                } else {
                    (spec.comparability, spec.difference, spec.output)
                };
                writeln!(
                    output,
                    "sci_rust_0.3.0,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                    spec.key,
                    spec.family,
                    phase.as_str(),
                    n,
                    spec.parameters,
                    spec.input.as_str(),
                    options.repeats,
                    options.batch,
                    t.median,
                    t.p25,
                    t.p75,
                    t.minimum,
                    comparability,
                    difference,
                    output_layout
                )?;
            }
        }
    }
    output.flush()?;
    Ok(())
}
