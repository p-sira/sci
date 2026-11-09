//! Matched real-domain special-function benchmark for Sci 0.3.0 vs SciPy 1.18.1.
//!
//! Pair with tests/scipy/benchmark_special_functions.py. Input grids and output
//! summaries agree by construction. `prepare_only` times input allocation;
//! `compute_only` prepares input outside the clock and includes output allocation.
//! Compare arithmetic and input domains, but account for SciPy's vectorized
//! ufunc dispatch and Rust's public scalar loop (recorded in CSV metadata).
//!
//! With exclusive CPU/Cargo ownership, from /sci:
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
//!   CARGO_BUILD_JOBS=6 cargo bench --bench special_performance_comparison --no-default-features \
//!   --features special -- --repeat 13 --warmup 3 --batch 4 \
//!   --out target/special_rust.csv

use sci::special;
use std::{
    env,
    error::Error,
    fs::{self, File},
    hint::black_box,
    io::{self, BufWriter, Write},
    path::Path,
    time::Instant,
};

const SIZES: &[usize] = &[64, 1024];

#[derive(Clone, Copy)]
enum Op {
    Gamma,
    Gammaln,
    Digamma,
    Betainc,
    Gammainc,
    Erf,
    Erfinv,
    Jv,
    Kv,
    Hyp1f1,
    Hyp2f1,
    Exp1,
    Ellipk,
    Ellipe,
    Ellipkinc,
    Ellipeinc,
    Elliprf,
    Elliprj,
    EvalLegendre,
    Logsumexp,
}

struct Spec {
    name: &'static str,
    family: &'static str,
    operation: Op,
    low: f64,
    high: f64,
    parameters: &'static str,
}

// Real, finite inputs strictly away from singularities and branch boundaries.
const SPECS: &[Spec] = &[
    Spec {
        name: "gamma",
        family: "gamma",
        operation: Op::Gamma,
        low: 0.5,
        high: 7.5,
        parameters: "gamma(x)",
    },
    Spec {
        name: "gammaln",
        family: "gamma",
        operation: Op::Gammaln,
        low: 0.5,
        high: 18.0,
        parameters: "gammaln(x)",
    },
    Spec {
        name: "digamma",
        family: "gamma",
        operation: Op::Digamma,
        low: 0.5,
        high: 18.0,
        parameters: "digamma(x)",
    },
    Spec {
        name: "betainc",
        family: "beta",
        operation: Op::Betainc,
        low: 0.05,
        high: 0.95,
        parameters: "a=2.5;b=3.5",
    },
    Spec {
        name: "gammainc",
        family: "gamma",
        operation: Op::Gammainc,
        low: 0.2,
        high: 12.0,
        parameters: "a=2.5",
    },
    Spec {
        name: "erf",
        family: "error",
        operation: Op::Erf,
        low: -3.0,
        high: 3.0,
        parameters: "erf(x)",
    },
    Spec {
        name: "erfinv",
        family: "error",
        operation: Op::Erfinv,
        low: -0.9,
        high: 0.9,
        parameters: "erfinv(x)",
    },
    Spec {
        name: "jv",
        family: "bessel",
        operation: Op::Jv,
        low: 0.2,
        high: 16.0,
        parameters: "v=1.5",
    },
    Spec {
        name: "kv",
        family: "bessel",
        operation: Op::Kv,
        low: 0.2,
        high: 16.0,
        parameters: "v=1.5",
    },
    Spec {
        name: "hyp1f1",
        family: "hypergeometric",
        operation: Op::Hyp1f1,
        low: -3.0,
        high: 3.0,
        parameters: "a=1.5;b=2.5",
    },
    Spec {
        name: "hyp2f1",
        family: "hypergeometric",
        operation: Op::Hyp2f1,
        low: -0.6,
        high: 0.6,
        parameters: "a=0.8;b=1.3;c=2.8",
    },
    Spec {
        name: "exp1",
        family: "integrals",
        operation: Op::Exp1,
        low: 0.2,
        high: 12.0,
        parameters: "E1(x);x>0",
    },
    Spec {
        name: "ellipk",
        family: "elliptic",
        operation: Op::Ellipk,
        low: -0.75,
        high: 0.85,
        parameters: "m=x;parameter_not_modulus",
    },
    Spec {
        name: "ellipe",
        family: "elliptic",
        operation: Op::Ellipe,
        low: -0.75,
        high: 0.85,
        parameters: "m=x;parameter_not_modulus",
    },
    Spec {
        name: "ellipkinc",
        family: "elliptic",
        operation: Op::Ellipkinc,
        low: 0.01,
        high: 1.1,
        parameters: "phi=x;m=0.35;radians",
    },
    Spec {
        name: "ellipeinc",
        family: "elliptic",
        operation: Op::Ellipeinc,
        low: 0.01,
        high: 1.1,
        parameters: "phi=x;m=0.35;radians",
    },
    Spec {
        name: "elliprf",
        family: "elliptic",
        operation: Op::Elliprf,
        low: 0.2,
        high: 3.0,
        parameters: "y=x+0.4;z=x+1.2",
    },
    Spec {
        name: "elliprj",
        family: "elliptic",
        operation: Op::Elliprj,
        low: 0.2,
        high: 3.0,
        parameters: "y=x+0.4;z=x+1.2;p=x+2.2",
    },
    Spec {
        name: "eval_legendre",
        family: "orthogonal",
        operation: Op::EvalLegendre,
        low: -0.8,
        high: 0.8,
        parameters: "degree=10",
    },
    Spec {
        name: "logsumexp",
        family: "reduction",
        operation: Op::Logsumexp,
        low: -5.0,
        high: 5.0,
        parameters: "axis=none;unweighted",
    },
];

struct Inputs {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    p: Vec<f64>,
}

fn prepare(spec: &Spec, n: usize) -> Inputs {
    let x: Vec<f64> = (0..n)
        .map(|k| spec.low + (spec.high - spec.low) * (k as f64 / (n - 1) as f64))
        .collect();
    let y = x.iter().map(|&v| v + 0.4).collect();
    let z = x.iter().map(|&v| v + 1.2).collect();
    let p = x.iter().map(|&v| v + 2.2).collect();
    Inputs { x, y, z, p }
}

fn unary(input: &Inputs, func: impl Fn(f64) -> f64) -> Vec<f64> {
    input.x.iter().map(|&x| func(x)).collect()
}

enum Output {
    Values(Vec<f64>),
    Scalar(f64),
}

fn compute(op: Op, i: &Inputs) -> Output {
    let values = match op {
        Op::Gamma => unary(i, special::gamma::gamma),
        Op::Gammaln => unary(i, special::gamma::gammaln),
        Op::Digamma => unary(i, special::gamma::digamma),
        Op::Betainc => unary(i, |x| special::beta::betainc(2.5, 3.5, x)),
        Op::Gammainc => unary(i, |x| special::gamma::gammainc(2.5, x)),
        Op::Erf => unary(i, special::elementary::erf),
        Op::Erfinv => unary(i, special::elementary::erfinv),
        Op::Jv => unary(i, |x| special::bessel::jv(1.5, x)),
        Op::Kv => unary(i, |x| special::bessel::kv(1.5, x)),
        Op::Hyp1f1 => unary(i, |x| special::hypergeometric::hyp1f1(1.5, 2.5, x)),
        Op::Hyp2f1 => unary(i, |x| special::hypergeometric::hyp2f1(0.8, 1.3, 2.8, x)),
        Op::Exp1 => unary(i, special::integrals::exp1),
        Op::Ellipk => unary(i, |x| special::elliptic::ellipk(x).unwrap()),
        Op::Ellipe => unary(i, |x| special::elliptic::ellipe(x).unwrap()),
        Op::Ellipkinc => unary(i, |x| special::elliptic::ellipkinc(x, 0.35).unwrap()),
        Op::Ellipeinc => unary(i, |x| special::elliptic::ellipeinc(x, 0.35).unwrap()),
        Op::Elliprf => {
            i.x.iter()
                .enumerate()
                .map(|(k, &x)| special::elliptic::elliprf(x, i.y[k], i.z[k]).unwrap())
                .collect()
        }
        Op::Elliprj => {
            i.x.iter()
                .enumerate()
                .map(|(k, &x)| special::elliptic::elliprj(x, i.y[k], i.z[k], i.p[k]).unwrap())
                .collect()
        }
        Op::EvalLegendre => unary(i, |x| special::orthogonal::eval_legendre(10, x)),
        Op::Logsumexp => return Output::Scalar(special::elementary::logsumexp(&i.x)),
    };
    Output::Values(values)
}

fn output_summary(out: Output, n: usize) -> Result<(f64, f64, f64), Box<dyn Error>> {
    let values = match out {
        Output::Values(values) => {
            if values.len() != n {
                return Err("incorrect output shape".into());
            }
            values
        }
        Output::Scalar(value) => vec![value],
    };
    if !values.iter().all(|v| v.is_finite()) {
        return Err("nonfinite output".into());
    }
    Ok((
        values[0],
        values[values.len() - 1],
        values.iter().copied().sum(),
    ))
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
    repeat: usize,
    warmup: usize,
    batch: usize,
    quick: bool,
}

fn parse_args() -> Result<(Options, Option<String>), Box<dyn Error>> {
    let mut opts = Options {
        repeat: 13,
        warmup: 3,
        batch: 4,
        quick: false,
    };
    let mut out = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bench" => {}
            "--quick" => opts.quick = true,
            "--repeat" => opts.repeat = args.next().ok_or("missing --repeat")?.parse()?,
            "--warmup" => opts.warmup = args.next().ok_or("missing --warmup")?.parse()?,
            "--batch" => opts.batch = args.next().ok_or("missing --batch")?.parse()?,
            "--out" => out = Some(args.next().ok_or("missing --out")?),
            "--help" | "-h" => {
                println!(
                    "Usage: special_performance_comparison [--quick] [--repeat N] [--warmup N] [--batch N] [--out FILE]"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unrecognized argument: {arg}").into()),
        }
    }
    if !(1..=100).contains(&opts.repeat) || opts.warmup > 100 || !(1..=64).contains(&opts.batch) {
        return Err("repeat must be 1..=100; warmup 0..=100; batch 1..=64".into());
    }
    Ok((opts, out))
}

struct Timing {
    median: u128,
    p25: u128,
    p75: u128,
    min: u128,
}

fn measure(spec: &Spec, n: usize, phase: Phase, opts: &Options) -> Timing {
    let mut times = Vec::with_capacity(opts.repeat);
    for iteration in 0..opts.warmup + opts.repeat {
        let elapsed = if phase == Phase::Prepare {
            let start = Instant::now();
            for _ in 0..opts.batch {
                black_box(prepare(black_box(spec), black_box(n)));
            }
            start.elapsed().as_nanos()
        } else {
            let inputs = prepare(spec, n);
            let start = Instant::now();
            for _ in 0..opts.batch {
                black_box(compute(spec.operation, black_box(&inputs)));
            }
            start.elapsed().as_nanos()
        };
        if iteration >= opts.warmup {
            times.push(elapsed / opts.batch as u128);
        }
    }
    times.sort_unstable();
    let mid = times.len() / 2;
    Timing {
        median: if times.len() % 2 == 0 {
            (times[mid - 1] + times[mid]) / 2
        } else {
            times[mid]
        },
        p25: times[times.len() / 4],
        p75: times[3 * times.len() / 4],
        min: times[0],
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let (opts, out) = parse_args()?;
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
        "implementation,case,family,phase,execution_mode,n,parameters,input_profile,repetitions,batch_calls,median_ns,p25_ns,p75_ns,min_ns,comparability,difference,output_layout,first_value,last_value,sum_value"
    )?;
    let sizes = if opts.quick { &SIZES[..1] } else { SIZES };
    for spec in SPECS {
        for &n in sizes {
            let (first, last, sum) = output_summary(compute(spec.operation, &prepare(spec, n)), n)
                .map_err(|e| format!("{} ({n}) failed verification: {e}", spec.name))?;
            for phase in [Phase::Prepare, Phase::Compute] {
                let t = measure(spec, n, phase, &opts);
                let (mode, comparable, difference, layout) = if phase == Phase::Prepare {
                    (
                        "prepare",
                        "adapted",
                        "rust_vec_vs_numpy_ndarray_preparation",
                        "prepared_four_vecs_vs_four_arrays",
                    )
                } else if matches!(spec.operation, Op::Logsumexp) {
                    (
                        "slice_reduction",
                        "matched",
                        "different_reduction_dispatch_and_order",
                        "scalar_f64_vs_numpy_scalar",
                    )
                } else {
                    (
                        "scalar_loop",
                        "adapted",
                        "rust_scalar_loop_vs_scipy_vectorized_ufunc",
                        "vec_f64_vs_numpy_ndarray",
                    )
                };
                writeln!(
                    output,
                    "sci_rust_0.3.0,{},{},{},{},{},{},linear_grid_v1,{},{},{},{},{},{},{},{},{},{:.17e},{:.17e},{:.17e}",
                    spec.name,
                    spec.family,
                    phase.as_str(),
                    mode,
                    n,
                    spec.parameters,
                    opts.repeat,
                    opts.batch,
                    t.median,
                    t.p25,
                    t.p75,
                    t.min,
                    comparable,
                    difference,
                    layout,
                    first,
                    last,
                    sum
                )?;
            }
        }
    }
    Ok(())
}
