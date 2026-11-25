//! Standalone, single-thread Sobol performance and accuracy comparison.
//!
//! Run (only with exclusive Cargo ownership):
//!   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
//!     cargo run --release --bench sobol_comparison --no-default-features --features stats \
//!     -- --quick --out target/sobol_baseline.csv
//!   # Omit --quick for the full dimension/bit-depth matrix.
//!
//! Benchmarks use the same n, d, bits and seed within each Sci pair. Euto's
//! handwritten 6D/32-bit design always produces 64 points with a digital shift
//! and half-bin offset; Sci's seeded generator uses LMS + digital shift. They
//! intentionally differ in scramble method and row storage (arrays vs Vecs).
//! SciPy's independent UNSCRAMBLED reference is checked before any timing via
//! the checked-in cases/expected CSVs. SciPy runtime timings can be measured
//! separately with the same parameters; no Python process runs here.
//! `draw_random_flat_only` uses a contiguous row-major output buffer and is
//! directly comparable with the existing `draw_random_only` workload.

use sci::stats::qmc::Sobol;
use std::env;
use std::fs::File;
use std::hint::black_box;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

const SEED: u64 = 65_537;
const MAX_VALUES: usize = 1_048_576;
const EUTO_DIM: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Algorithm {
    SciPlain,
    SciLms,
    EutoShift,
}

impl Algorithm {
    fn name(self) -> &'static str {
        match self {
            Self::SciPlain | Self::SciLms => "sci_rust_0.3.0",
            Self::EutoShift => "euto_handwritten_rust",
        }
    }

    fn scramble(self) -> &'static str {
        match self {
            Self::SciPlain => "none",
            Self::SciLms => "lms_plus_shift",
            Self::EutoShift => "digital_shift_plus_half_bin",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Operation {
    Construct,
    Random,
    Base2,
    Combined,
    Reset,
    FastForward,
    Flat,
}

impl Operation {
    fn name(self) -> &'static str {
        match self {
            Self::Construct => "construct_only",
            Self::Random => "draw_random_only",
            Self::Base2 => "draw_random_base2_only",
            Self::Combined => "construct_plus_draw",
            Self::Reset => "reset_only",
            Self::FastForward => "fast_forward_only",
            Self::Flat => "draw_random_flat_only",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Case {
    algorithm: Algorithm,
    operation: Operation,
    dimension: usize,
    bits: usize,
    n: usize,
    skip: usize,
}

fn sci_new(case: Case) -> Sobol {
    match case.algorithm {
        Algorithm::SciPlain => Sobol::new(case.dimension, case.bits).unwrap(),
        Algorithm::SciLms => Sobol::scrambled(case.dimension, case.bits, black_box(SEED)).unwrap(),
        Algorithm::EutoShift => unreachable!("Euto has its own generator"),
    }
}

// This is Euto's private sobol64 direction initialization, digital shift and
// 64-point drawing algorithm, decomposed solely to measure setup and draw apart.
struct EutoPrepared {
    directions: [[u32; 32]; EUTO_DIM],
    shift: [u32; EUTO_DIM],
}

impl EutoPrepared {
    fn new(seed: u64) -> Self {
        const S: [usize; EUTO_DIM] = [0, 1, 2, 3, 3, 4];
        const A: [u32; EUTO_DIM] = [0, 0, 1, 1, 2, 1];
        const M: [[u32; 4]; EUTO_DIM] = [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 3, 0, 0],
            [1, 3, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 3, 3],
        ];
        let mut directions = [[0u32; 32]; EUTO_DIM];
        for d in 0..EUTO_DIM {
            if d == 0 {
                for (j, item) in directions[d].iter_mut().enumerate() {
                    *item = 1u32 << (31 - j);
                }
                continue;
            }
            for j in 0..S[d] {
                directions[d][j] = M[d][j] << (31 - j);
            }
            for j in S[d]..32 {
                let mut v = directions[d][j - S[d]] ^ (directions[d][j - S[d]] >> S[d]);
                for k in 1..S[d] {
                    if (A[d] >> (S[d] - 1 - k)) & 1 != 0 {
                        v ^= directions[d][j - k];
                    }
                }
                directions[d][j] = v;
            }
        }
        let mut shift = [0u32; EUTO_DIM];
        let mut rng = seed;
        for s in &mut shift {
            rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *s = (z ^ (z >> 31)) as u32;
        }
        Self { directions, shift }
    }

    fn draw64(&self) -> Vec<[f64; EUTO_DIM]> {
        let mut state = [0u32; EUTO_DIM];
        let mut points = Vec::with_capacity(64);
        for idx in 0..64usize {
            if idx > 0 {
                let bit = (idx as u32).trailing_zeros() as usize;
                for (d, value) in state.iter_mut().enumerate() {
                    *value ^= self.directions[d][bit];
                }
            }
            points.push(std::array::from_fn(|d| {
                ((state[d] ^ self.shift[d]) as f64 + 0.5) / 4_294_967_296.0
            }));
        }
        points
    }
}

fn cases(full: bool) -> Vec<Case> {
    let mut cases = Vec::new();
    let mut append = |algorithm, dimension, bits, n, skip| {
        for operation in [
            Operation::Construct,
            Operation::Random,
            Operation::Base2,
            Operation::Combined,
            Operation::Reset,
            Operation::FastForward,
        ] {
            let case = Case {
                algorithm,
                operation,
                dimension,
                bits,
                n,
                skip,
            };
            if !cases.contains(&case) {
                cases.push(case);
            }
        }
    };
    // All shared Sci workloads are strictly identical in d, bits, n and skip.
    for bits in [30, 32] {
        for n in [64, 1_024, 16_384] {
            for algorithm in [Algorithm::SciPlain, Algorithm::SciLms] {
                append(algorithm, 6, bits, n, 65_536);
            }
        }
    }
    if full {
        for (dimension, count) in [
            (1, 16_384),
            (2, 16_384),
            (16, 4_096),
            (64, 1_024),
            (256, 256),
            (1_024, 64),
        ] {
            for algorithm in [Algorithm::SciPlain, Algorithm::SciLms] {
                append(algorithm, dimension, 32, count, 65_536);
            }
        }
        for bits in [8, 52, 64] {
            for algorithm in [Algorithm::SciPlain, Algorithm::SciLms] {
                let count = if bits == 8 { 64 } else { 4_096 };
                let skip = if bits == 8 { 128 } else { 65_536 };
                append(algorithm, 6, bits, count, skip);
            }
        }
    }
    for operation in [Operation::Construct, Operation::Random, Operation::Combined] {
        cases.push(Case {
            algorithm: Algorithm::EutoShift,
            operation,
            dimension: EUTO_DIM,
            bits: 32,
            n: 64,
            skip: 0,
        });
    }
    // Append contiguous-output counterparts after all original cases, so the
    // 183 pre/post workloads and their execution order remain unchanged.
    let flat_cases: Vec<_> = cases
        .iter()
        .filter(|case| {
            case.operation == Operation::Random && case.algorithm != Algorithm::EutoShift
        })
        .map(|&case| Case {
            operation: Operation::Flat,
            ..case
        })
        .collect();
    cases.extend(flat_cases);
    cases
}

fn validate_scipy_fixtures() -> Result<(), String> {
    let inputs = include_str!("../tests/scipy/sobol_cases.csv");
    let expected = include_str!("../tests/scipy/sobol_expected.csv");
    let a = inputs.lines().skip(1);
    let b = expected.lines().skip(1);
    let mut checked = 0;
    for (test, ref_line) in a.zip(b) {
        let c: Vec<_> = test.split(',').collect();
        let reference: Vec<_> = ref_line.split(',').collect();
        if c.len() != 6 || reference.len() != 4 || c[0] != reference[0] {
            return Err(format!("bad SciPy reference record: {test}"));
        }
        let parse = |v: &str| v.parse::<usize>().map_err(|e| e.to_string());
        let dimension = parse(c[1])?;
        let mut sobol = Sobol::new(dimension, parse(c[2])?).map_err(|e| e.to_string())?;
        sobol
            .fast_forward(parse(c[4])?)
            .map_err(|e| e.to_string())?;
        let points = match c[3] {
            "random" => sobol.random(parse(c[5])?),
            "base2" => sobol.random_base2(parse(c[5])?),
            _ => return Err(format!("bad SciPy case operation: {test}")),
        }
        .map_err(|e| e.to_string())?;
        if points.len() != parse(reference[1])? || dimension != parse(reference[2])? {
            return Err(format!("SciPy reference shape mismatch: {}", c[0]));
        }
        let desired: Vec<f64> = reference[3]
            .split(';')
            .map(|s| s.parse::<f64>().map_err(|e| e.to_string()))
            .collect::<Result<_, _>>()?;
        let actual: Vec<f64> = points.into_iter().flatten().collect();
        if actual != desired {
            return Err(format!("SciPy Sobol accuracy mismatch: {}", c[0]));
        }
        checked += 1;
    }
    if checked != 29 {
        return Err(format!("expected 29 SciPy fixtures, checked {checked}"));
    }
    Ok(())
}

fn validate_euto() -> Result<(), String> {
    let prepared = EutoPrepared::new(SEED);
    let points = prepared.draw64();
    if points != EutoPrepared::new(SEED).draw64() || points == EutoPrepared::new(SEED + 1).draw64()
    {
        return Err("Euto seeds lost reproducibility".into());
    }
    let plain = Sobol::new(6, 32).unwrap().random_base2(6).unwrap();
    for (i, row) in points.iter().enumerate() {
        for (axis, &value) in row.iter().enumerate() {
            if !(0.0..1.0).contains(&value) {
                return Err("Euto coordinate left [0,1)".into());
            }
            let shifted = (value * 4_294_967_296.0 - 0.5) as u32;
            let raw = shifted ^ prepared.shift[axis];
            if raw as f64 / 4_294_967_296.0 != plain[i][axis] {
                return Err(format!(
                    "Euto/Sci unscrambled directions differ: [{i},{axis}]"
                ));
            }
        }
    }
    Ok(())
}

// Each invocation measures the selected operation only. The constructor for
// draw/reset/skip and positioning before reset are outside the timed interval.
// Instant and black_box overhead affect short operations equally. There is no
// background worker, Rayon pool, Python callback or BLAS work in this target.
fn measure_once(case: Case) -> f64 {
    if case.algorithm == Algorithm::EutoShift {
        match case.operation {
            Operation::Construct => {
                let start = Instant::now();
                black_box(EutoPrepared::new(black_box(SEED)));
                return start.elapsed().as_nanos() as f64;
            }
            Operation::Random => {
                let prepared = EutoPrepared::new(SEED);
                let start = Instant::now();
                black_box(prepared.draw64());
                return start.elapsed().as_nanos() as f64;
            }
            Operation::Combined => {
                let start = Instant::now();
                black_box(EutoPrepared::new(black_box(SEED)).draw64());
                return start.elapsed().as_nanos() as f64;
            }
            Operation::Flat => unreachable!("Euto has no random_flat method"),
            _ => unreachable!("Euto exposes only a 64-point design"),
        }
    }
    match case.operation {
        Operation::Construct => {
            let start = Instant::now();
            black_box(sci_new(case));
            start.elapsed().as_nanos() as f64
        }
        Operation::Random => {
            let mut sobol = sci_new(case);
            let start = Instant::now();
            black_box(sobol.random(black_box(case.n)).unwrap());
            start.elapsed().as_nanos() as f64
        }
        Operation::Flat => {
            let mut sobol = sci_new(case);
            let start = Instant::now();
            black_box(sobol.random_flat(black_box(case.n)).unwrap());
            start.elapsed().as_nanos() as f64
        }
        Operation::Base2 => {
            let mut sobol = sci_new(case);
            let start = Instant::now();
            black_box(
                sobol
                    .random_base2(black_box(case.n.trailing_zeros() as usize))
                    .unwrap(),
            );
            start.elapsed().as_nanos() as f64
        }
        Operation::Combined => {
            let start = Instant::now();
            let mut sobol = sci_new(case);
            black_box(sobol.random(black_box(case.n)).unwrap());
            start.elapsed().as_nanos() as f64
        }
        Operation::Reset => {
            let mut sobol = sci_new(case);
            sobol.fast_forward(case.skip).unwrap();
            let start = Instant::now();
            sobol.reset();
            black_box(sobol.num_generated());
            start.elapsed().as_nanos() as f64
        }
        Operation::FastForward => {
            let mut sobol = sci_new(case);
            let start = Instant::now();
            sobol.fast_forward(black_box(case.skip)).unwrap();
            black_box(sobol.num_generated());
            start.elapsed().as_nanos() as f64
        }
    }
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    let p = (sorted.len() - 1) as f64 * fraction;
    let left = p.floor() as usize;
    let right = p.ceil() as usize;
    sorted[left] + (sorted[right] - sorted[left]) * p.fract()
}

struct Options {
    full: bool,
    rounds: usize,
    target_ns: f64,
    max_reps: usize,
    output: Option<PathBuf>,
}

fn options() -> Result<Options, String> {
    let mut result = Options {
        full: true,
        rounds: 9,
        target_ns: 5_000_000.0,
        max_reps: 256,
        output: None,
    };
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bench" => {} // Cargo's bench subcommand appends this for harness=false.
            "--quick" => result.full = false,
            "--full" => result.full = true,
            "--rounds" => {
                result.rounds = args
                    .next()
                    .ok_or("missing --rounds value")?
                    .parse()
                    .map_err(|_| "bad rounds")?
            }
            "--target-ms" => {
                let ms: f64 = args
                    .next()
                    .ok_or("missing --target-ms value")?
                    .parse()
                    .map_err(|_| "bad target-ms")?;
                result.target_ns = ms * 1_000_000.0;
            }
            "--max-reps" => {
                result.max_reps = args
                    .next()
                    .ok_or("missing --max-reps value")?
                    .parse()
                    .map_err(|_| "bad max-reps")?
            }
            "--out" => {
                result.output = Some(PathBuf::from(args.next().ok_or("missing --out path")?))
            }
            "--help" | "-h" => {
                println!(
                    "Usage: sobol_comparison [--quick|--full] [--rounds 9] [--target-ms 5] [--max-reps 256] [--out FILE.csv]"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unknown option: {arg}")),
        }
    }
    if !(3..=101).contains(&result.rounds)
        || !result.target_ns.is_finite()
        || !(100_000.0..=200_000_000.0).contains(&result.target_ns)
        || !(1..=10_000).contains(&result.max_reps)
    {
        return Err("rounds must be 3..101, target-ms 0.1..200, max-reps 1..10000".into());
    }
    Ok(result)
}

fn run() -> Result<(), String> {
    let opt = options()?;
    validate_scipy_fixtures()?;
    validate_euto()?;
    let out: Box<dyn Write> = match opt.output {
        Some(path) => Box::new(BufWriter::new(
            File::create(path).map_err(|e| e.to_string())?,
        )),
        None => Box::new(BufWriter::new(io::stdout())),
    };
    let mut out = out;
    writeln!(out, "implementation,operation,dimensions,bits,scramble,points,skipped,seed,threads,rounds,reps_per_round,median_ns,p05_ns,p95_ns,ns_per_coordinate").map_err(|e| e.to_string())?;
    let matrix = cases(opt.full);
    eprintln!(
        "Sobol: 29 SciPy fixture checks + Euto digital-shift parity passed; {} cases (1 Rust execution thread)",
        matrix.len()
    );
    let count = matrix.len();
    for (index, case) in matrix.into_iter().enumerate() {
        if case.n * case.dimension > MAX_VALUES {
            return Err(format!("oversized Sobol benchmark case: {case:?}"));
        }
        let warmup: Vec<_> = (0..3).map(|_| measure_once(case)).collect();
        let pilot_ns = percentile(
            &{
                let mut values = warmup;
                values.sort_by(f64::total_cmp);
                values
            },
            0.5,
        )
        .max(1.0);
        let reps = ((opt.target_ns / pilot_ns).round() as usize).clamp(1, opt.max_reps);
        let mut samples = Vec::with_capacity(opt.rounds);
        for _ in 0..opt.rounds {
            let mut total = 0.0;
            for _ in 0..reps {
                total += measure_once(case);
            }
            samples.push(total / reps as f64);
        }
        samples.sort_by(f64::total_cmp);
        let median = percentile(&samples, 0.5);
        let p05 = percentile(&samples, 0.05);
        let p95 = percentile(&samples, 0.95);
        let values = match case.operation {
            Operation::Random | Operation::Base2 | Operation::Combined | Operation::Flat => {
                case.n * case.dimension
            }
            _ => 0,
        };
        let normalized = if values > 0 {
            median / values as f64
        } else {
            0.0
        };
        writeln!(
            out,
            "{},{},{},{},{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.6}",
            case.algorithm.name(),
            case.operation.name(),
            case.dimension,
            case.bits,
            case.algorithm.scramble(),
            case.n,
            case.skip,
            SEED,
            1,
            opt.rounds,
            reps,
            median,
            p05,
            p95,
            normalized,
        )
        .map_err(|e| e.to_string())?;
        if index % 25 == 0 {
            eprintln!("Completed {} / {} cases", index + 1, count);
        }
    }
    out.flush().map_err(|e| e.to_string())?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Sobol comparison failed: {error}");
        std::process::exit(1);
    }
}
