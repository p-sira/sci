use super::*;
use statrs::distribution::*;

/// Native univariate distribution parameters for the functional probability APIs.
///
/// Parameters use f64, including integer counts, and are validated on evaluation.
/// All shapes and counts are at most 1,048,576; hypergeometric population is at
/// most 65,536 because its backend sums probabilities. Location/scale parameters
/// must be finite; scales are positive. There is no implicit loc/scale transform.
/// Unsupported SciPy distributions and methods are listed in STATISTICS_COVERAGE.md.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Distribution {
    /// Normal: `exp(−(x−μ)²/(2σ²))/(σ√(2π)) on R`. Finite mean μ and positive standard deviation σ.
    Normal {
        /// Mean, as specified in the variant definition.
        mean: f64,
        /// Std dev, as specified in the variant definition.
        std_dev: f64,
    },
    /// Beta: `x^(a−1)(1−x)^(b−1)/B(a,b) on [0,1]`. Positive shapes a and b.
    Beta {
        /// A, as specified in the variant definition.
        a: f64,
        /// B, as specified in the variant definition.
        b: f64,
    },
    /// Gamma: `x^(a−1)exp(−x/θ)/(Γ(a)θ^a) on [0,∞)`. Positive shape a and scale θ; scale rather than statrs rate.
    Gamma {
        /// Shape value, as specified in the variant definition.
        shape_value: f64,
        /// Scale, as specified in the variant definition.
        scale: f64,
    },
    /// ChiSquared: `x^(ν/2−1)exp(−x/2)/(2^(ν/2)Γ(ν/2)) on [0,∞)`. Positive degrees of freedom ν.
    ChiSquared {
        /// Df, as specified in the variant definition.
        df: f64,
    },
    /// Chi: `2^(1−ν/2)x^(ν−1)exp(−x²/2)/Γ(ν/2) on [0,∞)`. Positive integer degrees of freedom ν.
    Chi {
        /// Df, as specified in the variant definition.
        df: f64,
    },
    /// StudentT: `Γ((ν+1)/2)(1+x²/ν)^(-(ν+1)/2)/(√(πν)Γ(ν/2)) on R`. Positive degrees of freedom; standard location zero and scale one.
    StudentT {
        /// Df, as specified in the variant definition.
        df: f64,
    },
    /// F: `a^(a/2)b^(b/2)x^(a/2−1)/(B(a/2,b/2)(b+a*x)^((a+b)/2)) on [0,∞)`. Positive numerator and denominator degrees of freedom.
    F {
        /// Numerator df, as specified in the variant definition.
        numerator_df: f64,
        /// Denominator df, as specified in the variant definition.
        denominator_df: f64,
    },
    /// Exponential: `exp(−x/θ)/θ on [0,∞)`. Positive scale θ; reciprocal of statrs rate.
    Exponential {
        /// Scale, as specified in the variant definition.
        scale: f64,
    },
    /// Cauchy: `1/(πγ(1+((x−x₀)/γ)²)) on R`. Finite location x₀ and positive scale γ.
    Cauchy {
        /// Location, as specified in the variant definition.
        location: f64,
        /// Scale, as specified in the variant definition.
        scale: f64,
    },
    /// Laplace: `exp(−|x−μ|/b)/(2b) on R`. Finite location μ and positive scale b.
    Laplace {
        /// Location, as specified in the variant definition.
        location: f64,
        /// Scale, as specified in the variant definition.
        scale: f64,
    },
    /// LogNormal: `exp(−(ln(x)−μ)²/(2σ²))/(xσ√(2π)) on (0,∞)`. Finite log-mean μ and positive log-standard-deviation σ; SciPy s=σ and scale=exp(μ).
    LogNormal {
        /// Log mean, as specified in the variant definition.
        log_mean: f64,
        /// Log std, as specified in the variant definition.
        log_std: f64,
    },
    /// Uniform: `1/(high−low) on [low,high]`. Finite endpoints with high>low and finite width; SciPy loc=low, scale=high−low.
    Uniform {
        /// Low, as specified in the variant definition.
        low: f64,
        /// High, as specified in the variant definition.
        high: f64,
    },
    /// Weibull: `(k/λ)(x/λ)^(k−1)exp(−(x/λ)^k) on [0,∞)`. Positive shape k and scale λ; SciPy weibull_min.
    Weibull {
        /// Shape value, as specified in the variant definition.
        shape_value: f64,
        /// Scale, as specified in the variant definition.
        scale: f64,
    },
    /// Pareto: `a*m^a/x^(a+1) on [m,∞)`. Positive minimum m and shape a; SciPy pareto(a,scale=m).
    Pareto {
        /// Minimum, as specified in the variant definition.
        minimum: f64,
        /// Shape value, as specified in the variant definition.
        shape_value: f64,
    },
    /// InverseGamma: `θ^a*x^(−a−1)exp(−θ/x)/Γ(a) on (0,∞)`. Positive shape a and scale θ, matching SciPy invgamma.
    InverseGamma {
        /// Shape value, as specified in the variant definition.
        shape_value: f64,
        /// Scale, as specified in the variant definition.
        scale: f64,
    },
    /// Gumbel: `exp(−z−exp(−z))/β, z=(x−μ)/β, on R`. Finite location μ and positive scale β; SciPy gumbel_r.
    Gumbel {
        /// Location, as specified in the variant definition.
        location: f64,
        /// Scale, as specified in the variant definition.
        scale: f64,
    },
    /// Triangular: `2(x−low)/((high−low)(mode−low)) below mode; mirrored above mode`. Finite low≤mode≤high, low<high; SciPy triang c=(mode−low)/(high−low).
    Triangular {
        /// Low, as specified in the variant definition.
        low: f64,
        /// High, as specified in the variant definition.
        high: f64,
        /// Mode, as specified in the variant definition.
        mode: f64,
    },
    /// Erlang: gamma density with positive integer shape k and scale theta.
    Erlang {
        /// Positive integer shape, up to 1048576.
        shape_value: f64,
        /// Positive finite scale (reciprocal statrs rate).
        scale: f64,
    },
    /// Integer uniform mass 1/(high-low), low through high-1 (SciPy randint).
    DiscreteUniform {
        /// Inclusive integer lower bound, absolute value at most 1048576.
        low: f64,
        /// Exclusive integer upper bound, absolute value at most 1048576.
        high: f64,
    },
    /// Bernoulli: `p^k(1−p)^(1−k), k=0,1`. Probability p in \[0,1\].
    Bernoulli {
        /// Probability, as specified in the variant definition.
        probability: f64,
    },
    /// Binomial: `C(n,k)p^k(1−p)^(n−k), k=0,…,n`. Integer trials n≥0 and probability p in \[0,1\].
    Binomial {
        /// Trials, as specified in the variant definition.
        trials: f64,
        /// Probability, as specified in the variant definition.
        probability: f64,
    },
    /// Poisson: `exp(−λ)λ^k/k!, k≥0`. Positive rate λ; the zero-rate degenerate case is handled directly.
    Poisson {
        /// Rate, as specified in the variant definition.
        rate: f64,
    },
    /// Geometric: `p(1−p)^(k−1), k≥1`. Probability 0<p≤1; counts trials including the success, matching SciPy geom.
    Geometric {
        /// Probability, as specified in the variant definition.
        probability: f64,
    },
    /// NegativeBinomial: `Γ(k+r)/(k!Γ(r))*p^r*(1−p)^k, k≥0`. Positive successes r (need not be integer) and 0<p≤1; counts failures.
    NegativeBinomial {
        /// Successes, as specified in the variant definition.
        successes: f64,
        /// Probability, as specified in the variant definition.
        probability: f64,
    },
    /// Hypergeometric: `C(K,k)C(N−K,n−k)/C(N,n)`. Integer population N, successes K, and draws n, with K,n≤N; N≤65536.
    Hypergeometric {
        /// Population, as specified in the variant definition.
        population: f64,
        /// Successes, as specified in the variant definition.
        successes: f64,
        /// Draws, as specified in the variant definition.
        draws: f64,
    },
}

#[derive(Clone, Copy)]
pub(super) enum Operation {
    Density,
    LogDensity,
    Cdf,
    Sf,
    Ppf,
}

fn make<T, E>(value: std::result::Result<T, E>) -> Result<T> {
    value.map_err(|_| Error::InvalidInput("invalid distribution parameters"))
}
fn shape(x: f64) -> Result<f64> {
    positive(x)?;
    if x > MAX_SAMPLES as f64 {
        Err(Error::InvalidInput("shape exceeds 1048576"))
    } else {
        Ok(x)
    }
}
fn reciprocal(x: f64) -> Result<f64> {
    positive(x)?;
    finite(1. / x)
}
fn count(x: f64) -> Result<u64> {
    if x.is_finite() && (0. ..=MAX_SAMPLES as f64).contains(&x) && x.fract() == 0. {
        Ok(x as u64)
    } else {
        Err(Error::InvalidInput(
            "count must be an integer in [0,1048576]",
        ))
    }
}

fn normal_value(mean: f64, std_dev: f64, x: f64, op: Operation) -> Result<f64> {
    normal(mean, std_dev)?;
    let standard = Normal::standard();
    if matches!(op, Operation::Ppf) {
        let z = crate::special::ndtri(x);
        let value = std_dev.mul_add(z, mean);
        return if x == 0. || x == 1. {
            Ok(value)
        } else {
            finite(value)
        };
    }
    let delta = x - mean;
    let z = if delta.is_finite() || x.is_infinite() {
        delta / std_dev
    } else {
        // Opposite finite endpoints can have an overflowing difference while
        // their standardized separation remains finite.
        x / std_dev - mean / std_dev
    };
    match op {
        Operation::Cdf => finite(crate::special::ndtr(z)),
        Operation::Sf => finite(crate::special::ndtr(-z)),
        Operation::LogDensity => Ok(standard.ln_pdf(z) - std_dev.ln()),
        Operation::Density => {
            let density = standard.pdf(z) / std_dev;
            // A standardized tail can underflow before division by a tiny
            // scale restores a representable density. Log space also confirms
            // whether an overflowing quotient is a true density overflow.
            finite(if density == 0. || density.is_infinite() {
                (standard.ln_pdf(z) - std_dev.ln()).exp()
            } else {
                density
            })
        }
        Operation::Ppf => unreachable!(),
    }
}

pub(super) fn evaluate(distribution: Distribution, x: f64, op: Operation) -> Result<f64> {
    if x.is_nan() {
        return Err(Error::InvalidInput("distribution argument must not be NaN"));
    }
    if matches!(op, Operation::Ppf) {
        probability_check(x)?;
    }
    if let Distribution::NegativeBinomial {
        successes,
        probability: 1.,
    } = distribution
    {
        shape(successes)?;
        return evaluate(Distribution::Poisson { rate: 0. }, x, op);
    }
    let value = match distribution {
        Distribution::Normal { mean, std_dev } => normal_value(mean, std_dev, x, op)?,
        Distribution::Beta { a, b } => continuous(&make(Beta::new(shape(a)?, shape(b)?))?, x, op)?,
        Distribution::Gamma { shape_value, scale } => continuous(
            &make(Gamma::new(shape(shape_value)?, reciprocal(scale)?))?,
            x,
            op,
        )?,
        Distribution::ChiSquared { df } => continuous(&make(ChiSquared::new(shape(df)?))?, x, op)?,
        Distribution::Chi { df } => continuous(&make(Chi::new(count(df)?))?, x, op)?,
        Distribution::StudentT { df } => {
            let d = make(StudentsT::new(0., 1., shape(df)?))?;
            if matches!(op, Operation::Ppf) && x == 0.5 {
                0.
            } else {
                continuous(&d, x, op)?
            }
        }
        Distribution::F {
            numerator_df,
            denominator_df,
        } => continuous(
            &make(FisherSnedecor::new(
                shape(numerator_df)?,
                shape(denominator_df)?,
            ))?,
            x,
            op,
        )?,
        Distribution::Exponential { scale } => {
            continuous(&make(Exp::new(reciprocal(scale)?))?, x, op)?
        }
        Distribution::Cauchy { location, scale } => continuous(
            &{
                finite(location)?;
                positive(scale)?;
                make(Cauchy::new(location, scale))?
            },
            x,
            op,
        )?,
        Distribution::Laplace { location, scale } => continuous(
            &{
                finite(location)?;
                positive(scale)?;
                make(Laplace::new(location, scale))?
            },
            x,
            op,
        )?,
        Distribution::LogNormal { log_mean, log_std } => continuous(
            &{
                finite(log_mean)?;
                positive(log_std)?;
                make(LogNormal::new(log_mean, log_std))?
            },
            x,
            op,
        )?,
        Distribution::Uniform { low, high } => continuous(
            &{
                finite(low)?;
                finite(high)?;
                positive(high - low)?;
                make(Uniform::new(low, high))?
            },
            x,
            op,
        )?,
        Distribution::Weibull { shape_value, scale } => continuous(
            &{
                positive(scale)?;
                make(Weibull::new(shape(shape_value)?, scale))?
            },
            x,
            op,
        )?,
        Distribution::Pareto {
            minimum,
            shape_value,
        } => continuous(
            &{
                positive(minimum)?;
                make(Pareto::new(minimum, shape(shape_value)?))?
            },
            x,
            op,
        )?,
        Distribution::InverseGamma { shape_value, scale } => continuous(
            &{
                positive(scale)?;
                make(InverseGamma::new(shape(shape_value)?, scale))?
            },
            x,
            op,
        )?,
        Distribution::Gumbel { location, scale } => continuous(
            &{
                finite(location)?;
                positive(scale)?;
                make(Gumbel::new(location, scale))?
            },
            x,
            op,
        )?,
        Distribution::Triangular { low, high, mode } => continuous(
            &{
                finite(low)?;
                finite(high)?;
                finite(mode)?;
                positive(high - low)?;
                make(Triangular::new(low, high, mode))?
            },
            x,
            op,
        )?,
        Distribution::Erlang { shape_value, scale } => {
            let k = count(shape_value)?;
            if k == 0 {
                return Err(Error::InvalidInput("Erlang shape must be positive"));
            }
            continuous(&make(Gamma::new(k as f64, reciprocal(scale)?))?, x, op)?
        }
        Distribution::DiscreteUniform { low, high } => {
            if !low.is_finite()
                || !high.is_finite()
                || low.fract() != 0.
                || high.fract() != 0.
                || low.abs() > MAX_SAMPLES as f64
                || high.abs() > MAX_SAMPLES as f64
                || low >= high
            {
                return Err(Error::InvalidInput("invalid integer uniform bounds"));
            }
            let n = high - low;
            match op {
                Operation::Density => {
                    if x >= low && x < high && x.fract() == 0. {
                        1. / n
                    } else {
                        0.
                    }
                }
                Operation::LogDensity => {
                    if x >= low && x < high && x.fract() == 0. {
                        -n.ln()
                    } else {
                        f64::NEG_INFINITY
                    }
                }
                Operation::Cdf => ((x.floor() - low + 1.) / n).clamp(0., 1.),
                Operation::Sf => ((high - 1. - x.floor()) / n).clamp(0., 1.),
                Operation::Ppf => {
                    if x == 0. {
                        low - 1.
                    } else {
                        (low + (x * n).ceil() - 1.).clamp(low, high - 1.)
                    }
                }
            }
        }
        Distribution::Bernoulli { probability } => discrete(
            &{
                probability_check(probability)?;
                make(Bernoulli::new(probability))?
            },
            x,
            op,
        )?,
        Distribution::Binomial {
            trials,
            probability,
        } => discrete(
            &{
                probability_check(probability)?;
                make(Binomial::new(probability, count(trials)?))?
            },
            x,
            op,
        )?,
        Distribution::Poisson { rate: 0. } => match op {
            Operation::Density => {
                if x == 0. {
                    1.
                } else {
                    0.
                }
            }
            Operation::LogDensity => {
                if x == 0. {
                    0.
                } else {
                    f64::NEG_INFINITY
                }
            }
            Operation::Cdf => {
                if x >= 0. {
                    1.
                } else {
                    0.
                }
            }
            Operation::Sf => {
                if x >= 0. {
                    0.
                } else {
                    1.
                }
            }
            Operation::Ppf => {
                if x == 0. {
                    -1.
                } else if x == 1. {
                    f64::INFINITY
                } else {
                    0.
                }
            }
        },
        Distribution::Poisson { rate } => discrete(&make(Poisson::new(shape(rate)?))?, x, op)?,
        Distribution::Geometric { probability } => {
            probability_check(probability)?;
            positive(probability)?;
            let d = make(Geometric::new(probability))?;
            // SciPy retains the nominal unbounded support even when p = 1.
            if matches!(op, Operation::Ppf) && x == 1. {
                f64::INFINITY
            } else {
                discrete(&d, x, op)?
            }
        }
        Distribution::NegativeBinomial {
            successes,
            probability,
        } => discrete(
            &{
                probability_check(probability)?;
                positive(probability)?;
                make(NegativeBinomial::new(shape(successes)?, probability))?
            },
            x,
            op,
        )?,
        Distribution::Hypergeometric {
            population,
            successes,
            draws,
        } => discrete(
            &{
                let n = count(population)?;
                if n > 65536 {
                    return Err(Error::InvalidInput(
                        "hypergeometric population exceeds 65536",
                    ));
                }
                make(Hypergeometric::new(n, count(successes)?, count(draws)?))?
            },
            x,
            op,
        )?,
    };
    if value.is_nan() {
        return Err(Error::NumericalFailure(
            "distribution evaluation returned NaN",
        ));
    }
    if matches!(op, Operation::Cdf | Operation::Sf)
        && (!value.is_finite() || !(0. ..=1.).contains(&value))
    {
        return Err(Error::NumericalFailure(
            "distribution probability outside [0,1]",
        ));
    }
    Ok(value)
}

fn continuous<D: Continuous<f64, f64> + ContinuousCDF<f64, f64>>(
    d: &D,
    x: f64,
    op: Operation,
) -> Result<f64> {
    if x.is_infinite() && matches!(op, Operation::Density | Operation::LogDensity) {
        return Ok(if matches!(op, Operation::Density) {
            0.
        } else {
            f64::NEG_INFINITY
        });
    }
    if matches!(op, Operation::LogDensity) && (x == d.min() || x == d.max()) {
        return Ok(d.pdf(x).ln());
    }
    let value = match op {
        Operation::Density => d.pdf(x),
        Operation::LogDensity => d.ln_pdf(x),
        Operation::Cdf => d.cdf(x),
        Operation::Sf => d.sf(x),
        Operation::Ppf => inverse(d, x)?,
    };
    if matches!(op, Operation::Density) && value.is_infinite() && x != d.min() && x != d.max() {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok(value)
}

// The default statrs inverse uses only 16 bisections. Keep its probability
// evaluation, but bound and refine the inversion to binary64 resolution.
fn inverse<D: ContinuousCDF<f64, f64>>(d: &D, p: f64) -> Result<f64> {
    if p == 0. {
        return Ok(d.min());
    }
    if p == 1. {
        return Ok(d.max());
    }
    let mut low = if d.min().is_finite() { d.min() } else { -1. };
    let mut high = if d.max().is_finite() { d.max() } else { 1. };
    let below = |v: f64| -> Result<bool> {
        let q = if p <= 0.5 { d.cdf(v) } else { d.sf(v) };
        if !q.is_finite() {
            return Err(Error::NumericalFailure(
                "invalid probability during inversion",
            ));
        }
        Ok(if p <= 0.5 { q < p } else { q > 1. - p })
    };
    for _ in 0..2048 {
        if below(low)? {
            break;
        }
        if low <= -f64::MAX {
            return Err(Error::NumericalFailure("quantile outside finite range"));
        }
        low = (low * 2. - 1.).max(-f64::MAX);
    }
    for _ in 0..2048 {
        if !below(high)? {
            break;
        }
        if high >= f64::MAX {
            return Err(Error::NumericalFailure("quantile outside finite range"));
        }
        high = (high * 2. + 1.).min(f64::MAX);
    }
    for _ in 0..2048 {
        let middle = low / 2. + high / 2.;
        if middle == low || middle == high {
            return Ok(middle);
        }
        if below(middle)? {
            low = middle;
        } else {
            high = middle;
        }
    }
    Err(Error::NumericalFailure("quantile iteration limit exceeded"))
}

fn discrete<D: Discrete<u64, f64> + DiscreteCDF<u64, f64>>(
    d: &D,
    x: f64,
    op: Operation,
) -> Result<f64> {
    let lo = d.min();
    let hi = d.max();
    Ok(match op {
        Operation::Ppf => {
            if x == 0. {
                return Ok(lo as f64 - 1.);
            }
            if x == 1. {
                return Ok(if hi == u64::MAX {
                    f64::INFINITY
                } else {
                    hi as f64
                });
            }
            // No backend unbounded integer inversion; bracket within exactly
            // representable integers, then binary search in at most 53 steps.
            let limit = 9_007_199_254_740_991_u64;
            let mut left = lo;
            let mut right = hi.min(1).max(lo);
            let below = |k| {
                if x <= 0.5 {
                    d.cdf(k) < x
                } else {
                    d.sf(k) > 1. - x
                }
            };
            for _ in 0..53 {
                if !below(right) {
                    break;
                }
                right = (right * 2 + 1).min(hi).min(limit);
            }
            if below(right) {
                return Err(Error::NumericalFailure(
                    "discrete quantile exceeds exact integer range",
                ));
            }
            while left < right {
                let mid = left + (right - left) / 2;
                if below(mid) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            left as f64
        }
        Operation::Density | Operation::LogDensity => {
            if !x.is_finite() || x < lo as f64 || x > hi as f64 || x.fract() != 0. {
                if matches!(op, Operation::Density) {
                    0.
                } else {
                    f64::NEG_INFINITY
                }
            } else if x > 9_007_199_254_740_991. {
                return Err(Error::InvalidInput(
                    "observation exceeds exact integer range",
                ));
            } else if matches!(op, Operation::Density) {
                d.pmf(x as u64)
            } else {
                d.ln_pmf(x as u64)
            }
        }
        Operation::Cdf | Operation::Sf => {
            let cdf = matches!(op, Operation::Cdf);
            if x < lo as f64 {
                if cdf { 0. } else { 1. }
            } else if x >= hi as f64 || x == f64::INFINITY {
                if cdf { 1. } else { 0. }
            } else if x > 9_007_199_254_740_991. {
                return Err(Error::InvalidInput(
                    "observation exceeds exact integer range",
                ));
            } else if cdf {
                d.cdf(x.floor() as u64)
            } else {
                d.sf(x.floor() as u64)
            }
        }
    })
}

/// Computes `f(x)=dF(x)/dx; for discrete distributions, P(X=x)`.
///
/// ## Parameters
///
/// `distribution` specifies validated native parameters. `x` is a scalar observation.
/// Parameters, support, and normalization are defined by [`Distribution`].
/// [`DistributionSpec`] also accepts custom, truncated and order-statistic
/// descriptions; their constructors document additional domains and work limits.
///
/// ## Domain
///
/// NaN arguments are errors. Singular boundary densities may return positive infinity; outside support returns zero.
///
/// ## Returns
///
/// Density (units inverse to x), or dimensionless probability mass.
///
/// ## Errors
///
/// Invalid distribution parameters return an input error. NaN backend results
/// and quantiles outside the supported representable range return numerical errors.
///
/// ## Numerical behavior
///
/// Uses statrs privately. Quantiles use bounded bracketing and bisection
/// (at most 6144 continuous evaluations or 106 discrete evaluations).
/// Tail accuracy is limited by the backend probability functions and binary64.
///
/// # Examples
///
/// ```
/// let d = sci_stats::Distribution::Normal { mean: 0., std_dev: 1. };
/// assert!(sci_stats::pdf(d, 0.)? > 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy distributions](https://docs.scipy.org/doc/scipy/reference/stats.html).
pub fn pdf(distribution: impl Into<DistributionSpec>, x: f64) -> Result<f64> {
    composition::evaluate(distribution.into(), x, Operation::Density)
}

/// Computes `ln(f(x)); for discrete distributions, ln(P(X=x))`.
///
/// ## Parameters
///
/// `distribution` specifies validated native parameters. `x` is a scalar observation.
/// Parameters, support, and normalization are defined by [`Distribution`].
/// [`DistributionSpec`] also accepts custom, truncated and order-statistic
/// descriptions; their constructors document additional domains and work limits.
///
/// ## Domain
///
/// NaN arguments are errors. Zero density returns negative infinity; singular density may return positive infinity.
///
/// ## Returns
///
/// Natural logarithm of density or probability mass.
///
/// ## Errors
///
/// Invalid distribution parameters return an input error. NaN backend results
/// and quantiles outside the supported representable range return numerical errors.
///
/// ## Numerical behavior
///
/// Uses statrs privately. Quantiles use bounded bracketing and bisection
/// (at most 6144 continuous evaluations or 106 discrete evaluations).
/// Tail accuracy is limited by the backend probability functions and binary64.
///
/// # Examples
///
/// ```
/// let d = sci_stats::Distribution::Normal { mean: 0., std_dev: 1. };
/// assert!(sci_stats::logpdf(d, 0.)? < 0.);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy distributions](https://docs.scipy.org/doc/scipy/reference/stats.html).
pub fn logpdf(distribution: impl Into<DistributionSpec>, x: f64) -> Result<f64> {
    composition::evaluate(distribution.into(), x, Operation::LogDensity)
}

/// Computes `F(x)=P(X≤x)`.
///
/// ## Parameters
///
/// `distribution` specifies validated native parameters. `x` is a scalar observation.
/// Parameters, support, and normalization are defined by [`Distribution`].
/// [`DistributionSpec`] also accepts custom, truncated and order-statistic
/// descriptions; their constructors document additional domains and work limits.
///
/// ## Domain
///
/// NaN arguments are errors. Discrete arguments are floored; infinities return zero and one.
///
/// ## Returns
///
/// Dimensionless cumulative probability in \[0,1\].
///
/// ## Errors
///
/// Invalid distribution parameters return an input error. NaN backend results
/// and quantiles outside the supported representable range return numerical errors.
///
/// ## Numerical behavior
///
/// Uses statrs privately. Quantiles use bounded bracketing and bisection
/// (at most 6144 continuous evaluations or 106 discrete evaluations).
/// Tail accuracy is limited by the backend probability functions and binary64.
///
/// # Examples
///
/// ```
/// let d = sci_stats::Distribution::Normal { mean: 0., std_dev: 1. };
/// assert_eq!(sci_stats::cdf(d, 0.)?, 0.5);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy distributions](https://docs.scipy.org/doc/scipy/reference/stats.html).
pub fn cdf(distribution: impl Into<DistributionSpec>, x: f64) -> Result<f64> {
    composition::evaluate(distribution.into(), x, Operation::Cdf)
}

/// Computes `S(x)=P(X>x)`.
///
/// ## Parameters
///
/// `distribution` specifies validated native parameters. `x` is a scalar observation.
/// Parameters, support, and normalization are defined by [`Distribution`].
/// [`DistributionSpec`] also accepts custom, truncated and order-statistic
/// descriptions; their constructors document additional domains and work limits.
///
/// ## Domain
///
/// NaN arguments are errors. Evaluates the backend survival function directly to preserve upper-tail accuracy.
///
/// ## Returns
///
/// Dimensionless survival probability in \[0,1\].
///
/// ## Errors
///
/// Invalid distribution parameters return an input error. NaN backend results
/// and quantiles outside the supported representable range return numerical errors.
///
/// ## Numerical behavior
///
/// Uses statrs privately. Quantiles use bounded bracketing and bisection
/// (at most 6144 continuous evaluations or 106 discrete evaluations).
/// Tail accuracy is limited by the backend probability functions and binary64.
///
/// # Examples
///
/// ```
/// let d = sci_stats::Distribution::Normal { mean: 0., std_dev: 1. };
/// assert_eq!(sci_stats::sf(d, 0.)?, 0.5);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy distributions](https://docs.scipy.org/doc/scipy/reference/stats.html).
pub fn sf(distribution: impl Into<DistributionSpec>, x: f64) -> Result<f64> {
    composition::evaluate(distribution.into(), x, Operation::Sf)
}

/// Computes `inf{x:F(x)≥p}, the inverse cumulative distribution function`.
///
/// ## Parameters
///
/// `distribution` specifies validated native parameters. `x` is a probability p.
/// Parameters, support, and normalization are defined by [`Distribution`].
/// [`DistributionSpec`] also accepts custom, truncated and order-statistic
/// descriptions; their constructors document additional domains and work limits.
///
/// ## Domain
///
/// NaN arguments are errors. p is in \[0,1\]. Continuous endpoints return support bounds; discrete p=0 returns support minimum minus one, following SciPy. Unbounded p=1 returns infinity. The geometric distribution retains this endpoint
/// convention even when its success probability is one, matching SciPy.
///
/// ## Returns
///
/// Quantile in observation units; discrete quantiles are integer-valued f64.
///
/// ## Errors
///
/// Invalid distribution parameters return an input error. NaN backend results
/// and quantiles outside the supported representable range return numerical errors.
///
/// ## Numerical behavior
///
/// Uses statrs privately. Quantiles use bounded bracketing and bisection
/// (at most 6144 continuous evaluations or 106 discrete evaluations).
/// Tail accuracy is limited by the backend probability functions and binary64.
///
/// # Examples
///
/// ```
/// let d = sci_stats::Distribution::Normal { mean: 0., std_dev: 1. };
/// assert!(sci_stats::ppf(d, 0.5)?.abs() < 1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy distributions](https://docs.scipy.org/doc/scipy/reference/stats.html).
pub fn ppf(distribution: impl Into<DistributionSpec>, x: f64) -> Result<f64> {
    composition::evaluate(distribution.into(), x, Operation::Ppf)
}
