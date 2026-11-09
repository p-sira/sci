//! Bounded rejection samplers for univariate continuous densities.
//!
//! Envelope conditions are mathematical preconditions: finitely many evaluations
//! cannot certify a global upper bound or global log-concavity of user code.

use crate::{Error, Result};
use rand::{Rng, SeedableRng, rngs::StdRng};

const MAX_OUTPUT: usize = 1_048_576;
const MAX_ATTEMPTS: usize = 16_777_216;
const MAX_KNOTS: usize = 32;
// A uniform upward shift of the log hat preserves its normalized proposal
// distribution and makes the allowed tangent roundoff safe at acceptance.
const TDR_HAT_MARGIN: f64 = 1e-12;

fn valid_budget(max_attempts: usize) -> Result<()> {
    if !(1..=MAX_ATTEMPTS).contains(&max_attempts) {
        return Err(Error::InvalidInput(
            "rejection attempt budget must be 1..=16777216",
        ));
    }
    Ok(())
}

fn valid_count(n: usize, attempts: usize) -> Result<()> {
    if n > MAX_OUTPUT || n > attempts {
        return Err(Error::InvalidInput(
            "rejection sample count exceeds output or attempt budget",
        ));
    }
    Ok(())
}

/// A ratio-of-uniforms rejection generator for a nonnegative, unnormalized PDF.
///
/// Let `A = {(u,v): 0 < u <= sqrt(f(v/u + shift))}`. Uniform points in `A`
/// produce `X = v/u + shift`, with density proportional to `f(X)`.
/// The caller must supply a rectangle `[0,umax] × [vmin,vmax]` containing **all**
/// of `A`. A rectangle that misses part of `A` silently biases the distribution;
/// validating the rectangle's finite endpoints cannot establish containment.
///
/// ## Parameters and conventions
/// Construct with [`RatioUniforms::new`], a fallible PDF callback, upper `u`
/// bound, lower and upper `v` bounds, shift, seed and per-batch attempt budget.
/// The PDF can be unnormalized, but must be nonnegative and integrable.
///
/// ## Domain and special values
/// Finite rectangle and shift, `umax > 0`, `vmin < vmax`, and finite squared
/// `umax`. The callback must return a finite value ≥0 at evaluated points.
/// Zero densities are rejected; no assumption of a CDF is made.
///
/// ## Returns
/// [`RatioUniforms::random`] returns `n` scalar samples in original density units.
///
/// ## Errors
/// Invalid geometry, invalid PDF results, callback failures, budget exhaustion
/// and oversized requests return [`Error`]. A failing batch advances the RNG.
///
/// ## Numerical behavior
/// Uses seeded `StdRng`. At most 1,048,576 samples and 16,777,216 proposals per
/// call, with no infinite rejection loop. Finite-precision uniform points and
/// density comparisons can affect the final few bits; seeds are Rust-specific.
///
/// # Examples
/// ```
/// use sci_stats::sampling::RatioUniforms;
/// let v = 2.0_f64.sqrt() * (-0.5_f64).exp();
/// let mut sampler = RatioUniforms::new(
///     |x: f64| Ok((-0.5 * x * x).exp()), 1.0, -v, v, 0.0, 7, 50_000,
/// )?;
/// assert_eq!(sampler.random(20)?.len(), 20);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
/// Kinderman and Monahan (1977), *Computer Generation of Random Variables
/// Using the Ratio of Uniform Deviates*, and
/// [SciPy `RatioUniforms`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sampling.RatioUniforms.html).
pub struct RatioUniforms<F> {
    pdf: F,
    umax: f64,
    vmin: f64,
    width: f64,
    shift: f64,
    seed: u64,
    rng: StdRng,
    max_attempts: usize,
    support: Option<(f64, f64)>,
    density_upper: Option<f64>,
}

impl<F> RatioUniforms<F>
where
    F: Fn(f64) -> Result<f64>,
{
    /// Construct a sampler using a caller-certified ratio-of-uniforms rectangle.
    ///
    /// All bounds and `shift` must be finite, the rectangle must be nonempty,
    /// and `umax²` must be representable. `max_attempts` is a *total per-call*
    /// proposal limit in `1..=16777216`. Invalid values return
    /// [`Error::InvalidInput`]; no PDF evaluations occur during construction.
    /// The caller guarantees `sqrt(pdf(x)) <= umax` and
    /// `(x-shift)*sqrt(pdf(x))` lies in `[vmin,vmax]` for all positive-density `x`.
    pub fn new(
        pdf: F,
        umax: f64,
        vmin: f64,
        vmax: f64,
        shift: f64,
        seed: u64,
        max_attempts: usize,
    ) -> Result<Self> {
        valid_budget(max_attempts)?;
        if !umax.is_finite()
            || umax <= 0.0
            || !(umax * umax).is_finite()
            || !vmin.is_finite()
            || !vmax.is_finite()
            || !shift.is_finite()
            || vmax <= vmin
            || !(vmax - vmin).is_finite()
        {
            return Err(Error::InvalidInput(
                "invalid ratio-of-uniforms rectangle or shift",
            ));
        }
        Ok(Self {
            pdf,
            umax,
            vmin,
            width: vmax - vmin,
            shift,
            seed,
            rng: StdRng::seed_from_u64(seed),
            max_attempts,
            support: None,
            density_upper: None,
        })
    }

    /// Draw `n` samples; the number of attempts is bounded over the whole call.
    ///
    /// `n=0` returns an empty vector without advancing the RNG. An invalid
    /// count returns [`Error::InvalidInput`], callback errors are propagated,
    /// and an exhausted budget returns [`Error::NumericalFailure`]. The output
    /// preserves density units and generation order.
    pub fn random(&mut self, n: usize) -> Result<Vec<f64>> {
        valid_count(n, self.max_attempts)?;
        let mut samples = Vec::with_capacity(n);
        for _ in 0..self.max_attempts {
            if samples.len() == n {
                return Ok(samples);
            }
            let u = self.umax * self.rng.random::<f64>();
            if u == 0.0 {
                continue;
            }
            let v = self.vmin + self.width * self.rng.random::<f64>();
            let x = v / u + self.shift;
            if !x.is_finite() {
                continue;
            }
            if self.support.is_some_and(|(lo, hi)| x < lo || x > hi) {
                continue;
            }
            let density = (self.pdf)(x)?;
            if !density.is_finite() {
                return Err(Error::NonFiniteEvaluation);
            }
            if density < 0.0 {
                return Err(Error::InvalidInput("PDF must be nonnegative"));
            }
            if self.density_upper.is_some_and(|bound| density > bound) {
                return Err(Error::InvalidInput(
                    "PDF exceeds declared global upper bound",
                ));
            }
            // Check the geometry at every observed positive-density point.
            // Arbitrary callback bounds still require a global caller guarantee.
            if density == 0.0 {
                continue;
            }
            let root = density.sqrt();
            let vx = (x - self.shift) * root;
            let vmax = self.vmin + self.width;
            let tol_u = 32.0 * f64::EPSILON * self.umax.max(root);
            let tol_v = 32.0 * f64::EPSILON * self.vmin.abs().max(vmax.abs()).max(vx.abs());
            if !vx.is_finite()
                || root > self.umax + tol_u
                || vx < self.vmin - tol_v
                || vx > vmax + tol_v
            {
                return Err(Error::InvalidInput(
                    "observed density lies outside rejection rectangle",
                ));
            }
            if u <= root {
                samples.push(x);
            }
        }
        if samples.len() == n {
            Ok(samples)
        } else {
            Err(Error::NumericalFailure(
                "ratio-of-uniforms attempt budget exhausted",
            ))
        }
    }

    /// Reset the seeded random stream; does not change the PDF or rectangle.
    ///
    /// The next sequence of successful batches repeats the original sequence.
    /// This method is infallible and performs constant work.
    pub fn reset(&mut self) {
        self.rng = StdRng::seed_from_u64(self.seed);
    }
}

/// Bounded-support simple ratio-of-uniforms sampler with a known density bound.
///
/// If `f(x) <= M` throughout its finite support `[a,b]`, the valid enclosing
/// rectangle is `0 <= u <= sqrt(M)`,
/// `min(0,(a-c)*sqrt(M)) <= v <= max(0,(b-c)*sqrt(M))`.
/// This specializes [`RatioUniforms`] by constructing the rectangle from a
/// caller-certified global PDF bound; it does **not** implement SciPy's
/// universal `T(x)=-1/sqrt(x)` bounds for unbounded T-concave densities.
///
/// ## Parameters and conventions
/// [`SimpleRatioUniforms::new`] takes a fallible density callback, inclusive
/// support, global upper PDF bound, shift, seed and per-call proposal budget.
/// The callback need only be defined on the support and can be unnormalized.
///
/// ## Domain and special values
/// Finite nonempty support; positive finite upper bound. Values outside support
/// are excluded without callback evaluation. The bound must hold everywhere.
///
/// ## Returns
/// [`SimpleRatioUniforms::random`] returns continuous samples within support.
///
/// ## Errors
/// Invalid geometry, observed violations of the bound, callback errors,
/// nonfinite density, exhausted attempts and work limits return [`Error`].
///
/// ## Numerical behavior
/// Exact rectangle bounds follow from the assumed finite support and upper
/// density bound. No numerical search for a mode is required. Uses `StdRng`
/// and bounded rejection work; an overly loose global bound hurts efficiency.
///
/// # Examples
/// ```
/// use sci_stats::sampling::SimpleRatioUniforms;
/// let mut sampler = SimpleRatioUniforms::new(
///     |x: f64| Ok(2.0*x), (0.0, 1.0), 2.0, 0.5, 42, 50_000,
/// )?;
/// assert!(sampler.random(50)?.iter().all(|&x| (0.0..=1.0).contains(&x)));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
/// Kinderman and Monahan (1977) and
/// [SciPy `SimpleRatioUniforms`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sampling.SimpleRatioUniforms.html).
pub struct SimpleRatioUniforms<F> {
    inner: RatioUniforms<F>,
}

impl<F> SimpleRatioUniforms<F>
where
    F: Fn(f64) -> Result<f64>,
{
    /// Construct the finite-support rectangle from a certified density maximum.
    ///
    /// Requires finite ordered `support=(lower,upper)`, finite `shift`, and
    /// finite `density_upper > 0`; the computed rectangle must be representable.
    /// The callback must be nonnegative and at most `density_upper` on support.
    /// Invalid values and invalid attempt budgets return [`Error::InvalidInput`].
    pub fn new(
        pdf: F,
        support: (f64, f64),
        density_upper: f64,
        shift: f64,
        seed: u64,
        max_attempts: usize,
    ) -> Result<Self> {
        let (lower, upper) = support;
        if !lower.is_finite()
            || !upper.is_finite()
            || !(upper - lower).is_finite()
            || lower >= upper
            || !density_upper.is_finite()
            || density_upper <= 0.0
            || !shift.is_finite()
        {
            return Err(Error::InvalidInput(
                "invalid support or density upper bound",
            ));
        }
        let umax = density_upper.sqrt();
        let vmin = ((lower - shift) * umax).min(0.0);
        let vmax = ((upper - shift) * umax).max(0.0);
        let mut inner = RatioUniforms::new(pdf, umax, vmin, vmax, shift, seed, max_attempts)?;
        inner.support = Some(support);
        inner.density_upper = Some(density_upper);
        Ok(Self { inner })
    }

    /// Draw up to 1,048,576 samples using the certified density bound.
    ///
    /// Returns samples on the declared support or a callback, work-budget or
    /// envelope-violation [`Error`]. Invalid or exhausted calls yield no samples.
    pub fn random(&mut self, n: usize) -> Result<Vec<f64>> {
        self.inner.random(n)
    }

    /// Reset the deterministic `StdRng` stream, with constant work and no errors.
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

#[derive(Clone, Copy)]
struct Knot {
    x: f64,
    logf: f64,
    slope: f64,
}

#[derive(Clone, Copy)]
struct HatSegment {
    lo: f64,
    hi: f64,
    index: usize,
    log_mass: f64,
    cumulative_mass: f64,
}

/// Transformed-density rejection with the logarithmic (`c=0`) transformation.
///
/// For a positive, differentiable density on finite `[a,b]` with **concave**
/// `g(x)=ln(f(x))`, each tangent `t_i(x)=g(x_i)+g'(x_i)(x-x_i)` bounds `g(x)`
/// from above. This generator draws from the piecewise-exponential hat
/// `exp(min_i t_i(x))`, accepting with probability `exp(g(x)-hat_log(x))`.
/// Rejected points add tangents, at most 32 knots (adaptive rejection sampling).
///
/// ## Parameters and conventions
/// [`TransformedDensityRejection::new`] accepts fallible `log_pdf` and
/// `dlog_pdf` callbacks (derivative of log PDF), finite support, seed and
/// per-call attempt budget. The PDF need not be normalized; its logarithm may
/// include any finite additive constant. The initial knots lie at the first
/// and third quartiles of the support.
///
/// ## Domain and special values
/// The density must be log-concave and differentiable on the interior of the
/// finite support, with finite log density and derivative at the initial knots.
/// A `-∞` log density at later candidate points is treated as zero density.
/// This implements SciPy TDR's `c=0` transformation on a restricted support;
/// the general `c=-0.5` T-concave family and unbounded support are unsupported.
///
/// ## Returns
/// [`TransformedDensityRejection::random`] yields independent scalar samples
/// from the density restricted to `[a,b]`.
///
/// ## Errors
/// Invalid bounds, inconsistent observed log-concavity, nonfinite tangent
/// calculations, callback failures, exhausted attempts and oversized requests
/// return [`Error`]. Global log-concavity is a caller precondition: finite
/// checks cannot prove the envelope correct for an arbitrary callback.
///
/// ## Numerical behavior
/// Computes exponential hat integrals and component weights in log space and
/// inverts each piecewise exponential component analytically. Component weights
/// are cached until the envelope adapts. A uniform `1e-12` upward log-hat
/// margin makes the permitted tangent rounding harmless to acceptance. Uses
/// `StdRng`, 32 knots maximum, 1,048,576 output cap and 16,777,216 proposal
/// cap. Extreme derivatives or unrepresentable tangent intersections error.
///
/// # Examples
/// ```
/// use sci_stats::sampling::TransformedDensityRejection;
/// let mut sampler = TransformedDensityRejection::new(
///     |x: f64| Ok(-x*x/2.0), |x: f64| Ok(-x), (-5.0,5.0), 7, 100_000,
/// )?;
/// let sample = sampler.random(100)?;
/// assert_eq!(sample.len(), 100);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
/// Gilks and Wild (1992), *Adaptive Rejection Sampling for Gibbs Sampling*,
/// and [SciPy `TransformedDensityRejection`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sampling.TransformedDensityRejection.html).
pub struct TransformedDensityRejection<F, G> {
    log_pdf: F,
    dlog_pdf: G,
    support: (f64, f64),
    initial: [Knot; 2],
    knots: Vec<Knot>,
    rng: StdRng,
    seed: u64,
    max_attempts: usize,
}

impl<F, G> TransformedDensityRejection<F, G>
where
    F: Fn(f64) -> Result<f64>,
    G: Fn(f64) -> Result<f64>,
{
    /// Construct a logarithmic-transformation rejection sampler.
    ///
    /// `support` must be a finite nonempty interval with representable length;
    /// both callbacks must return finite values at the two interior knots.
    /// Initial derivatives must be nonincreasing, as required by log-concavity.
    /// The `max_attempts` budget is `1..=16777216` proposals per batch.
    /// Failed validation and callbacks return [`Error`].
    pub fn new(
        log_pdf: F,
        dlog_pdf: G,
        support: (f64, f64),
        seed: u64,
        max_attempts: usize,
    ) -> Result<Self> {
        valid_budget(max_attempts)?;
        let (a, b) = support;
        let width = b - a;
        if !a.is_finite() || !b.is_finite() || !width.is_finite() || width <= 0.0 {
            return Err(Error::InvalidInput("TDR requires finite, ordered support"));
        }
        let x0 = a + 0.25 * width;
        let x1 = a + 0.75 * width;
        if !(a < x0 && x0 < x1 && x1 < b) {
            return Err(Error::InvalidInput(
                "TDR support too narrow for construction knots",
            ));
        }
        let initial = [
            Knot {
                x: x0,
                logf: log_pdf(x0)?,
                slope: dlog_pdf(x0)?,
            },
            Knot {
                x: x1,
                logf: log_pdf(x1)?,
                slope: dlog_pdf(x1)?,
            },
        ];
        if initial
            .iter()
            .any(|p| !p.logf.is_finite() || !p.slope.is_finite())
        {
            return Err(Error::NonFiniteEvaluation);
        }
        let result = Self {
            log_pdf,
            dlog_pdf,
            support,
            initial,
            knots: initial.to_vec(),
            rng: StdRng::seed_from_u64(seed),
            seed,
            max_attempts,
        };
        result.validate_tangents()?;
        result.hat_segments()?;
        Ok(result)
    }

    fn validate_tangents(&self) -> Result<()> {
        for pair in self.knots.windows(2) {
            if pair[0].slope < pair[1].slope {
                return Err(Error::InvalidInput(
                    "log density derivatives must be nonincreasing",
                ));
            }
        }
        for p in &self.knots {
            for q in &self.knots {
                let upper = p.logf + p.slope * (q.x - p.x);
                if !upper.is_finite() {
                    return Err(Error::NonFiniteEvaluation);
                }
                // Log density is defined up to an arbitrary additive constant;
                // a tolerance scaled by |upper| could hide a large violation.
                if q.logf - upper > TDR_HAT_MARGIN {
                    return Err(Error::InvalidInput(
                        "log density exceeds one of its tangents",
                    ));
                }
            }
        }
        Ok(())
    }

    fn hat_segments(&self) -> Result<Vec<HatSegment>> {
        let mut boundaries = Vec::with_capacity(self.knots.len() + 1);
        boundaries.push(self.support.0);
        for pair in self.knots.windows(2) {
            let (p, q) = (pair[0], pair[1]);
            let dx = q.x - p.x;
            let ds = p.slope - q.slope;
            let crossing = if ds == 0.0 {
                (p.x + 0.5 * dx).min(q.x)
            } else {
                // t_p(x_p + offset) = t_q(x_p + offset).
                let offset = ((q.logf - p.logf) - q.slope * dx) / ds;
                if !offset.is_finite() || offset < -1e-10 * dx || offset > (1.0 + 1e-10) * dx {
                    return Err(Error::InvalidInput("inconsistent log-concave tangents"));
                }
                p.x + offset.clamp(0.0, dx)
            };
            if !crossing.is_finite() || crossing < *boundaries.last().unwrap() {
                return Err(Error::NonFiniteEvaluation);
            }
            boundaries.push(crossing);
        }
        boundaries.push(self.support.1);
        let mut segments = Vec::with_capacity(self.knots.len());
        for (index, pair) in boundaries.windows(2).enumerate() {
            let (lo, hi) = (pair[0], pair[1]);
            if lo >= hi {
                continue;
            }
            let knot = self.knots[index];
            let hl = knot.logf + knot.slope * (lo - knot.x);
            let hr = knot.logf + knot.slope * (hi - knot.x);
            let t = (hr - hl).abs();
            let ratio = if t < 1e-5 {
                1.0 - 0.5 * t + t * t / 6.0
            } else {
                -(-t).exp_m1() / t
            };
            let log_mass = hl.max(hr) + (hi - lo).ln() + ratio.ln();
            if !log_mass.is_finite() {
                return Err(Error::NonFiniteEvaluation);
            }
            segments.push(HatSegment {
                lo,
                hi,
                index,
                log_mass,
                cumulative_mass: 0.0,
            });
        }
        if segments.is_empty() {
            return Err(Error::NumericalFailure("TDR constructed an empty envelope"));
        }
        // The hat changes only when a new knot is inserted. Cache normalized
        // segment weights once, rather than exponentiating each weight for
        // every proposal.
        let max_log_mass = segments
            .iter()
            .map(|s| s.log_mass)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut total = 0.0;
        for segment in &mut segments {
            total += (segment.log_mass - max_log_mass).exp();
            segment.cumulative_mass = total;
        }
        if !total.is_finite() || total <= 0.0 {
            return Err(Error::NonFiniteEvaluation);
        }
        Ok(segments)
    }

    /// Draw a batch from the log-concave density using tangent rejection.
    ///
    /// `n=0` is a no-op; `n<=1048576` must also be at most the per-call
    /// proposal budget. Rejected points adapt the envelope up to 32 knots.
    /// Returns the complete vector or an [`Error`], never a partial batch.
    pub fn random(&mut self, n: usize) -> Result<Vec<f64>> {
        valid_count(n, self.max_attempts)?;
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut segments = self.hat_segments()?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..self.max_attempts {
            let area = segments.last().unwrap().cumulative_mass;
            // Multiplication can round to `area` even when the input is < 1.
            // Keep the search strictly inside the positive-mass CDF.
            let choice = (self.rng.random::<f64>() * area).min(area.next_down());
            let index = segments
                .partition_point(|part| part.cumulative_mass <= choice)
                .min(segments.len() - 1);
            let segment = segments[index];
            let knot = self.knots[segment.index];
            let width = segment.hi - segment.lo;
            let z = knot.slope * width;
            if !z.is_finite() {
                return Err(Error::NonFiniteEvaluation);
            }
            let u = self.rng.random::<f64>().max(f64::MIN_POSITIVE);
            let x = if z >= 1.0 {
                segment.hi + (u + (1.0 - u) * (-z).exp()).ln() / knot.slope
            } else if z != 0.0 {
                segment.lo + (u * z.exp_m1()).ln_1p() / knot.slope
            } else {
                segment.lo + width * u
            };
            if !x.is_finite() || x < segment.lo || x > segment.hi {
                return Err(Error::NonFiniteEvaluation);
            }
            let logf = (self.log_pdf)(x)?;
            if logf.is_nan() || logf == f64::INFINITY {
                return Err(Error::NonFiniteEvaluation);
            }
            let hat = knot.logf + knot.slope * (x - knot.x);
            if !hat.is_finite() {
                return Err(Error::NonFiniteEvaluation);
            }
            let gap = logf - hat;
            if gap > TDR_HAT_MARGIN {
                return Err(Error::InvalidInput(
                    "observed log density exceeds rejection envelope",
                ));
            }
            // The same constant margin raises every tangent, so the proposal
            // probabilities above remain unchanged. Subtracting it here
            // prevents accepting from a numerically undershooting envelope.
            // Draw from (0,1] to avoid ln(0) spuriously accepting tiny tails.
            let log_accept = (1.0 - self.rng.random::<f64>()).ln();
            if gap.is_finite() && log_accept <= gap - TDR_HAT_MARGIN {
                out.push(x);
                if out.len() == n {
                    return Ok(out);
                }
            } else if self.knots.len() < MAX_KNOTS && logf.is_finite() {
                let pos = self.knots.partition_point(|p| p.x < x);
                if self.knots.get(pos).is_none_or(|p| p.x != x)
                    && (pos == 0 || self.knots[pos - 1].x != x)
                {
                    let slope = (self.dlog_pdf)(x)?;
                    if !slope.is_finite() {
                        return Err(Error::NonFiniteEvaluation);
                    }
                    self.knots.insert(pos, Knot { x, logf, slope });
                    match self.validate_tangents().and_then(|()| self.hat_segments()) {
                        Ok(next) => segments = next,
                        Err(error) => {
                            // An invalid derivative must not poison subsequent
                            // calls to this generator after reporting an error.
                            self.knots.remove(pos);
                            return Err(error);
                        }
                    }
                }
            }
        }
        Err(Error::NumericalFailure("TDR attempt budget exhausted"))
    }

    /// Reset the seeded stream and the two initial tangent knots.
    ///
    /// A repeated sequence of calls after reset reproduces initial draws;
    /// the reset is infallible and takes constant time.
    pub fn reset(&mut self) {
        self.rng = StdRng::seed_from_u64(self.seed);
        self.knots.clear();
        self.knots.extend_from_slice(&self.initial);
    }

    /// Return the current number of tangent knots, in `2..=32`.
    ///
    /// Rejections may increase the count; `reset` restores it to two.
    pub fn knot_count(&self) -> usize {
        self.knots.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratio_uniforms_normal_moments_and_repeatability() -> Result<()> {
        let v = 2.0_f64.sqrt() * (-0.5_f64).exp();
        let pdf = |x: f64| Ok((-0.5 * x * x).exp());
        let mut sampler = RatioUniforms::new(pdf, 1.0, -v, v, 0.0, 42, 100_000)?;
        let a = sampler.random(8_000)?;
        let mean = a.iter().sum::<f64>() / a.len() as f64;
        let second = a.iter().map(|x| x * x).sum::<f64>() / a.len() as f64;
        assert!(mean.abs() < 0.06, "{mean}");
        assert!((second - 1.0).abs() < 0.07, "{second}");
        sampler.reset();
        assert_eq!(a, sampler.random(8_000)?);
        Ok(())
    }

    #[test]
    fn simple_ratio_uniforms_beta_two_one_and_guards() -> Result<()> {
        let mut sampler =
            SimpleRatioUniforms::new(|x: f64| Ok(2.0 * x), (0.0, 1.0), 2.0, 0.5, 6, 100_000)?;
        let x = sampler.random(5_000)?;
        assert!(x.iter().all(|x| (0.0..=1.0).contains(x)));
        let mean = x.iter().sum::<f64>() / x.len() as f64;
        assert!((mean - 2.0 / 3.0).abs() < 0.03, "{mean}");
        sampler.reset();
        assert_eq!(x, sampler.random(5_000)?);
        assert!(
            SimpleRatioUniforms::new(|_x: f64| Ok(1.0), (0.0, 1.0), 0.0, 0.0, 1, 500,).is_err()
        );
        let mut bad = SimpleRatioUniforms::new(|_x: f64| Ok(2.0), (0.0, 1.0), 1.0, 0.5, 1, 500)?;
        assert!(matches!(bad.random(10), Err(Error::InvalidInput(_))));
        Ok(())
    }

    #[test]
    fn tdr_log_concave_normal_moments_and_reset() -> Result<()> {
        let mut sampler = TransformedDensityRejection::new(
            |x: f64| Ok(-x * x / 2.0),
            |x: f64| Ok(-x),
            (-6.0, 6.0),
            15,
            100_000,
        )?;
        let x = sampler.random(8_000)?;
        let mean = x.iter().sum::<f64>() / x.len() as f64;
        let second = x.iter().map(|x| x * x).sum::<f64>() / x.len() as f64;
        assert!(mean.abs() < 0.055, "{mean}");
        assert!((second - 1.0).abs() < 0.06, "{second}");
        assert!(sampler.knot_count() > 2);
        sampler.reset();
        assert_eq!(sampler.knot_count(), 2);
        assert_eq!(sampler.random(8_000)?, x);
        Ok(())
    }

    #[test]
    fn tdr_linear_log_density_and_invalid_derivative() -> Result<()> {
        let mut sampler = TransformedDensityRejection::new(
            |x: f64| Ok(-x),
            |_x: f64| Ok(-1.0),
            (0.0, 4.0),
            3,
            12_000,
        )?;
        let x = sampler.random(6_000)?;
        let true_mean = 1.0 - 4.0 / (4.0_f64.exp() - 1.0);
        let mean = x.iter().sum::<f64>() / x.len() as f64;
        assert!((mean - true_mean).abs() < 0.04, "{mean}");
        assert_eq!(sampler.knot_count(), 2);
        assert!(
            TransformedDensityRejection::new(
                |x: f64| Ok(x * x),
                |x: f64| Ok(2.0 * x),
                (-1.0, 1.0),
                3,
                1_000,
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn tdr_caches_monotone_hat_weights_and_samples_steep_exponential() -> Result<()> {
        // An exponential with slope 1000 concentrates almost all mass in the
        // last segment; CDF selection must not choose a zero-mass segment.
        let mut sampler = TransformedDensityRejection::new(
            |x: f64| Ok(1000.0 * x),
            |_x: f64| Ok(1000.0),
            (0.0, 1.0),
            77,
            10_000,
        )?;
        let segments = sampler.hat_segments()?;
        assert_eq!(segments.len(), 2);
        assert!(segments[0].cumulative_mass < segments[1].cumulative_mass);
        assert!(segments[1].cumulative_mass.is_finite());
        let draws = sampler.random(2_000)?;
        assert!(draws.iter().all(|&x| (0.0..=1.0).contains(&x)));
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        assert!((mean - 0.999).abs() < 0.00015, "{mean}");
        sampler.reset();
        assert_eq!(draws, sampler.random(2_000)?);
        Ok(())
    }

    #[test]
    fn tdr_detects_density_above_tangent_despite_large_log_offset() -> Result<()> {
        // Checking `logf > hat + tolerance` loses the tolerance for a large
        // offset; compare the log gap directly and reject an invalid hat.
        let mut sampler = TransformedDensityRejection::new(
            |x: f64| {
                Ok(if (0.4..0.6).contains(&x) {
                    1e6 + 1e-6
                } else {
                    1e6
                })
            },
            |_x: f64| Ok(0.0),
            (0.0, 1.0),
            23,
            1_000,
        )?;
        assert!(matches!(sampler.random(100), Err(Error::InvalidInput(_))));
        Ok(())
    }

    #[test]
    fn tdr_nearly_flat_increasing_log_density_remains_continuous() -> Result<()> {
        // exp(-z) rounds to one for slopes this small. Sampling with an
        // unstabilized positive-slope inverse would create artificial atoms
        // at the hat-segment right endpoints.
        let mut sampler = TransformedDensityRejection::new(
            |x: f64| Ok(1e-16 * x),
            |_x: f64| Ok(1e-16),
            (0.0, 1.0),
            11,
            10_000,
        )?;
        let values = sampler.random(1_000)?;
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        assert!((mean - 0.5).abs() < 0.05);
        assert!(values.iter().any(|&x| x < 0.1));
        assert!(values.iter().any(|&x| x > 0.9));
        assert!(values.iter().filter(|&&x| x == 0.5 || x == 1.0).count() < 10);
        Ok(())
    }

    #[test]
    fn invalid_adaptive_tangent_preserves_previous_envelope() -> Result<()> {
        let mut sampler = TransformedDensityRejection::new(
            |x: f64| Ok(-0.5 * x * x),
            |x: f64| Ok(if x == -2.5 || x == 2.5 { -x } else { 100.0 }),
            (-5.0, 5.0),
            13,
            1_000,
        )?;
        assert!(matches!(sampler.random(50), Err(Error::InvalidInput(_))));
        assert_eq!(sampler.knot_count(), 2);
        assert!(sampler.validate_tangents().is_ok());
        assert!(sampler.hat_segments().is_ok());
        Ok(())
    }

    #[test]
    fn tdr_asymmetric_concave_hat_matches_exact_tangent_crossing() -> Result<()> {
        // The two tangents for -x^4 generally cross away from the midpoint.
        // An incorrect complementary crossing produces a biased proposal hat.
        let mut sampler = TransformedDensityRejection::new(
            |x: f64| Ok(-x.powi(4)),
            |x: f64| Ok(-4.0 * x.powi(3)),
            (-1.0, 2.0),
            17,
            100_000,
        )?;
        let parts = sampler.hat_segments()?;
        assert_eq!(parts.len(), 2);
        let crossing = parts[0].hi;
        let p = sampler.knots[0];
        let q = sampler.knots[1];
        assert!((crossing - (p.x + q.x) / 2.0).abs() > 0.1);
        let left = p.logf + p.slope * (crossing - p.x);
        let right = q.logf + q.slope * (crossing - q.x);
        assert!((left - right).abs() < 1e-12);
        for i in 0..=30 {
            let x = -1.0 + 0.1 * i as f64;
            let log_h = sampler
                .knots
                .iter()
                .map(|k| k.logf + k.slope * (x - k.x))
                .fold(f64::INFINITY, f64::min);
            assert!(log_h >= -x.powi(4) - 1e-12);
        }

        // A deterministic Simpson reference tests the resulting distribution,
        // independently of the ratio/rejection implementation.
        let mut area = 0.0;
        let mut first = 0.0;
        let mut second = 0.0;
        for i in 0..=20_000 {
            let x = -1.0 + 3.0 * i as f64 / 20_000.0;
            let weight = if i == 0 || i == 20_000 {
                1.0
            } else if i % 2 == 0 {
                2.0
            } else {
                4.0
            };
            let mass = weight * (-x.powi(4)).exp();
            area += mass;
            first += mass * x;
            second += mass * x * x;
        }
        let x = sampler.random(7_000)?;
        let mean = x.iter().sum::<f64>() / x.len() as f64;
        let second_moment = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
        assert!((mean - first / area).abs() < 0.035);
        assert!((second_moment - second / area).abs() < 0.035);
        Ok(())
    }

    #[test]
    fn exhaustion_and_callbacks_are_errors() -> Result<()> {
        let mut empty = RatioUniforms::new(|_x: f64| Ok(0.0), 1.0, -1.0, 1.0, 0.0, 2, 10)?;
        assert!(matches!(empty.random(1), Err(Error::NumericalFailure(_))));
        assert!(matches!(empty.random(11), Err(Error::InvalidInput(_))));
        // All u² underflow for this rectangle; zero density must still reject.
        let mut subzero =
            RatioUniforms::new(|_x: f64| Ok(0.0), 1e-200, -1e-200, 1e-200, 0.0, 2, 100)?;
        assert!(matches!(subzero.random(1), Err(Error::NumericalFailure(_))));
        let tiny_pdf = f64::from_bits(1);
        let umax = tiny_pdf.sqrt();
        let mut tiny = RatioUniforms::new(
            move |x: f64| Ok(if x.abs() <= 1.0 { tiny_pdf } else { 0.0 }),
            umax,
            -umax,
            umax,
            0.0,
            7,
            10_000,
        )?;
        let draw = tiny.random(500)?;
        assert!(draw.iter().all(|x| (-1.0..=1.0).contains(x)));
        let mut invalid = RatioUniforms::new(
            |_x: f64| Err(Error::InvalidInput("callback")),
            1.0,
            -1.0,
            1.0,
            0.0,
            3,
            30,
        )?;
        assert_eq!(invalid.random(1), Err(Error::InvalidInput("callback")));
        assert!(
            RatioUniforms::new(|_x: f64| Ok(1.0), f64::INFINITY, -1.0, 1.0, 0.0, 3, 30,).is_err()
        );
        assert!(
            TransformedDensityRejection::new(
                |_x: f64| Ok(0.0),
                |_x: f64| Ok(0.0),
                (0.0, f64::INFINITY),
                5,
                100,
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn tdr_rejects_exact_zero_density_tails() -> Result<()> {
        let mut sampler = TransformedDensityRejection::new(
            |x: f64| {
                Ok(if x.abs() < 0.9 {
                    0.0
                } else {
                    f64::NEG_INFINITY
                })
            },
            |_x: f64| Ok(0.0),
            (-1.0, 1.0),
            71,
            20_000,
        )?;
        let x = sampler.random(2_000)?;
        assert!(x.iter().all(|x| x.abs() < 0.9));
        Ok(())
    }
}
