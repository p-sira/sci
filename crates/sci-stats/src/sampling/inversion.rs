//! Inverse-transform sampling for native continuous distributions.
//!
//! These are Rust adaptations of the similarly named SciPy sampling classes.
//! They do not use UNU.RAN or reproduce its spline construction algorithms.

use crate::implementation::{Distribution, cdf, ppf, sf};
use crate::{Error, Result};
use rand::{Rng, SeedableRng, rngs::StdRng};

const MAX_SAMPLES: usize = 1_048_576;
const MAX_NODES: usize = 8_192;
const MAX_REFINEMENTS: usize = 64;
const MAX_BISECTIONS: usize = 96;

fn continuous(distribution: Distribution) -> Result<()> {
    if matches!(
        distribution,
        Distribution::DiscreteUniform { .. }
            | Distribution::Bernoulli { .. }
            | Distribution::Binomial { .. }
            | Distribution::Poisson { .. }
            | Distribution::Geometric { .. }
            | Distribution::NegativeBinomial { .. }
            | Distribution::Hypergeometric { .. }
    ) {
        return Err(Error::InvalidInput(
            "inverse sampling requires a continuous distribution",
        ));
    }
    cdf(distribution, 0.)?;
    Ok(())
}

fn valid_probability(probability: f64) -> Result<()> {
    if !(0. ..=1.).contains(&probability) {
        Err(Error::InvalidInput("probability must be in [0,1]"))
    } else {
        Ok(())
    }
}

fn valid_batch(count: usize) -> Result<()> {
    if count > MAX_SAMPLES {
        Err(Error::InvalidInput("sample count exceeds 1048576"))
    } else {
        Ok(())
    }
}

fn uniform_interior(generator: &mut StdRng) -> f64 {
    let u = generator.random::<f64>();
    if u == 0. { f64::EPSILON * 0.5 } else { u }
}

fn quantiles(
    probabilities: &[f64],
    mut inverse: impl FnMut(f64) -> Result<f64>,
) -> Result<Vec<f64>> {
    valid_batch(probabilities.len())?;
    for &probability in probabilities {
        valid_probability(probability)?;
    }
    probabilities.iter().map(|&u| inverse(u)).collect()
}

/// Maximum and average absolute probability errors of an inverse-CDF sampler.
///
/// Each residual is `|CDF_conditional(PPF_approx(u)) - u|` for an interior
/// probability `u`. The conditional CDF uses the original distribution's CDF
/// or survival function independently of the interpolation table. Both values
/// are dimensionless.
/// These are measured errors, not global analytic error bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UError {
    /// Largest absolute CDF residual among the evaluated probabilities.
    pub max_error: f64,
    /// Arithmetic mean absolute CDF residual among the evaluated probabilities.
    pub mean_absolute_error: f64,
}

fn measured_error(
    count: usize,
    mut inverse: impl FnMut(f64) -> Result<f64>,
    mut cumulative: impl FnMut(f64) -> Result<f64>,
) -> Result<UError> {
    if count == 0 || count > MAX_SAMPLES {
        return Err(Error::InvalidInput(
            "error probe count must be in 1..=1048576",
        ));
    }
    let mut maximum: f64 = 0.;
    let mut sum = 0.;
    for i in 0..count {
        let u = (i as f64 + 0.5) / count as f64;
        let difference = (cumulative(inverse(u)?)? - u).abs();
        if !difference.is_finite() {
            return Err(Error::NonFiniteEvaluation);
        }
        maximum = maximum.max(difference);
        sum += difference;
    }
    Ok(UError {
        max_error: maximum,
        mean_absolute_error: sum / count as f64,
    })
}

/// Seeded inverse-transform sampler backed by the native distribution quantile.
///
/// For a continuous CDF `F`, `X = F⁻¹(U)` with `U ~ Uniform[0,1)` has CDF `F`.
/// This Rust adaptation delegates to the native [`ppf`](crate::distributions::ppf) on every draw; SciPy's
/// `FastGeneratorInversion` instead uses a precomputed numerical generator.
///
/// ## Parameters and conventions
///
/// `distribution` is a native continuous [`Distribution`](crate::distributions::Distribution) with valid parameters;
/// `seed` seeds a `rand` 0.9 `StdRng`. Samples and quantiles retain the
/// distribution's observation units.
///
/// ## Domain and special values
///
/// Quantile probabilities belong to `[0,1]`; their endpoints follow native
/// distribution support conventions, including infinite support endpoints.
/// Zero-length sample batches return an empty vector. Discrete laws are rejected.
///
/// ## Returns
///
/// A stateful scalar sampler; `rvs(n)` returns `n` observations, and `qrvs`
/// maps caller-supplied one-dimensional uniform inputs to quantiles.
///
/// ## Errors
///
/// Invalid distributions, probabilities, sample counts above 1,048,576 and
/// unrepresentable interior quantiles produce errors.
///
/// ## Numerical behavior
///
/// The seed determines reproducible draws within the same Rust/rand version.
/// Accuracy and work per draw inherit the native distribution quantile backend;
/// no UNU.RAN approximation tolerance or SciPy RNG stream parity is implied.
///
/// ## Examples
///
/// ```
/// use sci_stats::{distributions::Distribution, sampling::FastGeneratorInversion};
/// let mut sampler = FastGeneratorInversion::new(Distribution::Uniform { low: 2., high: 4. }, 7)?;
/// assert_eq!(sampler.ppf(0.25)?, 2.5);
/// assert!(sampler.rvs(5)?.iter().all(|x| (2. ..4.).contains(x)));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy FastGeneratorInversion](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sampling.FastGeneratorInversion.html).
#[derive(Clone, Debug)]
pub struct FastGeneratorInversion {
    distribution: Distribution,
    generator: StdRng,
}

impl FastGeneratorInversion {
    /// Construct a seeded native-quantile inverse sampler.
    ///
    /// Validates a continuous distribution and a finite median. Invalid
    /// parameters and unrepresentable median quantiles return an error.
    /// The seed determines the subsequent `rvs` stream.
    pub fn new(distribution: Distribution, seed: u64) -> Result<Self> {
        continuous(distribution)?;
        if !ppf(distribution, 0.5)?.is_finite() {
            return Err(Error::NonFiniteEvaluation);
        }
        Ok(Self {
            distribution,
            generator: StdRng::seed_from_u64(seed),
        })
    }

    /// Return `F⁻¹(p)` in observation units, with `p` in `[0,1]`.
    ///
    /// This uses the native quantile evaluator, preserving its endpoint
    /// infinities and reporting invalid probabilities or evaluation failures.
    pub fn ppf(&self, probability: f64) -> Result<f64> {
        valid_probability(probability)?;
        ppf(self.distribution, probability)
    }

    /// Evaluate the original dimensionless CDF `F(x)`.
    ///
    /// `x` has observation units; NaN is invalid, while infinite endpoints
    /// follow the native distribution's CDF behavior.
    pub fn cdf(&self, x: f64) -> Result<f64> {
        cdf(self.distribution, x)
    }

    /// Return the distribution's support as `(F⁻¹(0), F⁻¹(1))`.
    ///
    /// Bounds retain observation units and can be infinite. Backend quantile
    /// errors are propagated.
    pub fn support(&self) -> Result<(f64, f64)> {
        Ok((self.ppf(0.)?, self.ppf(1.)?))
    }

    /// Map at most 1,048,576 supplied uniform probabilities through `F⁻¹`.
    ///
    /// Inputs must lie in `[0,1]`, including valid support endpoints. The
    /// output has the same length and ordering; RNG state is unaffected.
    pub fn qrvs(&self, probabilities: &[f64]) -> Result<Vec<f64>> {
        quantiles(probabilities, |u| self.ppf(u))
    }

    /// Draw `n` independent pseudorandom inverse-CDF variates.
    ///
    /// At most 1,048,576 observations per call. `n=0` leaves RNG state
    /// unchanged. Native quantile errors propagate.
    pub fn rvs(&mut self, n: usize) -> Result<Vec<f64>> {
        valid_batch(n)?;
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            let u = uniform_interior(&mut self.generator);
            samples.push(self.ppf(u)?);
        }
        Ok(samples)
    }

    /// Measure `|F(F⁻¹(u))-u|` at `count` equally spaced interior probabilities.
    ///
    /// Returns maximum/mean absolute dimensionless residuals. Counts outside
    /// `1..=1048576` are errors. This is a deterministic diagnostic rather than
    /// a certified worst-case quantile error.
    pub fn u_error(&self, count: usize) -> Result<UError> {
        measured_error(count, |u| self.ppf(u), |x| self.cdf(x))
    }

    /// Measure the native inverse-CDF probability residual on `count` probes.
    ///
    /// This is the Rust diagnostic corresponding to SciPy's `evaluate_error`,
    /// returning measured maximum and mean absolute u-errors only; no separate
    /// quantile x-error is reported. Invalid counts return errors.
    pub fn evaluate_error(&self, count: usize) -> Result<UError> {
        self.u_error(count)
    }
}

#[derive(Clone, Copy, Debug)]
struct Node {
    u: f64,
    x: f64,
}

#[derive(Clone, Copy, Debug)]
struct ConditionalCdf {
    distribution: Distribution,
    lower: f64,
    upper: f64,
    base_mass: f64,
    mass: f64,
    upper_tail: bool,
}

impl ConditionalCdf {
    fn new(distribution: Distribution, lower: f64, upper: f64) -> Result<Self> {
        continuous(distribution)?;
        if !lower.is_finite()
            || !upper.is_finite()
            || lower >= upper
            || !(upper - lower).is_finite()
        {
            return Err(Error::InvalidInput(
                "inverse interpolation requires increasing finite domain bounds",
            ));
        }
        let lower_cdf = cdf(distribution, lower)?;
        if !lower_cdf.is_finite() || !(0. ..=1.).contains(&lower_cdf) {
            return Err(Error::NonFiniteEvaluation);
        }
        // Upper-tail CDF values can both round to 1 although S(lower)-S(upper)
        // is representable. Select the better-conditioned difference.
        let (base_mass, edge_mass, upper_tail) = if lower_cdf > 0.5 {
            (sf(distribution, lower)?, sf(distribution, upper)?, true)
        } else {
            (lower_cdf, cdf(distribution, upper)?, false)
        };
        let mass = if upper_tail {
            base_mass - edge_mass
        } else {
            edge_mass - base_mass
        };
        if !base_mass.is_finite()
            || !edge_mass.is_finite()
            || !(0. ..=1.).contains(&base_mass)
            || !(0. ..=1.).contains(&edge_mass)
            || !mass.is_finite()
            || mass <= 0.
        {
            return Err(Error::InvalidInput(
                "domain must contain representable positive probability mass",
            ));
        }
        Ok(Self {
            distribution,
            lower,
            upper,
            base_mass,
            mass,
            upper_tail,
        })
    }

    fn cdf(&self, x: f64) -> Result<f64> {
        if x.is_nan() {
            return Err(Error::InvalidInput("CDF argument must not be NaN"));
        }
        if x <= self.lower {
            return Ok(0.);
        }
        if x >= self.upper {
            return Ok(1.);
        }
        let probability = if self.upper_tail {
            sf(self.distribution, x)?
        } else {
            cdf(self.distribution, x)?
        };
        if !probability.is_finite() || !(0. ..=1.).contains(&probability) {
            return Err(Error::NonFiniteEvaluation);
        }
        let relative_mass = if self.upper_tail {
            self.base_mass - probability
        } else {
            probability - self.base_mass
        };
        Ok((relative_mass / self.mass).clamp(0., 1.))
    }
}

#[derive(Clone, Copy, Debug)]
enum Interpolation {
    Linear,
    Hermite,
}

fn secant(a: Node, b: Node) -> Result<f64> {
    let slope = (b.x - a.x) / (b.u - a.u);
    if slope.is_finite() && slope > 0. {
        Ok(slope)
    } else {
        Err(Error::InvalidInput(
            "ill-conditioned inverse CDF tabulation",
        ))
    }
}

// PCHIP / Fritsch-Butland slopes for the monotone inverse function x(u).
fn hermite_slopes(nodes: &[Node]) -> Result<Vec<f64>> {
    let slopes = nodes
        .windows(2)
        .map(|pair| secant(pair[0], pair[1]))
        .collect::<Result<Vec<_>>>()?;
    if slopes.len() == 1 {
        return Ok(vec![slopes[0]; 2]);
    }
    let mut derivatives = vec![0.; nodes.len()];
    for i in 1..slopes.len() {
        let h_prev = nodes[i].u - nodes[i - 1].u;
        let h_next = nodes[i + 1].u - nodes[i].u;
        // Work with normalized weights to avoid overflow for valid tiny steps.
        let ratio = h_prev / (h_prev + h_next);
        let w_prev = (2. - ratio) / 3.;
        let w_next = (1. + ratio) / 3.;
        // Normalize the harmonic mean by its smallest secant. Directly
        // taking reciprocal slopes can overflow for subnormal derivatives.
        let scale = slopes[i - 1].min(slopes[i]);
        derivatives[i] = scale / (w_prev * (scale / slopes[i - 1]) + w_next * (scale / slopes[i]));
    }
    let endpoint = |first: f64, second: f64, h0: f64, h1: f64| {
        let ratio = h0 / (h0 + h1);
        (first + ratio * (first - second)).clamp(0., 3. * first)
    };
    derivatives[0] = endpoint(
        slopes[0],
        slopes[1],
        nodes[1].u - nodes[0].u,
        nodes[2].u - nodes[1].u,
    );
    let last = nodes.len() - 1;
    derivatives[last] = endpoint(
        slopes[last - 1],
        slopes[last - 2],
        nodes[last].u - nodes[last - 1].u,
        nodes[last - 1].u - nodes[last - 2].u,
    );
    if derivatives.iter().all(|m| m.is_finite() && *m >= 0.) {
        Ok(derivatives)
    } else {
        Err(Error::InvalidInput("nonfinite inverse Hermite derivative"))
    }
}

fn interpolate(nodes: &[Node], derivatives: &[f64], index: usize, u: f64) -> f64 {
    let a = nodes[index];
    let b = nodes[index + 1];
    let h = b.u - a.u;
    let t = ((u - a.u) / h).clamp(0., 1.);
    if derivatives.is_empty() {
        return a.x + t * (b.x - a.x);
    }
    // Interpolate the offset from a.x instead of combining large absolute
    // endpoint values with opposite-sign basis coefficients.
    let delta = b.x - a.x;
    let secant = delta / h;
    let m0 = derivatives[index] / secant;
    let m1 = derivatives[index + 1] / secant;
    let shape = t * t * (3. - 2. * t) + t * (1. - t) * ((1. - t) * m0 - t * m1);
    (a.x + delta * shape.clamp(0., 1.)).clamp(a.x, b.x)
}

fn invert_cdf(cumulative: ConditionalCdf, a: Node, b: Node, u_resolution: f64) -> Result<Node> {
    let target = a.u + 0.5 * (b.u - a.u);
    let mut left = a.x;
    let mut right = b.x;
    let mut left_probability = a.u;
    let mut right_probability = b.u;
    // Root error is kept far below the requested interpolation residual.
    // Stop once it is resolved rather than always bisecting to x-machine-ulp.
    let root_tolerance = (u_resolution * 1e-3).min((b.u - a.u) * 1e-3);
    for _ in 0..MAX_BISECTIONS {
        let midpoint = left + 0.5 * (right - left);
        if midpoint <= left || midpoint >= right {
            break;
        }
        let probability = cumulative.cdf(midpoint)?;
        if (probability - target).abs() <= root_tolerance && a.u < probability && probability < b.u
        {
            return Ok(Node {
                x: midpoint,
                u: probability,
            });
        }
        if probability < target {
            left = midpoint;
            left_probability = probability;
        } else {
            right = midpoint;
            right_probability = probability;
        }
    }
    let left_node = Node {
        x: left,
        u: left_probability,
    };
    let right_node = Node {
        x: right,
        u: right_probability,
    };
    let middle = if (left_node.u - target).abs() < (right_node.u - target).abs() {
        left_node
    } else {
        right_node
    };
    if !(a.x < middle.x && middle.x < b.x && a.u < middle.u && middle.u < b.u) {
        return Err(Error::InvalidInput(
            "CDF is insufficiently resolved for inverse interpolation",
        ));
    }
    Ok(middle)
}

#[derive(Clone, Debug)]
struct InverseTable {
    conditional: ConditionalCdf,
    nodes: Vec<Node>,
    derivatives: Vec<f64>,
    generator: StdRng,
}

impl InverseTable {
    fn new(
        distribution: Distribution,
        domain: (f64, f64),
        u_resolution: f64,
        max_nodes: usize,
        seed: u64,
        kind: Interpolation,
    ) -> Result<Self> {
        if !u_resolution.is_finite() || !(0. < u_resolution && u_resolution <= 0.1) {
            return Err(Error::InvalidInput(
                "u_resolution must be finite and in (0,0.1]",
            ));
        }
        if !(2..=MAX_NODES).contains(&max_nodes) {
            return Err(Error::InvalidInput("max_nodes must be in 2..=8192"));
        }
        let conditional = ConditionalCdf::new(distribution, domain.0, domain.1)?;
        let mut nodes = vec![Node { u: 0., x: domain.0 }, Node { u: 1., x: domain.1 }];
        let mut derivatives = Vec::new();
        let mut converged = false;
        for _ in 0..MAX_REFINEMENTS {
            if matches!(kind, Interpolation::Hermite) {
                derivatives = hermite_slopes(&nodes)?;
            }
            let mut next = Vec::with_capacity(max_nodes.min(nodes.len().saturating_mul(2)));
            next.push(nodes[0]);
            let mut split = false;
            for i in 0..nodes.len() - 1 {
                let a = nodes[i];
                let b = nodes[i + 1];
                let mut max_error: f64 = 0.;
                for fraction in [0.25, 0.5, 0.75] {
                    let u = a.u + fraction * (b.u - a.u);
                    let x = interpolate(&nodes, &derivatives, i, u);
                    if !x.is_finite() {
                        return Err(Error::NonFiniteEvaluation);
                    }
                    max_error = max_error.max((conditional.cdf(x)? - u).abs());
                }
                if max_error > u_resolution {
                    if next.len() + nodes.len() - i > max_nodes {
                        return Err(Error::InvalidInput(
                            "max_nodes exhausted before requested u_resolution",
                        ));
                    }
                    next.push(invert_cdf(conditional, a, b, u_resolution)?);
                    split = true;
                }
                next.push(b);
            }
            nodes = next;
            if !split {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(Error::InvalidInput(
                "inverse interpolation refinement limit exceeded",
            ));
        }
        if matches!(kind, Interpolation::Hermite) {
            derivatives = hermite_slopes(&nodes)?;
        }
        Ok(Self {
            conditional,
            nodes,
            derivatives,
            generator: StdRng::seed_from_u64(seed),
        })
    }

    fn ppf(&self, probability: f64) -> Result<f64> {
        valid_probability(probability)?;
        if probability == 0. {
            return Ok(self.conditional.lower);
        }
        if probability == 1. {
            return Ok(self.conditional.upper);
        }
        let index = self.nodes.partition_point(|node| node.u <= probability) - 1;
        let result = interpolate(&self.nodes, &self.derivatives, index, probability);
        if !result.is_finite() {
            return Err(Error::NonFiniteEvaluation);
        }
        Ok(result)
    }

    fn qrvs(&self, probabilities: &[f64]) -> Result<Vec<f64>> {
        quantiles(probabilities, |u| self.ppf(u))
    }

    fn rvs(&mut self, n: usize) -> Result<Vec<f64>> {
        valid_batch(n)?;
        let mut values = Vec::with_capacity(n);
        for _ in 0..n {
            let u = uniform_interior(&mut self.generator);
            values.push(self.ppf(u)?);
        }
        Ok(values)
    }

    fn u_error(&self, count: usize) -> Result<UError> {
        measured_error(count, |u| self.ppf(u), |x| self.conditional.cdf(x))
    }
}

/// Shape-preserving cubic-Hermite approximation to a conditional inverse CDF.
///
/// The interpolant is a monotone piecewise cubic `H(u) ≈ F_T⁻¹(u)` using
/// Fritsch-Butland slopes, where `F_T(x) = (F(x)-F(a))/(F(b)-F(a))` for
/// `a <= x <= b`. An adaptive CDF inversion constructs the knots.
///
/// ## Parameters and conventions
///
/// `distribution` is a native continuous law; `domain=(a,b)` selects an
/// increasing, finite interval in observation units, conditioning the law to
/// that interval. `u_resolution` controls tested absolute CDF residuals,
/// `max_nodes` caps adaptive knots, and `seed` initializes `StdRng`.
///
/// ## Domain and special values
///
/// Require `0 < u_resolution <= 0.1`, `2 <= max_nodes <= 8192`, finite
/// `a<b`, and positive representable probability `F(b)-F(a)`. Probabilities
/// are in `[0,1]`; the endpoints map exactly to `a` and `b`.
///
/// ## Returns
///
/// Scalar conditional quantiles in observation units and dimensionless CDF
/// values, or vectors of pseudorandom/supplied quasi-random quantiles.
///
/// ## Errors
///
/// Invalid distribution parameters, degenerate domains, non-monotone or
/// unresolved CDF values, exceeded node/work limits, and invalid probability
/// inputs return errors instead of producing an unvalidated interpolation.
///
/// ## Numerical behavior
///
/// Each refinement checks CDF residuals at three interior probabilities per
/// interval and bisects the true conditional CDF where needed (up to 96
/// bisections per split, 64 passes). Knot inversion stops once its probability
/// residual is below 0.1% of the requested resolution. The resolution bounds
/// these *tested* errors, not the global approximation error. Unlike SciPy HINV,
/// this implementation uses source CDF/survival evaluations and cubic splines
/// on a finite caller-specified interval. `cdf(x)` evaluates the conditioned
/// source probability independently of the interpolation table; upper-tail
/// intervals use survival probabilities to preserve small differences.
///
/// ## Examples
///
/// ```
/// use sci_stats::{distributions::Distribution, sampling::NumericalInverseHermite};
/// let d = Distribution::Uniform { low: 0., high: 1. };
/// let mut inv = NumericalInverseHermite::new(d, (0., 1.), 1e-6, 128, 3)?;
/// assert!((inv.ppf(0.3)? - 0.3).abs() < 1e-12);
/// assert_eq!(inv.rvs(4)?.len(), 4);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy NumericalInverseHermite](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sampling.NumericalInverseHermite.html).
#[derive(Clone, Debug)]
pub struct NumericalInverseHermite(InverseTable);

impl NumericalInverseHermite {
    /// Build an adaptive monotone cubic quantile spline for `[a,b]`.
    ///
    /// The requested absolute probability resolution is checked at three
    /// points per interval; failure to reach it within `max_nodes` is an error.
    /// The seed selects a deterministic `rvs` stream.
    pub fn new(
        distribution: Distribution,
        domain: (f64, f64),
        u_resolution: f64,
        max_nodes: usize,
        seed: u64,
    ) -> Result<Self> {
        Ok(Self(InverseTable::new(
            distribution,
            domain,
            u_resolution,
            max_nodes,
            seed,
            Interpolation::Hermite,
        )?))
    }

    /// Approximate `F_T⁻¹(p)` on `[0,1]` by a monotone cubic spline.
    ///
    /// Quantiles have observation units; endpoints return the exact truncation
    /// bounds. Invalid probabilities and numerical failures return errors.
    pub fn ppf(&self, probability: f64) -> Result<f64> {
        self.0.ppf(probability)
    }

    /// Evaluate `F_T(x)` of the conditioned source distribution.
    ///
    /// Returns zero below the lower domain bound and one above the upper;
    /// interior results come from the underlying CDF. NaN returns an error.
    pub fn cdf(&self, x: f64) -> Result<f64> {
        self.0.conditional.cdf(x)
    }

    /// Return the finite interval `(a,b)` used for conditioning.
    ///
    /// Bounds have the same physical units as the observations.
    pub fn support(&self) -> (f64, f64) {
        (self.0.conditional.lower, self.0.conditional.upper)
    }

    /// Return the number of adaptive CDF knots used by the spline.
    ///
    /// The count is between two and the constructor's `max_nodes` limit.
    pub fn node_count(&self) -> usize {
        self.0.nodes.len()
    }

    /// Transform up to 1,048,576 supplied probabilities using the cubic spline.
    ///
    /// Inputs must be in `[0,1]`. Output ordering matches input ordering;
    /// the pseudorandom generator state is not consumed.
    pub fn qrvs(&self, probabilities: &[f64]) -> Result<Vec<f64>> {
        self.0.qrvs(probabilities)
    }

    /// Draw at most 1,048,576 inverse-CDF pseudorandom observations.
    ///
    /// Each draw evaluates the cubic spline at a seeded uniform variate.
    /// Zero-length draws return an empty vector without advancing RNG state.
    pub fn rvs(&mut self, n: usize) -> Result<Vec<f64>> {
        self.0.rvs(n)
    }

    /// Measure maximum and mean `|F_T(H(u))-u|` on an interior uniform grid.
    ///
    /// `count` must be in `1..=1048576`. These residuals are deterministic
    /// diagnostics and do not certify a global error bound.
    pub fn u_error(&self, count: usize) -> Result<UError> {
        self.0.u_error(count)
    }
}

/// Piecewise-linear approximation to a conditional inverse CDF.
///
/// On each knot interval `[u_i,u_(i+1)]`, `Q(u) = x_i +
/// (u-u_i)(x_(i+1)-x_i)/(u_(i+1)-u_i)`. The knots are adaptively obtained by
/// inverting the source CDF `F`, conditioned to a finite `[a,b]` interval.
///
/// ## Parameters and conventions
///
/// `distribution` is a native continuous law, `domain=(a,b)` is a finite
/// conditioning interval, `u_resolution` is the requested tested probability
/// residual, `max_nodes` limits knots, and `seed` controls pseudorandom draws.
///
/// ## Domain and special values
///
/// Require finite `a<b`, positive representable `F(b)-F(a)`,
/// `0 < u_resolution <= 0.1`, `2 <= max_nodes <= 8192`, and `p` in `[0,1]`.
/// Endpoint quantiles return `a` and `b` exactly.
///
/// ## Returns
///
/// Approximate conditional quantiles in observation units, normalized CDF
/// values in `[0,1]`, or vectors of random/externally supplied quantiles.
///
/// ## Errors
///
/// Invalid distributions, nonincreasing CDFs, impossible resolution targets,
/// exceeded node limits and invalid probabilities return errors.
///
/// ## Numerical behavior
///
/// Each segment is validated against three CDF residual probes, splitting
/// segments with excessive error. Midpoint roots use at most 96 bisections,
/// stopping when knot probability error is below 0.1% of the requested
/// resolution; construction uses at most 64 refinement passes. The residual threshold is
/// sampled, not a rigorous uniform error guarantee. Unlike SciPy PINV, this
/// adaptation uses monotone degree-one interpolation only, and always requires
/// a finite conditioning interval. `cdf(x)` uses the original distribution's
/// CDF or survival function, selecting the more accurate upper-tail difference.
///
/// ## Examples
///
/// ```
/// use sci_stats::{distributions::Distribution, sampling::NumericalInversePolynomial};
/// let d = Distribution::Uniform { low: -2., high: 2. };
/// let inv = NumericalInversePolynomial::new(d, (-2., 2.), 1e-6, 64, 11)?;
/// assert_eq!(inv.qrvs(&[0., 0.25, 1.])?, vec![-2., -1., 2.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// ## References
///
/// [SciPy NumericalInversePolynomial](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sampling.NumericalInversePolynomial.html).
#[derive(Clone, Debug)]
pub struct NumericalInversePolynomial(InverseTable);

impl NumericalInversePolynomial {
    /// Build an adaptive linear quantile table on a finite conditioned domain.
    ///
    /// Refinement must reach the requested probability residual at three
    /// interior points per segment before the `max_nodes` limit is exhausted.
    pub fn new(
        distribution: Distribution,
        domain: (f64, f64),
        u_resolution: f64,
        max_nodes: usize,
        seed: u64,
    ) -> Result<Self> {
        Ok(Self(InverseTable::new(
            distribution,
            domain,
            u_resolution,
            max_nodes,
            seed,
            Interpolation::Linear,
        )?))
    }

    /// Approximate the conditioned quantile `F_T⁻¹(p)` by linear interpolation.
    ///
    /// `p` must belong to `[0,1]`; endpoint quantiles equal the finite domain
    /// bounds. Invalid probabilities or nonfinite interpolation return errors.
    pub fn ppf(&self, probability: f64) -> Result<f64> {
        self.0.ppf(probability)
    }

    /// Evaluate `F_T(x) = (F(x)-F(a))/(F(b)-F(a))`, clipped to `[0,1]`.
    ///
    /// Uses the source CDF for interior values; x may be infinite, but NaN is
    /// invalid. Results are dimensionless.
    pub fn cdf(&self, x: f64) -> Result<f64> {
        self.0.conditional.cdf(x)
    }

    /// Return the finite conditioning interval `(a,b)` in observation units.
    pub fn support(&self) -> (f64, f64) {
        (self.0.conditional.lower, self.0.conditional.upper)
    }

    /// Return the number of adaptive linear interpolation knots.
    ///
    /// The count is at most the constructor's `max_nodes` limit.
    pub fn node_count(&self) -> usize {
        self.0.nodes.len()
    }

    /// Transform at most 1,048,576 supplied uniform probabilities in `[0,1]`.
    ///
    /// Output order matches the input; RNG state is not affected. Invalid
    /// probabilities or overlong input are errors.
    pub fn qrvs(&self, probabilities: &[f64]) -> Result<Vec<f64>> {
        self.0.qrvs(probabilities)
    }

    /// Sample `n` variates from the interpolated quantile using seeded uniforms.
    ///
    /// `n` must be at most 1,048,576; zero draws do not advance the RNG.
    pub fn rvs(&mut self, n: usize) -> Result<Vec<f64>> {
        self.0.rvs(n)
    }

    /// Measure max/mean `|F_T(Q(u))-u|` on an interior uniform probability grid.
    ///
    /// Counts must be in `1..=1048576`. This deterministic diagnostic does not
    /// prove a worst-case bound between the grid points.
    pub fn u_error(&self, count: usize) -> Result<UError> {
        self.0.u_error(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_is_exact_for_all_three_engines() -> Result<()> {
        let law = Distribution::Uniform { low: -2., high: 2. };
        let mut fast = FastGeneratorInversion::new(law, 42)?;
        let mut hermite = NumericalInverseHermite::new(law, (-2., 2.), 1e-8, 16, 42)?;
        let mut linear = NumericalInversePolynomial::new(law, (-2., 2.), 1e-8, 16, 42)?;
        let probabilities = [0., 0.125, 0.5, 0.875, 1.];
        for (&p, &x) in probabilities.iter().zip(&[-2., -1.5, 0., 1.5, 2.]) {
            assert!((fast.ppf(p)? - x).abs() < 1e-12);
            assert!((hermite.ppf(p)? - x).abs() < 1e-12);
            assert!((linear.ppf(p)? - x).abs() < 1e-12);
            assert!((hermite.cdf(x)? - p).abs() < 1e-12);
            assert!((linear.cdf(x)? - p).abs() < 1e-12);
        }
        assert_eq!(hermite.node_count(), 2);
        assert_eq!(linear.node_count(), 2);
        let expected = fast.rvs(8)?;
        for (actual, reference) in hermite.rvs(8)?.iter().zip(&expected) {
            assert!((actual - reference).abs() < 1e-12);
        }
        for (actual, reference) in linear.rvs(8)?.iter().zip(&expected) {
            assert!((actual - reference).abs() < 1e-12);
        }
        Ok(())
    }

    #[test]
    fn nonlinear_cdf_converges_and_samples_inside_domain() -> Result<()> {
        let law = Distribution::Exponential { scale: 1. };
        let domain = (0., 5.);
        let mut cubic = NumericalInverseHermite::new(law, domain, 3e-5, 512, 9)?;
        let mut linear = NumericalInversePolynomial::new(law, domain, 3e-5, 512, 9)?;
        assert!(cubic.node_count() > 2);
        assert!(linear.node_count() > 2);
        let mut last_cubic = f64::NEG_INFINITY;
        let mut last_linear = f64::NEG_INFINITY;
        for i in 0..=256 {
            let p = i as f64 / 256.;
            let cubic_x = cubic.ppf(p)?;
            let linear_x = linear.ppf(p)?;
            assert!(cubic_x >= last_cubic);
            assert!(linear_x >= last_linear);
            last_cubic = cubic_x;
            last_linear = linear_x;
        }
        for p in [0.001, 0.01, 0.2, 0.5, 0.7, 0.95, 0.999] {
            let exact = -(1. - p * (1. - (-domain.1).exp())).ln();
            // A bounded CDF residual does not imply a uniform x-error near
            // low-density endpoints, where the inverse derivative is large.
            let x_tolerance = 6e-5 * (1. - (-domain.1).exp()) * exact.exp() + 1e-6;
            assert!((cubic.ppf(p)? - exact).abs() < x_tolerance);
            assert!((linear.ppf(p)? - exact).abs() < x_tolerance);
            assert!((cubic.cdf(cubic.ppf(p)?)? - p).abs() < 6e-5);
            assert!((linear.cdf(linear.ppf(p)?)? - p).abs() < 6e-5);
        }
        assert!(cubic.u_error(1024)?.max_error < 6e-5);
        assert!(linear.u_error(1024)?.max_error < 6e-5);
        assert!(cubic.rvs(128)?.iter().all(|x| (0. ..=5.).contains(x)));
        assert!(linear.rvs(128)?.iter().all(|x| (0. ..=5.).contains(x)));
        Ok(())
    }

    #[test]
    fn conditional_normal_and_reproducibility() -> Result<()> {
        let law = Distribution::Normal {
            mean: 0.,
            std_dev: 1.,
        };
        let mut a = NumericalInverseHermite::new(law, (-3., 3.), 1e-4, 512, 11)?;
        let mut b = NumericalInverseHermite::new(law, (-3., 3.), 1e-4, 512, 11)?;
        assert_eq!(a.rvs(64)?, b.rvs(64)?);
        assert!((a.ppf(0.5)?).abs() < 1e-8);
        assert_eq!(a.support(), (-3., 3.));
        assert!((a.cdf(0.)? - 0.5).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn upper_tail_conditioning_preserves_mass_after_cdf_rounds_to_one() -> Result<()> {
        let law = Distribution::Exponential { scale: 1. };
        // statrs computes F(x)=1-exp(-x): at x=37 the result is still
        // 0.9999999999999999, while at x>=40 it rounds to exactly 1.
        // Its survival function exp(-x) retains the interval's true mass.
        let domain = (40., 43.);
        assert_eq!(cdf(law, domain.0)?, 1.);
        assert_eq!(cdf(law, domain.1)?, 1.);
        assert!(sf(law, domain.0)? > sf(law, domain.1)?);
        let cubic = NumericalInverseHermite::new(law, domain, 1e-4, 512, 1)?;
        let linear = NumericalInversePolynomial::new(law, domain, 1e-4, 512, 1)?;
        assert!(cubic.0.conditional.upper_tail);
        assert!(cubic.0.conditional.mass > 0.);
        for u in [0., 0.1, 0.25, 0.5, 0.75, 0.9, 1.] {
            let expected = domain.0 - (1. - u * (1. - (-3.0_f64).exp())).ln();
            for actual in [cubic.ppf(u)?, linear.ppf(u)?] {
                assert!((actual - expected).abs() < 1e-3);
                assert!((cubic.cdf(actual)? - u).abs() < 3e-4);
            }
        }
        Ok(())
    }

    #[test]
    fn hermite_preserves_linear_quantiles_at_large_offsets() -> Result<()> {
        // Expanding absolute-coordinate cubic basis terms can lose an entire
        // 16384-unit ULP even though the linear quantiles are representable.
        let low = 1e20;
        let high = low + 65536.;
        let law = Distribution::Uniform { low, high };
        let cubic = NumericalInverseHermite::new(law, (low, high), 1e-8, 16, 7)?;
        assert_eq!(cubic.node_count(), 2);
        for p in [0., 0.25, 0.5, 0.75, 1.] {
            assert_eq!(cubic.ppf(p)?, low + p * (high - low));
        }
        Ok(())
    }

    #[test]
    fn hermite_harmonic_mean_handles_subnormal_secants() -> Result<()> {
        let tiny = f64::from_bits(1);
        let nodes = [
            Node { u: 0., x: 0. },
            Node { u: 0.5, x: tiny },
            Node {
                u: 1.,
                x: 2. * tiny,
            },
        ];
        let slopes = hermite_slopes(&nodes)?;
        assert_eq!(slopes, vec![2. * tiny; 3]);
        Ok(())
    }

    #[test]
    fn invalid_inputs_and_budget_fail_cleanly() -> Result<()> {
        let law = Distribution::Uniform { low: 0., high: 1. };
        let discrete = Distribution::Poisson { rate: 2. };
        assert!(FastGeneratorInversion::new(discrete, 1).is_err());
        assert!(NumericalInverseHermite::new(discrete, (0., 5.), 1e-3, 128, 1).is_err());
        assert!(NumericalInversePolynomial::new(law, (0., 0.), 1e-3, 128, 1).is_err());
        assert!(NumericalInverseHermite::new(law, (0., 1.), f64::NAN, 128, 1).is_err());
        assert!(NumericalInverseHermite::new(law, (0., 1.), 1e-3, 1, 1).is_err());
        assert!(
            NumericalInversePolynomial::new(
                Distribution::Exponential { scale: 1. },
                (0., 5.),
                1e-8,
                2,
                1,
            )
            .is_err()
        );
        let mut fast = FastGeneratorInversion::new(law, 1)?;
        assert!(fast.ppf(f64::NAN).is_err());
        assert!(fast.qrvs(&[0.1, 2.]).is_err());
        assert!(fast.rvs(MAX_SAMPLES + 1).is_err());
        assert!(fast.u_error(0).is_err());
        Ok(())
    }
}
