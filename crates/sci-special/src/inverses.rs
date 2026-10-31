//! Bounded numerical inversions and noncentral distribution mixtures.

use super::{betainc, gammainc, gammaincc, gammaln, ndtr};

use crate::backend::utils::brent_root;

/// Largest point of the logarithmic scan in [`positive_inverse`]: `1e-10·1.3^180`.
const POSITIVE_SCAN_MAX: f64 = 3.2e10;

/// Finds the root of `function(x) = target` in a bracket whose residuals at
/// the endpoints are known to have opposite signs.
///
/// Brent's method converges superlinearly; bisection takes over if it meets
/// a non-finite residual.
fn refine_bracket(
    function: impl Fn(f64) -> f64,
    target: f64,
    mut low: f64,
    mut high: f64,
    left: f64,
    right: f64,
) -> f64 {
    if let Some(root) = brent_root(low, high, left, right, |x| function(x) - target) {
        return root;
    }
    for _ in 0..256 {
        let middle = low + (high - low) * 0.5;
        if middle == low || middle == high {
            return middle;
        }
        let value = function(middle) - target;
        if !value.is_finite() {
            return f64::NAN;
        }
        if value == 0.0 {
            return middle;
        }
        if value.is_sign_positive() == left.is_sign_positive() {
            low = middle;
        } else {
            high = middle;
        }
    }
    low + (high - low) * 0.5
}

fn bisect(function: impl Fn(f64) -> f64, target: f64, low: f64, high: f64) -> f64 {
    if !(0.0..=1.0).contains(&target) {
        return f64::NAN;
    }
    let left = function(low) - target;
    let right = function(high) - target;
    if left == 0.0 {
        return low;
    }
    if right == 0.0 {
        return high;
    }
    if !left.is_finite()
        || !right.is_finite()
        || left.is_sign_positive() == right.is_sign_positive()
    {
        return f64::NAN;
    }
    refine_bracket(function, target, low, high, left, right)
}

/// Root of a function that is monotone on `[low, high]`, bracketed by
/// stepping from `start` toward the root with doubling steps.
///
/// Monotonicity makes any root in `[low, high]` the unique one, so this
/// returns the same root as [`bisect`] on the whole interval while avoiding
/// evaluations far from the root, which are expensive for Poisson mixtures
/// with large noncentrality. Non-finite residuals, or no sign change before
/// the end of the interval, fall back to [`bisect`].
fn monotone_bisect(
    function: impl Fn(f64) -> f64,
    target: f64,
    (low, high): (f64, f64),
    start: f64,
    step: f64,
    increasing: bool,
) -> f64 {
    if !(0.0..=1.0).contains(&target) {
        return f64::NAN;
    }
    let mut a = start;
    let mut ra = function(a) - target;
    if ra == 0.0 {
        return a;
    }
    if ra.is_finite() {
        // Residual below zero: an increasing function's root lies above.
        let up = (ra < 0.0) == increasing;
        let bound = if up { high } else { low };
        let mut width = step;
        loop {
            let b = if up {
                (a + width).min(bound)
            } else {
                (a - width).max(bound)
            };
            let rb = function(b) - target;
            if !rb.is_finite() {
                break;
            }
            if rb == 0.0 {
                return b;
            }
            if rb.is_sign_positive() != ra.is_sign_positive() {
                let (lo, hi, rl, rh) = if up { (a, b, ra, rb) } else { (b, a, rb, ra) };
                return refine_bracket(function, target, lo, hi, rl, rh);
            }
            if b == bound {
                break;
            }
            a = b;
            ra = rb;
            width *= 2.0;
        }
    }
    bisect(function, target, low, high)
}

/// Inverts a function that is monotone in a positive parameter.
///
/// Monotonicity makes the root unique, so the result agrees with the first
/// bracket of [`positive_inverse`]'s logarithmic scan, but the bracket is
/// found by stepping outward from 1 by factors of 4 and then refined with
/// Brent's method. Inputs that meet non-finite values or no sign change fall
/// back to the scan.
fn monotone_positive_inverse(
    function: impl Fn(f64) -> f64,
    probability: f64,
    increasing: bool,
) -> f64 {
    if !(0.0..=1.0).contains(&probability) {
        return f64::NAN;
    }
    let mut a = 1.0;
    let mut ra = function(a) - probability;
    if ra == 0.0 {
        return a;
    }
    if ra.is_finite() {
        // Residual below zero: an increasing function's root lies above.
        let up = (ra < 0.0) == increasing;
        loop {
            let b = if up {
                (a * 4.0).min(POSITIVE_SCAN_MAX)
            } else {
                (a * 0.25).max(1e-10)
            };
            let rb = function(b) - probability;
            if !rb.is_finite() {
                break;
            }
            if rb == 0.0 {
                return b;
            }
            if rb.is_sign_positive() != ra.is_sign_positive() {
                let (lo, hi, rl, rh) = if up { (a, b, ra, rb) } else { (b, a, rb, ra) };
                return refine_bracket(function, probability, lo, hi, rl, rh);
            }
            if b == POSITIVE_SCAN_MAX || b == 1e-10 {
                break;
            }
            a = b;
            ra = rb;
        }
    }
    positive_inverse(function, probability)
}

fn positive_inverse(function: impl Fn(f64) -> f64, probability: f64) -> f64 {
    if !(0.0..=1.0).contains(&probability) {
        return f64::NAN;
    }
    let mut low = 1e-10;
    let mut left = function(low) - probability;
    // Scan logarithmically: some parameter inverses have more than one root.
    // Select the first bracket, rather than claiming uniqueness.
    for _ in 0..180 {
        let high = low * 1.3;
        let right = function(high) - probability;
        if left == 0.0 {
            return low;
        }
        if left.is_finite()
            && right.is_finite()
            && left.is_sign_positive() != right.is_sign_positive()
        {
            return refine_bracket(function, probability, low, high, left, right);
        }
        low = high;
        left = right;
    }
    f64::NAN
}

fn poisson_mixture(lambda: f64, function: impl Fn(usize) -> f64) -> f64 {
    if !lambda.is_finite() || !(0.0..=10_000.0).contains(&lambda) {
        return f64::NAN;
    }
    if lambda == 0.0 {
        return function(0);
    }
    let center = lambda.floor() as usize;
    let central = (-lambda + (center as f64) * lambda.ln() - gammaln(center as f64 + 1.0)).exp();
    let mut sum = central * function(center);
    let mut mass = central;
    let mut weight = central;
    for j in (0..center).rev() {
        weight *= (j + 1) as f64 / lambda;
        sum += weight * function(j);
        mass += weight;
        if weight < 1e-16 * mass {
            break;
        }
    }
    weight = central;
    for j in center + 1..=center + 4096 {
        weight *= lambda / (j as f64);
        sum += weight * function(j);
        mass += weight;
        if weight < 1e-16 * mass {
            return sum / mass;
        }
    }
    f64::NAN
}

fn adaptive_simpson(f: &impl Fn(f64) -> f64, a: f64, b: f64, tol: f64, depth: u32) -> f64 {
    fn refine(
        f: &impl Fn(f64) -> f64,
        a: f64,
        b: f64,
        values: [f64; 3],
        whole: f64,
        tol: f64,
        depth: u32,
    ) -> f64 {
        let [fa, fm, fb] = values;
        let middle = (a + b) / 2.0;
        let fl = f((a + middle) / 2.0);
        let fr = f((middle + b) / 2.0);
        let left = (middle - a) * (fa + 4.0 * fl + fm) / 6.0;
        let right = (b - middle) * (fm + 4.0 * fr + fb) / 6.0;
        let delta = left + right - whole;
        if !delta.is_finite() {
            return f64::NAN;
        }
        if delta.abs() <= 15.0 * tol {
            return left + right + delta / 15.0;
        }
        if depth == 0 {
            return f64::NAN;
        }
        refine(f, a, middle, [fa, fl, fm], left, tol / 2.0, depth - 1)
            + refine(f, middle, b, [fm, fr, fb], right, tol / 2.0, depth - 1)
    }
    let values = [f(a), f((a + b) / 2.0), f(b)];
    let whole = (b - a) * (values[0] + 4.0 * values[1] + values[2]) / 6.0;
    refine(f, a, b, values, whole, tol, depth)
}

pub(super) fn integrate(f: impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    adaptive_simpson(&f, a, b, 2e-13, 24)
}

/// Inverse I_x(a,b)=p with respect to shape a.
///
/// ## Parameters and domain
///
/// b>0, x,p in (0,1).
///
/// ## Numerical behavior
///
/// Inversion brackets the root within bounds and refines it with Brent's method; failure to bracket or
/// converge returns NaN. Positive parameter searches cover `1e-10..1e10`.
/// If an inverse is not unique, the lowest bracketed positive root is selected.
/// Endpoint probabilities can be singular; use interior probabilities for inversion.
///
/// ## Examples
///
/// ```
/// let value = sci_special::probability::btdtria(0.3,2.5,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy btdtria](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.btdtria.html).
pub fn btdtria(p: f64, b: f64, x: f64) -> f64 {
    monotone_positive_inverse(|a| betainc(a, b, x), p, false)
}

/// Inverse I_x(a,b)=p with respect to shape b.
///
/// ## Parameters and domain
///
/// a>0, x,p in (0,1).
///
/// ## Numerical behavior
///
/// Inversion brackets the root within bounds and refines it with Brent's method; failure to bracket or
/// converge returns NaN. Positive parameter searches cover `1e-10..1e10`.
/// If an inverse is not unique, the lowest bracketed positive root is selected.
/// Endpoint probabilities can be singular; use interior probabilities for inversion.
///
/// ## Examples
///
/// ```
/// let value = sci_special::probability::btdtrib(2.5,0.3,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy btdtrib](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.btdtrib.html).
pub fn btdtrib(a: f64, p: f64, x: f64) -> f64 {
    monotone_positive_inverse(|b| betainc(a, b, x), p, true)
}

/// Continuous inverse binomial CDF with respect to count k.
///
/// ## Parameters and domain
///
/// n>0, p,y in (0,1); k interpolates the CDF through the incomplete beta function.
///
/// ## Numerical behavior
///
/// Inversion brackets the root within bounds and refines it with Brent's method; failure to bracket or
/// converge returns NaN. Positive parameter searches cover `1e-10..1e10`.
/// If an inverse is not unique, the lowest bracketed positive root is selected.
/// Endpoint probabilities can be singular; use interior probabilities for inversion.
///
/// ## Examples
///
/// ```
/// let value = sci_special::probability::bdtrik(0.3,5.0,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy bdtrik](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.bdtrik.html).
pub fn bdtrik(y: f64, n: f64, p: f64) -> f64 {
    if !n.is_finite() || n <= 0.0 || !(0.0..1.0).contains(&p) {
        f64::NAN
    } else {
        // The CDF increases in k; start at the mean and step by about one
        // standard deviation.
        monotone_bisect(
            |k| {
                if k <= -1.0 {
                    0.0
                } else if k >= n {
                    1.0
                } else {
                    betainc(n - k, k + 1.0, 1.0 - p)
                }
            },
            y,
            (-1.0, n),
            (n * p).clamp(-1.0, n),
            1.0 + (n * p * (1.0 - p)).sqrt(),
            true,
        )
    }
}

/// Continuous inverse binomial CDF with respect to trial count n.
///
/// ## Parameters and domain
///
/// k≥0, p,y in (0,1); returned n is a real-valued interpolation.
///
/// ## Numerical behavior
///
/// Inversion brackets the root within bounds and refines it with Brent's method; failure to bracket or
/// converge returns NaN. Positive parameter searches cover `1e-10..1e10`.
/// If an inverse is not unique, the lowest bracketed positive root is selected.
/// Endpoint probabilities can be singular; use interior probabilities for inversion.
///
/// ## Examples
///
/// ```
/// let value = sci_special::probability::bdtrin(2.0,0.3,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy bdtrin](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.bdtrin.html).
pub fn bdtrin(k: f64, y: f64, p: f64) -> f64 {
    monotone_positive_inverse(|d| betainc(d, k + 1.0, 1.0 - p), y, false) + k
}

/// Continuous inverse negative-binomial CDF with respect to failures k.
///
/// ## Parameters and domain
///
/// n>0, p,y in (0,1).
///
/// ## Numerical behavior
///
/// Inversion brackets the root within bounds and refines it with Brent's method; failure to bracket or
/// converge returns NaN. Positive parameter searches cover `1e-10..1e10`.
/// If an inverse is not unique, the lowest bracketed positive root is selected.
/// Endpoint probabilities can be singular; use interior probabilities for inversion.
///
/// ## Examples
///
/// ```
/// let value = sci_special::probability::nbdtrik(0.3,5.0,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy nbdtrik](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.nbdtrik.html).
pub fn nbdtrik(y: f64, n: f64, p: f64) -> f64 {
    monotone_positive_inverse(|s| betainc(n, s, p), y, true) - 1.0
}

/// Continuous inverse negative-binomial CDF with respect to successes n.
///
/// ## Parameters and domain
///
/// k≥0, p,y in (0,1).
///
/// ## Numerical behavior
///
/// Inversion brackets the root within bounds and refines it with Brent's method; failure to bracket or
/// converge returns NaN. Positive parameter searches cover `1e-10..1e10`.
/// If an inverse is not unique, the lowest bracketed positive root is selected.
/// Endpoint probabilities can be singular; use interior probabilities for inversion.
///
/// ## Examples
///
/// ```
/// let value = sci_special::probability::nbdtrin(2.0,0.3,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy nbdtrin](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.nbdtrin.html).
pub fn nbdtrin(k: f64, y: f64, p: f64) -> f64 {
    monotone_positive_inverse(|n| betainc(n, k + 1.0, p), y, false)
}

/// Continuous inverse Poisson CDF with respect to count k.
///
/// ## Parameters and domain
///
/// m>0, p in (0,1); the result interpolates Q(k+1,m).
///
/// ## Numerical behavior
///
/// Inversion brackets the root within bounds and refines it with Brent's method; failure to bracket or
/// converge returns NaN. Positive parameter searches cover `1e-10..1e10`.
/// If an inverse is not unique, the lowest bracketed positive root is selected.
/// Endpoint probabilities can be singular; use interior probabilities for inversion.
///
/// ## Examples
///
/// ```
/// let value = sci_special::probability::pdtrik(0.3,5.0);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy pdtrik](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pdtrik.html).
pub fn pdtrik(p: f64, m: f64) -> f64 {
    monotone_positive_inverse(|s| gammaincc(s, m), p, true) - 1.0
}

/// Inverse Student t CDF with respect to degrees of freedom.
///
/// ## Parameters and domain
///
/// Interior p, finite nonzero t; the root need not be unique.
///
/// ## Numerical behavior
///
/// Inversion brackets the root within bounds and refines it with Brent's method; failure to bracket or
/// converge returns NaN. Positive parameter searches cover `1e-10..1e10`.
/// If an inverse is not unique, the lowest bracketed positive root is selected.
/// Endpoint probabilities can be singular; use interior probabilities for inversion.
///
/// ## Examples
///
/// ```
/// let value = sci_special::probability::stdtridf(0.6742760175759245,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy stdtridf](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.stdtridf.html).
pub fn stdtridf(p: f64, t: f64) -> f64 {
    positive_inverse(|df| super::stdtr(df, t), p)
}

/// Inverse F CDF with respect to denominator degrees of freedom.
///
/// ## Parameters and domain
///
/// Positive dfn and x, interior p; the root need not be unique.
///
/// ## Numerical behavior
///
/// Inversion brackets the root within bounds and refines it with Brent's method; failure to bracket or
/// converge returns NaN. Positive parameter searches cover `1e-10..1e10`.
/// If an inverse is not unique, the lowest bracketed positive root is selected.
/// Endpoint probabilities can be singular; use interior probabilities for inversion.
///
/// ## Examples
///
/// ```
/// let value = sci_special::probability::fdtridfd(3.0,0.30154736269507604,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy fdtridfd](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.fdtridfd.html).
pub fn fdtridfd(dfn: f64, p: f64, x: f64) -> f64 {
    positive_inverse(|dfd| super::fdtr(dfn, dfd, x), p)
}

/// Noncentral chi-square CDF Σⱼ Poisson(j;nc/2) P(df/2+j,x/2).
///
/// ## Parameters and domain
///
/// df>0, x≥0, and 0≤nc≤20000. The sum is centered on the Poisson mode to avoid underflow.
///
/// ## Numerical behavior
///
/// Inversion brackets the root within bounds and refines it with Brent's method; failure to bracket or
/// converge returns NaN. Positive parameter searches cover `1e-10..1e10`.
/// If an inverse is not unique, the lowest bracketed positive root is selected.
/// Endpoint probabilities can be singular; use interior probabilities for inversion.
///
/// ## Examples
///
/// ```
/// let value = sci_special::probability::chndtr(2.5,3.0,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy chndtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chndtr.html).
pub fn chndtr(x: f64, df: f64, nc: f64) -> f64 {
    if !df.is_finite() || df <= 0.0 || !(0.0..=20_000.0).contains(&nc) || x.is_nan() || x < 0.0 {
        f64::NAN
    } else if x == f64::INFINITY {
        1.0
    } else {
        poisson_mixture(nc / 2.0, |j| gammainc(df / 2.0 + j as f64, x / 2.0))
    }
}

/// Inverse noncentral chi-square CDF with respect to x.
///
/// ## Parameters and domain
///
/// df>0, 0≤nc≤20000, and p in (0,1).
///
/// ## Numerical behavior
///
/// Inversion brackets the root within bounds and refines it with Brent's method; failure to bracket or
/// converge returns NaN. Positive parameter searches cover `1e-10..1e10`.
/// If an inverse is not unique, the lowest bracketed positive root is selected.
/// Endpoint probabilities can be singular; use interior probabilities for inversion.
///
/// ## Examples
///
/// ```
/// let value = sci_special::probability::chndtrix(0.3,3.0,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy chndtrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chndtrix.html).
pub fn chndtrix(p: f64, df: f64, nc: f64) -> f64 {
    monotone_positive_inverse(|x| chndtr(x, df, nc), p, true)
}

/// Inverse noncentral chi-square CDF with respect to degrees of freedom.
///
/// ## Parameters and domain
///
/// x>0, 0≤nc≤20000, and p in (0,1).
///
/// ## Numerical behavior
///
/// Inversion brackets the root within bounds and refines it with Brent's method; failure to bracket or
/// converge returns NaN. Positive parameter searches cover `1e-10..1e10`.
/// If an inverse is not unique, the lowest bracketed positive root is selected.
/// Endpoint probabilities can be singular; use interior probabilities for inversion.
///
/// ## Examples
///
/// ```
/// let value = sci_special::probability::chndtridf(2.5,0.3,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy chndtridf](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chndtridf.html).
pub fn chndtridf(x: f64, p: f64, nc: f64) -> f64 {
    monotone_positive_inverse(|df| chndtr(x, df, nc), p, false)
}

/// Inverse noncentral chi-square CDF with respect to noncentrality.
///
/// ## Parameters and domain
///
/// x,df>0, interior p; returns NaN if p exceeds the central CDF.
///
/// ## Numerical behavior
///
/// Inversion brackets the root within bounds and refines it with Brent's method; failure to bracket or
/// converge returns NaN. Positive parameter searches cover `1e-10..1e10`.
/// If an inverse is not unique, the lowest bracketed positive root is selected.
/// Endpoint probabilities can be singular; use interior probabilities for inversion.
///
/// ## Examples
///
/// ```
/// let value = sci_special::probability::chndtrinc(2.5,3.0,0.3);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy chndtrinc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chndtrinc.html).
pub fn chndtrinc(x: f64, df: f64, p: f64) -> f64 {
    monotone_bisect(|nc| chndtr(x, df, nc), p, (0.0, 20_000.0), 0.0, 1.0, false)
}

/// Noncentral F CDF as a Poisson mixture of regularized incomplete beta functions.
///
/// ## Parameters and domain
///
/// Positive finite degrees of freedom, x≥0, and 0≤nc≤20000.
///
/// ## Numerical behavior
///
/// Inversion brackets the root within bounds and refines it with Brent's method; failure to bracket or
/// converge returns NaN. Positive parameter searches cover `1e-10..1e10`.
/// If an inverse is not unique, the lowest bracketed positive root is selected.
/// Endpoint probabilities can be singular; use interior probabilities for inversion.
///
/// ## Examples
///
/// ```
/// let value = sci_special::probability::ncfdtr(3.0,5.0,0.5,2.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy ncfdtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ncfdtr.html).
pub fn ncfdtr(dfn: f64, dfd: f64, nc: f64, x: f64) -> f64 {
    if !dfn.is_finite()
        || !dfd.is_finite()
        || dfn <= 0.0
        || dfd <= 0.0
        || !(0.0..=20_000.0).contains(&nc)
        || x.is_nan()
        || x < 0.0
    {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return 1.0;
    }
    let z = 1.0 / (1.0 + dfd / (dfn * x));
    poisson_mixture(nc / 2.0, |j| betainc(dfn / 2.0 + j as f64, dfd / 2.0, z))
}

/// Inverse noncentral F CDF with respect to x.
///
/// ## Parameters and domain
///
/// Positive degrees of freedom, 0≤nc≤20000, and p in (0,1).
///
/// ## Numerical behavior
///
/// Inversion brackets the root within bounds and refines it with Brent's method; failure to bracket or
/// converge returns NaN. Positive parameter searches cover `1e-10..1e10`.
/// If an inverse is not unique, the lowest bracketed positive root is selected.
/// Endpoint probabilities can be singular; use interior probabilities for inversion.
///
/// ## Examples
///
/// ```
/// let value = sci_special::probability::ncfdtri(3.0,5.0,0.5,0.3);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy ncfdtri](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ncfdtri.html).
pub fn ncfdtri(dfn: f64, dfd: f64, nc: f64, p: f64) -> f64 {
    monotone_positive_inverse(|x| ncfdtr(dfn, dfd, nc, x), p, true)
}

/// Inverse noncentral F CDF with respect to denominator degrees of freedom.
///
/// ## Parameters and domain
///
/// Positive dfn,x, nonnegative nc≤20000, and interior p. Roots need not be unique.
///
/// ## Numerical behavior
///
/// Inversion brackets the root within bounds and refines it with Brent's method; failure to bracket or
/// converge returns NaN. Positive parameter searches cover `1e-10..1e10`.
/// If an inverse is not unique, the lowest bracketed positive root is selected.
/// Endpoint probabilities can be singular; use interior probabilities for inversion.
///
/// ## Examples
///
/// ```
/// let value = sci_special::probability::ncfdtridfd(3.0,0.25637506749643246,0.5,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy ncfdtridfd](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ncfdtridfd.html).
pub fn ncfdtridfd(dfn: f64, p: f64, nc: f64, x: f64) -> f64 {
    positive_inverse(|dfd| ncfdtr(dfn, dfd, nc, x), p)
}

/// Inverse noncentral F CDF with respect to numerator degrees of freedom.
///
/// ## Parameters and domain
///
/// Positive dfd,x, nonnegative nc≤20000, and interior p. Roots need not be unique.
///
/// ## Numerical behavior
///
/// Inversion brackets the root within bounds and refines it with Brent's method; failure to bracket or
/// converge returns NaN. Positive parameter searches cover `1e-10..1e10`.
/// If an inverse is not unique, the lowest bracketed positive root is selected.
/// Endpoint probabilities can be singular; use interior probabilities for inversion.
///
/// ## Examples
///
/// ```
/// let value = sci_special::probability::ncfdtridfn(0.25637506749643246,5.0,0.5,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy ncfdtridfn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ncfdtridfn.html).
pub fn ncfdtridfn(p: f64, dfd: f64, nc: f64, x: f64) -> f64 {
    positive_inverse(|dfn| ncfdtr(dfn, dfd, nc, x), p)
}

/// Inverse noncentral F CDF with respect to noncentrality.
///
/// ## Parameters and domain
///
/// Positive degrees of freedom and x, interior p not exceeding the central CDF.
///
/// ## Numerical behavior
///
/// Inversion brackets the root within bounds and refines it with Brent's method; failure to bracket or
/// converge returns NaN. Positive parameter searches cover `1e-10..1e10`.
/// If an inverse is not unique, the lowest bracketed positive root is selected.
/// Endpoint probabilities can be singular; use interior probabilities for inversion.
///
/// ## Examples
///
/// ```
/// let value = sci_special::probability::ncfdtrinc(3.0,5.0,0.3,2.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy ncfdtrinc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ncfdtrinc.html).
pub fn ncfdtrinc(dfn: f64, dfd: f64, p: f64, x: f64) -> f64 {
    monotone_bisect(
        |nc| ncfdtr(dfn, dfd, nc, x),
        p,
        (0.0, 20_000.0),
        0.0,
        1.0,
        false,
    )
}

/// Noncentral Student t CDF E`[Φ(t√(V/df)−nc)]`, V~chi-square(df).
///
/// ## Parameters and domain
///
/// Finite df in `[0.01,10000]`, abs(nc)≤100, and real t. Uses Poisson-weighted incomplete beta series. Cancellation can reduce relative accuracy in extreme tails.
///
/// ## Numerical behavior
///
/// Inversion brackets the root within bounds and refines it with Brent's method; failure to bracket or
/// converge returns NaN. Positive parameter searches cover `1e-10..1e10`.
/// If an inverse is not unique, the lowest bracketed positive root is selected.
/// Endpoint probabilities can be singular; use interior probabilities for inversion.
///
/// ## Examples
///
/// ```
/// let value = sci_special::probability::nctdtr(3.0,0.5,0.3);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy nctdtr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.nctdtr.html).
pub fn nctdtr(df: f64, nc: f64, t: f64) -> f64 {
    if !df.is_finite()
        || !(0.01..=10_000.0).contains(&df)
        || !nc.is_finite()
        || nc.abs() > 100.0
        || t.is_nan()
    {
        return f64::NAN;
    }
    if t == 0.0 {
        return ndtr(-nc);
    }
    if t == f64::INFINITY {
        return 1.0;
    }
    if t == f64::NEG_INFINITY {
        return 0.0;
    }
    if nc == 0.0 {
        return super::stdtr(df, t);
    }
    if t < 0.0 {
        return 1.0 - nctdtr(df, -nc, -t);
    }
    let x = 1.0 / (1.0 + df / (t * t));
    let b = df / 2.0;
    let series = poisson_mixture(nc * nc / 2.0, |j| {
        let j = j as f64;
        let ratio = nc / std::f64::consts::SQRT_2 * (gammaln(j + 1.0) - gammaln(j + 1.5)).exp();
        0.5 * (betainc(j + 0.5, b, x) + ratio * betainc(j + 1.0, b, x))
    });
    (ndtr(-nc) + series).clamp(0.0, 1.0)
}

/// Inverse noncentral Student t CDF with respect to t.
///
/// ## Parameters and domain
///
/// df in `[0.01,10000]`, abs(nc)≤100, interior p; t is searched in `[−1e8,1e8]`.
///
/// ## Numerical behavior
///
/// Inversion brackets the root within bounds and refines it with Brent's method; failure to bracket or
/// converge returns NaN. Positive parameter searches cover `1e-10..1e10`.
/// If an inverse is not unique, the lowest bracketed positive root is selected.
/// Endpoint probabilities can be singular; use interior probabilities for inversion.
///
/// ## Examples
///
/// ```
/// let value = sci_special::probability::nctdtrit(3.0,0.5,0.3);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy nctdtrit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.nctdtrit.html).
pub fn nctdtrit(df: f64, nc: f64, p: f64) -> f64 {
    monotone_bisect(|t| nctdtr(df, nc, t), p, (-1e8, 1e8), nc, 1.0, true)
}

/// Inverse noncentral Student t CDF with respect to noncentrality.
///
/// ## Parameters and domain
///
/// df in `[0.01,10000]`, finite t, interior p; noncentrality in `[−100,100]`.
///
/// ## Numerical behavior
///
/// Inversion brackets the root within bounds and refines it with Brent's method; failure to bracket or
/// converge returns NaN. Positive parameter searches cover `1e-10..1e10`.
/// If an inverse is not unique, the lowest bracketed positive root is selected.
/// Endpoint probabilities can be singular; use interior probabilities for inversion.
///
/// ## Examples
///
/// ```
/// let value = sci_special::probability::nctdtrinc(3.0,0.3,0.5);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy nctdtrinc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.nctdtrinc.html).
pub fn nctdtrinc(df: f64, p: f64, t: f64) -> f64 {
    monotone_bisect(
        |nc| nctdtr(df, nc, t),
        p,
        (-100.0, 100.0),
        t.clamp(-100.0, 100.0),
        1.0,
        false,
    )
}

/// Inverse noncentral Student t CDF with respect to degrees of freedom.
///
/// ## Parameters and domain
///
/// Interior p, finite t, abs(nc)≤100; degrees of freedom restricted to `[0.01,10000]`.
///
/// ## Numerical behavior
///
/// Inversion brackets the root within bounds and refines it with Brent's method; failure to bracket or
/// converge returns NaN. Positive parameter searches cover `1e-10..1e10`.
/// If an inverse is not unique, the lowest bracketed positive root is selected.
/// Endpoint probabilities can be singular; use interior probabilities for inversion.
///
/// ## Examples
///
/// ```
/// let value = sci_special::probability::nctdtridf(0.4120716320853016,0.5,0.3);
/// assert!(value.is_finite());
/// ```
///
/// ## References
///
/// - [SciPy nctdtridf](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.nctdtridf.html).
pub fn nctdtridf(p: f64, nc: f64, t: f64) -> f64 {
    positive_inverse(|df| nctdtr(df, nc, t), p)
}
