//! Vector distances, pairwise matrices, and condensed distance storage.
//!
//! Optional weights are finite, nonnegative, and match the coordinates. Zero
//! total weight is rejected. Boolean metrics require 0/1 coordinates, except
//! Jaccard, which converts every nonzero coordinate to true. This deliberately
//! excludes SciPy's arithmetic extensions of boolean metrics to arbitrary reals.

use std::borrow::Cow;

use super::{MAX_ENTRIES, MAX_WORK, bound, evaluated, pair, points};
use crate::{Error, Result, error::is_finite_slice};

/// A distance definition for [`pdist`] and [`cdist`].
///
/// Parameters are explicit: variances and inverse covariance are never inferred
/// from observations. Pairwise calls use unit weights and centered correlation.
#[derive(Clone, Copy, Debug)]
pub enum Metric<'a> {
    /// L2 distance.
    Euclidean,
    /// Squared L2 distance.
    SqEuclidean,
    /// L1 distance.
    Cityblock,
    /// Maximum absolute coordinate difference.
    Chebyshev,
    /// Lp distance; p is positive or positive infinity.
    Minkowski(f64),
    /// Sum of relative coordinate differences.
    Canberra,
    /// Ratio of absolute difference and absolute sum totals.
    Braycurtis,
    /// One minus the cosine similarity.
    Cosine,
    /// One minus centered correlation.
    Correlation,
    /// Fraction of unequal coordinates.
    Hamming,
    /// Boolean Jaccard dissimilarity.
    Jaccard,
    /// Boolean Dice dissimilarity.
    Dice,
    /// Boolean Rogers–Tanimoto dissimilarity.
    Rogerstanimoto,
    /// Boolean Russell–Rao dissimilarity.
    Russellrao,
    /// Boolean Sokal–Sneath dissimilarity.
    Sokalsneath,
    /// Boolean Yule dissimilarity.
    Yule,
    /// Standardized Euclidean distance with explicit positive variances.
    Seuclidean(&'a [f64]),
    /// Mahalanobis distance with an explicit symmetric positive definite matrix.
    Mahalanobis(&'a [Vec<f64>]),
    /// Jensen–Shannon distance; None uses natural logarithms.
    JensenShannon(Option<f64>),
}

fn weights(w: Option<&[f64]>, n: usize) -> Result<Cow<'_, [f64]>> {
    let Some(w) = w else {
        return Ok(Cow::Owned(vec![1.0; n]));
    };
    if w.len() != n {
        return Err(Error::DimensionMismatch);
    }
    is_finite_slice(w)?;
    if w.iter().any(|&x| x < 0.0) {
        return Err(Error::InvalidInput("weights must be nonnegative"));
    }
    let scale = w.iter().copied().fold(0.0, f64::max);
    if scale == 0.0 {
        return Err(Error::InvalidInput("weights must have positive total"));
    }
    Ok(Cow::Borrowed(w))
}

fn lp(u: &[f64], v: &[f64], p: f64, w: &[f64]) -> Result<f64> {
    if p.is_nan() || p <= 0.0 {
        return Err(Error::InvalidInput("p must be positive"));
    }
    let active = || u.iter().zip(v).zip(w).filter(|(_, w)| **w > 0.0);
    if p == 1.0 || p == 2.0 || p == f64::INFINITY {
        let mut norm = 0.0_f64;
        for ((&a, &b), &w) in active() {
            let factor = if p == 1.0 {
                w
            } else if p == 2.0 {
                w.sqrt()
            } else {
                1.0
            };
            let delta = a - b;
            let term = if delta.is_finite() {
                delta.abs() * factor
            } else if factor <= 1.0 {
                (a * factor - b * factor).abs()
            } else {
                f64::INFINITY
            };
            norm = if p == 1.0 {
                norm + term
            } else if p == 2.0 {
                norm.hypot(term)
            } else {
                norm.max(term)
            };
        }
        return evaluated(norm);
    }
    // Log-domain weighted components avoid overflowing sums of weights and
    // allow small weights to balance large coordinate differences.
    let logs = active()
        .filter_map(|((&a, &b), &w)| {
            if a == b {
                return None;
            }
            let delta = (a - b).abs();
            let log_delta = if delta.is_finite() {
                delta.ln()
            } else {
                let scale = a.abs().max(b.abs());
                scale.ln() + (a / scale - b / scale).abs().ln()
            };
            Some(log_delta + w.ln() / p)
        })
        .collect::<Vec<_>>();
    let maximum = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return evaluated(maximum.exp());
    }
    let sum = logs.iter().map(|&v| (p * (v - maximum)).exp()).sum::<f64>();
    evaluated((maximum + sum.ln() / p).exp())
}

fn active_scale(values: &[f64], w: &[f64]) -> f64 {
    values
        .iter()
        .zip(w)
        .filter(|(_, w)| **w > 0.0)
        .map(|(v, _)| v.abs())
        .fold(0.0, f64::max)
}

fn boolean_counts(u: &[f64], v: &[f64], w: &[f64], convert: bool) -> Result<[f64; 4]> {
    let mut counts = [0.0; 4];
    for ((&a, &b), &w) in u.iter().zip(v).zip(w) {
        if !convert && (a != 0.0 && a != 1.0 || b != 0.0 && b != 1.0) {
            return Err(Error::InvalidInput("boolean metric requires zero or one"));
        }
        counts[usize::from(a != 0.0) * 2 + usize::from(b != 0.0)] += w;
    }
    if counts.iter().any(|x| !x.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok(counts)
}

fn calculate(u: &[f64], v: &[f64], metric: Metric<'_>, w: Option<&[f64]>) -> Result<f64> {
    pair(u, v)?;
    let mut w = weights(w, u.len())?;
    // These ratios are invariant under a common weight scale.
    if matches!(
        metric,
        Metric::Braycurtis
            | Metric::Hamming
            | Metric::Jaccard
            | Metric::Dice
            | Metric::Rogerstanimoto
            | Metric::Russellrao
            | Metric::Sokalsneath
            | Metric::Yule
    ) {
        let scale = w.iter().copied().fold(0.0, f64::max);
        if scale > 0.0 {
            for weight in w.to_mut() {
                *weight /= scale;
            }
        }
    }
    let sum = |f: &dyn Fn(f64, f64) -> f64| -> f64 {
        u.iter()
            .zip(v)
            .zip(w.iter())
            .filter(|(_, w)| **w > 0.0)
            .map(|((&a, &b), &w)| w * f(a, b))
            .sum()
    };
    let value = match metric {
        Metric::Euclidean => return lp(u, v, 2.0, &w),
        Metric::Minkowski(p) => return lp(u, v, p, &w),
        Metric::Chebyshev => return lp(u, v, f64::INFINITY, &w),
        Metric::SqEuclidean => {
            let d = lp(u, v, 2.0, &w)?;
            d * d
        }
        Metric::Cityblock => return lp(u, v, 1.0, &w),
        Metric::Canberra => sum(&|a, b| {
            let den = a.abs() + b.abs();
            if den == 0.0 {
                0.0
            } else {
                // Scaling avoids overflow in both the numerator and denominator.
                let s = a.abs().max(b.abs());
                ((a / s) - (b / s)).abs() / ((a / s).abs() + (b / s).abs())
            }
        }),
        Metric::Braycurtis => {
            let scale = active_scale(u, &w).max(active_scale(v, &w));
            if scale == 0.0 {
                f64::NAN
            } else {
                sum(&|a, b| (a / scale - b / scale).abs())
                    / sum(&|a, b| (a / scale + b / scale).abs())
            }
        }
        Metric::Hamming => sum(&|a, b| f64::from(a != b)) / w.iter().sum::<f64>(),
        Metric::Cosine | Metric::Correlation => {
            let su = active_scale(u, &w);
            let sv = active_scale(v, &w);
            if su == 0.0 || sv == 0.0 {
                return Ok(f64::NAN);
            }
            let sw = w.iter().copied().fold(0.0, f64::max);
            let wn: Vec<_> = w.iter().map(|x| x / sw).collect();
            let total = wn.iter().sum::<f64>();
            let center = matches!(metric, Metric::Correlation);
            let mu = if center {
                u.iter()
                    .zip(&wn)
                    .filter(|(_, w)| **w > 0.0)
                    .map(|(&a, &w)| a / su * w)
                    .sum::<f64>()
                    / total
            } else {
                0.0
            };
            let mv = if center {
                v.iter()
                    .zip(&wn)
                    .filter(|(_, w)| **w > 0.0)
                    .map(|(&a, &w)| a / sv * w)
                    .sum::<f64>()
                    / total
            } else {
                0.0
            };
            let (mut uv, mut uu, mut vv) = (0.0, 0.0, 0.0);
            for ((&a, &b), &w) in u.iter().zip(v).zip(&wn) {
                if w == 0.0 {
                    continue;
                }
                let (a, b) = (a / su - mu, b / sv - mv);
                uv += w * a * b;
                uu += w * a * a;
                vv += w * b * b;
            }
            (1.0 - uv / (uu.sqrt() * vv.sqrt())).clamp(0.0, 2.0)
        }
        Metric::Jaccard
        | Metric::Dice
        | Metric::Rogerstanimoto
        | Metric::Russellrao
        | Metric::Sokalsneath
        | Metric::Yule => {
            let [ff, ft, tf, tt] = boolean_counts(u, v, &w, matches!(metric, Metric::Jaccard))?;
            let unequal = ft + tf;
            match metric {
                Metric::Jaccard => {
                    if unequal + tt == 0.0 {
                        0.0
                    } else {
                        unequal / (unequal + tt)
                    }
                }
                Metric::Dice => unequal / (2.0 * tt + unequal),
                Metric::Rogerstanimoto => 2.0 * unequal / (tt + ff + 2.0 * unequal),
                Metric::Russellrao => (ff + unequal) / (ff + unequal + tt),
                Metric::Sokalsneath => {
                    if tt + unequal == 0.0 {
                        return Err(Error::InvalidInput(
                            "Sokal-Sneath is undefined for all-false vectors",
                        ));
                    }
                    2.0 * unequal / (tt + 2.0 * unequal)
                }
                Metric::Yule => {
                    if ft == 0.0 || tf == 0.0 {
                        0.0
                    } else {
                        let s = ff.max(ft).max(tf).max(tt);
                        2.0 * (ft / s) * (tf / s) / ((tt / s) * (ff / s) + (ft / s) * (tf / s))
                    }
                }
                _ => unreachable!(),
            }
        }
        Metric::Seuclidean(variance) => {
            if variance.len() != u.len() {
                return Err(Error::DimensionMismatch);
            }
            is_finite_slice(variance)?;
            if variance.iter().any(|&x| x <= 0.0) {
                return Err(Error::InvalidInput("variances must be positive"));
            }
            let mut norm: f64 = 0.0;
            for ((&a, &b), &v) in u.iter().zip(v).zip(variance) {
                norm = norm.hypot((a - b) / v.sqrt());
            }
            norm
        }
        Metric::Mahalanobis(vi) => {
            let factor = mahalanobis_factor(vi, u.len())?;
            mahalanobis_distance(u, v, &factor)?
        }
        Metric::JensenShannon(base) => {
            if base.is_some_and(|b| !b.is_finite() || b <= 1.0) {
                return Err(Error::InvalidInput("logarithm base must exceed one"));
            }
            let normalized = |x: &[f64]| -> Result<Vec<f64>> {
                if x.iter().any(|&x| x < 0.0) {
                    return Err(Error::InvalidInput("probabilities must be nonnegative"));
                }
                let s = x.iter().copied().fold(0.0, f64::max);
                if s == 0.0 {
                    return Err(Error::InvalidInput("probability mass must be positive"));
                }
                let total = x.iter().map(|x| x / s).sum::<f64>();
                Ok(x.iter().map(|x| (x / s) / total).collect())
            };
            let (p, q) = (normalized(u)?, normalized(v)?);
            let mut divergence = 0.0;
            for (&p, &q) in p.iter().zip(&q) {
                let m = (p + q) / 2.0;
                if p > 0.0 {
                    divergence += p * (p / m).ln();
                }
                if q > 0.0 {
                    divergence += q * (q / m).ln();
                }
            }
            (divergence.max(0.0) / (2.0 * base.map_or(1.0, f64::ln))).sqrt()
        }
    };
    evaluated(value)
}

macro_rules! vector_distance {
    ($name:ident, $metric:ident, $definition:literal, $domain:literal) => {
        #[doc = concat!("Computes ", $definition, ".\n\n",
        "# Parameters and conventions\n`u` and `v` are equally sized finite f64 vectors.\n",
        "`w` supplies optional nonnegative coordinate weights; None uses ones.\n\n",
        "# Domain and special values\n", $domain, " NaN and infinity inputs are rejected.\n\n",
        "# Returns\nThe scalar dissimilarity.\n\n",
        "# Errors\nReturns a shared error for mismatched lengths, invalid weights or domain,\n",
        "work limits, or arithmetic overflow. Zero total explicit weight is invalid.\n\n",
        "# Numerical behavior\nUses f64 arithmetic and at most 1,000,000 coordinates.\n",
        "L2 uses weighted components and hypot; zero-weight coordinates are excluded\n",
        "from Lp arithmetic and ratio-metric normalization scales.\n",
        "Rounding affects near-zero distances; no uniform relative accuracy is promised.\n\n",
        "# Examples\n```\nlet d = sci_spatial::distance::", stringify!($name),
        "(&[1.0, 0.0, 1.0], &[0.0, 1.0, 1.0], None)?;\nassert!(d >= 0.0);\n",
        "# Ok::<(), sci_core::Error>(())\n```\n\n# References\n",
        "[SciPy definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.",
        stringify!($name), ".html).")]
        pub fn $name(u: &[f64], v: &[f64], w: Option<&[f64]>) -> Result<f64> {
            calculate(u,v,Metric::$metric,w)
        }
    }
}
vector_distance!(
    euclidean,
    Euclidean,
    "the weighted L2 norm `sqrt(Σ wᵢ(uᵢ-vᵢ)²)`",
    "Empty vectors return zero."
);
vector_distance!(
    sqeuclidean,
    SqEuclidean,
    "the squared L2 distance `Σ wᵢ(uᵢ-vᵢ)²`",
    "Empty vectors return zero."
);
vector_distance!(
    cityblock,
    Cityblock,
    "the L1 distance `Σ wᵢ|uᵢ-vᵢ|`",
    "Empty vectors return zero."
);
vector_distance!(
    chebyshev,
    Chebyshev,
    "the distance `maxᵢ |uᵢ-vᵢ|` over positive-weight coordinates",
    "Weight magnitudes do not matter. Empty vectors return zero (a SciPy adaptation)."
);
vector_distance!(
    canberra,
    Canberra,
    "the distance `Σ wᵢ|uᵢ-vᵢ|/(|uᵢ|+|vᵢ|)`",
    "A 0/0 coordinate contributes zero. Empty vectors return zero."
);
vector_distance!(
    braycurtis,
    Braycurtis,
    "the distance `Σ wᵢ|uᵢ-vᵢ| / Σ wᵢ|uᵢ+vᵢ|`",
    "Zero numerator and denominator return NaN; a positive numerator with zero denominator returns an overflow error."
);
vector_distance!(
    cosine,
    Cosine,
    "`1 - Σ wᵢuᵢvᵢ / sqrt((Σ wᵢuᵢ²)(Σ wᵢvᵢ²))`",
    "Zero norms return NaN. Results are clipped to `[0,2]`."
);
vector_distance!(
    hamming,
    Hamming,
    "the unequal-coordinate fraction `Σ wᵢ[uᵢ≠vᵢ]/Σ wᵢ`",
    "Coordinates may be any finite real numbers. Empty vectors return NaN."
);
vector_distance!(
    jaccard,
    Jaccard,
    "`(cTF+cFT)/(cTT+cTF+cFT)`",
    "Nonzero coordinates represent true; cAB is the weighted count of the boolean pair (A,B). An empty union returns zero."
);
vector_distance!(
    dice,
    Dice,
    "`(cTF+cFT)/(2cTT+cTF+cFT)`",
    "Requires 0/1 coordinates; cAB counts weighted boolean pairs. All-false vectors return NaN."
);
vector_distance!(
    rogerstanimoto,
    Rogerstanimoto,
    "`2(cTF+cFT)/(cTT+cFF+2(cTF+cFT))`",
    "Requires 0/1 coordinates; cAB counts weighted boolean pairs. Empty vectors return NaN."
);
vector_distance!(
    russellrao,
    Russellrao,
    "`(cFF+cTF+cFT)/(cTT+cFF+cTF+cFT)`",
    "Requires 0/1 coordinates; cAB counts weighted boolean pairs. Empty vectors return NaN."
);
vector_distance!(
    sokalsneath,
    Sokalsneath,
    "`2(cTF+cFT)/(cTT+2(cTF+cFT))`",
    "Requires 0/1 coordinates; cAB counts weighted boolean pairs. All-false vectors are invalid."
);
vector_distance!(
    yule,
    Yule,
    "`2cTF cFT/(cTT cFF+cTF cFT)`",
    "Requires 0/1 coordinates; cAB counts weighted boolean pairs. If either mismatch count is zero, returns zero."
);

/// Computes the weighted Lp distance `(Σ wᵢ|uᵢ-vᵢ|ᵖ)^(1/p)`.
///
/// ## Parameters
///
/// `u`, `v` are equal-length vectors; `w` is optional nonnegative weights.
/// `p` is positive; positive infinity takes the maximum over positive weights.
/// Values below one define a quasinorm.
///
/// ## Domain
///
/// Coordinates and weights must be finite. Empty vectors return zero.
///
/// ## Returns
///
/// A scalar in coordinate units.
///
/// ## Errors
///
/// Rejects invalid p, weights, shapes, overflow, or more than 1,000,000 coordinates.
///
/// ## Numerical behavior
///
/// L1 uses weighted differences, L2 uses weighted components and hypot, and
/// infinity takes the maximum positive-weight difference. Other powers use
/// log-domain weighted components and maximum-shifted exponentials to avoid
/// intermediate weight-sum overflow. Zero weights exclude coordinates before
/// arithmetic. Logarithm rounding, tiny weights and subnormal component rounding
/// can limit relative accuracy; very small p can cause true result overflow.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_spatial::distance::minkowski(&[0.0], &[2.0], 3.0, None)?, 2.0);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.minkowski.html).
pub fn minkowski(u: &[f64], v: &[f64], p: f64, w: Option<&[f64]>) -> Result<f64> {
    calculate(u, v, Metric::Minkowski(p), w)
}

/// Computes `1 - <u-μu,v-μv>_w/(||u-μu||_w ||v-μv||_w)`.
///
/// ## Parameters
///
/// Vectors have equal length; optional weights are nonnegative. `centered`
/// chooses weighted mean subtraction; false is equivalent to [`cosine`].
///
/// ## Domain
///
/// Finite coordinates are required; constant centered vectors or zero norms
/// return NaN. Results lie in `[0,2]`.
///
/// ## Returns
///
/// The scalar correlation distance.
///
/// ## Errors
///
/// Rejects shapes, weights, overflow, or more than 1,000,000 coordinates.
///
/// ## Numerical behavior
///
/// Rescales coordinates and weights; clips rounding excursions to `[0,2]`.
/// Nearly constant vectors may lose relative accuracy in centering.
///
/// # Examples
///
/// ```
/// assert!(sci_spatial::distance::correlation(&[1.0,2.0], &[2.0,4.0], None, true)? < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.correlation.html).
pub fn correlation(u: &[f64], v: &[f64], w: Option<&[f64]>, centered: bool) -> Result<f64> {
    calculate(
        u,
        v,
        if centered {
            Metric::Correlation
        } else {
            Metric::Cosine
        },
        w,
    )
}

/// Computes standardized Euclidean distance `sqrt(Σ (uᵢ-vᵢ)²/Vᵢ)`.
///
/// ## Parameters
///
/// `u`, `v`, and explicit coordinate variances `variance` have equal length.
///
/// ## Domain
///
/// Coordinates are finite; variances must be finite and strictly positive.
/// Empty vectors return zero. Zero or negative variances are rejected.
///
/// ## Returns
///
/// A dimensionless scalar distance.
///
/// ## Errors
///
/// Rejects shapes, domains, overflow, or more than 1,000,000 coordinates.
///
/// ## Numerical behavior
///
/// Uses hypot accumulation; very small variances can overflow.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_spatial::distance::seuclidean(&[0.0], &[2.0], &[4.0])?, 1.0);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.seuclidean.html).
pub fn seuclidean(u: &[f64], v: &[f64], variance: &[f64]) -> Result<f64> {
    calculate(u, v, Metric::Seuclidean(variance), None)
}

/// Computes Mahalanobis distance `sqrt((u-v)ᵀ VI (u-v))`.
///
/// ## Parameters
///
/// `vi` is the explicit row-major inverse covariance matrix of size d×d;
/// `u` and `v` each have d coordinates.
///
/// ## Domain
///
/// All inputs are finite; `vi` must be exactly symmetric and positive definite.
/// Indefinite and semidefinite matrices are rejected.
///
/// ## Returns
///
/// The dimensionless distance.
///
/// ## Errors
///
/// Rejects shapes, domains, overflow, or dimension greater than 64.
///
/// ## Numerical behavior
///
/// Private nalgebra Cholesky factorization, O(d³); ill-conditioned matrices
/// may fail factorization or lose accuracy. No inverse is computed.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_spatial::distance::mahalanobis(&[0.0], &[2.0], &[vec![0.25]])?, 1.0);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.mahalanobis.html).
pub fn mahalanobis(u: &[f64], v: &[f64], vi: &[Vec<f64>]) -> Result<f64> {
    calculate(u, v, Metric::Mahalanobis(vi), None)
}

/// Computes Jensen–Shannon distance `sqrt((KL(p||m)+KL(q||m))/2)`, m=(p+q)/2.
///
/// ## Parameters
///
/// Equal-length nonnegative probability vectors are normalized separately.
/// `base` is None for natural logs, or a finite logarithm base greater than one.
/// SciPy's axis and keepdims options are replaced by a single-vector API.
///
/// ## Domain
///
/// Finite values and positive total masses are required; zero probability terms
/// contribute zero. Negative probabilities and empty vectors are invalid.
///
/// ## Returns
///
/// A nonnegative distance, at most sqrt(ln(2)/ln(base)), with ln(base)=1 for None.
///
/// ## Errors
///
/// Rejects shapes, domains, overflow, or more than 1,000,000 coordinates.
///
/// ## Numerical behavior
///
/// Scaled normalization avoids overflowing total mass. Roundoff can erase
/// very small divergences; negative roundoff is clipped to zero.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_spatial::distance::jensenshannon(&[1.0,0.0], &[0.0,1.0], Some(2.0))?, 1.0);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html).
pub fn jensenshannon(p: &[f64], q: &[f64], base: Option<f64>) -> Result<f64> {
    calculate(p, q, Metric::JensenShannon(base), None)
}

fn mahalanobis_factor(vi: &[Vec<f64>], dimension: usize) -> Result<nalgebra::DMatrix<f64>> {
    let d = points(vi)?;
    if vi.len() != d || d != dimension {
        return Err(Error::DimensionMismatch);
    }
    bound(Some(d), 64)?;
    for (i, row) in vi.iter().enumerate() {
        for (j, other) in vi.iter().enumerate().take(i) {
            if row[j] != other[i] {
                return Err(Error::InvalidInput("inverse covariance must be symmetric"));
            }
        }
    }
    nalgebra::DMatrix::from_fn(d, d, |i, j| vi[i][j])
        .cholesky()
        .map(|factor| factor.l())
        .ok_or(Error::InvalidInput(
            "inverse covariance must be positive definite",
        ))
}

fn mahalanobis_distance(u: &[f64], v: &[f64], lower: &nalgebra::DMatrix<f64>) -> Result<f64> {
    // VI = L L^T, so the distance is ||L^T (u-v)||. Reuse L for every pair.
    let mut norm = 0.0_f64;
    for j in 0..u.len() {
        let mut component = 0.0;
        for i in j..u.len() {
            component += lower[(i, j)] * (u[i] - v[i]);
        }
        norm = norm.hypot(component);
    }
    evaluated(norm)
}

fn prepare_metric(metric: Metric<'_>, d: usize) -> Result<Option<nalgebra::DMatrix<f64>>> {
    if let Metric::Mahalanobis(vi) = metric {
        mahalanobis_factor(vi, d).map(Some)
    } else {
        validate_metric(metric, d)?;
        Ok(None)
    }
}

fn pairwise_bound(n: usize, m: usize, d: usize, metric: Metric<'_>) -> Result<()> {
    let pairs = n.checked_mul(m);
    bound(pairs, MAX_ENTRIES)?;
    let work = if matches!(metric, Metric::Mahalanobis(_)) {
        d.checked_mul(d).and_then(|square| {
            pairs
                .and_then(|pairs| pairs.checked_mul(square))
                .and_then(|pairs| square.checked_mul(d)?.checked_add(pairs))
        })
    } else {
        pairs.and_then(|pairs| pairs.checked_mul(d.max(1)))
    };
    bound(work, MAX_WORK)
}

/// Computes cross distances `D[i][j] = metric(x[i], y[j])`.
///
/// ## Parameters
///
/// `x`, `y` are row-major observations with the same coordinate count.
/// [`Metric`] supplies the definition and explicit parameters; weights are ones.
/// No callbacks, output buffers, broadcasting, or inferred covariance are used.
///
/// ## Domain
///
/// Finite rectangular inputs; each metric's domain applies. An empty collection
/// accepts the other collection's coordinate count. Undefined metrics yield NaN.
///
/// ## Returns
///
/// A matrix with x.len() rows and y.len() columns in input order.
///
/// ## Errors
///
/// Rejects ragged/incompatible inputs, metric domains, overflow, more than
/// 1,000,000 entries or 10,000,000 coordinate operations. Mahalanobis work
/// includes one shared factorization and quadratic work per pair. Empty outputs
/// still validate parameters.
///
/// ## Numerical behavior
///
/// Evaluates the same kernels as the vector functions; no parallelism or approximation.
/// Mahalanobis distances share one Cholesky factor per batch, using
/// O(d³ + pairs*d²) work and O(d²) additional storage.
///
/// # Examples
///
/// ```
/// use sci_spatial::{distance::cdist, distance::Metric};
/// assert_eq!(cdist(&[vec![0.0]], &[vec![2.0]], Metric::Euclidean)?, vec![vec![2.0]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html).
pub fn cdist(x: &[Vec<f64>], y: &[Vec<f64>], metric: Metric<'_>) -> Result<Vec<Vec<f64>>> {
    let (dx, dy) = (points(x)?, points(y)?);
    if !x.is_empty() && !y.is_empty() && dx != dy {
        return Err(Error::DimensionMismatch);
    }
    let d = dx.max(dy);
    pairwise_bound(x.len(), y.len(), d, metric)?;
    let factor = prepare_metric(metric, d)?;
    x.iter()
        .map(|a| {
            y.iter()
                .map(|b| match &factor {
                    Some(factor) => mahalanobis_distance(a, b, factor),
                    None => calculate(a, b, metric, None),
                })
                .collect()
        })
        .collect()
}

fn validate_metric(metric: Metric<'_>, d: usize) -> Result<()> {
    match metric {
        Metric::Minkowski(p) if p.is_nan() || p <= 0.0 => {
            Err(Error::InvalidInput("p must be positive"))
        }
        Metric::JensenShannon(Some(b)) if !b.is_finite() || b <= 1.0 => {
            Err(Error::InvalidInput("logarithm base must exceed one"))
        }
        Metric::Seuclidean(_) | Metric::Mahalanobis(_) => {
            calculate(&vec![0.0; d], &vec![0.0; d], metric, None).map(|_| ())
        }
        _ => Ok(()),
    }
}

/// Computes within-sample distances `metric(x[i],x[j])` for i<j.
///
/// ## Parameters
///
/// `x` has observations as rows; [`Metric`] supplies explicit parameters.
/// Weights are ones; centered correlation is used.
///
/// ## Domain
///
/// Finite rectangular observations and the selected metric's domain are required.
/// Zero or one observation gives an empty vector; parameters are still validated.
///
/// ## Returns
///
/// Condensed distances ordered (0,1), (0,2), …, (1,2), …, (n-2,n-1).
///
/// ## Errors
///
/// Rejects domains, shapes, overflow, more than 1,000,000 output entries or
/// 10,000,000 coordinate operations (including Mahalanobis factorization).
///
/// ## Numerical behavior
///
/// Uses the public vector kernels without approximate neighbor searches.
/// Mahalanobis batches share one Cholesky factor and require O(d³ + pairs*d²)
/// work, with O(d²) additional storage.
///
/// # Examples
///
/// ```
/// use sci_spatial::{distance::pdist, distance::Metric};
/// assert_eq!(pdist(&[vec![0.0],vec![2.0],vec![5.0]],Metric::Euclidean)?,vec![2.0,5.0,3.0]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html).
pub fn pdist(x: &[Vec<f64>], metric: Metric<'_>) -> Result<Vec<f64>> {
    let d = points(x)?;
    let count = x
        .len()
        .checked_mul(x.len().saturating_sub(1))
        .map(|x| x / 2);
    bound(count, MAX_ENTRIES)?;
    pairwise_bound(count.unwrap(), 1, d, metric)?;
    let factor = prepare_metric(metric, d)?;
    let mut result = Vec::with_capacity(count.unwrap());
    for i in 0..x.len() {
        for j in i + 1..x.len() {
            result.push(match &factor {
                Some(factor) => mahalanobis_distance(&x[i], &x[j], factor)?,
                None => calculate(&x[i], &x[j], metric, None)?,
            });
        }
    }
    Ok(result)
}

fn observation_count(k: usize) -> Option<usize> {
    // Integer binary search avoids rounding in triangular-number recognition.
    let (mut lo, mut hi) = (1usize, k.checked_add(2)?);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match mid.checked_mul(mid - 1).map(|x| x / 2) {
            Some(t) if t < k => lo = mid + 1,
            _ => hi = mid,
        }
    }
    (lo.checked_mul(lo - 1)? / 2 == k).then_some(lo)
}

/// Checks whether D is square, symmetric, and has a zero diagonal within tol.
///
/// ## Parameters
///
/// Row-major `d`; nonnegative finite absolute `tol`. Triangle inequalities and
/// nonnegativity are not checked. SciPy throw/warning/name options are omitted.
///
/// ## Domain
///
/// Nonfinite matrix values, ragged rows, invalid tolerance, or excessive work
/// return false. Unlike SciPy, negative diagonals must also be within abs(tol).
///
/// ## Returns
///
/// True exactly when all shape, finiteness, and absolute tolerance checks pass.
///
/// ## Numerical behavior
///
/// O(n²) comparisons, at most 1,000,000 entries. Symmetry uses absolute differences.
///
/// # Examples
///
/// ```
/// assert!(sci_spatial::distance::is_valid_dm(&[vec![0.0,2.0],vec![2.0,0.0]],0.0));
/// ```
/// # References
///
/// [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.is_valid_dm.html).
pub fn is_valid_dm(d: &[Vec<f64>], tol: f64) -> bool {
    if !tol.is_finite() || tol < 0.0 || points(d).ok() != Some(d.len()) {
        return false;
    }
    for (i, row) in d.iter().enumerate() {
        if row[i].abs() > tol {
            return false;
        }
        for (j, other) in d.iter().enumerate().take(i) {
            if (row[j] - other[i]).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Checks whether y.len() equals `n(n-1)/2` for an integer n≥1.
///
/// ## Parameters
///
/// `y` is a one-dimensional condensed distance vector.
///
/// ## Domain
///
/// Only length is checked, as in SciPy; NaN, infinity, and negative entries are allowed.
/// Empty input is valid. At most 1,000,000 entries are permitted.
///
/// ## Returns
///
/// Whether the length is triangular and within the work limit.
///
/// ## Numerical behavior
///
/// Exact integer recognition in O(log(y.len()+1)) work.
///
/// # Examples
///
/// ```
/// assert!(sci_spatial::distance::is_valid_y(&[1.0,2.0,3.0]));
/// ```
/// # References
///
/// [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.is_valid_y.html).
pub fn is_valid_y(y: &[f64]) -> bool {
    y.len() <= MAX_ENTRIES && observation_count(y.len()).is_some()
}

/// Returns n from the square shape `n×n` of a redundant distance matrix.
///
/// ## Parameters
///
/// `d` is row-major; symmetry and zero diagonal are not required, matching
/// SciPy's shape-focused count API.
///
/// ## Domain
///
/// This adaptation rejects nonfinite entries and permits a 0×0 empty matrix.
///
/// ## Returns
///
/// The number of observations.
///
/// ## Errors
///
/// Rejects nonsquare/ragged matrices, nonfinite entries, or over 1,000,000 entries.
///
/// ## Numerical behavior
///
/// O(n²) validation; the count is exact.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_spatial::distance::num_obs_dm(&[vec![0.0]])?,1);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.num_obs_dm.html).
pub fn num_obs_dm(d: &[Vec<f64>]) -> Result<usize> {
    if points(d)? != d.len() {
        return Err(Error::DimensionMismatch);
    }
    Ok(d.len())
}

/// Recovers n≥2 from a condensed vector length `n(n-1)/2`.
///
/// ## Parameters
///
/// `y` is a condensed distance vector; its values are not inspected.
///
/// ## Domain
///
/// Empty input is ambiguous and rejected, matching SciPy. NaN/infinity values are allowed.
///
/// ## Returns
///
/// The exact observation count.
///
/// ## Errors
///
/// Rejects empty/nontriangular lengths and over 1,000,000 entries.
///
/// ## Numerical behavior
///
/// Exact integer binary search, O(log(y.len()+1)).
///
/// # Examples
///
/// ```
/// assert_eq!(sci_spatial::distance::num_obs_y(&[1.0,2.0,3.0])?,3);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.num_obs_y.html).
pub fn num_obs_y(y: &[f64]) -> Result<usize> {
    if y.is_empty() || !is_valid_y(y) {
        return Err(Error::InvalidInput("invalid or ambiguous condensed length"));
    }
    Ok(observation_count(y.len()).unwrap())
}

/// Input to [`squareform`], determining conversion direction without strings.
#[derive(Clone, Copy, Debug)]
pub enum DistanceForm<'a> {
    /// Upper-triangle distances in pdist order.
    Condensed(&'a [f64]),
    /// Square redundant matrix; validated for symmetry and zero diagonal.
    Square(&'a [Vec<f64>]),
}

/// Converted distance storage returned by [`squareform`].
#[derive(Clone, Debug, PartialEq)]
pub enum DistanceOutput {
    /// Upper-triangle values in pdist order.
    Condensed(Vec<f64>),
    /// Symmetric row-major matrix with zero diagonal.
    Square(Vec<Vec<f64>>),
}

/// Converts `y[k(i,j)] = D[i][j]`, i<j, between condensed and square storage.
///
/// ## Parameters
///
/// [`DistanceForm`] selects direction. `k(i,j)=ni-i(i+1)/2+j-i-1`.
/// SciPy's force/checks arguments are replaced by typed input with mandatory checks.
///
/// ## Domain
///
/// All values must be finite. Square input must be symmetric with zero diagonal.
/// Empty condensed input expands to `[[0]]`; a 0×0 square matrix condenses to [].
/// Negative distances are allowed; triangle inequalities are not checked.
///
/// ## Returns
///
/// The opposite storage form, with input observation ordering preserved.
///
/// ## Errors
///
/// Rejects invalid lengths/shapes/nonfinite values and over 1,000,000 output entries.
///
/// ## Numerical behavior
///
/// Copies entries exactly; uses exact integer shape inference.
///
/// # Examples
///
/// ```
/// use sci_spatial::{distance::squareform, distance::DistanceForm, distance::DistanceOutput};
/// assert_eq!(squareform(DistanceForm::Condensed(&[2.0]))?,
///     DistanceOutput::Square(vec![vec![0.0,2.0],vec![2.0,0.0]]));
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html).
pub fn squareform(input: DistanceForm<'_>) -> Result<DistanceOutput> {
    match input {
        DistanceForm::Condensed(y) => {
            bound(Some(y.len()), MAX_ENTRIES)?;
            is_finite_slice(y)?;
            let n = observation_count(y.len())
                .ok_or(Error::InvalidInput("condensed length must be triangular"))?;
            bound(n.checked_mul(n), MAX_ENTRIES)?;
            let mut d = vec![vec![0.0; n]; n];
            let mut k = 0;
            for i in 0..n {
                let (prefix, tail) = d.split_at_mut(i + 1);
                for (value, other) in prefix[i].iter_mut().skip(i + 1).zip(tail) {
                    *value = y[k];
                    other[i] = y[k];
                    k += 1;
                }
            }
            Ok(DistanceOutput::Square(d))
        }
        DistanceForm::Square(d) => {
            if !is_valid_dm(d, 0.0) {
                return Err(Error::InvalidInput("invalid square distance matrix"));
            }
            let mut y = Vec::new();
            for (i, row) in d.iter().enumerate() {
                y.extend_from_slice(&row[i + 1..]);
            }
            Ok(DistanceOutput::Condensed(y))
        }
    }
}

/// A directed Hausdorff distance and its witness indices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HausdorffResult {
    /// `max_i min_j ||u[i]-v[j]||₂`, in coordinate units.
    pub distance: f64,
    /// Index of a maximizing point in the first collection.
    pub index_u: usize,
    /// Index of its nearest point in the second collection.
    pub index_v: usize,
}

/// Computes the directed Hausdorff distance `max_i min_j ||u[i]-v[j]||₂`.
///
/// ## Parameters
///
/// `u`, `v` are nonempty row-major point sets with equal positive dimension.
/// SciPy's RNG/seed options are omitted; ties choose the first indices in input order.
///
/// ## Domain
///
/// Finite rectangular point sets are required; duplicates are permitted.
///
/// ## Returns
///
/// The distance and a pair of witnessing original indices.
///
/// ## Errors
///
/// Rejects shapes, domains, overflow, over 1,000,000 input entries, or
/// 10,000,000 pair-coordinate comparisons.
///
/// ## Numerical behavior
///
/// Exact exhaustive nearest-distance search with scaled L2 kernels; O(nmd) work.
/// Witness indices can differ from SciPy when the maximizing pair is tied.
///
/// # Examples
///
/// ```
/// let r = sci_spatial::distance::directed_hausdorff(&[vec![0.0],vec![3.0]], &[vec![1.0]])?;
/// assert_eq!((r.distance,r.index_u,r.index_v),(2.0,1,0));
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.directed_hausdorff.html).
pub fn directed_hausdorff(u: &[Vec<f64>], v: &[Vec<f64>]) -> Result<HausdorffResult> {
    let (du, dv) = (points(u)?, points(v)?);
    if du != dv {
        return Err(Error::DimensionMismatch);
    }
    if u.is_empty() || v.is_empty() || du == 0 {
        return Err(Error::InvalidInput("point sets must be nonempty"));
    }
    bound(
        u.len().checked_mul(v.len()).and_then(|n| n.checked_mul(du)),
        MAX_WORK,
    )?;
    let mut result = HausdorffResult {
        distance: -1.0,
        index_u: 0,
        index_v: 0,
    };
    for (i, a) in u.iter().enumerate() {
        let (mut nearest, mut index) = (f64::INFINITY, 0);
        for (j, b) in v.iter().enumerate() {
            let d = euclidean(a, b, None)?;
            if d < nearest {
                nearest = d;
                index = j;
            }
        }
        if nearest > result.distance {
            result = HausdorffResult {
                distance: nearest,
                index_u: i,
                index_v: index,
            };
        }
    }
    Ok(result)
}
