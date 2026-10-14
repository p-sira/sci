//! Lamé polynomial eigensystems and ellipsoidal integral normalization.
//!
//! The tridiagonal formulas follow SciPy's `_ellip_harm.pxd` (BSD licensed;
//! see `LICENSES/scipy-ellipsoidal.txt`). Integration uses endpoint substitutions
//! and bounded adaptive Simpson quadrature, rather than SciPy's QUADPACK backend.

use crate::{Error, Result};

struct Lame {
    coefficients: Vec<f64>,
    family: u8,
    n: usize,
    h2: f64,
    k2: f64,
}

fn lame(h2: f64, k2: f64, n: usize, p: usize) -> Result<Lame> {
    if !h2.is_finite()
        || !k2.is_finite()
        || h2 <= 0.0
        || k2 <= h2
        || n > 64
        || p == 0
        || p > 2 * n + 1
    {
        return Err(Error::InvalidInput(
            "Lamé parameters require 0<h2<k2, n<=64, and 1<=p<=2n+1",
        ));
    }
    let r = n / 2;
    let (family, index, size) = if p <= r + 1 {
        (0, p - 1, r + 1)
    } else if p <= n + 1 {
        (1, p - r - 2, n - r)
    } else if p <= 2 * n - r + 1 {
        (2, p - n - 2, n - r)
    } else {
        (3, p - 2 * n + r - 2, r)
    };
    let alpha = h2;
    let beta = k2 - h2;
    let gamma = alpha - beta;
    let rf = r as f64;
    let odd = n % 2 == 1;
    let mut diagonal = Vec::with_capacity(size);
    let mut g = Vec::with_capacity(size);
    let mut f = Vec::with_capacity(size);
    for j in 0..size {
        let j = j as f64;
        let g0 = -(2.0 * j + 2.0)
            * (2.0 * j + if family == 1 || family == 3 { 3.0 } else { 1.0 })
            * beta;
        let (f0, d) = match family {
            0 if odd => (
                -alpha * (2.0 * (rf - j - 1.0) + 2.0) * (2.0 * (j + 1.0 + rf) + 1.0),
                ((2.0 * rf + 1.0) * (2.0 * rf + 2.0) - 4.0 * j * j) * alpha
                    + (2.0 * j + 1.0).powi(2) * beta,
            ),
            0 => (
                -alpha * (2.0 * (rf - j - 1.0) + 2.0) * (2.0 * (rf + j + 1.0) - 1.0),
                2.0 * rf * (2.0 * rf + 1.0) * alpha - 4.0 * j * j * gamma,
            ),
            1 if odd => (
                -alpha * (2.0 * (rf - j - 1.0) + 2.0) * (2.0 * (j + 1.0 + rf) + 1.0),
                (2.0 * rf + 1.0) * (2.0 * rf + 2.0) * alpha - (2.0 * j + 1.0).powi(2) * gamma,
            ),
            1 => (
                -alpha * 2.0 * (rf - j - 1.0) * (2.0 * (rf + j + 1.0) + 1.0),
                (2.0 * rf * (2.0 * rf + 1.0) - (2.0 * j + 1.0).powi(2)) * alpha
                    + (2.0 * j + 2.0).powi(2) * beta,
            ),
            2 if odd => (
                -alpha * (2.0 * (rf - j - 1.0) + 2.0) * (2.0 * (j + 1.0 + rf) + 1.0),
                ((2.0 * rf + 1.0) * (2.0 * rf + 2.0) - (2.0 * j + 1.0).powi(2)) * alpha
                    + 4.0 * j * j * beta,
            ),
            2 => (
                -alpha * 2.0 * (rf - j - 1.0) * (2.0 * (rf + j + 1.0) + 1.0),
                2.0 * rf * (2.0 * rf + 1.0) * alpha - (2.0 * j + 1.0).powi(2) * gamma,
            ),
            _ if odd => (
                -alpha * 2.0 * (rf - j - 1.0) * (2.0 * (j + 1.0 + rf) + 3.0),
                (2.0 * rf + 1.0) * (2.0 * rf + 2.0) * alpha - (2.0 * j + 2.0).powi(2) * gamma,
            ),
            _ => (
                -alpha * 2.0 * (rf - j - 1.0) * (2.0 * (rf + j + 1.0) + 1.0),
                2.0 * rf * (2.0 * rf + 1.0) * alpha - (2.0 * j + 2.0).powi(2) * alpha
                    + (2.0 * j + 1.0).powi(2) * beta,
            ),
        };
        diagonal.push(d);
        g.push(g0);
        f.push(f0);
    }
    let mut scales = vec![1.0; size];
    for i in 1..size {
        scales[i] = (g[i - 1] / f[i - 1]).sqrt() * scales[i - 1];
    }
    let off: Vec<_> = (0..size.saturating_sub(1))
        .map(|i| g[i] * scales[i] / scales[i + 1])
        .collect();
    let (_, vectors) = super::polynomials::eigensystem(&diagonal, &off)?;
    let mut coefficients: Vec<_> = vectors[index]
        .iter()
        .zip(&scales)
        .map(|(v, s)| v / s)
        .collect();
    let normalization = coefficients[size - 1] / (-h2).powi((size - 1) as i32);
    for c in &mut coefficients {
        *c /= normalization;
    }
    if coefficients.iter().any(|x| !x.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok(Lame {
        coefficients,
        family,
        n,
        h2,
        k2,
    })
}

impl Lame {
    fn value(&self, s: f64, signm: f64, signn: f64) -> f64 {
        let s2 = s * s;
        let psi = match self.family {
            0 => s.powi((self.n % 2) as i32),
            1 => s.powi((1 - self.n % 2) as i32) * signm * (s2 - self.h2).abs().sqrt(),
            2 => s.powi((1 - self.n % 2) as i32) * signn * (s2 - self.k2).abs().sqrt(),
            _ => {
                s.powi((self.n % 2) as i32)
                    * signm
                    * signn
                    * ((s2 - self.h2) * (s2 - self.k2)).abs().sqrt()
            }
        };
        let t = 1.0 - s2 / self.h2;
        self.coefficients
            .iter()
            .rev()
            .fold(0.0, |acc, c| acc * t + c)
            * psi
    }
}

/// Ellipsoidal harmonic E_n^p(s), normalized so the leading power is s^n.
///
/// ## Parameters and domain
///
/// Finite squared focal parameters `0<h2<k2`, degree `n≤64`, index
/// `1≤p≤2n+1`, and finite real coordinate `s`. The square-root branch signs
/// `signm` and `signn` must each equal +1 or −1. All four Lamé families are
/// indexed in SciPy's K,L,M,N order. The polynomial coefficients are obtained
/// from a symmetric tridiagonal eigensystem.
///
/// ## Errors
///
/// Invalid arguments return [`Error::InvalidInput`]. Eigensolver failure returns
/// [`Error::NumericalFailure`]; overflow returns [`Error::NonFiniteEvaluation`].
///
/// # Examples
///
/// ```
/// assert_eq!(sci_special::ellip_harm(0.5,1.5,0,1,2.0,1.0,1.0)?,1.0);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// - [SciPy ellip_harm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellip_harm.html).
pub fn ellip_harm(
    h2: f64,
    k2: f64,
    n: usize,
    p: usize,
    s: f64,
    signm: f64,
    signn: f64,
) -> Result<f64> {
    if !s.is_finite() || signm.abs() != 1.0 || signn.abs() != 1.0 {
        return Err(Error::InvalidInput(
            "coordinate must be finite and branch signs must be ±1",
        ));
    }
    crate::error::finite(lame(h2, k2, n, p)?.value(s, signm, signn))
}

/// Second-kind ellipsoidal harmonic F_n^p(s).
///
/// The definition is `(2n+1) E_n^p(s) ∫_s^∞ dt / (E_n^p(t)² √(t²−h2) √(t²−k2))`.
///
/// ## Parameters and domain
///
/// Focal parameters and indices follow [`ellip_harm`]; finite `s>sqrt(k2)`.
/// Branch signs are positive. A reciprocal-coordinate substitution maps the
/// infinite integral onto a finite interval. Adaptive quadrature uses absolute
/// tolerance 2e-13; this is not a relative-accuracy guarantee for tiny results.
///
/// ## Errors
///
/// Invalid parameters return [`Error::InvalidInput`]; eigensolver or quadrature
/// failure and nonfinite evaluations use the shared errors.
///
/// # Examples
///
/// ```
/// assert!(sci_special::ellip_harm_2(0.5,1.5,0,1,2.0)?.is_finite());
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// - [SciPy ellip_harm_2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellip_harm_2.html).
pub fn ellip_harm_2(h2: f64, k2: f64, n: usize, p: usize, s: f64) -> Result<f64> {
    let polynomial = lame(h2, k2, n, p)?;
    if !s.is_finite() || s <= k2.sqrt() {
        return Err(Error::InvalidInput(
            "second-kind coordinate must exceed sqrt(k2)",
        ));
    }
    let value = super::inverses::integrate(
        |t| {
            if t == 0.0 {
                return if n == 0 { 1.0 } else { 0.0 };
            }
            let e = polynomial.value(1.0 / t, 1.0, 1.0);
            1.0 / (e * e * (1.0 - k2 * t * t).sqrt() * (1.0 - h2 * t * t).sqrt())
        },
        0.0,
        1.0 / s,
    );
    crate::error::finite((2 * n + 1) as f64 * polynomial.value(s, 1.0, 1.0) * value)
}

/// Ellipsoidal surface normalization constant γ_n^p.
///
/// ## Parameters and domain
///
/// Focal parameters and indices follow [`ellip_harm`]. Endpoint substitutions
/// remove square-root singularities before adaptive quadrature. The returned
/// normalization is `8(A₂ B₀−A₀ B₂)`, where A integrates over `[sqrt(h2),sqrt(k2)]`
/// and B over `[0,sqrt(h2)]` with the Lamé surface weight.
///
/// ## Errors
///
/// Invalid parameters, eigensolver failure, and nonfinite quadrature results
/// use the same errors as [`ellip_harm_2`]. Absolute quadrature tolerance is 2e-13.
///
/// # Examples
///
/// ```
/// assert!(sci_special::ellip_normal(0.5,1.5,0,1)? > 0.0);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// - [SciPy ellip_normal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellip_normal.html).
pub fn ellip_normal(h2: f64, k2: f64, n: usize, p: usize) -> Result<f64> {
    let polynomial = lame(h2, k2, n, p)?;
    let integral = |upper: bool, power: i32| {
        super::inverses::integrate(
            |theta| {
                let t2 = if upper {
                    h2 + (k2 - h2) * theta.sin().powi(2)
                } else {
                    h2 * theta.sin().powi(2)
                };
                let t = t2.sqrt();
                let e = polynomial.value(t, 1.0, 1.0);
                if upper {
                    e * e * t.powi(power - 1)
                } else {
                    e * e * t.powi(power) / (k2 - t2).sqrt()
                }
            },
            0.0,
            std::f64::consts::FRAC_PI_2,
        )
    };
    crate::error::finite(
        8.0 * (integral(true, 2) * integral(false, 0) - integral(true, 0) * integral(false, 2)),
    )
}
