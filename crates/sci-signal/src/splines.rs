// Boundary recurrences adapted from SciPy; see LICENSES/scipy-signal.txt.
use super::{checked, nonempty, positive, samples, sepfir2d, size, work};
use crate::{Error, Result};
use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;

fn precision(p: f64) -> Result<()> {
    if !p.is_finite() || p <= 0. || p >= 1. {
        return Err(Error::InvalidInput("precision must be in (0,1)"));
    }
    Ok(())
}
fn hc(k: i32, cs: f64, r: f64, omega: f64) -> f64 {
    if k < 0 {
        return 0.;
    }
    let base = cs * r.powi(k);
    if omega == 0. {
        base * (k + 1) as f64
    } else if omega == PI {
        base * (k + 1) as f64 * if k % 2 == 0 { 1. } else { -1. }
    } else {
        base * (omega * (k + 1) as f64).sin() / omega.sin()
    }
}
fn hs(k: i32, cs: f64, r: f64, omega: f64) -> f64 {
    let k = k.abs();
    let rsq = r * r;
    let base = cs * cs * (1. + rsq) / (1. - rsq).powi(3) * r.powi(k);
    if omega == 0. {
        return base * (1. + (1. - rsq) / (1. + rsq) * k as f64);
    }
    if omega == PI {
        return base
            * (1. + (1. - rsq) / (1. + rsq) * if k % 2 == 0 { 1. } else { -1. } * k as f64);
    }
    let c0 = cs * cs * (1. + rsq) / (1. - rsq) / (1. - 2. * rsq * (2. * omega).cos() + rsq * rsq);
    let gamma = (1. - rsq) / (1. + rsq) / omega.tan();
    c0 * r.powi(k) * ((omega * k as f64).cos() + gamma * (omega * k as f64).sin())
}
fn initial_sum(
    x: &[f64],
    coefficient: impl Fn(i32) -> f64,
    threshold: f64,
    reverse: bool,
) -> Result<f64> {
    let mut sum = 0.;
    for k in 0..x.len() {
        let c = coefficient(k as i32);
        sum += c * x[if reverse { x.len() - 1 - k } else { k }];
        if c * c <= threshold && k + 1 < x.len() {
            return Ok(sum);
        }
    }
    Err(Error::NumericalFailure(
        "mirror boundary sum did not converge within signal length",
    ))
}
fn smooth_parameters(lambda: f64) -> Result<(f64, f64)> {
    if !lambda.is_finite() || lambda <= 1. / 144. {
        return Err(Error::InvalidInput("smoothing lambda must exceed 1/144"));
    }
    let xi = 1. - 96. * lambda + 24. * lambda * (3. + 144. * lambda).sqrt();
    let omega = (144. * lambda - 1.).sqrt().atan2(xi.sqrt());
    let rho = (24. * lambda - 1. - xi.sqrt()) / (24. * lambda)
        * ((48. * lambda + 24. * lambda * (3. + 144. * lambda).sqrt()) / xi).sqrt();
    if !rho.is_finite() || !(0. ..1.).contains(&rho) {
        return Err(Error::NumericalFailure(
            "ill-conditioned smoothing parameter",
        ));
    }
    Ok((rho, omega))
}
fn coefficient1d(x: &[f64], z: f64, scale: f64) -> Result<Vec<f64>> {
    nonempty(x)?;
    let mut power = 1.;
    let initial = x
        .iter()
        .map(|v| {
            let p = power;
            power *= z;
            p * v
        })
        .sum::<f64>();
    let mut forward = vec![0.; x.len()];
    forward[0] = x[0] + z * initial;
    for k in 1..x.len() {
        forward[k] = x[k] + z * forward[k - 1];
    }
    let last = x.len() - 1;
    let mut out = vec![0.; x.len()];
    out[last] = z / (z - 1.) * forward[last];
    if x.len() == 1 {
        return checked(out);
    }
    for k in (0..last).rev() {
        out[k] = z * (out[k + 1] - forward[k]);
    }
    checked(out.into_iter().map(|v| v * scale).collect())
}
fn basis(x: f64, cubic: bool) -> f64 {
    let x = x.abs();
    if cubic {
        if x < 1. {
            2. / 3. - x * x + x.powi(3) / 2.
        } else if x < 2. {
            (2. - x).powi(3) / 6.
        } else {
            0.
        }
    } else if x < 0.5 {
        0.75 - x * x
    } else if x < 1.5 {
        (1.5 - x).powi(2) / 2.
    } else {
        0.
    }
}
fn evaluate(c: &[f64], x: &[f64], dx: f64, x0: f64, cubic: bool) -> Result<Vec<f64>> {
    nonempty(c)?;
    samples(x)?;
    positive(dx)?;
    if !x0.is_finite() {
        return Err(Error::InvalidInput("origin must be finite"));
    }
    work(x.len(), 4)?;
    x.iter()
        .map(|&x| {
            let mut t = (x - x0) / dx;
            if !t.is_finite() {
                return Err(Error::NonFiniteEvaluation);
            }
            if c.len() == 1 {
                t = 0.;
            } else {
                let period = 2. * (c.len() - 1) as f64;
                t = t.rem_euclid(period);
                if t > (c.len() - 1) as f64 {
                    t = period - t;
                }
            }
            let lower = (t - if cubic { 2. } else { 1.5 }).floor() as isize + 1;
            let mut value = 0.;
            for j in lower..lower + if cubic { 4 } else { 3 } {
                let index = j.clamp(0, c.len() as isize - 1) as usize;
                value += c[index] * basis(t - j as f64, cubic);
            }
            if value.is_finite() {
                Ok(value)
            } else {
                Err(Error::NonFiniteEvaluation)
            }
        })
        .collect()
}
fn separable(x: &[Vec<f64>], filter: impl Fn(&[f64]) -> Result<Vec<f64>>) -> Result<Vec<Vec<f64>>> {
    if x.is_empty() || x[0].is_empty() || x.iter().any(|r| r.len() != x[0].len()) {
        return Err(Error::DimensionMismatch);
    }
    work(x.len(), x[0].len())?;
    size(x.len() * x[0].len())?;
    let mut out = x.iter().map(|r| filter(r)).collect::<Result<Vec<_>>>()?;
    for j in 0..x[0].len() {
        let col = out.iter().map(|r| r[j]).collect::<Vec<_>>();
        let col = filter(&col)?;
        for (i, v) in col.into_iter().enumerate() {
            out[i][j] = v;
        }
    }
    Ok(out)
}

/// Applies the mirror-symmetric filter `H(z)=c0/((1-z1/z)(1-z1 z))`.
///
/// ## Parameters
///
/// x is nonempty real samples, c0 is finite gain, abs(z1)<1, precision in (0,1) is the boundary-series convergence threshold. Returns length x.len(); rejects boundary sums that do not converge within the signal length.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses causal and anticausal recurrences with finite mirror boundary sums, matching SciPy convergence behavior; short signals or poles near the unit circle can fail.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _=symiirorder1(&[1.;32],1.,0.2,1e-8)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy symiirorder1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.symiirorder1.html).
pub fn symiirorder1(x: &[f64], c0: f64, z1: f64, p: f64) -> Result<Vec<f64>> {
    nonempty(x)?;
    precision(p)?;
    if !c0.is_finite() || !z1.is_finite() || z1.abs() >= 1. {
        return Err(Error::InvalidInput("finite gain and |pole|<1 required"));
    }
    let first = x[0] + initial_sum(x, |k| z1.powi(k + 1), p * p, false)?;
    let mut y = vec![0.; x.len()];
    y[0] = first;
    for k in 1..x.len() {
        y[k] = x[k] + z1 * y[k - 1];
    }
    let last = x.len() - 1;
    let mut out = vec![0.; x.len()];
    out[last] = c0 / (1. - z1) * y[last];
    for k in (0..last).rev() {
        out[k] = c0 * y[k] + z1 * out[k + 1];
    }
    checked(out)
}

/// Applies a mirror-symmetric cascade of `cs/(1-2r cos(omega) z^-1+r^2 z^-2)` and its reverse.
///
/// ## Parameters
///
/// x requires at least two samples, 0<=r<1, omega in \[0,pi\] radians/sample; cs=1-2r cos(omega)+r^2. precision in (0,1) controls boundary sums, which must converge within x.len(). Returns equally many samples.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Boundary series follow SciPy: forward coefficients compare squared magnitude to precision squared, backward coefficients to precision. Filtering uses f64 recurrences.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _=symiirorder2(&[1.;64],0.3,0.6,1e-8)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy symiirorder2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.symiirorder2.html).
pub fn symiirorder2(x: &[f64], r: f64, omega: f64, p: f64) -> Result<Vec<f64>> {
    nonempty(x)?;
    precision(p)?;
    if x.len() < 2
        || !r.is_finite()
        || !(0. ..1.).contains(&r)
        || !omega.is_finite()
        || !(0. ..=PI).contains(&omega)
    {
        return Err(Error::InvalidInput(
            "two samples, radius in [0,1), angle in [0,pi] required",
        ));
    }
    let a2 = 2. * r * omega.cos();
    let a3 = -r * r;
    let cs = 1. - a2 - a3;
    let mut y = vec![0.; x.len()];
    y[0] = hc(0, cs, r, omega) * x[0] + initial_sum(x, |k| hc(k + 1, cs, r, omega), p * p, false)?;
    y[1] = hc(0, cs, r, omega) * x[1]
        + hc(1, cs, r, omega) * x[0]
        + initial_sum(x, |k| hc(k + 2, cs, r, omega), p * p, false)?;
    for k in 2..x.len() {
        y[k] = cs * x[k] + a2 * y[k - 1] + a3 * y[k - 2];
    }
    let mut out = vec![0.; x.len()];
    let n = x.len();
    out[n - 1] = initial_sum(
        x,
        |k| hs(k, cs, r, omega) + hs(k + 1, cs, r, omega),
        p,
        true,
    )?;
    out[n - 2] = initial_sum(
        x,
        |k| hs(k - 1, cs, r, omega) + hs(k + 2, cs, r, omega),
        p,
        true,
    )?;
    for k in (0..n - 2).rev() {
        out[k] = cs * y[k] + a2 * out[k + 1] + a3 * out[k + 2];
    }
    checked(out)
}

/// Computes cardinal cubic B-spline coefficients for `s(x)=sum_j c[j] beta_3(x-j)`.
///
/// ## Parameters
///
/// x is nonempty unit-spaced samples. lambda=0 gives interpolation; lambda>1/144 gives cubic smoothing. Small positive lambda and negative lambda are rejected. Returns coefficients in input order; the single-sample lambda=0 case preserves SciPy historical normalization.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic; cancellation and ill conditioning can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _=cspline1d(&[1.,2.,3.,2.,1.],0.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy cspline1d](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cspline1d.html).
pub fn cspline1d(x: &[f64], lambda: f64) -> Result<Vec<f64>> {
    nonempty(x)?;
    if lambda == 0. {
        return coefficient1d(x, -2. + 3f64.sqrt(), 6.);
    }
    if x.len() < 2 {
        return Err(Error::InvalidInput("smoothing requires two samples"));
    }
    let (r, w) = smooth_parameters(lambda)?;
    let cs = 1. - 2. * r * w.cos() + r * r;
    let a2 = 2. * r * w.cos();
    let a3 = -r * r;
    let mut y = vec![0.; x.len()];
    y[0] = hc(0, cs, r, w) * x[0]
        + x.iter()
            .enumerate()
            .map(|(k, x)| hc(k as i32 + 1, cs, r, w) * x)
            .sum::<f64>();
    y[1] = hc(0, cs, r, w) * x[0]
        + hc(1, cs, r, w) * x[1]
        + x.iter()
            .enumerate()
            .map(|(k, x)| hc(k as i32 + 2, cs, r, w) * x)
            .sum::<f64>();
    for k in 2..x.len() {
        y[k] = cs * x[k] + a2 * y[k - 1] + a3 * y[k - 2];
    }
    let n = x.len();
    let mut out = vec![0.; n];
    out[n - 1] = x
        .iter()
        .rev()
        .enumerate()
        .map(|(k, x)| (hs(k as i32, cs, r, w) + hs(k as i32 + 1, cs, r, w)) * x)
        .sum();
    out[n - 2] = x
        .iter()
        .rev()
        .enumerate()
        .map(|(k, x)| (hs(k as i32 - 1, cs, r, w) + hs(k as i32 + 2, cs, r, w)) * x)
        .sum();
    for k in (0..n - 2).rev() {
        out[k] = cs * y[k] + a2 * out[k + 1] + a3 * out[k + 2];
    }
    checked(out)
}

/// Computes cardinal quadratic B-spline coefficients for `s(x)=sum_j c[j] beta_2(x-j)`.
///
/// ## Parameters
///
/// x is nonempty unit-spaced samples. Returns interpolation coefficients; quadratic smoothing is unsupported, as in SciPy. Single-sample output preserves SciPy historical normalization.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic; cancellation and ill conditioning can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _=qspline1d(&[1.,2.,3.,2.,1.])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy qspline1d](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.qspline1d.html).
pub fn qspline1d(x: &[f64]) -> Result<Vec<f64>> {
    coefficient1d(x, -3. + 2. * 2f64.sqrt(), 8.)
}

/// Evaluates the cardinal cubic spline `s(x)=sum_j c[j] beta((x-x0)/dx-j)` with mirror extension.
///
/// ## Parameters
///
/// c contains nonempty coefficients. x contains arbitrary finite evaluation positions, dx>0 is original sample spacing, x0 is the original origin in the same units. Returns values in x order. Positions reflect about both sample endpoints; coefficient indices clamp at edges as in SciPy.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses closed-form compact-support cardinal basis functions and modulo reflection; distant evaluation positions can lose phase precision.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _=cspline1d_eval(&[1.,2.,3.],&[-1.,0.,0.5,2.],1.,0.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy cspline1d_eval](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cspline1d_eval.html).
pub fn cspline1d_eval(c: &[f64], x: &[f64], dx: f64, x0: f64) -> Result<Vec<f64>> {
    evaluate(c, x, dx, x0, true)
}

/// Evaluates the cardinal quadratic spline `s(x)=sum_j c[j] beta((x-x0)/dx-j)` with mirror extension.
///
/// ## Parameters
///
/// c contains nonempty coefficients. x contains arbitrary finite evaluation positions, dx>0 is original sample spacing, x0 is the original origin in the same units. Returns values in x order. Positions reflect about both sample endpoints; coefficient indices clamp at edges as in SciPy.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses closed-form compact-support cardinal basis functions and modulo reflection; distant evaluation positions can lose phase precision.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _=qspline1d_eval(&[1.,2.,3.],&[-1.,0.,0.5,2.],1.,0.)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy qspline1d_eval](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.qspline1d_eval.html).
pub fn qspline1d_eval(c: &[f64], x: &[f64], dx: f64, x0: f64) -> Result<Vec<f64>> {
    evaluate(c, x, dx, x0, false)
}

/// Computes separable cubic B-spline coefficients on a unit-spaced two-dimensional grid.
///
/// ## Parameters
///
/// x is a nonempty rectangular row-major matrix. lambda>=0; lambda<=1/144 uses interpolation, larger lambda smoothing. precision in (0,1) controls mirror boundary sums along both axes. Returns the original matrix shape.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic; cancellation and ill conditioning can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _=cspline2d(&vec![vec![1.;32];32],0.,1e-6)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy cspline2d](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cspline2d.html).
pub fn cspline2d(x: &[Vec<f64>], lambda: f64, p: f64) -> Result<Vec<Vec<f64>>> {
    if !lambda.is_finite() || lambda < 0. {
        return Err(Error::InvalidInput("lambda must be nonnegative"));
    }
    precision(p)?;
    if lambda <= 1. / 144. {
        let r = -2. + 3f64.sqrt();
        separable(x, |v| symiirorder1(v, -6. * r, r, p))
    } else {
        let (r, w) = smooth_parameters(lambda)?;
        separable(x, |v| symiirorder2(v, r, w, p))
    }
}

/// Computes separable quadratic B-spline interpolation coefficients on a unit-spaced grid.
///
/// ## Parameters
///
/// x is a nonempty rectangular row-major matrix. precision in (0,1) controls mirror boundary convergence along both axes. Returns the original shape. Quadratic smoothing is unsupported.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic; cancellation and ill conditioning can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _=qspline2d(&vec![vec![1.;32];32],1e-6)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy qspline2d](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.qspline2d.html).
pub fn qspline2d(x: &[Vec<f64>], p: f64) -> Result<Vec<Vec<f64>>> {
    let r = -3. + 2. * 2f64.sqrt();
    separable(x, |v| symiirorder1(v, -8. * r, r, p))
}

/// Smooths a grid by cubic spline analysis followed by separable reconstruction with `[1,4,1]/6`.
///
/// ## Parameters
///
/// x is a nonempty rectangular matrix; lambda>=0 follows cspline2d. Uses SciPy default precision=1e-6 and mirror boundaries. Returns the original shape. Supports real f64 matrices only. The reconstruction taps retain SciPy float32 rounding before f64 filtering.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic; cancellation and ill conditioning can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _=spline_filter(&vec![vec![1.;32];32],0.1)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy spline_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spline_filter.html).
pub fn spline_filter(x: &[Vec<f64>], lambda: f64) -> Result<Vec<Vec<f64>>> {
    let c = cspline2d(x, lambda, 1e-6)?;
    sepfir2d(
        &c,
        &[(1f32 / 6.) as f64, (4f32 / 6.) as f64, (1f32 / 6.) as f64],
        &[(1f32 / 6.) as f64, (4f32 / 6.) as f64, (1f32 / 6.) as f64],
    )
}

/// Minimizes `sum_i weight[i](x[i]-y[i])^2 + lambda sum_i (Delta^order x[i])^2`.
///
/// ## Parameters
///
/// y contains 2..256 finite samples; order is 1..8 and below y.len(); lambda>=0 is explicit, replacing SciPy REML dispatch. weights contains one nonnegative finite weight per sample. Returns smoothed samples; singular weight/penalty combinations are errors. lambda=0 returns y as SciPy does.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Delegates a dense positive-definite solve to private nalgebra Cholesky, bounded to 256 samples. Large penalties can make the system ill conditioned; no automatic penalty selection is performed.
///
/// # Examples
///
/// ```
/// use sci_signal::*;
/// let _=whittaker_henderson(&[1.,2.,4.,3.],1.,2,&[1.;4])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy whittaker_henderson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.whittaker_henderson.html).
pub fn whittaker_henderson(
    y: &[f64],
    lambda: f64,
    order: usize,
    weights: &[f64],
) -> Result<Vec<f64>> {
    nonempty(y)?;
    samples(weights)?;
    if y.len() > 256
        || weights.len() != y.len()
        || order == 0
        || order > 8
        || order >= y.len()
        || !lambda.is_finite()
        || lambda < 0.
        || weights.iter().any(|v| *v < 0.)
    {
        return Err(Error::InvalidInput(
            "invalid smoothing shape, weights, order, or lambda",
        ));
    }
    if lambda == 0. {
        return Ok(y.to_vec());
    }
    if weights.iter().filter(|&&w| w > 0.).count() < order {
        return Err(Error::InvalidInput(
            "positive weights must anchor every unpenalized polynomial degree",
        ));
    }
    let n = y.len();
    let mut matrix = DMatrix::from_diagonal(&DVector::from_column_slice(weights));
    let mut diff = vec![1.; order + 1];
    for k in 1..=order {
        diff[k] = diff[k - 1] * (order - k + 1) as f64 / k as f64;
    }
    for (k, v) in diff.iter_mut().enumerate() {
        if (order - k) % 2 == 1 {
            *v = -*v;
        }
    }
    for row in 0..n - order {
        for i in 0..=order {
            for j in 0..=order {
                matrix[(row + i, row + j)] += lambda * diff[i] * diff[j];
            }
        }
    }
    let rhs = DVector::from_iterator(n, y.iter().zip(weights).map(|(y, w)| y * w));
    let out = matrix
        .cholesky()
        .ok_or(Error::NumericalFailure("singular smoothing system"))?
        .solve(&rhs);
    checked(out.as_slice().to_vec())
}
