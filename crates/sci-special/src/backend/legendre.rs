//! Integer-degree associated Legendre functions and spherical harmonics.
use crate::backend::{StrErr, utils::finish};
use core::f64::consts::PI;
use num_complex::Complex;
use sci_core::Float;

const ERROR: StrErr = "Associated Legendre functions require degree <= 1,000,000 and a finite argument in their domain.";

fn validate_degree(n: u32) -> Result<(), StrErr> {
    if n > 1_000_000 { Err(ERROR) } else { Ok(()) }
}

// Spherical normalization: sqrt((2n+1)/(4pi)*(n-m)!/(n+m)!).
// Keep the diagonal in logarithmic form and rescale the recurrence, so tiny
// diagonals can recover in later degrees and large factorials never overflow.
fn associated(n: u32, m: i32, x: f64, sine: f64, spherical: bool) -> f64 {
    let order = m.unsigned_abs();
    if order > n {
        return 0.0;
    }
    if order == 0 {
        let value = crate::backend::orthogonal::eval_legendre(n as i32, x).unwrap_or(f64::NAN);
        return if spherical {
            value * ((2.0 * n as f64 + 1.0) / (4.0 * PI)).sqrt()
        } else {
            value
        };
    }
    if x == 0.0 && (n - order) % 2 == 1 {
        return 0.0;
    }
    if sine == 0.0 {
        return 0.0;
    }
    let mut logarithm = -0.5 * (4.0 * PI).ln() + order as f64 * sine.abs().ln();
    for k in 1..=order {
        logarithm += 0.5 * (0.5 / k as f64).ln_1p();
    }
    let mut previous = 0.0;
    let mut current = if order % 2 == 1 { -1.0 } else { 1.0 };
    for degree in order + 1..=n {
        let l = degree as f64;
        let q = order as f64;
        let denominator = (l - q) * (l + q);
        let a = ((4.0 * l * l - 1.0) / denominator).sqrt();
        let b = (((2.0 * l + 1.0) * ((l - 1.0 - q) * (l - 1.0 + q)))
            / ((2.0 * l - 3.0) * denominator))
            .sqrt();
        let next = a * x * current - b * previous;
        previous = current;
        current = next;
        let magnitude = current.abs().max(previous.abs());
        if magnitude > 1e150 {
            current *= 1e-150;
            previous *= 1e-150;
            logarithm += 150.0 * core::f64::consts::LN_10;
        } else if magnitude != 0.0 && magnitude < 1e-150 {
            current *= 1e150;
            previous *= 1e150;
            logarithm -= 150.0 * core::f64::consts::LN_10;
        }
    }
    if !spherical {
        logarithm += 0.5 * ((4.0 * PI).ln() - (2.0 * n as f64 + 1.0).ln());
        let sign = if m < 0 { -0.5 } else { 0.5 };
        for k in n - order + 1..=n + order {
            logarithm += sign * (k as f64).ln();
        }
    }
    let parity = if (m < 0) && order % 2 == 1 { -1.0 } else { 1.0 };
    parity * (current.abs().ln() + logarithm).exp().copysign(current)
}

/// Ferrers associated Legendre P_n^m(x), including the Condon–Shortley phase.
///
/// Requires integer `n <= 1_000_000` and `-1 <= x <= 1`. Negative orders use
/// P_n^(-m) = (-1)^m (n-m)!/(n+m)! P_n^m; `|m| > n` returns zero.
/// Factorials and tiny diagonal values are handled without intermediate overflow.
pub fn assoc_legendre_p<T: Float>(n: u32, m: i32, x: T) -> Result<T, StrErr> {
    validate_degree(n)?;
    let x = x.to_f64().ok_or(ERROR)?;
    if !(-1.0..=1.0).contains(&x) {
        return Err(ERROR);
    }
    finish(
        associated(n, m, x, ((1.0 - x) * (1.0 + x)).sqrt(), false),
        ERROR,
    )
}

/// Spherically normalized associated Legendre function at polar angle `theta`.
///
/// Equals sqrt((2n+1)/(4pi)*(n-m)!/(n+m)!) P_n^m(cos(theta)).
/// Includes the Condon–Shortley phase. Requires `0 <= theta <= pi` and
/// `n <= 1_000_000`; orders outside `-n..=n` return zero.
pub fn sph_legendre_p<T: Float>(n: u32, m: i32, theta: T) -> Result<T, StrErr> {
    validate_degree(n)?;
    let theta = if theta == T::from(PI).ok_or(ERROR)? {
        PI
    } else {
        theta.to_f64().ok_or(ERROR)?
    };
    if !(0.0..=PI).contains(&theta) {
        return Err(ERROR);
    }
    let (sine, cosine) = theta.sin_cos();
    // pi denotes the exact south pole even though sin(pi) is rounded away from zero.
    let sine = if theta == PI { 0.0 } else { sine };
    finish(associated(n, m, cosine, sine, true), ERROR)
}

/// Complex spherical harmonic Y_n^m(theta,phi), with polar angle first.
///
/// Uses spherical normalization and the Condon–Shortley phase. Requires
/// `0 <= theta <= pi`, finite azimuth `phi`, and `n <= 1_000_000`.
pub fn sph_harm_y<T: Float>(n: u32, m: i32, theta: T, phi: T) -> Result<Complex<T>, StrErr> {
    let amplitude = sph_legendre_p(n, m, theta)?;
    let phi = phi.to_f64().ok_or(ERROR)?;
    if !phi.is_finite() {
        return Err(ERROR);
    }
    let phase = m as f64 * phi;
    let (sine, cosine) = if phase.is_finite() {
        phase.sin_cos()
    } else {
        let (sine, cosine) = phi.sin_cos();
        let phase = num_complex::Complex64::new(cosine, sine).powu(m.unsigned_abs());
        (if m < 0 { -phase.im } else { phase.im }, phase.re)
    };
    Ok(Complex::new(
        amplitude * T::from(cosine).ok_or(ERROR)?,
        amplitude * T::from(sine).ok_or(ERROR)?,
    ))
}
