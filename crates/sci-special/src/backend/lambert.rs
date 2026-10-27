//! Real branches of Lambert W, backed by the `lambert_w` dependency used by `special`.
use crate::backend::StrErr;
use crate::backend::utils::real_kernel;
use sci_core::Float;

fn branch_series(x: f64, lower: bool) -> Option<f64> {
    let distance = x - lambert_w::NEG_INV_E;
    if !(0.0..1e-6).contains(&distance) || distance == 0.0 {
        return None;
    }
    // Correct the rounded -1/e before taking the square root: otherwise its
    // rounding error is amplified near the branch point. Use the Puiseux series.
    let delta = distance - 1.2428753672788363168e-17;
    let p = (2.0 * core::f64::consts::E * delta).sqrt() * if lower { -1.0 } else { 1.0 };
    Some(
        -1.0 + p
            * (1.0
                + p * (-1.0 / 3.0 + p * (11.0 / 72.0 + p * (-43.0 / 540.0 + p * 769.0 / 17280.0)))),
    )
}

/// Real Lambert W on branch k = 0 or k = -1.
///
/// Other branches return an error; use [`crate::backend::complex::lambertw`] for complex results.
pub fn lambertw<T: Float>(x: T, k: i32) -> Result<T, StrErr> {
    match k {
        0 => lambertw0(x),
        -1 => lambertwm1(x),
        _ => Err("lambertw: Real branches require k = 0 or k = -1."),
    }
}

real_kernel! {
    /// Principal real Lambert W branch, for x >= -1/e.
    ///
    /// Inverts w*exp(w) for w >= -1. Preserves signed zero.
    lambertw0(x) {
        // The backend's rational approximation has absolute error near zero.
        // This Taylor expansion preserves relative accuracy and signed subnormals.
        if x.abs() <= 1e-3 {
            x * (1.0 + x * (-1.0 + x * (1.5 + x * (-8.0 / 3.0
                + x * (125.0 / 24.0 - 10.8 * x)))))
        } else if let Some(value) = branch_series(x, false) {
            value
        } else {
            lambert_w::lambert_w0(x)
        }
    }
}

real_kernel! {
    /// Lower real Lambert W branch, for -1/e <= x <= 0.
    ///
    /// Inverts w*exp(w) for w <= -1. At either signed zero returns -infinity.
    lambertwm1(x) {
        branch_series(x, true).unwrap_or_else(|| lambert_w::lambert_wm1(x))
    }
}
