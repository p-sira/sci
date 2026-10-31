//! Complex Lambert W branches, adapted from SciPy XSF's BSD-licensed lambertw.h.
//! See LICENSES/xsf-BSD-3-Clause.txt. Work and branch indices are bounded.

use super::StrErr;
use core::f64::consts::{E, PI};
use num_complex::Complex64;

const NEG_INV_E: f64 = -1.0 / E;
const TOL: f64 = 64.0 * f64::EPSILON;

fn finite(z: Complex64) -> bool {
    z.re.is_finite() && z.im.is_finite()
}

fn pade(z: Complex64) -> Complex64 {
    // (3, 2) rational approximation to the Taylor series about zero.
    let num = (12.851_063_829_787_234 * z + 12.340_425_531_914_894) * z + 1.0;
    let den = (32.531_914_893_617_02 * z + 14.340_425_531_914_894) * z + 1.0;
    z * num / den
}

fn branch_point(z: Complex64, side: f64) -> Complex64 {
    let p = side * (2.0 * (E * z + 1.0)).sqrt();
    // The Puiseux series avoids Halley's ill-conditioned double root at -1/e.
    -1.0 + p * (1.0 + p * (-1.0 / 3.0 + p * (11.0 / 72.0 - p * (43.0 / 540.0))))
}

fn asymptotic(z: Complex64, k: i32) -> Complex64 {
    let l = z.ln() + Complex64::new(0.0, 2.0 * PI * f64::from(k));
    l - l.ln()
}

fn initial(z: Complex64, k: i32) -> Complex64 {
    if k == 0 {
        if (z - NEG_INV_E).norm() < 0.3 {
            branch_point(z, 1.0)
        } else if z.re > -1.0 && z.re < 1.5 && z.im.abs() < 1.0 && z.re > -2.5 * z.im.abs() - 0.2 {
            pade(z)
        } else {
            asymptotic(z, k)
        }
    } else if k == -1 && z.im >= 0.0 && (z - NEG_INV_E).norm() < 0.3 {
        branch_point(z, -1.0)
    } else if k == -1 && z.im == 0.0 && z.re < 0.0 && z.norm() <= -NEG_INV_E {
        Complex64::new((-z.re).ln(), 0.0)
    } else {
        asymptotic(z, k)
    }
}

/// Inverse of `w * exp(w) = z` on an integer branch, evaluated in binary64.
pub(crate) fn lambertw(z: Complex64, k: i32) -> Result<Complex64, StrErr> {
    if !finite(z) {
        return Err("lambertw: Argument must be finite.");
    }
    if k.unsigned_abs() > 1_000_000 {
        return Err("lambertw: Branch index exceeds the supported bound.");
    }
    if z == Complex64::new(0.0, 0.0) {
        return if k == 0 {
            Ok(z)
        } else {
            Err("lambertw: Nonprincipal branches are singular at zero.")
        };
    }
    if z.im == 0.0 && k == 0 && z.re >= NEG_INV_E {
        return Ok(Complex64::new(super::lambertw0(z.re)?, z.im));
    }
    if z.im == 0.0 && k == -1 && (NEG_INV_E..0.0).contains(&z.re) {
        return Ok(Complex64::new(super::lambertwm1(z.re)?, z.im));
    }
    if z.re == NEG_INV_E && z.im == 0.0 && (k == 0 || k == -1) {
        return Ok(Complex64::new(-1.0, z.im));
    }
    if z == Complex64::new(1.0, 0.0) && k == 0 {
        return Ok(Complex64::new(0.567_143_290_409_783_8, z.im));
    }

    // At the double root, derivative-based iterations amplify roundoff.
    // Four Puiseux correction orders provide a smaller forward error than
    // a converged Halley step when |z+1/e| is this small.
    if (z - NEG_INV_E).norm() < 1e-6 {
        if k == 0 {
            return Ok(branch_point(z, 1.0));
        }
        if k == -1 && z.im >= 0.0 {
            return Ok(branch_point(z, -1.0));
        }
    }

    let mut w = initial(z, k);
    if !finite(w) {
        return Err("lambertw: Initial approximation is not finite.");
    }
    for _ in 0..100 {
        // Rearrange the Halley step for Re(w)>=0 to avoid exp(w) overflow.
        let (residual, derivative) = if w.re >= 0.0 {
            let difference = w - z * (-w).exp();
            (difference, w + 1.0)
        } else {
            let ew = w.exp();
            let difference = w * ew - z;
            (difference / ew, w + 1.0)
        };
        let denominator = derivative - (w + 2.0) * residual / (2.0 * derivative);
        if denominator.norm() == 0.0 || !finite(denominator) {
            return Err("lambertw: Ill-conditioned iteration near branch point.");
        }
        let next = w - residual / denominator;
        if !finite(next) {
            return Err("lambertw: Iteration overflowed.");
        }
        if (next - w).norm() <= TOL * next.norm().max(1e-300) {
            // A small step alone does not prove a correct root: reject stalled
            // iterations and wrong values in the branch-point neighborhood.
            let (r, scale) = if next.re >= 0.0 {
                (next - z * (-next).exp(), next.norm())
            } else {
                (next * next.exp() - z, z.norm())
            };
            if r.norm() <= 2e-13 * scale.max(1e-300) {
                return Ok(next);
            }
        }
        w = next;
    }
    Err("lambertw: Failed to converge in 100 iterations.")
}
