//! Additional Gamma-related functions.
use crate::backend::{cephes64, utils::real_kernel};
#[allow(unused_imports)]
use sci_core::Float;

// Minka's initial estimate followed by Newton refinement, as in xsf/digammainv.h.

pub(crate) fn sign_gamma(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x == 0.0 {
        1.0_f64.copysign(x)
    } else if !x.is_finite() {
        f64::NAN
    } else {
        let lower_integer = x.floor();
        if x == lower_integer {
            return f64::NAN;
        }
        // Every negative noninteger f64 has |floor(x)| < 2^53, so its
        // integer part can be inspected exactly without a floating remainder.
        if (lower_integer as i64) & 1 != 0 {
            -1.0
        } else {
            1.0
        }
    }
}
real_kernel! {
    /// Sign of the Gamma function. Negative integers are outside the domain;
    /// signed zero returns the sign of the corresponding Gamma pole.
    gammasgn(x) { sign_gamma(x) }
}
real_kernel! {
    /// Pochhammer symbol (a)_m = Gamma(a+m)/Gamma(a), with pole cancellation.
    poch(a, m) {
        let mut m = m;
        let mut product = 1.0;
        // Bound the recurrence: above 2^53, subtraction by 1 need not make progress.
        if m.abs() <= 10000.0 {
            while m >= 1.0 {
                if a + m == 1.0 { break; }
                m -= 1.0;
                product *= a + m;
                if !product.is_finite() || product == 0.0 { break; }
            }
            while m <= -1.0 {
                if a + m == 0.0 { break; }
                product /= a + m;
                m += 1.0;
                if !product.is_finite() || product == 0.0 { break; }
            }
        }
        if m == 0.0 { product }
        else if a > 1e4 && m.abs() <= 1.0 {
            product * a.powf(m) * (1.0 + m * (m - 1.0) / (2.0 * a)
                + m * (m - 1.0) * (m - 2.0) * (3.0 * m - 1.0) / (24.0 * a * a)
                + m * m * (m - 1.0) * (m - 1.0) * (m - 2.0) * (m - 3.0) / (48.0 * a * a * a))
        } else {
            let pole = |x: f64| x <= 0.0 && x == x.ceil() && x.abs() < 1e13;
            if pole(a + m) && !pole(a) && a + m != m { f64::INFINITY }
            else if !pole(a + m) && pole(a) { 0.0 }
            else { product * (cephes64::lgam(a + m) - cephes64::lgam(a)).exp() * sign_gamma(a + m) * sign_gamma(a) }
        }
    }
}
real_kernel! {
    /// Real binomial coefficient Gamma(n+1)/(Gamma(k+1)*Gamma(n-k+1)).
    /// Negative integer n is outside the domain.
    binom(n, k) {
        if n < 0.0 && n == n.floor() { f64::NAN }
        else {
            let mut kx = k.floor();
            if k == kx && (n.abs() > 1e-8 || n == 0.0) && n == n.floor() && kx > n / 2.0 && n > 0.0 { kx = n - kx; }
            if k == k.floor() && (n.abs() > 1e-8 || n == 0.0) && (0.0..20.0).contains(&kx) {
                let (mut num, mut den) = (1.0, 1.0);
                for i in 1..=kx as i32 {
                    num *= i as f64 + n - kx; den *= i as f64;
                    if num.abs() > 1e50 { num /= den; den = 1.0; }
                }
                num / den
            } else if n >= 1e10 * k && k > 0.0 { (-cephes64::lbeta(1.0 + n - k, 1.0 + k) - (n + 1.0).ln()).exp() }
            else if k > 1e8 * n.abs() {
                let g = cephes64::gamma(1.0 + n);
                let num = (g / k.abs() + g * n / (2.0 * k * k)) / (core::f64::consts::PI * k.abs().powf(n));
                let kx = k.floor();
                let (dk, sign) = if kx.abs() <= i32::MAX as f64 { (k-kx, if kx % 2.0 == 0.0 { 1.0 } else { -1.0 }) } else { (k,1.0) };
                num * ((dk - n) * core::f64::consts::PI).sin() * sign
            } else { 1.0 / (n + 1.0) / cephes64::beta(1.0 + n - k, 1.0 + k) }
        }
    }
}

// Refine Cephes' inverse in the smaller probability tail. Its interval-halving
// stopping criterion otherwise loses digits for strongly unequal beta shapes.
pub(crate) fn inverse_beta(a: f64, b: f64, p: f64) -> f64 {
    // Select the smaller tail before refinement. Keep a direct-coordinate
    // estimate when subtracting the reflected quantile would lose digits.
    let reflected = if p > 0.5 {
        let reflected = 1.0 - inverse_beta(b, a, 1.0 - p);
        if reflected >= 0.125 {
            return reflected;
        }
        Some(reflected)
    } else {
        None
    };
    let (mut x, converged) = cephes64::incbi_status(a, b, p);
    let mut converged = converged;
    if let Some(reflected) = reflected {
        // Here the direct coordinate is small. Cephes can instead return a
        // spurious near-one estimate when its CDF saturates, so reject that
        // estimate in favor of the reflected seed before Newton refinement.
        if !(0.0..0.125).contains(&x) || x == 0.0 {
            x = reflected;
        }
        // The direct-coordinate residual of a p > 1/2 solve loses the small
        // upper tail, so always refine this case.
        converged = false;
    }
    // A converged Cephes step in unreflected coordinates is already accurate
    // relative to x; refinement would only re-evaluate the same residual.
    if converged || p == 0.0 || p == 1.0 || x == 0.0 || x == 1.0 {
        return x;
    }
    let log_beta = cephes64::lbeta(a, b);
    // Safeguarded Halley iteration on the smaller tail. Far from the root the
    // step uses ln(F/target), which stays well scaled in the convex far tails
    // where a plain Newton step on F overshoots or crawls. The iterate with
    // the smallest log residual is returned, so refinement never makes the
    // Cephes estimate worse.
    let (mut best_x, mut best_g) = (x, f64::INFINITY);
    for _ in 0..12 {
        let density = ((a - 1.0) * x.ln() + (b - 1.0) * (-x).ln_1p() - log_beta).exp();
        // For p > 1/2, the upper tail Q = 1 - p is known exactly, but forming
        // 1 - x costs up to eps/2 in x. The direct residual instead costs about
        // eps*p/density in x, so take the upper tail unless that is smaller.
        let upper = p > 0.5 && (x >= 0.125 || 0.5 * density < 2.0 * p - 1.0);
        let (tail, target) = if upper {
            (cephes64::incbet(b, a, 1.0 - x), 1.0 - p)
        } else {
            (cephes64::incbet(a, b, x), p)
        };
        if tail.is_nan() || tail <= 0.0 {
            break;
        }
        let ratio = tail / target;
        let g = ratio.ln().abs();
        if g < best_g {
            (best_x, best_g) = (x, g);
        }
        if tail == target {
            break;
        }
        // `sign` orients the tail as an increasing function of x.
        let sign = if upper { -1.0 } else { 1.0 };
        // c = f''/f' = (a-1)/x - (b-1)/(1-x) for the beta density.
        let c = (a - 1.0) / x - (b - 1.0) / (1.0 - x);
        let near = (ratio - 1.0).abs() < 0.1;
        let (mut step, curvature) = if near {
            (sign * (tail - target) / density, c)
        } else {
            // Newton on ln(tail): g'/g'' terms of the log-transformed tail.
            (
                sign * ratio.ln() * tail / density,
                c - sign * density / tail,
            )
        };
        let h = 0.5 * step * curvature;
        if h.abs() < 0.5 {
            step /= 1.0 - h;
        }
        if !step.is_finite() {
            break;
        }
        // Keep the iterate inside (0,1) without taking more than 7/8 of the
        // distance to either endpoint in one step.
        let next = (x - step).clamp(0.125 * x, 1.0 - 0.125 * (1.0 - x));
        if next == x || next <= 0.0 || next >= 1.0 {
            break;
        }
        let clamped = next != x - step;
        x = next;
        // Stop once the predicted error of this step is far below rounding.
        if near
            && !clamped
            && h.abs() < 1e-2
            && cephes64::halley_converged(a, b, x, step, x.min(1.0 - x))
        {
            return x;
        }
    }
    best_x
}

real_kernel! {
    /// Real logarithm of Gamma, for x >= 0. Use gammaln for log(|Gamma(x)|).
    loggamma(x) { if x < 0.0 { f64::NAN } else { cephes64::lgam(x) } }
}

#[cfg(test)]
mod tests {
    use super::sign_gamma;

    #[test]
    fn sign_gamma_handles_negative_intervals_and_large_exact_parity() {
        for (x, expected) in [
            (-0.5, -1.0),
            (-1.5, 1.0),
            (-2.5, -1.0),
            (-3.5, 1.0),
            (-(2.0_f64.powi(52) - 0.5), 1.0),
            (-(2.0_f64.powi(52) - 1.5), -1.0),
            (0.5, 1.0),
        ] {
            assert_eq!(sign_gamma(x), expected, "x = {x}");
        }
        for x in [-2.0, -1.0, f64::NEG_INFINITY, f64::NAN] {
            assert!(sign_gamma(x).is_nan(), "x = {x}");
        }
        assert_eq!(sign_gamma(0.0).to_bits(), 1.0_f64.to_bits());
        assert_eq!(sign_gamma(-0.0).to_bits(), (-1.0_f64).to_bits());
    }
}
