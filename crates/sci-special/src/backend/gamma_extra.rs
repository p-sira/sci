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
    } else if !x.is_finite() || x == x.floor() {
        f64::NAN
    } else if x.floor() % 2.0 != 0.0 {
        -1.0
    } else {
        1.0
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
    let mut x = cephes64::incbi(a, b, p);
    if p == 0.0 || p == 1.0 || x == 0.0 || x == 1.0 {
        return x;
    }
    let log_beta = cephes64::lbeta(a, b);
    for _ in 0..4 {
        let residual = if p > 0.5 {
            (1.0 - p) - cephes64::incbet(b, a, 1.0 - x)
        } else {
            cephes64::incbet(a, b, x) - p
        };
        let derivative = ((a - 1.0) * x.ln() + (b - 1.0) * (-x).ln_1p() - log_beta).exp();
        let next = x - residual / derivative;
        if !next.is_finite() || next <= 0.0 || next >= 1.0 || next == x {
            break;
        }
        if (next - x).abs() > 0.1 * x.min(1.0 - x) {
            break;
        }
        x = next;
    }
    x
}

real_kernel! {
    /// Real logarithm of Gamma, for x >= 0. Use gammaln for log(|Gamma(x)|).
    loggamma(x) { if x < 0.0 { f64::NAN } else { cephes64::lgam(x) } }
}
