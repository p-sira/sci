//! Integer-degree classical orthogonal polynomials.
//!
//! Kernels follow xsf/orthogonal_eval.h and evaluate in double precision.
//! Degrees with absolute value above 1,000,000 return an error to bound recurrence work.
use crate::backend::{StrErr, utils::finish};
#[allow(unused_imports)]
use sci_core::Float;

macro_rules! polynomial {
    ($(#[$meta:meta])* $name:ident($n:ident, $($arg:ident),+) $body:block) => {
        $(#[$meta])*
        ///
        /// Returns an error for NaN input, undefined results, or |n| > 1,000,000.
        pub fn $name<T: Float>($n: i32, $($arg: T),+) -> Result<T, StrErr> {
            if $n.unsigned_abs() > 1_000_000 {
                return Err(concat!(stringify!($name), ": Invalid argument or degree exceeds 1,000,000."));
            }
            $(if $arg.is_nan() {
                return Err(concat!(stringify!($name), ": Arguments cannot be NAN."));
            }
            let $arg = $arg.to_f64().ok_or("Polynomial: Cannot represent argument as f64.")?;)+
            let value: f64 = $body;
            finish(value, concat!(stringify!($name), ": Undefined real result."))
        }
    };
}

fn hermite(n: i32, x: f64, scale: f64) -> f64 {
    if n < 0 {
        return f64::NAN;
    }
    if n == 0 {
        return 1.0;
    }
    if x.is_infinite() {
        return if n % 2 == 0 { f64::INFINITY } else { x };
    }
    // Odd polynomials vanish exactly at zero, including after even terms overflow.
    if x == 0.0 {
        if n % 2 != 0 {
            return x;
        }
        return (1..n)
            .step_by(2)
            .fold(1.0, |value, k| -scale * k as f64 * value);
    }
    let (mut previous, mut current) = (0.0, 1.0);
    for k in (2..=n).rev() {
        let next = scale * x * current - scale * k as f64 * previous;
        previous = current;
        current = next;
    }
    scale * x * current - scale * previous
}

fn chebyshev(n: i32, x: f64, second: bool) -> f64 {
    let (n, sign) = if n < 0 && second {
        if n == -1 {
            return 0.0;
        }
        (-n - 2, -1.0)
    } else {
        (n.abs(), 1.0)
    };
    if n == 0 {
        return sign;
    }
    if x.is_infinite() {
        return sign * if n % 2 == 0 { f64::INFINITY } else { x };
    }
    if x.abs() == 1.0 {
        let parity = if x < 0.0 && n % 2 != 0 { -1.0 } else { 1.0 };
        return sign * parity * if second { n as f64 + 1.0 } else { 1.0 };
    }
    let parity = if x < 0.0 && n % 2 != 0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let (mut previous, mut current) = (1.0, if second { 2.0 * x } else { x });
    for _ in 1..n {
        let next = 2.0 * x * current - previous;
        previous = current;
        current = next;
        // For x > 1 the magnitudes grow monotonically; the final sign is known.
        if current.is_infinite() {
            break;
        }
    }
    sign * parity * current
}

polynomial! {
    /// Physicists' Hermite polynomial H_n(x), for n >= 0.
    eval_hermite(n, x) { hermite(n, x, 2.0) }
}
polynomial! {
    /// Probabilists' Hermite polynomial He_n(x), for n >= 0.
    eval_hermitenorm(n, x) { hermite(n, x, 1.0) }
}
polynomial! {
    /// Chebyshev polynomial T_n(x), with T_{-n} = T_n.
    eval_chebyt(n, x) { chebyshev(n, x, false) }
}
polynomial! {
    /// Chebyshev polynomial U_n(x), with U_{-1} = 0 and U_{-n-2} = -U_n.
    eval_chebyu(n, x) { chebyshev(n, x, true) }
}
polynomial! {
    /// Scaled Chebyshev polynomial C_n(x) = 2*T_n(x/2).
    eval_chebyc(n, x) { 2.0 * chebyshev(n, 0.5 * x, false) }
}
polynomial! {
    /// Scaled Chebyshev polynomial S_n(x) = U_n(x/2).
    eval_chebys(n, x) { chebyshev(n, 0.5 * x, true) }
}
polynomial! {
    /// Shifted Chebyshev polynomial T_n(2*x-1).
    eval_sh_chebyt(n, x) { chebyshev(n, 2.0 * x - 1.0, false) }
}
polynomial! {
    /// Shifted Chebyshev polynomial U_n(2*x-1).
    eval_sh_chebyu(n, x) { chebyshev(n, 2.0 * x - 1.0, true) }
}

fn legendre(n: i32, x: f64) -> f64 {
    let n = if n < 0 { -(n + 1) } else { n };
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }
    let sign = if x < 0.0 && n % 2 != 0 { -1.0 } else { 1.0 };
    let x = x.abs();
    if x == 1.0 {
        return sign;
    }
    if x.is_infinite() {
        return sign * f64::INFINITY;
    }
    if x < 1e-5 {
        // xsf's power series avoids subtracting adjacent, nearly equal values.
        let a = n / 2;
        let mut term = if a % 2 == 0 { 1.0 } else { -1.0 };
        term *= if n % 2 == 0 {
            -2.0 / crate::backend::cephes64::beta(a as f64 + 1.0, -0.5)
        } else {
            2.0 * x / crate::backend::cephes64::beta(a as f64 + 1.0, 0.5)
        };
        let mut sum = 0.0;
        for k in 0..=a {
            sum += term;
            term *= -2.0 * x * x * (a - k) as f64 * (2 * n + 1 - 2 * a + 2 * k) as f64
                / ((n + 1 - 2 * a + 2 * k) as f64 * (n + 2 - 2 * a + 2 * k) as f64);
            if term.abs() <= f64::EPSILON * sum.abs() {
                break;
            }
        }
        return sign * sum;
    }
    let (mut difference, mut value) = (x - 1.0, x);
    for k in 1..n {
        let k = k as f64;
        difference = ((2.0 * k + 1.0) * (x - 1.0) * value + k * difference) / (k + 1.0);
        value += difference;
        if value.is_infinite() {
            break;
        }
    }
    sign * value
}

fn endpoint_binomial(n: i32, alpha: f64) -> f64 {
    (1..=n).fold(1.0, |value, k| value * ((alpha + k as f64) / k as f64))
}

fn laguerre(n: i32, alpha: f64, x: f64) -> f64 {
    if !alpha.is_finite() || alpha <= -1.0 {
        return f64::NAN;
    }
    if n < 0 {
        return 0.0;
    }
    if n == 0 {
        return 1.0;
    }
    if x.is_infinite() {
        return if x > 0.0 && n % 2 != 0 {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }
    // Retain the rounding residual of alpha+1 when the linear polynomial cancels.
    let sum = alpha + 1.0;
    let error = if alpha.abs() <= 1.0 {
        alpha - (sum - 1.0)
    } else {
        1.0 - (sum - alpha)
    };
    let initial = (sum - x) + error;
    if n == 1 {
        return initial;
    }
    let scale = endpoint_binomial(n, alpha);
    if x == 0.0 {
        return scale;
    }
    if scale.is_finite() {
        let mut difference = -x / (alpha + 1.0);
        let mut value = 1.0 + difference;
        for k in 1..n {
            let k = k as f64;
            difference = -x / (k + alpha + 1.0) * value + k / (k + alpha + 1.0) * difference;
            value += difference;
        }
        return scale * value;
    }
    // Avoid normalization overflow times a tiny value by falling back to the
    // unnormalized three-term recurrence when the scale cannot be represented.
    let (mut previous, mut value) = (1.0, initial);
    for k in 1..n {
        let k = k as f64;
        let next = ((2.0 * k + 1.0 + alpha - x) / (k + 1.0)) * value
            - ((k + alpha) / (k + 1.0)) * previous;
        previous = value;
        value = next;
    }
    value
}

fn gegenbauer(n: i32, alpha: f64, x: f64) -> f64 {
    if !alpha.is_finite() {
        return f64::NAN;
    }
    // Match xsf's convention at alpha = 0, including n = 0.
    if alpha == 0.0 || n < 0 {
        return 0.0;
    }
    if n == 0 {
        return 1.0;
    }
    if alpha < 0.0 && alpha == alpha.floor() && n as f64 > -2.0 * alpha {
        return 0.0;
    }
    if x.abs() == 1.0 {
        let value = (0..n).fold(1.0, |value, k| {
            value * ((2.0 * alpha + k as f64) / (k as f64 + 1.0))
        });
        return if x < 0.0 && n % 2 != 0 { -value } else { value };
    }
    let (mut previous, mut value) = (1.0, 2.0 * alpha * x);
    for k in 1..n {
        let k = k as f64;
        let next = (2.0 * (k + alpha) / (k + 1.0)) * x * value
            - (((k - 1.0) + 2.0 * alpha) / (k + 1.0)) * previous;
        previous = value;
        value = next;
    }
    value
}

fn jacobi(n: i32, alpha: f64, beta: f64, x: f64) -> f64 {
    if n < 0 || !alpha.is_finite() || !beta.is_finite() || alpha <= -1.0 || beta <= -1.0 {
        return f64::NAN;
    }
    if n == 0 {
        return 1.0;
    }
    // Reflection preserves parity and uses the positive endpoint consistently.
    if x < 0.0 {
        return if n % 2 == 0 { 1.0 } else { -1.0 } * jacobi(n, beta, alpha, -x);
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }
    if x == 1.0 {
        return endpoint_binomial(n, alpha);
    }
    let sum = alpha + beta;
    let (mut previous, mut value) = (1.0, 0.5 * ((alpha - beta) + (sum + 2.0) * x));
    for k in 1..n {
        let k = k as f64;
        let t = 2.0 * k + sum;
        let denominator = 2.0 * (k + 1.0) * (k + sum + 1.0) * t;
        let next = ((t + 1.0) * (t * (t + 2.0) * x + (alpha - beta) * sum) * value
            - 2.0 * (k + alpha) * (k + beta) * (t + 2.0) * previous)
            / denominator;
        previous = value;
        value = next;
    }
    value
}

// Monic shifted-Jacobi recurrence. Cancelling the binomial normalization in
// each step avoids overflow of both numerator and denominator at large degree.
fn shifted_jacobi(n: i32, p: f64, q: f64, x: f64) -> f64 {
    if n < 0 || !p.is_finite() || !q.is_finite() || q <= 0.0 || p - q <= -1.0 {
        return f64::NAN;
    }
    if n == 0 {
        return 1.0;
    }
    if x.is_infinite() {
        return if n % 2 == 0 { f64::INFINITY } else { x };
    }
    let (mut previous, mut value) = (1.0, x - q / (p + 1.0));
    for k in 1..n {
        let k = k as f64;
        let t = 2.0 * k + p - 1.0;
        let a = x - 0.5 + (p - 2.0 * q + 1.0) * (p - 1.0) / (2.0 * t * (t + 2.0));
        let ratio = if k == 1.0 {
            1.0 / (p + 1.0)
        } else {
            k * (k + p - 1.0) / ((t - 1.0) * t)
        };
        let b = ((k + p - q) / t) * ((k + q - 1.0) / (t + 1.0)) * ratio;
        let next = a * value - b * previous;
        previous = value;
        value = next;
    }
    value
}

polynomial! {
    /// Legendre polynomial P_n(x), with P_{-n-1} = P_n.
    eval_legendre(n, x) { legendre(n, x) }
}
polynomial! {
    /// Shifted Legendre polynomial P_n(2*x-1).
    eval_sh_legendre(n, x) { legendre(n, 2.0*x-1.0) }
}
polynomial! {
    /// Generalized Laguerre polynomial L_n^(alpha)(x), for finite alpha > -1.
    /// Negative degrees return zero.
    eval_genlaguerre(n, alpha, x) { laguerre(n, alpha, x) }
}
polynomial! {
    /// Laguerre polynomial L_n(x). Negative degrees return zero.
    eval_laguerre(n, x) { laguerre(n, 0.0, x) }
}
polynomial! {
    /// Gegenbauer polynomial C_n^(alpha)(x), for finite alpha.
    /// Negative degrees and alpha = 0 return zero, following xsf.
    eval_gegenbauer(n, alpha, x) { gegenbauer(n, alpha, x) }
}
polynomial! {
    /// Jacobi polynomial P_n^(alpha,beta)(x), for n >= 0 and finite alpha,beta > -1.
    eval_jacobi(n, alpha, beta, x) { jacobi(n, alpha, beta, x) }
}
polynomial! {
    /// Shifted Jacobi G_n^(p,q)(x), for n >= 0, q > 0 and p-q > -1.
    /// Both parameters must be finite.
    eval_sh_jacobi(n, p, q, x) {
        shifted_jacobi(n, p, q, x)
    }
}
