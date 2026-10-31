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

#[inline]
fn chebyshev(n: i32, x: f64, second: bool) -> f64 {
    if n == 10 && x.abs() < 1.0 {
        // For the common even degree, reflection does not change the value.
        // Apply T_10 = 2*T_5² - 1 and U_10 = 2*T_5*U_5 - 1 directly.
        let y = x * x;
        let t5 = x * ((16.0 * y - 20.0) * y + 5.0);
        let partner = if second {
            x * ((32.0 * y - 32.0) * y + 6.0)
        } else {
            t5
        };
        return 2.0 * t5 * partner - 1.0;
    }
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
    if n >= 64 && x < 1.0 {
        // Binary doubling computes (T_k(x), U_{k-1}(x)) in O(log n).
        // Keeping both polynomials avoids division by 1-x² near endpoints.
        // The short recurrence below retains its low overhead for small n.
        let target = if second { n + 1 } else { n } as u32;
        let mut t = 1.0;
        let mut u = 0.0;
        let mut bit = 1 << (31 - target.leading_zeros());
        let delta = (x - 1.0) * (x + 1.0);
        while bit != 0 {
            // (T_{2k}, U_{2k-1}) from (T_k, U_{k-1}).
            let doubled_t = 2.0 * t * t - 1.0;
            let doubled_u = 2.0 * t * u;
            if target & bit != 0 {
                // T_{k+1}=x T_k+(x²-1) U_{k-1}; U_k=T_k+x U_{k-1}.
                t = x * doubled_t + delta * doubled_u;
                u = doubled_t + x * doubled_u;
            } else {
                t = doubled_t;
                u = doubled_u;
            }
            bit >>= 1;
        }
        return sign * parity * if second { u } else { t };
    }
    let (mut previous, mut current) = (1.0, if second { 2.0 * x } else { x });
    let twice = 2.0 * x;
    if x < 1.0 {
        // |T_n|, |U_n| <= n + 1 on [-1, 1], so the recurrence cannot overflow.
        for _ in 1..n {
            (previous, current) = (current, twice * current - previous);
        }
    } else {
        for _ in 1..n {
            (previous, current) = (current, twice * current - previous);
            // For x > 1 the magnitudes grow monotonically; the final sign is known.
            if current.is_infinite() {
                break;
            }
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
    #[inline]
    eval_chebyt(n, x) { chebyshev(n, x, false) }
}
polynomial! {
    /// Chebyshev polynomial U_n(x), with U_{-1} = 0 and U_{-n-2} = -U_n.
    #[inline]
    eval_chebyu(n, x) { chebyshev(n, x, true) }
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
    if x < 0.9 {
        // The ordinary three-term recurrence is stable well inside [-1, 1].
        // It is substantially cheaper than the difference recurrence below,
        // which is retained near x=1 to avoid cancellation of P_n and P_{n-1}.
        let (mut previous, mut value) = (1.0, x);
        for k in 1..n {
            let k = k as f64;
            let next = ((2.0 * k + 1.0) * x * value - k * previous) / (k + 1.0);
            previous = value;
            value = next;
        }
        return sign * value;
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

/// P_0(x), ..., P_n(x) in one pass of the recurrence used by [`legendre`];
/// each entry is bit-identical to the corresponding single-degree value.
pub(crate) fn legendre_all(n: usize, x: f64) -> Vec<f64> {
    let per_degree = |k: usize| {
        if x.is_nan() {
            f64::NAN
        } else {
            legendre(k as i32, x)
        }
    };
    let a = x.abs();
    if x.is_nan() || a < 1e-5 || a == 1.0 || a.is_infinite() || n < 2 {
        return (0..=n).map(per_degree).collect();
    }
    let mut out = Vec::with_capacity(n + 1);
    out.push(1.0);
    out.push(x);
    if a < 0.9 {
        let (mut previous, mut value) = (1.0, a);
        for k in 1..n {
            let kf = k as f64;
            let next = ((2.0 * kf + 1.0) * a * value - kf * previous) / (kf + 1.0);
            previous = value;
            value = next;
            let degree = k + 1;
            out.push(if x < 0.0 && degree % 2 != 0 {
                -value
            } else {
                value
            });
        }
        return out;
    }
    let (mut difference, mut value) = (a - 1.0, a);
    for k in 1..n {
        if !value.is_infinite() {
            let kf = k as f64;
            difference = ((2.0 * kf + 1.0) * (a - 1.0) * value + kf * difference) / (kf + 1.0);
            value += difference;
        }
        let degree = k + 1;
        let sign = if x < 0.0 && degree % 2 != 0 {
            -1.0
        } else {
            1.0
        };
        out.push(sign * value);
    }
    out
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
    if alpha == 0.0 {
        // The normalization is exactly one for L_n = L_n^(0).
        // Avoid recomputing its n-factor product on every scalar call.
        if x == 0.0 {
            return 1.0;
        }
        let mut difference = -x;
        let mut value = 1.0 + difference;
        for k in 1..n {
            let k = k as f64;
            difference = -x / (k + 1.0) * value + k / (k + 1.0) * difference;
            value += difference;
        }
        return value;
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

polynomial! {
    /// Legendre polynomial P_n(x), with P_{-n-1} = P_n.
    eval_legendre(n, x) { legendre(n, x) }
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

#[cfg(test)]
mod chebyshev_fast_tests {
    use super::chebyshev;

    fn recurrence(n: i32, x: f64, second: bool) -> f64 {
        let (mut previous, mut value) = (1.0, if second { 2.0 * x } else { x });
        for _ in 1..n {
            (previous, value) = (value, 2.0 * x * value - previous);
        }
        value
    }

    #[test]
    fn degree_ten_factorization_matches_recurrence_and_public_wrappers() {
        // Include nearby roots, exact endpoints, their adjacent floating-point
        // numbers, and arguments outside the polynomial's orthogonality range.
        let mut values = vec![
            -2.0,
            -1.0,
            -1.0 + f64::EPSILON,
            -0.999999999999,
            -0.95,
            -0.5,
            -0.0,
            0.0,
            0.5,
            0.95,
            0.999999999999,
            1.0 - f64::EPSILON,
            1.0,
            2.0,
        ];
        for k in 1..=10 {
            let root = (k as f64 * std::f64::consts::PI / 11.0).cos();
            values.extend([root, root - 1e-12, root + 1e-12]);
        }
        for k in 0..10 {
            let root = ((k as f64 + 0.5) * std::f64::consts::PI / 10.0).cos();
            values.extend([root, root - 1e-12, root + 1e-12]);
        }
        for i in 0..=1000 {
            values.push(i as f64 / 500.0 - 1.0);
        }
        for x in values {
            for second in [false, true] {
                let expected = recurrence(10, x, second);
                let value = chebyshev(10, x, second);
                let tolerance = 2e-14 * (1.0 + expected.abs());
                assert!(
                    (value - expected).abs() <= tolerance,
                    "x={x:?}, second={second}: value={value}, expected={expected}"
                );
                // Private kernel extends negative degrees; public wrappers
                // retain their documented restriction to nonnegative degrees.
                let reflected = chebyshev(if second { -12 } else { -10 }, x, second);
                let reflection = if second { -value } else { value };
                assert!((reflected - reflection).abs() <= tolerance);
                let wrapper = if second {
                    crate::eval_chebyu(10, x)
                } else {
                    crate::eval_chebyt(10, x)
                };
                assert!((wrapper - expected).abs() <= tolerance);
                let shifted = if second {
                    crate::eval_sh_chebyu(10, (x + 1.0) / 2.0)
                } else {
                    crate::eval_sh_chebyt(10, (x + 1.0) / 2.0)
                };
                assert!((shifted - expected).abs() <= 5e-13 * (1.0 + expected.abs()));
                let scaled = if second {
                    crate::eval_chebys(10, 2.0 * x)
                } else {
                    crate::eval_chebyc(10, 2.0 * x) * 0.5
                };
                assert!((scaled - expected).abs() <= tolerance);
            }
        }
        assert!(crate::eval_chebyt(-10, 0.5_f64).is_nan());
        assert!(crate::eval_chebyu(-12, 0.5_f32).is_nan());
        let typed = crate::eval_chebyt(10, 0.3_f32);
        let reference = recurrence(10, f64::from(0.3_f32), false) as f32;
        assert!((typed - reference).abs() <= 8.0 * f32::EPSILON);
        assert_eq!(chebyshev(-1, 0.5, true), 0.0);
    }

    #[test]
    fn doubling_matches_independent_three_term_recurrence() {
        for x in [-0.999999, -0.9, -0.3, 0.0, 0.3, 0.9, 0.999999] {
            for second in [false, true] {
                let (mut previous, mut current) = (1.0_f64, if second { 2.0 * x } else { x });
                for n in 1..=4096 {
                    if n >= 64 && (n == 64 || n == 65 || n == 127 || n == 1024 || n == 4096) {
                        let actual = chebyshev(n, x, second);
                        let tolerance = 2e-8 * (1.0 + current.abs());
                        assert!(
                            (actual - current).abs() <= tolerance,
                            "n={n}, x={x}, second={second}, actual={actual}, reference={current}"
                        );
                    }
                    (previous, current) = (current, 2.0 * x * current - previous);
                }
            }
        }
    }
}

#[cfg(test)]
mod recurrence_fast_tests {
    use super::{legendre, legendre_all};

    fn legendre_difference(n: i32, x: f64) -> f64 {
        let (mut difference, mut value) = (x - 1.0, x);
        for k in 1..n {
            let k = k as f64;
            difference = ((2.0 * k + 1.0) * (x - 1.0) * value + k * difference) / (k + 1.0);
            value += difference;
        }
        value
    }

    fn laguerre_three_term(n: i32, x: f64) -> f64 {
        if n == 0 {
            return 1.0;
        }
        let (mut previous, mut value) = (1.0, 1.0 - x);
        for k in 1..n {
            let k = k as f64;
            let next = ((2.0 * k + 1.0 - x) * value - k * previous) / (k + 1.0);
            previous = value;
            value = next;
        }
        value
    }

    #[test]
    fn legendre_interior_recurrence_matches_endpoint_stable_version() {
        // Both sides of the fast-path cutoff, small arguments, and parity.
        for x in [
            -0.9999999999_f64,
            -0.9,
            -0.8999999999,
            -0.65,
            -0.2,
            -1e-5,
            0.0,
            1e-5,
            0.2,
            0.65,
            0.8999999999,
            0.9,
            0.9999999999,
        ] {
            let a = x.abs();
            let table = legendre_all(128, x);
            for n in [2, 3, 7, 10, 11, 32, 64, 128] {
                let expected =
                    legendre_difference(n, a) * if x < 0.0 && n % 2 != 0 { -1.0 } else { 1.0 };
                let actual = legendre(n, x);
                let tolerance = 2e-12 * (1.0 + expected.abs());
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "P_{n}({x}): actual={actual}, expected={expected}"
                );
                assert_eq!(table[n as usize], actual, "P_{n}({x}) table differs");
            }
        }
    }

    #[test]
    fn classical_laguerre_matches_independent_recurrence() {
        for x in [
            -4.0, -1.0, -0.0, 0.0, 1e-10, 0.1, 0.5, 1.0, 2.0, 4.0, 9.5, 10.0,
        ] {
            for n in [0, 1, 2, 3, 10, 16, 32] {
                let expected = laguerre_three_term(n, x);
                let actual = crate::eval_laguerre(n, x);
                assert!(
                    (actual - expected).abs() <= 1e-11 * (1.0 + expected.abs()),
                    "L_{n}({x}): actual={actual}, expected={expected}"
                );
                assert_eq!(actual, crate::eval_genlaguerre(n, 0.0, x));
            }
        }
    }
}
