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
    if n == 10 && scale == 1.0 && x.abs() <= 3.0 {
        // He_10(x) is even. Evaluating its degree-five polynomial in y=x²
        // avoids nine dependent Clenshaw steps. Center the polynomial in
        // three ranges to limit cancellation near the roots; retain Clenshaw
        // outside this interval, where the large polynomial terms grow.
        let y = x * x;
        if y < 1.0 {
            return ((((y - 45.0) * y + 630.0) * y - 3150.0) * y + 4725.0) * y - 945.0;
        }
        if y < 4.0 {
            let z = y - 2.0;
            return ((((z - 35.0) * z + 310.0) * z - 370.0) * z - 1675.0) * z + 257.0;
        }
        let z = y - 6.0;
        return ((((z - 15.0) * z - 90.0) * z + 630.0) * z + 2565.0) * z - 459.0;
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

// Precompute the first recurrence ratios once. The first 64 degrees are
// especially common, and using multiplication avoids division in each step.
// Higher degrees continue with the ordinary recurrence.
const LEGENDRE_RECURRENCE_RATIOS: [(f64, f64); 64] = {
    let mut ratios = [(0.0, 0.0); 64];
    let mut i = 0;
    while i < ratios.len() {
        let k = (i + 1) as f64;
        ratios[i] = ((2.0 * k + 1.0) / (k + 1.0), k / (k + 1.0));
        i += 1;
    }
    ratios
};

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
        let table_steps = ((n - 1) as usize).min(LEGENDRE_RECURRENCE_RATIOS.len());
        for &(a, b) in &LEGENDRE_RECURRENCE_RATIOS[..table_steps] {
            let next = a * x * value - b * previous;
            previous = value;
            value = next;
        }
        for k in (table_steps as i32 + 1)..n {
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
#[inline]
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
    if a < 0.9 {
        // Fill a fixed-length output slice instead of checking a Vec's spare
        // capacity on every degree. The signed recurrence already has the
        // required parity, so no per-degree sign correction is necessary.
        let mut table = vec![0.0; n + 1];
        table[0] = 1.0;
        table[1] = x;
        let (mut previous, mut value) = (1.0, x);
        let table_steps = (n - 1).min(LEGENDRE_RECURRENCE_RATIOS.len());
        for (slot, &(coefficient, prior_coefficient)) in table[2..table_steps + 2]
            .iter_mut()
            .zip(&LEGENDRE_RECURRENCE_RATIOS[..table_steps])
        {
            let next = coefficient * x * value - prior_coefficient * previous;
            previous = value;
            value = next;
            *slot = value;
        }
        for (offset, slot) in table[table_steps + 2..].iter_mut().enumerate() {
            let k = table_steps + 1 + offset;
            let kf = k as f64;
            let next = ((2.0 * kf + 1.0) * x * value - kf * previous) / (kf + 1.0);
            previous = value;
            value = next;
            *slot = value;
        }
        return table;
    }
    let mut out = Vec::with_capacity(n + 1);
    out.push(1.0);
    out.push(x);
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
    use super::{hermite, legendre, legendre_all};

    fn hermite_clenshaw(n: i32, x: f64, scale: f64) -> f64 {
        let (mut previous, mut current) = (0.0, 1.0);
        for k in (2..=n).rev() {
            let next = scale * x * current - scale * k as f64 * previous;
            previous = current;
            current = next;
        }
        scale * x * current - scale * previous
    }

    #[test]
    fn probabilists_hermite_ten_polynomial_matches_clenshaw_and_public_api() {
        let mut xs = vec![
            -3.01,
            -3.0,
            -3.0 + f64::EPSILON,
            -2.999999999,
            -2.000000001,
            -2.0,
            -1.999999999,
            -1.000000001,
            -1.0,
            -0.999999999,
            -0.0,
            0.0,
            0.999999999,
            1.0,
            1.000000001,
            1.999999999,
            2.0,
            2.000000001,
            2.999999999,
            3.0,
            3.01,
        ];
        // Every positive root in the optimized interval and its neighbors.
        for root in [0.484935707515497, 1.4659890943911582, 2.484325841638954] {
            for x in [root, root - 1e-12, root + 1e-12, root - 1e-9, root + 1e-9] {
                xs.extend([x, -x]);
            }
        }
        for i in 0..=20_000 {
            xs.push(-3.0 + i as f64 * (6.0 / 20_000.0));
        }
        for x in xs {
            let expected = hermite_clenshaw(10, x, 1.0);
            let actual = hermite(10, x, 1.0);
            let tolerance = 4e-12 * (1.0 + expected.abs());
            assert!(
                (actual - expected).abs() <= tolerance,
                "He_10({x}): actual={actual}, expected={expected}"
            );
            assert_eq!(actual, crate::eval_hermitenorm(10, x));
            assert_eq!(actual, hermite(10, -x, 1.0));
        }
        assert_eq!(crate::eval_hermitenorm(10, 0.0_f64), -945.0);
        assert_eq!(crate::eval_hermitenorm(10, -0.0_f64), -945.0);
        assert_eq!(crate::eval_hermitenorm(10, f64::INFINITY), f64::INFINITY);
        assert_eq!(
            crate::eval_hermitenorm(10, f64::NEG_INFINITY),
            f64::INFINITY
        );
        assert!(crate::eval_hermitenorm(10, f64::NAN).is_nan());
        assert!(crate::eval_hermitenorm(-10, 0.3_f64).is_nan());
        let x = 0.3_f32;
        let expected = hermite_clenshaw(10, f64::from(x), 1.0) as f32;
        assert!((crate::eval_hermitenorm(10, x) - expected).abs() <= f32::EPSILON * expected.abs());
        // Physicists' H_n and other probabilists' degrees remain on Clenshaw.
        for n in [1, 2, 3, 9, 11, 20] {
            for x in [-2.5, 0.2, 2.5] {
                assert_eq!(hermite(n, x, 2.0), hermite_clenshaw(n, x, 2.0));
                assert_eq!(hermite(n, x, 1.0), hermite_clenshaw(n, x, 1.0));
            }
        }
    }

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
            let table = legendre_all(4096, x);
            for n in [
                2, 3, 7, 10, 11, 32, 63, 64, 65, 66, 127, 128, 256, 512, 4096,
            ] {
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
    fn legendre_polynomial_roots_and_public_float_types() {
        // Positive zeros of P_10, as independently tabulated for
        // Gauss-Legendre quadrature. Include both signs and nearby points.
        for root in [
            0.14887433898163122_f64,
            0.4333953941292472,
            0.6794095682990244,
            0.8650633666889845,
            0.9739065285171717,
        ] {
            for x in [root - 1e-12, root, root + 1e-12] {
                for x in [x, -x] {
                    let value = crate::eval_legendre(10, x);
                    assert!(value.abs() < 5e-11, "P_10({x})={value}");
                    let shifted = crate::eval_sh_legendre(10, (x + 1.0) * 0.5);
                    assert!(shifted.abs() < 5e-11, "shifted P_10({x})={shifted}");
                    let x32 = x as f32;
                    let typed = crate::eval_legendre(10, x32);
                    let reference = crate::eval_legendre(10, f64::from(x32)) as f32;
                    assert!(
                        (typed - reference).abs() <= 4.0 * f32::EPSILON * (1.0 + reference.abs()),
                        "f32 P_10({x32})={typed}, f64 reference={reference}"
                    );
                }
            }
        }
        for n in [0, 1, 2, 10, 32, 63, 64, 65, 66, 4096] {
            for x in [
                -1.0,
                -1.0 + f64::EPSILON,
                -0.9,
                -0.5,
                0.0,
                0.5,
                0.9,
                1.0 - f64::EPSILON,
                1.0,
            ] {
                let actual = crate::eval_legendre(n, x);
                let x32 = x as f32;
                let typed = crate::eval_legendre(n, x32);
                let reference = crate::eval_legendre(n, f64::from(x32)) as f32;
                assert_eq!(
                    typed.to_bits(),
                    reference.to_bits(),
                    "f32 degree {n}, x={x}"
                );
                if x.abs() == 1.0 {
                    assert_eq!(actual, if x < 0.0 && n % 2 == 1 { -1.0 } else { 1.0 });
                }
            }
        }
        assert!(crate::eval_legendre(10, f64::NAN).is_nan());
        assert!(crate::eval_legendre(-1, 0.5_f32).is_nan());
    }

    #[test]
    fn legendre_table_matches_every_scalar_entry_across_recurrence_boundaries() {
        // The public table stops at 128. The private recurrence also supports
        // larger degrees, so cover the coefficient-table handoff and a long tail.
        for x in [
            f64::NEG_INFINITY,
            -10.0,
            -2.0,
            -1.0,
            -1.0 + f64::EPSILON,
            -0.90000000000001,
            -0.9,
            -0.89999999999999,
            -0.5,
            -1e-5,
            -0.0,
            0.0,
            1e-5,
            0.3,
            0.5,
            0.89999999999999,
            0.9,
            0.90000000000001,
            1.0 - f64::EPSILON,
            1.0,
            2.0,
            10.0,
            f64::INFINITY,
            f64::NAN,
        ] {
            let values = legendre_all(128, x);
            assert_eq!(values.len(), 129);
            for (degree, &value) in values.iter().enumerate() {
                let scalar = if x.is_nan() {
                    f64::NAN
                } else {
                    legendre(degree as i32, x)
                };
                if scalar.is_nan() {
                    assert!(value.is_nan(), "P_{degree}({x}) expected NaN");
                } else {
                    assert_eq!(value.to_bits(), scalar.to_bits(), "P_{degree}({x})");
                }
            }
        }
        for x in [-0.899, -0.3, 0.3, 0.899] {
            let values = legendre_all(4096, x);
            for &degree in &[0, 1, 2, 10, 63, 64, 65, 66, 127, 128, 256, 1024, 4096] {
                assert_eq!(
                    values[degree].to_bits(),
                    legendre(degree as i32, x).to_bits(),
                    "P_{degree}({x})"
                );
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

#[cfg(test)]
mod exceptional_orthogonal_coverage_tests {
    use super::{chebyshev, gegenbauer, hermite, jacobi, laguerre, legendre};

    fn close(actual: f64, expected: f64, relative: f64) {
        assert!(
            (actual - expected).abs() <= relative * expected.abs().max(f64::MIN_POSITIVE),
            "actual={actual:e}, independent reference={expected:e}"
        );
    }

    #[test]
    fn laguerre_generalized_parameter_and_normalization_overflow() {
        // L_1^(a)(x)=a+1-x and
        // L_2^(a)(x)=[x²-2(a+2)x+(a+1)(a+2)]/2.
        let alpha = 2.5;
        let x = 1.25;
        close(laguerre(1, alpha, x), alpha + 1.0 - x, 4e-15);
        let second = (x * x - 2.0 * (alpha + 2.0) * x + (alpha + 1.0) * (alpha + 2.0)) / 2.0;
        close(laguerre(2, alpha, x), second, 4e-15);
        close(
            laguerre(2, alpha, 0.0),
            (alpha + 1.0) * (alpha + 2.0) / 2.0,
            4e-15,
        );
        assert_eq!(laguerre(-2, alpha, x), 0.0);

        // The normalization binomial overflows, but L_2^(a)(a)=1-a/2
        // stays representable. This must use the unnormalized recurrence
        // rather than multiplying an infinite scale by a tiny polynomial.
        let huge = 1e308_f64;
        close(laguerre(2, huge, huge), -huge / 2.0, 8e-15);
        assert!(laguerre(2, -1.0, x).is_nan());
        assert!(laguerre(2, f64::INFINITY, x).is_nan());
        assert!(laguerre(2, f64::NAN, x).is_nan());
        assert!(super::eval_genlaguerre(2, -1.0_f64, x).is_err());
        assert_eq!(super::eval_genlaguerre(-2, alpha, x).unwrap(), 0.0);
    }

    #[test]
    fn gegenbauer_zero_parameter_truncation_and_endpoint_formulas() {
        // At alpha=0 this backend follows xsf's all-zero convention,
        // including the degree-zero case.
        assert_eq!(gegenbauer(0, 0.0, 0.3), 0.0);
        assert_eq!(gegenbauer(4, 0.0, 0.3), 0.0);
        assert_eq!(gegenbauer(-3, 1.5, 0.3), 0.0);
        assert_eq!(gegenbauer(0, 1.5, 0.3), 1.0);

        // (1-2xt+t²)^2 generates C_n^(-2); its t^4 coefficient
        // equals one and every coefficient with n>4 is zero.
        close(gegenbauer(4, -2.0, 0.3), 1.0, 4e-15);
        assert_eq!(gegenbauer(5, -2.0, 0.3), 0.0);
        // C_2^(-1)(x)=1, whereas C_3^(-1)(x)=0.
        assert_eq!(gegenbauer(2, -1.0, -0.4), 1.0);
        assert_eq!(gegenbauer(3, -1.0, -0.4), 0.0);

        // C_n^(a)(1)=(2a)_n/n!, and C_n^(a)(-1)=(-1)^n C_n^(a)(1).
        let endpoint = (3.0 * 4.0 * 5.0) / 6.0;
        assert_eq!(gegenbauer(3, 1.5, 1.0), endpoint);
        assert_eq!(gegenbauer(3, 1.5, -1.0), -endpoint);
        assert_eq!(gegenbauer(4, 1.5, -1.0), 15.0);
        assert!(gegenbauer(2, f64::INFINITY, 0.3).is_nan());
        assert_eq!(super::eval_gegenbauer(0, 0.0_f64, 0.3).unwrap(), 0.0);
    }

    #[test]
    fn jacobi_constant_endpoints_reflection_and_infinite_arguments() {
        let (alpha, beta) = (2.5, 0.5);
        assert_eq!(jacobi(0, alpha, beta, 0.3), 1.0);
        assert_eq!(jacobi(0, alpha, beta, f64::INFINITY), 1.0);
        // P_2^(5/2,1/2)(x)=(42x²+24x-3)/8.
        let x = 0.3_f64;
        close(
            jacobi(2, alpha, beta, x),
            (42.0 * x * x + 24.0 * x - 3.0) / 8.0,
            4e-14,
        );
        // P_2^(a,b)(1)=(a+1)(a+2)/2; left endpoint exchanges a,b.
        let at_one = (alpha + 1.0) * (alpha + 2.0) / 2.0;
        let at_minus_one = (beta + 1.0) * (beta + 2.0) / 2.0;
        assert_eq!(jacobi(2, alpha, beta, 1.0), at_one);
        assert_eq!(jacobi(2, alpha, beta, -1.0), at_minus_one);
        assert_eq!(
            jacobi(3, alpha, beta, -1.0),
            -(beta + 1.0) * (beta + 2.0) * (beta + 3.0) / 6.0
        );
        // The leading coefficient is positive for alpha,beta > -1.
        // Thus the high-degree polynomial diverges with degree parity.
        assert_eq!(jacobi(2, alpha, beta, f64::INFINITY), f64::INFINITY);
        assert_eq!(jacobi(3, alpha, beta, f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert!(jacobi(-1, alpha, beta, 0.3).is_nan());
        assert!(jacobi(2, -1.0, beta, 0.3).is_nan());
        assert!(jacobi(2, alpha, f64::INFINITY, 0.3).is_nan());
    }

    #[test]
    fn chebyshev_and_legendre_overflow_and_tiny_arguments() {
        // T_3(x)=4x^3-3x, U_3(x)=8x^3-4x.
        assert_eq!(chebyshev(3, 2.0, false), 26.0);
        assert_eq!(chebyshev(3, 2.0, true), 56.0);
        // T_n(2)=cosh(n acosh(2)) and U_n(2)=sinh((n+1)
        // acosh(2))/sinh(acosh(2)); for n=1200 both exceed f64::MAX.
        assert!(1200.0_f64 * 2.0_f64.acosh() > f64::MAX.ln());
        assert_eq!(chebyshev(1200, 2.0, false), f64::INFINITY);
        assert_eq!(chebyshev(1201, -2.0, false), f64::NEG_INFINITY);
        assert_eq!(chebyshev(1200, 2.0, true), f64::INFINITY);
        // For x>1, P_n(x)>0 and grows exponentially in n.
        assert_eq!(legendre(4096, 2.0), f64::INFINITY);
        assert_eq!(legendre(4095, -2.0), f64::NEG_INFINITY);
        // P_2(x)=(3x²-1)/2; P_3(x)=(5x³-3x)/2.
        for x in [1e-300_f64, -1e-300] {
            close(legendre(2, x), (3.0 * x * x - 1.0) / 2.0, 2e-15);
            let p3 = (5.0 * x * x - 3.0) * x / 2.0;
            close(legendre(3, x), p3, 2e-14);
        }
        assert_eq!(legendre(1, -0.0).to_bits(), (-0.0_f64).to_bits());
        assert!(hermite(-2, 0.3, 1.0).is_nan());
        // The private macro limits work before entering a recurrence.
        assert!(super::eval_legendre(1_000_001, 0.3_f64).is_err());
    }
}
