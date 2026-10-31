//! Elementary combinatorics, reductions, and derivative recurrences.

use crate::{Error, Result, complex::Complex};
use sci_core::Float;

/// Stable `logsumexp(x) = ln(Σᵢ exp(xᵢ))` over a one-dimensional slice.
///
/// ## Parameters and conventions
///
/// `values` are dimensionless real values; the logarithm is natural.
///
/// ## Returns
///
/// The scalar logarithm of the exponential sum in the input precision.
///
/// ## Numerical behavior
///
/// Maximum shifting bounds each exponential by one. Linear work in the slice
/// length, without a configurable work limit. Rounding and underflow of small
/// relative contributions remain; no precision promotion is performed.
///
/// ## References
///
/// [SciPy logsumexp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html).
///
/// ## Domain
///
/// Real inputs. Empty slices return −∞; NaN propagates, and any +∞ gives +∞.
/// Shifting by the largest element avoids overflow. No axis, weights, or signed
/// log-sum options are provided; compose operations on ordinary vectors.
///
/// ## Examples
///
/// ```
/// assert!((sci_special::logsumexp(&[1000.0,1000.0])-1000.0-2.0_f64.ln()).abs()<1e-12);
/// ```
pub fn logsumexp<T: Float>(values: &[T]) -> T {
    if values.iter().any(|x| x.is_nan()) {
        return T::nan();
    }
    let maximum = values.iter().copied().fold(T::neg_infinity(), T::max);
    if !maximum.is_finite() {
        return maximum;
    }
    maximum
        + values
            .iter()
            .map(|&x| (x - maximum).exp())
            .fold(T::zero(), |a, b| a + b)
            .ln()
}

/// Softmax `softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)` over a one-dimensional slice.
///
/// ## Parameters and conventions
///
/// `values` are dimensionless real logits; normalization covers the entire slice.
///
/// ## Returns
///
/// Probabilities in input order, with the same length and scalar precision.
///
/// ## Numerical behavior
///
/// Linear work and storage, without a configurable work limit. Maximum shifting
/// prevents exponential overflow. Small probabilities can underflow to zero;
/// the sum is one up to rounding in the selected precision.
///
/// ## References
///
/// [SciPy softmax](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html).
///
/// ## Domain
///
/// Finite real inputs; nonfinite normalization propagates NaNs. An empty slice
/// returns an empty vector. Maximum-shifting keeps the exponential bounded.
///
/// ## Examples
///
/// ```
/// assert_eq!(sci_special::softmax(&[1000.0,1000.0]),vec![0.5,0.5]);
/// ```
pub fn softmax<T: Float>(values: &[T]) -> Vec<T> {
    let maximum = values.iter().copied().fold(T::neg_infinity(), T::max);
    let exp: Vec<_> = values.iter().map(|&x| (x - maximum).exp()).collect();
    let sum = exp.iter().copied().fold(T::zero(), |a, b| a + b);
    exp.into_iter().map(|x| x / sum).collect()
}

/// Log softmax `log_softmax(x)ᵢ = xᵢ − ln(Σⱼ exp(xⱼ))`.
///
/// ## Parameters and conventions
///
/// `values` are dimensionless real logits; logarithms are natural and
/// normalization covers the entire slice.
///
/// ## Returns
///
/// Log probabilities in input order, with the same length and precision.
///
/// ## Numerical behavior
///
/// Linear work and storage, without a configurable work limit. Subtracts the
/// maximum before computing the log sum to preserve small log probabilities
/// at large common offsets. Rounding and cancellation remain type dependent.
///
/// ## References
///
/// [SciPy log_softmax](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_softmax.html).
///
/// ## Domain
///
/// Finite real inputs; nonfinite normalization propagates NaNs. Empty inputs
/// return an empty vector. This preserves small differences even at large offsets.
///
/// ## Examples
///
/// ```
/// assert!((sci_special::log_softmax(&[1000.0,1000.0])[0]+2.0_f64.ln()).abs()<1e-14);
/// ```
pub fn log_softmax<T: Float>(values: &[T]) -> Vec<T> {
    let maximum = values.iter().copied().fold(T::neg_infinity(), T::max);
    let shifted: Vec<_> = values.iter().map(|&x| x - maximum).collect();
    let logsum = logsumexp(&shifted);
    shifted.into_iter().map(|x| x - logsum).collect()
}

fn derivative(v: f64, x: f64, n: u32, function: fn(f64, f64) -> f64, phase: f64) -> f64 {
    if n > 1024 {
        return f64::NAN;
    }
    let mut weight = 2.0_f64.powi(-(n as i32));
    let mut sum = weight * function(v - f64::from(n), x);
    for i in 1..=n {
        weight *= phase * f64::from(n - i + 1) / f64::from(i);
        sum += weight * function(v - f64::from(n) + 2.0 * f64::from(i), x);
    }
    sum
}

fn bernoulli_numbers(n: usize) -> Vec<f64> {
    let mut values = vec![1.0];
    for m in 1..=n {
        if m > 1 && m % 2 == 1 {
            values.push(0.0);
            continue;
        }
        if m == 1 {
            values.push(-0.5);
            continue;
        }
        // ζ(2k) = (−1)^(k+1) B_(2k) (2π)^(2k)/(2(2k)!).
        let magnitude = (2.0_f64.ln() + super::gammaln(m as f64 + 1.0)
            - (m as f64) * (2.0 * std::f64::consts::PI).ln())
        .exp()
            * (super::zetac(m as f64) + 1.0);
        values.push(if (m / 2) % 2 == 0 {
            -magnitude
        } else {
            magnitude
        });
    }
    values
}

/// Order-n derivative of jv(v,x) with respect to x, computed by the adjacent-order recurrence.
///
/// ## Parameters and domain
///
/// Real order v, argument in the corresponding Bessel domain, and n≤1024. High-order derivatives may suffer cancellation.
/// Results use IEEE NaN for invalid inputs and infinities for overflow or poles.
///
/// ## Examples
///
/// ```
/// let value = sci_special::jvp(0.5,2.0,2);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy jvp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jvp.html).
pub fn jvp(v: f64, x: f64, n: u32) -> f64 {
    derivative(v, x, n, super::jv, -1.0)
}

/// Order-n derivative of yv(v,x) with respect to x, computed by the adjacent-order recurrence.
///
/// ## Parameters and domain
///
/// Real order v, argument in the corresponding Bessel domain, and n≤1024. High-order derivatives may suffer cancellation.
/// Results use IEEE NaN for invalid inputs and infinities for overflow or poles.
///
/// ## Examples
///
/// ```
/// let value = sci_special::yvp(0.5,2.0,2);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy yvp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.yvp.html).
pub fn yvp(v: f64, x: f64, n: u32) -> f64 {
    derivative(v, x, n, super::yv, -1.0)
}

/// Order-n derivative of iv(v,x) with respect to x, computed by the adjacent-order recurrence.
///
/// ## Parameters and domain
///
/// Real order v, argument in the corresponding Bessel domain, and n≤1024. High-order derivatives may suffer cancellation.
/// Results use IEEE NaN for invalid inputs and infinities for overflow or poles.
///
/// ## Examples
///
/// ```
/// let value = sci_special::ivp(0.5,2.0,2);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ivp.html).
pub fn ivp(v: f64, x: f64, n: u32) -> f64 {
    derivative(v, x, n, super::iv, 1.0)
}

/// Order-n derivative of kv(v,x) with respect to x, computed by the adjacent-order recurrence.
///
/// ## Parameters and domain
///
/// Real order v, argument in the corresponding Bessel domain, and n≤1024. High-order derivatives may suffer cancellation.
/// Results use IEEE NaN for invalid inputs and infinities for overflow or poles.
///
/// ## Examples
///
/// ```
/// let value = sci_special::kvp(0.5,2.0,2);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy kvp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kvp.html).
pub fn kvp(v: f64, x: f64, n: u32) -> f64 {
    derivative(v, x, n, super::kv, 1.0) * if n.is_multiple_of(2) { 1.0 } else { -1.0 }
}

/// Order-n derivative of a Hankel function J_v(x) ± iY_v(x).
///
/// ## Parameters and domain
///
/// Real order v, positive real x, and n≤1024.
/// Results use IEEE NaN for invalid inputs and infinities for overflow or poles.
///
/// ## Examples
///
/// ```
/// let value = sci_special::h1vp(0.5,2.0,2);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy h1vp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.h1vp.html).
pub fn h1vp(v: f64, x: f64, n: u32) -> Complex {
    Complex {
        re: jvp(v, x, n),
        im: 1.0 * yvp(v, x, n),
    }
}

/// Order-n derivative of a Hankel function J_v(x) ± iY_v(x).
///
/// ## Parameters and domain
///
/// Real order v, positive real x, and n≤1024.
/// Results use IEEE NaN for invalid inputs and infinities for overflow or poles.
///
/// ## Examples
///
/// ```
/// let value = sci_special::h2vp(0.5,2.0,2);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy h2vp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.h2vp.html).
pub fn h2vp(v: f64, x: f64, n: u32) -> Complex {
    Complex {
        re: jvp(v, x, n),
        im: -yvp(v, x, n),
    }
}

/// Spherical Bessel jn and optionally its first derivative.
///
/// ## Parameters and domain
///
/// Integer degree 0≤n≤4096; real x in the corresponding Bessel domain. The modified second kind uses SciPy's π/(2x) normalization.
/// Results use IEEE NaN for invalid inputs and infinities for overflow or poles.
///
/// ## Examples
///
/// ```
/// let value = sci_special::spherical_jn(2,0.5_f64,false);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy spherical_jn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.spherical_jn.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn spherical_jn<T: sci_core::Float>(n: i64, x: T, derivative: bool) -> T {
    if !(0..=4096).contains(&n) {
        T::nan()
    } else if derivative {
        crate::backend::sph_bessel_j_jac(n as i32, x).unwrap_or(T::nan())
    } else {
        crate::backend::sph_bessel_j(n as i32, x).unwrap_or(T::nan())
    }
}

/// Spherical Bessel yn and optionally its first derivative.
///
/// ## Parameters and domain
///
/// Integer degree 0≤n≤4096; real x in the corresponding Bessel domain. The modified second kind uses SciPy's π/(2x) normalization.
/// Results use IEEE NaN for invalid inputs and infinities for overflow or poles.
///
/// ## Examples
///
/// ```
/// let value = sci_special::spherical_yn(2,0.5_f64,false);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy spherical_yn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.spherical_yn.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn spherical_yn<T: sci_core::Float>(n: i64, x: T, derivative: bool) -> T {
    if !(0..=4096).contains(&n) {
        T::nan()
    } else if derivative {
        crate::backend::sph_bessel_y_jac(n as i32, x).unwrap_or(T::nan())
    } else {
        crate::backend::sph_bessel_y(n as i32, x).unwrap_or(T::nan())
    }
}

/// Spherical Bessel in and optionally its first derivative.
///
/// ## Parameters and domain
///
/// Integer degree 0≤n≤4096; real x in the corresponding Bessel domain. The modified second kind uses SciPy's π/(2x) normalization.
/// Results use IEEE NaN for invalid inputs and infinities for overflow or poles.
///
/// ## Examples
///
/// ```
/// let value = sci_special::spherical_in(2,0.5_f64,false);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy spherical_in](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.spherical_in.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn spherical_in<T: sci_core::Float>(n: i64, x: T, derivative: bool) -> T {
    if !(0..=4096).contains(&n) {
        T::nan()
    } else if derivative {
        crate::backend::sph_bessel_i_jac(n as i32, x).unwrap_or(T::nan())
    } else {
        crate::backend::sph_bessel_i(n as i32, x).unwrap_or(T::nan())
    }
}

/// Spherical Bessel kn and optionally its first derivative.
///
/// ## Parameters and domain
///
/// Integer degree 0≤n≤4096; real x in the corresponding Bessel domain. The modified second kind uses SciPy's π/(2x) normalization.
/// Results use IEEE NaN for invalid inputs and infinities for overflow or poles.
///
/// ## Examples
///
/// ```
/// let value = sci_special::spherical_kn(2,0.5_f64,false);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy spherical_kn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.spherical_kn.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn spherical_kn<T: sci_core::Float>(n: i64, x: T, derivative: bool) -> T {
    if !(0..=4096).contains(&n) {
        T::nan()
    } else if derivative {
        crate::backend::sph_bessel_k_jac(n as i32, x).unwrap_or(T::nan())
    } else {
        crate::backend::sph_bessel_k(n as i32, x).unwrap_or(T::nan())
    }
}

/// Scaled or shifted orthogonal polynomial, using its standard three-term recurrence.
///
/// ## Parameters and domain
///
/// Integer degree 0≤n≤4096 and real x.
/// Results use IEEE NaN for invalid inputs and infinities for overflow or poles.
///
/// ## Examples
///
/// ```
/// let value = sci_special::eval_chebyc(3,0.3);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy eval_chebyc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_chebyc.html).
#[inline]
pub fn eval_chebyc(n: i32, x: f64) -> f64 {
    2.0 * super::eval_chebyt(n, x / 2.0)
}

/// Scaled or shifted orthogonal polynomial, using its standard three-term recurrence.
///
/// ## Parameters and domain
///
/// Integer degree 0≤n≤4096 and real x.
/// Results use IEEE NaN for invalid inputs and infinities for overflow or poles.
///
/// ## Examples
///
/// ```
/// let value = sci_special::eval_chebys(3,0.3);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy eval_chebys](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_chebys.html).
#[inline]
pub fn eval_chebys(n: i32, x: f64) -> f64 {
    super::eval_chebyu(n, x / 2.0)
}

/// Scaled or shifted orthogonal polynomial, using its standard three-term recurrence.
///
/// ## Parameters and domain
///
/// Integer degree 0≤n≤4096 and real x.
/// Results use IEEE NaN for invalid inputs and infinities for overflow or poles.
///
/// ## Examples
///
/// ```
/// let value = sci_special::eval_sh_chebyt(3,0.3);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy eval_sh_chebyt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_sh_chebyt.html).
#[inline]
pub fn eval_sh_chebyt(n: i32, x: f64) -> f64 {
    super::eval_chebyt(n, 2.0 * x - 1.0)
}

/// Scaled or shifted orthogonal polynomial, using its standard three-term recurrence.
///
/// ## Parameters and domain
///
/// Integer degree 0≤n≤4096 and real x.
/// Results use IEEE NaN for invalid inputs and infinities for overflow or poles.
///
/// ## Examples
///
/// ```
/// let value = sci_special::eval_sh_chebyu(3,0.3);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy eval_sh_chebyu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_sh_chebyu.html).
#[inline]
pub fn eval_sh_chebyu(n: i32, x: f64) -> f64 {
    super::eval_chebyu(n, 2.0 * x - 1.0)
}

/// Scaled or shifted orthogonal polynomial, using its standard three-term recurrence.
///
/// ## Parameters and domain
///
/// Integer degree 0≤n≤4096 and real x.
/// Results use IEEE NaN for invalid inputs and infinities for overflow or poles.
///
/// ## Examples
///
/// ```
/// let value = sci_special::eval_sh_legendre(3,0.3);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy eval_sh_legendre](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_sh_legendre.html).
pub fn eval_sh_legendre(n: i32, x: f64) -> f64 {
    super::eval_legendre(n, 2.0 * x - 1.0)
}

/// Shifted Jacobi G_n^(p,q)(x) = n! Γ(n+p)/Γ(2n+p) P_n^(p−q,q−1)(2x−1).
///
/// ## Parameters and domain
///
/// Integer 0≤n≤4096, q>0, p−q>−1, and real x.
/// Results use IEEE NaN for invalid inputs and infinities for overflow or poles.
///
/// ## Examples
///
/// ```
/// let value = sci_special::eval_sh_jacobi(3,2.5,1.5,0.3);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy eval_sh_jacobi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_sh_jacobi.html).
pub fn eval_sh_jacobi(n: i32, p: f64, q: f64, x: f64) -> f64 {
    let nf = f64::from(n);
    (super::gammaln(nf + 1.0) + super::gammaln(nf + p) - super::gammaln(2.0 * nf + p)).exp()
        * super::eval_jacobi(n, p - q, q - 1.0, 2.0 * x - 1.0)
}

/// Periodic sinc sin(nx/2)/(n sin(x/2)), with the continuous limiting value at multiples of 2π.
///
/// ## Parameters and domain
///
/// Finite real x in radians and positive integer n.
/// Results use IEEE NaN for invalid inputs and infinities for overflow or poles.
///
/// ## Examples
///
/// ```
/// let value = sci_special::diric(0.3,3);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy diric](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.diric.html).
pub fn diric(x: f64, n: u32) -> f64 {
    if n == 0 || !x.is_finite() {
        return f64::NAN;
    }
    let r = x.rem_euclid(2.0 * std::f64::consts::PI);
    if r == 0.0 {
        let k = (x / (2.0 * std::f64::consts::PI)).round();
        if n.is_multiple_of(2) && k.rem_euclid(2.0) != 0.0 {
            -1.0
        } else {
            1.0
        }
    } else {
        let nf = f64::from(n);
        (nf * x / 2.0).sin() / (nf * (x / 2.0).sin())
    }
}

/// Factorial Γ(n+1), with zero for negative n and gamma extension for real nonnegative n.
///
/// ## Parameters and domain
///
/// Nonnegative real n; overflow gives +∞. Exact arbitrary-precision integer arithmetic is not provided.
/// Results use IEEE NaN for invalid inputs and infinities for overflow or poles.
///
/// ## Examples
///
/// ```
/// let value = sci_special::factorial(5.0);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy factorial](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.factorial.html).
pub fn factorial(n: f64) -> f64 {
    if n < 0.0 { 0.0 } else { super::gamma(n + 1.0) }
}

/// Double factorial n(n−2)(n−4)… in f64 arithmetic.
///
/// ## Parameters and domain
///
/// Integer n; negative n gives zero in SciPy's default convention. Values above the overflow threshold return +∞.
/// Results use IEEE NaN for invalid inputs and infinities for overflow or poles.
///
/// ## Examples
///
/// ```
/// let value = sci_special::factorial2(7);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy factorial2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.factorial2.html).
pub fn factorial2(n: i32) -> f64 {
    if n < 0 {
        0.0
    } else if n > 300 {
        f64::INFINITY
    } else {
        (1..=n).rev().step_by(2).map(f64::from).product()
    }
}

/// Multifactorial n(n−k)(n−2k)… in f64 arithmetic.
///
/// ## Parameters and domain
///
/// Positive integer k; negative n gives zero and n=0 gives one. n≤100000 to bound work.
/// Results use IEEE NaN for invalid inputs and infinities for overflow or poles.
///
/// ## Examples
///
/// ```
/// let value = sci_special::factorialk(8,3);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy factorialk](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.factorialk.html).
pub fn factorialk(n: i32, k: u32) -> f64 {
    if k == 0 || n > 100_000 {
        f64::NAN
    } else if n < 0 {
        0.0
    } else {
        let mut out = 1.0;
        let mut value = i64::from(n);
        while value > 0 {
            out *= value as f64;
            if out.is_infinite() {
                break;
            }
            value -= i64::from(k);
        }
        out
    }
}

/// Combinations n choose k in f64 arithmetic.
///
/// ## Parameters and domain
///
/// n,k≥0 and k≤n; outside gives zero. No exact integer or repetition option is provided.
/// Results use IEEE NaN for invalid inputs and infinities for overflow or poles.
///
/// ## Examples
///
/// ```
/// let value = sci_special::comb(5.0_f64,2.0);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy comb](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.comb.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn comb<T: sci_core::Float>(n: T, k: T) -> T {
    if n < 0.0 || k < 0.0 || k > n {
        0.0
    } else {
        crate::backend::binom(n, k).unwrap_or(T::nan())
    }
}

/// Permutations n(n−1)…(n−k+1), in f64 arithmetic.
///
/// ## Parameters and domain
///
/// n≥k≥0; outside gives zero. k≤100000 to bound work.
/// Results use IEEE NaN for invalid inputs and infinities for overflow or poles.
///
/// ## Examples
///
/// ```
/// let value = sci_special::perm(5.0,2);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy perm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.perm.html).
pub fn perm(n: f64, k: u32) -> f64 {
    if k > 100_000 {
        f64::NAN
    } else if n < f64::from(k) || n < 0.0 {
        0.0
    } else {
        (0..k).fold(1.0, |acc, i| acc * (n - f64::from(i)))
    }
}

/// Pochhammer rising factorial (a)_m = Γ(a+m)/Γ(a), including removable-pole limits.
///
/// ## Parameters and domain
///
/// Real a and m; noninteger continuations follow SciPy's real convention.
/// Results use IEEE NaN for invalid inputs and infinities for overflow or poles.
///
/// ## Examples
///
/// ```
/// let value = sci_special::poch(2.5_f64,3.0);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy poch](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.poch.html).
///
/// ## Numerical behavior
///
/// Accepts the shared `sci_core::Float` type. Specialized kernels evaluate in
/// binary64 and convert the result to `T`; this does not provide extra precision.
/// Domain or numerical backend errors return NaN. Iterations are bounded by
/// the private algorithm; correct rounding is not guaranteed.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn poch<T: sci_core::Float>(a: T, m: T) -> T {
    crate::backend::poch(a, m).unwrap_or(T::nan())
}

/// Stirling number of the second kind S(n,k), counting partitions of n elements into k nonempty subsets.
///
/// ## Parameters and domain
///
/// n,k≤4096; k>n gives zero. Values use f64 and may overflow; exact integer mode is omitted.
/// Results use IEEE NaN for invalid inputs and infinities for overflow or poles.
///
/// ## Examples
///
/// ```
/// let value = sci_special::stirling2(5,2);
/// let _ = value;
/// ```
///
/// ## References
///
/// - [SciPy stirling2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.stirling2.html).
pub fn stirling2(n: u32, k: u32) -> f64 {
    if n > 4096 || k > 4096 {
        return f64::NAN;
    }
    if k > n {
        return 0.0;
    }
    let mut row = vec![0.0; k as usize + 1];
    row[0] = 1.0;
    for i in 1..=n {
        for j in (1..=k.min(i)).rev() {
            row[j as usize] = f64::from(j) * row[j as usize] + row[j as usize - 1];
        }
        row[0] = 0.0;
    }
    row[k as usize]
}

/// Bernoulli numbers B₀ through B_n, with B₁=−1/2.
///
/// ## Parameters and domain
///
/// n≤256. Even entries use the gamma–zeta relation; odd entries above one are exactly zero.
/// Invalid sizes or parameters return [`Error::InvalidInput`]. Numerical overflow returns [`Error::NonFiniteEvaluation`] where validated.
///
/// ## Examples
///
/// ```
/// let value = sci_special::bernoulli(8)?;
/// let _ = value;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy bernoulli](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.bernoulli.html).
pub fn bernoulli(n: usize) -> Result<Vec<f64>> {
    if n > 256 {
        return Err(Error::InvalidInput("Bernoulli order must not exceed 256"));
    }
    let values = bernoulli_numbers(n);
    if values.iter().any(|x| !x.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok(values)
}

/// Euler numbers E₀ through E_n from secant/hyperbolic-secant recurrence, E₂=−1.
///
/// ## Parameters and domain
///
/// n≤256; odd entries are zero. Overflow returns an error.
/// Invalid sizes or parameters return [`Error::InvalidInput`]. Numerical overflow returns [`Error::NonFiniteEvaluation`] where validated.
///
/// ## Examples
///
/// ```
/// let value = sci_special::euler(8)?;
/// let _ = value;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy euler](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.euler.html).
pub fn euler(n: usize) -> Result<Vec<f64>> {
    if n > 256 {
        return Err(Error::InvalidInput("Euler order must not exceed 256"));
    }
    let mut values = vec![0.0; n + 1];
    values[0] = 1.0;
    for m in (2..=n).step_by(2) {
        // All binomial arguments are small nonnegative integers here. Avoid
        // evaluating a general gamma-based binomial coefficient for each term:
        // C(m, j+2) = C(m, j) * (m-j)(m-j-1) / ((j+1)(j+2)).
        let mut coefficient = 1.0;
        let mut sum = 0.0;
        for j in (0..m).step_by(2) {
            sum += coefficient * values[j];
            coefficient *= ((m - j) * (m - j - 1)) as f64 / ((j + 1) * (j + 2)) as f64;
        }
        values[m] = -sum;
    }
    if values.iter().any(|x| !x.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok(values)
}

/// Parabolic cylinder values and derivatives for orders frac(v) through frac(v)+max(trunc(v),1).
///
/// ## Parameters and domain
///
/// Finite real v≤256 and finite real x. For negative v, only the two lowest fractional orders are returned, following SciPy's sequence convention.
/// Invalid sizes or parameters return [`Error::InvalidInput`]. Numerical overflow returns [`Error::NonFiniteEvaluation`] where validated.
///
/// ## Examples
///
/// ```
/// let value = sci_special::pbdv_seq(2.5,0.3)?;
/// let _ = value;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy pbdv_seq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pbdv_seq.html).
pub fn pbdv_seq(v: f64, x: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    if !v.is_finite() || v > 256.0 || !x.is_finite() {
        return Err(Error::InvalidInput(
            "invalid parabolic cylinder sequence arguments",
        ));
    }
    let base = v - v.trunc();
    Ok((0..=(v.trunc().max(1.0) as usize))
        .map(|i| super::pbdv(base + i as f64, x))
        .unzip())
}

/// Parabolic cylinder values and derivatives for orders frac(v) through frac(v)+max(trunc(v),1).
///
/// ## Parameters and domain
///
/// Finite real v≤256 and finite real x. For negative v, only the two lowest fractional orders are returned, following SciPy's sequence convention.
/// Invalid sizes or parameters return [`Error::InvalidInput`]. Numerical overflow returns [`Error::NonFiniteEvaluation`] where validated.
///
/// ## Examples
///
/// ```
/// let value = sci_special::pbvv_seq(2.5,0.3)?;
/// let _ = value;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy pbvv_seq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pbvv_seq.html).
pub fn pbvv_seq(v: f64, x: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    if !v.is_finite() || v > 256.0 || !x.is_finite() {
        return Err(Error::InvalidInput(
            "invalid parabolic cylinder sequence arguments",
        ));
    }
    let base = v - v.trunc();
    Ok((0..=(v.trunc().max(1.0) as usize))
        .map(|i| super::pbvv(base + i as f64, x))
        .unzip())
}

/// Integer-order parabolic cylinder values and coordinate derivatives.
///
/// ## Definition
///
/// `D_k(x) = exp(-x²/4) He_k(x)`, where `He_k` is the probabilists' Hermite
/// polynomial. The derivative satisfies `D_k′(x) = x D_k(x)/2 - D_(k+1)(x)`.
///
/// ## Parameters and conventions
///
/// `n <= 256` is the largest requested integer order; `x` is a dimensionless
/// real coordinate. SciPy's complex-input overload is not provided.
///
/// ## Domain and special values
///
/// All finite real coordinates are supported. Nonfinite coordinates produce
/// NaN entries. Intermediate overflow and underflow follow IEEE arithmetic.
///
/// ## Returns
///
/// `(values, derivatives)`, each in ascending order `0..=max(n, 1)`, with
/// `max(n, 1) + 1` entries, retaining the two-entry convention at `n = 0`.
///
/// ## Errors
///
/// Returns [`Error::InvalidInput`] when `n > 256`.
///
/// ## Numerical behavior
///
/// Uses one bounded forward recurrence shared with the scalar cylinder kernel.
/// The initial `exp(-x²/4)` can underflow for large coordinates; subsequent
/// recurrence steps do not recover values lost to that initial underflow.
///
/// ## Examples
///
/// ```
/// let (values, derivatives) = sci_special::pbdn_seq(3, 0.3)?;
/// assert_eq!(values.len(), 4);
/// assert_eq!(derivatives.len(), 4);
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [NIST DLMF §12.7](https://dlmf.nist.gov/12.7).
/// - [SciPy pbdn_seq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pbdn_seq.html).
pub fn pbdn_seq(n: usize, x: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    if n > 256 {
        return Err(Error::InvalidInput("parabolic cylinder degree exceeds 256"));
    }
    Ok(crate::backend::legacy::pbdn_sequence(n, x))
}

/// Riccati–Bessel values x*j_i(x) and derivatives through order n.
///
/// ## Parameters and domain
///
/// n≤4096 and positive finite real x.
/// Invalid sizes or parameters return [`Error::InvalidInput`]. Numerical overflow returns [`Error::NonFiniteEvaluation`] where validated.
///
/// ## Examples
///
/// ```
/// let value = sci_special::riccati_jn(3,0.5)?;
/// let _ = value;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy riccati_jn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.riccati_jn.html).
pub fn riccati_jn(n: usize, x: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    if n > 4096 || !x.is_finite() || x <= 0.0 {
        return Err(Error::InvalidInput(
            "Riccati sequence requires positive finite x and degree at most 4096",
        ));
    }
    Ok((0..=n)
        .map(|i| {
            let value = super::spherical_jn(i as i64, x, false);
            (
                x * value,
                value + x * super::spherical_jn(i as i64, x, true),
            )
        })
        .unzip())
}

/// Riccati–Bessel values x*y_i(x) and derivatives through order n.
///
/// ## Parameters and domain
///
/// n≤4096 and positive finite real x.
/// Invalid sizes or parameters return [`Error::InvalidInput`]. Numerical overflow returns [`Error::NonFiniteEvaluation`] where validated.
///
/// ## Examples
///
/// ```
/// let value = sci_special::riccati_yn(3,0.5)?;
/// let _ = value;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy riccati_yn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.riccati_yn.html).
pub fn riccati_yn(n: usize, x: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    if n > 4096 || !x.is_finite() || x <= 0.0 {
        return Err(Error::InvalidInput(
            "Riccati sequence requires positive finite x and degree at most 4096",
        ));
    }
    Ok((0..=n)
        .map(|i| {
            let value = super::spherical_yn(i as i64, x, false);
            (
                x * value,
                value + x * super::spherical_yn(i as i64, x, true),
            )
        })
        .unzip())
}

/// Jahnke–Emden Lambda function Γ(v+1)(2/x)^v J_v(x) and its derivative sequence.
///
/// ## Parameters and domain
///
/// Finite nonnegative real order v≤256 and positive finite real x.
/// Invalid sizes or parameters return [`Error::InvalidInput`]. Numerical overflow returns [`Error::NonFiniteEvaluation`] where validated.
///
/// ## Returns
///
/// `(values, derivatives)` for orders `frac(v)..=frac(v)+floor(v)`, in
/// ascending order. Both vectors contain `floor(v) + 1` entries. Derivatives
/// are with respect to the dimensionless coordinate `x`.
///
/// ## Numerical behavior
///
/// For well-conditioned `x <= 50` cases with exactly representable anchor
/// orders, a paired, compensated binary64 series evaluates two adjacent orders
/// and a backward recurrence computes the sequence. A conservative bound on
/// series and recurrence roundoff rejects cases where absolute error could
/// exceed about `4e-12 * max(1, |value|)`. Other `x <= 50` inputs retain the
/// double-double series anchors and backward recurrence; larger arguments use
/// individually evaluated hypergeometric values and stable recurrences.
/// Work is bounded by the order limit and the private series' 4,096-term cap.
///
/// ## Examples
///
/// ```
/// let value = sci_special::lmbda(2.5,0.5)?;
/// let _ = value;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// ## References
///
/// - [SciPy lmbda](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.lmbda.html).
pub fn lmbda(v: f64, x: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    if !v.is_finite() || !(0.0..=256.0).contains(&v) || !x.is_finite() || x <= 0.0 {
        return Err(Error::InvalidInput("invalid Lambda function arguments"));
    }
    Ok(crate::backend::legacy::jahnke_emden_lambda(v, x))
}
