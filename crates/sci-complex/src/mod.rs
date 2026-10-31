use crate::{Error, Result};
use sci_core::Float;

/// A complex value `re + i im`, shared across Sci modules.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex<T = f64> {
    /// Real component.
    pub re: T,
    /// Imaginary component.
    pub im: T,
}

pub(crate) fn native<T: Float>(z: Complex<T>) -> num_complex::Complex<T> {
    num_complex::Complex::<T>::new(z.re, z.im)
}

pub(crate) fn owned<T: Float>(z: num_complex::Complex<T>) -> Complex<T> {
    Complex { re: z.re, im: z.im }
}

/// Adds `a + b = (a.re + b.re) + i(a.im + b.im)`.
///
/// ## Parameters
///
/// `a` and `b` are Cartesian complex operands.
///
/// ## Domain
///
/// All components are accepted. NaN and infinity follow IEEE arithmetic.
///
/// ## Returns
///
/// The complex result.
///
/// ## Numerical behavior
///
/// Constant work using the selected floating-point type; rounding, overflow, and underflow apply.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::add(Complex { re: 1.0, im: 2.0 }, Complex { re: 3.0, im: -1.0 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
pub fn add<T: Float>(a: Complex<T>, b: Complex<T>) -> Complex<T> {
    Complex {
        re: a.re + b.re,
        im: a.im + b.im,
    }
}

/// Subtracts `a − b = (a.re − b.re) + i(a.im − b.im)`.
///
/// ## Parameters
///
/// `a` and `b` are Cartesian complex operands.
///
/// ## Domain
///
/// All components are accepted. NaN and infinity follow IEEE arithmetic.
///
/// ## Returns
///
/// The complex result.
///
/// ## Numerical behavior
///
/// Constant work using the selected floating-point type; rounding, overflow, and underflow apply.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::subtract(Complex { re: 1.0, im: 2.0 }, Complex { re: 3.0, im: -1.0 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
pub fn subtract<T: Float>(a: Complex<T>, b: Complex<T>) -> Complex<T> {
    Complex {
        re: a.re - b.re,
        im: a.im - b.im,
    }
}

/// Multiplies `(x + iy)(u + iv) = (xu − yv) + i(xv + yu)`.
///
/// ## Parameters
///
/// `a` and `b` are Cartesian complex operands.
///
/// ## Domain
///
/// All components are accepted. NaN and infinity follow IEEE arithmetic.
///
/// ## Returns
///
/// The complex result.
///
/// ## Numerical behavior
///
/// Finite products use binary scaling and fused multiply-add residuals, combining
/// each pair of products before restoring its exponent. This preserves small
/// cancellation residuals, subnormal results, and representable components when
/// the individual products overflow. Work is constant; true component overflow
/// returns signed infinity. Rounding remains; correct rounding is not guaranteed.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::multiply(Complex { re: 1.0, im: 2.0 }, Complex { re: 3.0, im: -1.0 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
pub fn multiply<T: Float>(a: Complex<T>, b: Complex<T>) -> Complex<T> {
    if [a.re, a.im, b.re, b.im].iter().all(|x| x.is_finite()) {
        Complex {
            re: product_sum(a.re, b.re, -a.im, b.im),
            im: product_sum(a.re, b.im, a.im, b.re),
        }
    } else {
        owned(native(a) * native(b))
    }
}

// Exact binary decomposition, including subnormals; zero keeps its sign.
fn binary_parts<T: Float>(value: T) -> (T, i32) {
    if value == T::zero() {
        return (value, 0);
    }
    let (mantissa, exponent, sign) = value.integer_decode();
    let leading = 63 - mantissa.leading_zeros();
    let fraction = T::from(mantissa).unwrap() / T::from(1_u64 << leading).unwrap();
    (
        fraction * T::from(sign).unwrap(),
        i32::from(exponent) + leading as i32,
    )
}

#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
fn restore_binary_scale<T: Float>(mut value: T, mut exponent: i32) -> T {
    let (_, maximum) = binary_parts(T::max_value());
    let (_, minimum) = binary_parts(T::min_positive_value());
    while exponent > maximum {
        value = value * 2.0.powi(maximum);
        exponent -= maximum;
    }
    while exponent < minimum {
        value = value * T::min_positive_value();
        exponent -= minimum;
    }
    value * 2.0.powi(exponent)
}

// Two products are combined before restoring their shared exponent. FMA
// residuals preserve cancellation that separate rounded products would lose.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
fn product_sum<T: Float>(a: T, b: T, c: T, d: T) -> T {
    let (mut a, ae) = binary_parts(a);
    let (b, be) = binary_parts(b);
    let (mut c, ce) = binary_parts(c);
    let (d, de) = binary_parts(d);
    if a == 0.0 || b == 0.0 {
        return if c == 0.0 || d == 0.0 {
            a * b + c * d
        } else {
            restore_binary_scale(c * d, ce + de)
        };
    }
    if c == 0.0 || d == 0.0 {
        return restore_binary_scale(a * b, ae + be);
    }
    let exponent = (ae + be).max(ce + de);
    a = a * (2.0_f64.powi(ae + be - exponent));
    c = c * (2.0_f64.powi(ce + de - exponent));
    let (p, q) = (a * b, c * d);
    let residual = a.mul_add(b, -p) + c.mul_add(d, -q);
    restore_binary_scale((p + q) + residual, exponent)
}

/// Divides `a / b = a conjugate(b) / |b|²`.
///
/// ## Parameters
///
/// `a` is the numerator; `b` is the denominator.
///
/// ## Domain
///
/// All components are accepted. NaN and infinity follow IEEE arithmetic. A zero denominator returns two NaNs.
///
/// ## Returns
///
/// The complex quotient.
///
/// ## Numerical behavior
///
/// Finite denominators are scaled before division to avoid squaring large or
/// tiny components. Intermediate overflow remains possible for extreme ratios.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// assert_eq!(complex::divide(Complex { re: 3.0, im: 4.0 }, Complex { re: 1.0, im: 0.0 }), Complex { re: 3.0, im: 4.0 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn divide<T: Float>(a: Complex<T>, b: Complex<T>) -> Complex<T> {
    let scale = b.re.abs().max(b.im.abs());
    if scale == 0.0 {
        return Complex {
            re: T::nan(),
            im: T::nan(),
        };
    }
    if !scale.is_finite() {
        return owned(native(a) / native(b));
    }
    let c = b.re / scale;
    let d = b.im / scale;
    let denominator = c * c + d * d;
    // Normalize weights before the sums. Scale the numerator down only:
    // scaling it up can overflow even when the quotient is representable.
    let numerator_scale = scale.max(1.0);
    let ar = a.re / numerator_scale;
    let ai = a.im / numerator_scale;
    let c = c / denominator;
    let d = d / denominator;
    let remaining_scale = scale / numerator_scale;
    Complex {
        re: (ar * c + ai * d) / remaining_scale,
        im: (ai * c - ar * d) / remaining_scale,
    }
}

/// Negates `z`: `−z = −re − i im`.
///
/// ## Parameters
///
/// `z` is a Cartesian complex value.
///
/// ## Domain
///
/// All components are accepted. NaN and infinity follow IEEE arithmetic. Signed zeros are preserved or negated componentwise.
///
/// ## Returns
///
/// The complex result.
///
/// ## Numerical behavior
///
/// Constant work using the selected floating-point type; rounding, overflow, and underflow apply.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::negate(Complex { re: 0.5, im: 0.25 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
pub fn negate<T: Float>(z: Complex<T>) -> Complex<T> {
    Complex {
        re: -z.re,
        im: -z.im,
    }
}

/// Conjugates `z`: `conjugate(z) = re − i im`.
///
/// ## Parameters
///
/// `z` is a Cartesian complex value.
///
/// ## Domain
///
/// All components are accepted. NaN and infinity follow IEEE arithmetic. Signed zeros are preserved or negated componentwise.
///
/// ## Returns
///
/// The complex result.
///
/// ## Numerical behavior
///
/// Constant work using the selected floating-point type; rounding, overflow, and underflow apply.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::conjugate(Complex { re: 0.5, im: 0.25 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
pub fn conjugate<T: Float>(z: Complex<T>) -> Complex<T> {
    Complex {
        re: z.re,
        im: -z.im,
    }
}

/// Computes `1/z = conjugate(z)/|z|²`.
///
/// ## Parameters
///
/// `z` is a Cartesian complex value.
///
/// ## Domain
///
/// All components are accepted. NaN and infinity follow IEEE arithmetic. Zero returns two NaNs.
///
/// ## Returns
///
/// The complex result.
///
/// ## Numerical behavior
///
/// Constant work using the selected floating-point type; rounding, overflow, and underflow apply.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::reciprocal(Complex { re: 0.5, im: 0.25 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn reciprocal<T: Float>(z: Complex<T>) -> Complex<T> {
    divide(Complex { re: 1.0, im: 0.0 }, z)
}

/// Extracts `Re(z) = z.re`.
///
/// ## Parameters
///
/// `z` is Cartesian; phase angles are in radians.
///
/// ## Domain
///
/// All components are accepted. NaN and infinity follow IEEE arithmetic.
///
/// ## Returns
///
/// The real scalar.
///
/// ## Numerical behavior
///
/// Magnitude uses scaled `hypot`; squared magnitude may overflow or underflow.
/// Component extraction is exact; phase uses `atan2`.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::real(Complex { re: 0.5, im: 0.25 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
pub fn real<T: Float>(z: Complex<T>) -> T {
    z.re
}

/// Extracts `Im(z) = z.im`.
///
/// ## Parameters
///
/// `z` is Cartesian; phase angles are in radians.
///
/// ## Domain
///
/// All components are accepted. NaN and infinity follow IEEE arithmetic.
///
/// ## Returns
///
/// The real scalar.
///
/// ## Numerical behavior
///
/// Magnitude uses scaled `hypot`; squared magnitude may overflow or underflow.
/// Component extraction is exact; phase uses `atan2`.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::imag(Complex { re: 0.5, im: 0.25 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
pub fn imag<T: Float>(z: Complex<T>) -> T {
    z.im
}

/// Computes `|z| = √(re² + im²)`.
///
/// ## Parameters
///
/// `z` is Cartesian; phase angles are in radians.
///
/// ## Domain
///
/// Uses `f64::hypot`: infinity dominates NaN, and zero has positive sign.
///
/// ## Returns
///
/// The real scalar.
///
/// ## Numerical behavior
///
/// Magnitude uses scaled `hypot`; squared magnitude may overflow or underflow.
/// Component extraction is exact; phase uses `atan2`.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::abs(Complex { re: 0.5, im: 0.25 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
pub fn abs<T: Float>(z: Complex<T>) -> T {
    z.re.hypot(z.im)
}

/// Computes the squared magnitude `|z|² = re² + im²`.
///
/// ## Parameters
///
/// `z` is Cartesian; phase angles are in radians.
///
/// ## Domain
///
/// All components are accepted. NaN and infinity follow IEEE arithmetic.
///
/// ## Returns
///
/// The real scalar.
///
/// ## Numerical behavior
///
/// Scaled products combine the squares before restoring their exponent,
/// preserving representable subnormal sums when individual squares underflow.
/// Constant work; true overflow returns infinity and true underflow may return
/// zero. Floating-point rounding remains; correct rounding is not guaranteed.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::norm_sqr(Complex { re: 0.5, im: 0.25 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
pub fn norm_sqr<T: Float>(z: Complex<T>) -> T {
    if z.re.is_finite() && z.im.is_finite() {
        product_sum(z.re, z.re, z.im, z.im)
    } else {
        z.re * z.re + z.im * z.im
    }
}

/// Computes the principal phase `arg(z) = atan2(im, re)`.
///
/// ## Parameters
///
/// `z` is Cartesian; phase angles are in radians.
///
/// ## Domain
///
/// Signed zeros select the side of the negative real axis, including at zero.
/// NaN yields NaN; infinities follow `f64::atan2`.
///
/// ## Returns
///
/// The real scalar in `[-π, π]`.
///
/// ## Numerical behavior
///
/// Magnitude uses scaled `hypot`; squared magnitude may overflow or underflow.
/// Component extraction is exact; phase uses `atan2`.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::arg(Complex { re: 0.5, im: 0.25 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
pub fn arg<T: Float>(z: Complex<T>) -> T {
    z.im.atan2(z.re)
}

/// Converts `z = r exp(iθ)` to `(r, θ) = (|z|, arg(z))`.
///
/// ## Parameters
///
/// `z` is Cartesian; `θ` is in radians.
///
/// ## Domain
///
/// See [`abs`] and [`arg`] for NaN, infinity, and signed-zero conventions.
///
/// ## Returns
///
/// Magnitude first, principal phase second, with `θ ∈ [-π, π]`.
///
/// ## Numerical behavior
///
/// Constant work using the selected floating-point type; rounding, overflow, and underflow apply.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::to_polar(Complex { re: 0.5, im: 0.25 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
pub fn to_polar<T: Float>(z: Complex<T>) -> (T, T) {
    (abs(z), arg(z))
}

/// Constructs `z = r exp(iθ) = r cos(θ) + i r sin(θ)`.
///
/// ## Parameters
///
/// `radius` is `r ≥ 0`; `phase` is `θ` in radians and need not be principal.
///
/// ## Domain
///
/// Finite nonnegative radius and finite phase are required. Signed zero is accepted.
///
/// ## Returns
///
/// The Cartesian complex value.
///
/// ## Errors
///
/// `Error::InvalidInput` for negative radius, NaN, or infinity in either argument.
///
/// ## Numerical behavior
///
/// Uses standard sine and cosine; large angles lose argument-reduction precision.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let z = complex::from_polar(2.0, 0.0)?;
/// assert_eq!(z, Complex { re: 2.0, im: 0.0 });
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn from_polar<T: Float>(radius: T, phase: T) -> Result<Complex<T>> {
    if !radius.is_finite() || radius < 0.0 || !phase.is_finite() {
        return Err(Error::InvalidInput(
            "polar radius must be finite and nonnegative; phase must be finite",
        ));
    }
    Ok(owned(num_complex::Complex::<T>::from_polar(radius, phase)))
}

// Split a large exponential into at most four chunks in the selected format.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
fn exponential_component<T: Float>(x: T, coefficient: T, half: bool) -> T {
    if coefficient == 0.0 {
        return coefficient;
    }
    let chunk_log = (T::max_value().ln() * 0.75).floor();
    if x > T::max_value().ln() - T::min_positive_value().ln() - T::epsilon().ln() + 1.0 {
        return T::infinity().copysign(coefficient);
    }
    let chunk = chunk_log.exp();
    // Scale before halving so the smallest subnormal coefficient survives.
    let first = x.min(chunk_log);
    let mut value = coefficient * first.exp();
    if half {
        value = value * 0.5;
    }
    let mut remainder = x - first;
    for _ in 0..2 {
        if remainder <= chunk_log {
            break;
        }
        value = value * chunk;
        remainder = remainder - chunk_log;
    }
    value * remainder.exp()
}

/// Computes `exp(z) = exp(re)(cos(im) + i sin(im))`.
///
/// ## Parameters
///
/// `z` is dimensionless; its imaginary part is in radians.
///
/// ## Domain
///
/// Every finite complex value is accepted. Nonfinite inputs follow backend IEEE
/// behavior; componentwise NumPy compatibility is not guaranteed.
///
/// ## Returns
///
/// The complex exponential.
///
/// ## Numerical behavior
///
/// Large finite exponential factors are split into bounded chunks before
/// multiplication by trigonometric coefficients. This preserves representable
/// components and signed zeros. True overflow, underflow, and angle reduction
/// error remain; work is constant.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::exp(Complex { re: 0.5, im: 0.25 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn exp<T: Float>(z: Complex<T>) -> Complex<T> {
    if z.re.is_finite() && z.im.is_finite() && z.re > (T::max_value().ln() - 10.0) {
        let (sine, cosine) = z.im.sin_cos();
        Complex {
            re: exponential_component(z.re, cosine, false),
            im: exponential_component(z.re, sine, false),
        }
    } else {
        owned(native(z).exp())
    }
}

/// Computes `ln(z) = ln|z| + i arg(z)` on the principal branch.
///
/// ## Parameters
///
/// `z` is dimensionless; imaginary output is in radians.
///
/// ## Domain
///
/// The cut is the negative real axis; signed imaginary zero selects ±π.
/// Zero returns real `−∞` and its signed-zero phase. Infinity gives real `+∞`;
/// NaN follows `hypot` and `atan2`.
///
/// ## Returns
///
/// Principal logarithm with imaginary part in `[-π, π]`.
///
/// ## Numerical behavior
///
/// Uses scaled components for the logarithm when finite, avoiding overflow
/// of the magnitude. Near the unit circle, `log1p` preserves small real parts.
/// Underflow in negligible components is harmless.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::ln(Complex { re: 0.5, im: 0.25 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn ln<T: Float>(z: Complex<T>) -> Complex<T> {
    let scale = z.re.abs().max(z.im.abs());
    let re = if scale > 0.0 && z.re.is_finite() && z.im.is_finite() {
        if (0.5..=1.5).contains(&scale) {
            // Preserve the small log modulus near the unit circle.
            let x = z.re.abs();
            let y = z.im.abs();
            let (large, small) = (x.max(y), x.min(y));
            0.5 * ((large - 1.0) * (large + 1.0) + small * small).ln_1p()
        } else {
            scale.ln() + (z.re / scale).hypot(z.im / scale).ln()
        }
    } else {
        abs(z).ln()
    };
    Complex { re, im: arg(z) }
}

/// Computes the principal root `sqrt(z) = √|z| exp(i arg(z)/2)`.
///
/// ## Parameters
///
/// `z` is Cartesian; the output has nonnegative real part.
///
/// ## Domain
///
/// The cut is the negative real axis; signed imaginary zero selects the side.
/// Zero preserves imaginary signed zero. Nonfinite values follow the backend.
///
/// ## Returns
///
/// The principal square root.
///
/// ## Numerical behavior
///
/// Finite components are scaled before the magnitude calculation. The smaller
/// root component uses `im/(2*large)` to avoid cancellation near the real axis.
/// Floating-point rounding and subnormal underflow remain.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::sqrt(Complex { re: 0.5, im: 0.25 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn sqrt<T: Float>(z: Complex<T>) -> Complex<T> {
    if !z.re.is_finite() || !z.im.is_finite() {
        return owned(native(z).sqrt());
    }
    let scale = z.re.abs().max(z.im.abs());
    if scale == 0.0 {
        return Complex { re: 0.0, im: z.im };
    }
    let x = z.re / scale;
    let y = z.im / scale;
    let large = scale.sqrt() * (0.5 * (x.hypot(y) + x.abs())).sqrt();
    let small = z.im / (2.0 * large);
    if z.re >= 0.0 {
        Complex {
            re: large,
            im: small,
        }
    } else {
        Complex {
            re: small.abs(),
            im: large.copysign(z.im),
        }
    }
}

/// Computes `sin(z) = sin(re) cosh(im) + i cos(re) sinh(im)`.
///
/// ## Parameters
///
/// `z` is dimensionless; trigonometric angles are in radians.
///
/// ## Domain
///
/// Every finite complex value is accepted. Nonfinite inputs follow backend IEEE
/// behavior; componentwise NumPy compatibility is not guaranteed.
///
/// ## Returns
///
/// The complex function value.
///
/// ## Numerical behavior
///
/// Large finite exponential factors are split into bounded chunks before
/// multiplication by trigonometric coefficients. This preserves representable
/// components and signed zeros. True overflow, underflow, and angle reduction
/// error remain; work is constant.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::sin(Complex { re: 0.5, im: 0.25 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn sin<T: Float>(z: Complex<T>) -> Complex<T> {
    if z.re.is_finite() && z.im.is_finite() && z.im.abs() > (T::max_value().ln() - 10.0) {
        let (sine, cosine) = z.re.sin_cos();
        Complex {
            re: exponential_component(z.im.abs(), sine, true),
            im: exponential_component(z.im.abs(), cosine * 1.0_f64.copysign(z.im), true),
        }
    } else {
        owned(native(z).sin())
    }
}

/// Computes `cos(z) = cos(re) cosh(im) − i sin(re) sinh(im)`.
///
/// ## Parameters
///
/// `z` is dimensionless; trigonometric angles are in radians.
///
/// ## Domain
///
/// Every finite complex value is accepted. Nonfinite inputs follow backend IEEE
/// behavior; componentwise NumPy compatibility is not guaranteed.
///
/// ## Returns
///
/// The complex function value.
///
/// ## Numerical behavior
///
/// Large finite exponential factors are split into bounded chunks before
/// multiplication by trigonometric coefficients. This preserves representable
/// components and signed zeros. True overflow, underflow, and angle reduction
/// error remain; work is constant.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::cos(Complex { re: 0.5, im: 0.25 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn cos<T: Float>(z: Complex<T>) -> Complex<T> {
    if z.re.is_finite() && z.im.is_finite() && z.im.abs() > (T::max_value().ln() - 10.0) {
        let (sine, cosine) = z.re.sin_cos();
        Complex {
            re: exponential_component(z.im.abs(), cosine, true),
            im: exponential_component(z.im.abs(), -sine * 1.0_f64.copysign(z.im), true),
        }
    } else {
        owned(native(z).cos())
    }
}

/// Computes `sinh(z) = sinh(re) cos(im) + i cosh(re) sin(im)`.
///
/// ## Parameters
///
/// `z` is dimensionless; trigonometric angles are in radians.
///
/// ## Domain
///
/// Every finite complex value is accepted. Nonfinite inputs follow backend IEEE
/// behavior; componentwise NumPy compatibility is not guaranteed.
///
/// ## Returns
///
/// The complex function value.
///
/// ## Numerical behavior
///
/// Large finite exponential factors are split into bounded chunks before
/// multiplication by trigonometric coefficients. This preserves representable
/// components and signed zeros. True overflow, underflow, and angle reduction
/// error remain; work is constant.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::sinh(Complex { re: 0.5, im: 0.25 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn sinh<T: Float>(z: Complex<T>) -> Complex<T> {
    if z.re.is_finite() && z.im.is_finite() && z.re.abs() > (T::max_value().ln() - 10.0) {
        let value = sin(Complex { re: z.im, im: z.re });
        Complex {
            re: value.im,
            im: value.re,
        }
    } else {
        owned(native(z).sinh())
    }
}

/// Computes `cosh(z) = cosh(re) cos(im) + i sinh(re) sin(im)`.
///
/// ## Parameters
///
/// `z` is dimensionless; trigonometric angles are in radians.
///
/// ## Domain
///
/// Every finite complex value is accepted. Nonfinite inputs follow backend IEEE
/// behavior; componentwise NumPy compatibility is not guaranteed.
///
/// ## Returns
///
/// The complex function value.
///
/// ## Numerical behavior
///
/// Large finite exponential factors are split into bounded chunks before
/// multiplication by trigonometric coefficients. This preserves representable
/// components and signed zeros. True overflow, underflow, and angle reduction
/// error remain; work is constant.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::cosh(Complex { re: 0.5, im: 0.25 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn cosh<T: Float>(z: Complex<T>) -> Complex<T> {
    if z.re.is_finite() && z.im.is_finite() && z.re.abs() > (T::max_value().ln() - 10.0) {
        cos(Complex {
            re: -z.im,
            im: z.re,
        })
    } else {
        owned(native(z).cosh())
    }
}

/// Computes `tan(z) = sin(z)/cos(z)`.
///
/// ## Parameters
///
/// `z` is in radians.
///
/// ## Domain
///
/// Poles occur at real `π/2 + kπ` (not exactly representable in `f64`).
/// Nonfinite inputs follow IEEE arithmetic.
///
/// ## Returns
///
/// The complex tangent.
///
/// ## Numerical behavior
///
/// For finite inputs, a scaled sum of squares avoids denominator cancellation
/// near poles. Exponential scaling for `|im| > 1` prevents hyperbolic overflow;
/// the leading factor is retained before subnormal rounding.
/// Accuracy degrades near poles and for large real angles.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::tan(Complex { re: 0.5, im: 0.25 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn tan<T: Float>(z: Complex<T>) -> Complex<T> {
    if !z.re.is_finite() || !z.im.is_finite() {
        return owned(native(z).tan());
    }
    let (s, c) = z.re.sin_cos();
    if z.im.abs() <= 1.0 {
        // cos(2x)+cosh(2y) = 2(cos(x)^2+sinh(y)^2).
        let sh = z.im.sinh();
        let scale = c.abs().max(sh.abs());
        let cr = c / scale;
        let hr = sh / scale;
        let denominator = cr * cr + hr * hr;
        return Complex {
            re: (s * cr / denominator) / scale,
            im: (z.im.cosh() * hr / denominator) / scale,
        };
    }
    let t = (-2.0 * z.im.abs()).exp();
    let denominator = 1.0 + 2.0 * t * (c * c - s * s) + t * t;
    Complex {
        // exp(-y) remains normal across the subnormal transition of exp(-2y).
        re: (((4.0 * s * c) * (-z.im.abs()).exp()) * (-z.im.abs()).exp()) / denominator,
        im: ((1.0 - t * t) / denominator).copysign(z.im),
    }
}

/// Computes `tanh(z) = sinh(z)/cosh(z)`.
///
/// ## Parameters
///
/// `z` is dimensionless.
///
/// ## Domain
///
/// Poles occur at imaginary `i(π/2 + kπ)`. Nonfinite inputs follow IEEE arithmetic.
///
/// ## Returns
///
/// The complex hyperbolic tangent.
///
/// ## Numerical behavior
///
/// Uses `tanh(z) = −i tan(iz)` to reuse the scaled tangent algorithm.
/// Accuracy degrades near poles and for large imaginary angles.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::tanh(Complex { re: 0.5, im: 0.25 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
pub fn tanh<T: Float>(z: Complex<T>) -> Complex<T> {
    let t = tan(Complex {
        re: -z.im,
        im: z.re,
    });
    Complex {
        re: t.im,
        im: -t.re,
    }
}

// `std`'s `acosh` evaluates `ln(x + sqrt(x² − 1))`, which loses the small
// result next to 1 and overflows near the largest finite values. Form `x − 1`
// exactly and use `log1p` near 1, and `ln(2x)` once `x² − 1` rounds to `x²`.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
fn real_acosh<T: Float>(x: T) -> T {
    if x < 2.0 {
        let t = x - 1.0;
        (t + (t * (t + 2.0)).sqrt()).ln_1p()
    } else if x < 1e8 {
        (x + (x * x - 1.0).sqrt()).ln()
    } else {
        // sqrt(x² − 1) rounds to x; avoid overflowing x + x.
        x.ln() + T::from(std::f64::consts::LN_2).unwrap()
    }
}

// First-quadrant inverse geometry. Rationalize hypot differences before
// evaluating A-1 and A-x; direct subtraction loses the small branch component.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
fn inverse_geometry<T: Float>(x: T, y: T) -> (T, T) {
    if x.max(y) > (T::max_value().sqrt() * 0.01) {
        let scale = x.max(y);
        let height = scale.ln()
            + (x / scale).hypot(y / scale).ln()
            + T::from(std::f64::consts::LN_2).unwrap();
        return (y, height);
    }
    if y == 0.0 {
        return if x <= 1.0 {
            (((1.0 - x) * (1.0 + x)).sqrt(), 0.0)
        } else {
            (0.0, real_acosh(x))
        };
    }
    // Away from the branch point, first-order limits preserve subnormals.
    if y < (T::min_positive_value().sqrt() * 10000.0) && x != 1.0 {
        let root = (x - 1.0).abs().sqrt() * (x + 1.0).sqrt();
        return if x < 1.0 {
            (root, y / root)
        } else {
            ((y / root) * x, real_acosh(x))
        };
    }
    let r = (x + 1.0).hypot(y);
    let s = (x - 1.0).hypot(y);
    let a = 0.5 * r + 0.5 * s;
    let correction = 0.5 * (y / (r + x + 1.0) + y / (s + (x - 1.0).abs()));
    let root_delta = y.sqrt() * correction.sqrt();
    let distance = if x <= 1.0 {
        (1.0 - x).sqrt().hypot(root_delta) * (a + x).sqrt()
    } else {
        root_delta * (a + x).sqrt()
    };
    let height = if x <= 1.0 {
        let u = y * correction;
        // sqrt(A-1) stays representable even when A-1 underflows.
        (u + root_delta * (u + 2.0).sqrt()).ln_1p()
    } else {
        let u = (x - 1.0) + y * correction;
        (u + u.sqrt() * (u + 2.0).sqrt()).ln_1p()
    };
    (distance, height)
}

/// Computes the principal inverse `asin(z) = −i ln(iz + sqrt(1 − z²))`.
///
/// ## Parameters
///
/// `z` is dimensionless; angle outputs use radians.
///
/// ## Domain
///
/// Branch cuts: real `(-∞, −1)` and `(1, ∞)`. Signed zeros select the side of each cut.
/// Every finite complex value is accepted. Nonfinite inputs follow backend IEEE
/// behavior; componentwise NumPy compatibility is not guaranteed.
///
/// ## Returns
///
/// Principal value with real part in `[-π/2, π/2]`.
///
/// ## Numerical behavior
///
/// Constant work using rationalized distances, `log1p`, and scaled large-input
/// limits for finite values. This preserves small branch components and avoids
/// intermediate overflow. Floating-point rounding and subnormal underflow remain.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::asin(Complex { re: 0.5, im: 0.25 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn asin<T: Float>(z: Complex<T>) -> Complex<T> {
    if z.im == 0.0 {
        if z.re.abs() <= 1.0 {
            return Complex {
                re: z.re.asin(),
                im: z.im,
            };
        }
        return Complex {
            re: T::from(std::f64::consts::FRAC_PI_2).unwrap().copysign(z.re),
            im: real_acosh(z.re.abs()).copysign(z.im),
        };
    }
    if !z.re.is_finite() || !z.im.is_finite() {
        return owned(native(z).asin());
    }
    let (angle, height) = inverse_geometry(z.re.abs(), z.im.abs());
    Complex {
        re: z.re.abs().atan2(angle).copysign(z.re),
        im: height.copysign(z.im),
    }
}

/// Computes the principal inverse `acos(z) = π/2 − asin(z)`.
///
/// ## Parameters
///
/// `z` is dimensionless; angle outputs use radians.
///
/// ## Domain
///
/// Branch cuts: real `(-∞, −1)` and `(1, ∞)`. Signed zeros select the side of each cut.
/// Every finite complex value is accepted. Nonfinite inputs follow backend IEEE
/// behavior; componentwise NumPy compatibility is not guaranteed.
///
/// ## Returns
///
/// Principal value with real part in `[0, π]`.
///
/// ## Numerical behavior
///
/// Constant work using rationalized distances, `log1p`, and scaled large-input
/// limits for finite values. This preserves small branch components and avoids
/// intermediate overflow. Floating-point rounding and subnormal underflow remain.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::acos(Complex { re: 0.5, im: 0.25 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn acos<T: Float>(z: Complex<T>) -> Complex<T> {
    if !z.re.is_finite() || !z.im.is_finite() {
        return owned(native(z).acos());
    }
    let (distance, height) = inverse_geometry(z.re.abs(), z.im.abs());
    let angle = distance.atan2(z.re.abs());
    Complex {
        re: if z.re < 0.0 {
            T::from(std::f64::consts::PI).unwrap() - angle
        } else {
            angle
        },
        im: -height.copysign(z.im),
    }
}

/// Computes the principal inverse `atan(z) = (ln(1 + iz) − ln(1 − iz))/(2i)`.
///
/// ## Parameters
///
/// `z` is dimensionless; angle outputs use radians.
///
/// ## Domain
///
/// Branch cuts: imaginary `(-∞i, −i]` and `[i, ∞i)`. Signed zeros select the side of each cut.
/// At `z = ±i`, the imaginary result is signed infinity.
///
/// ## Returns
///
/// Principal value with real part in `[-π/2, π/2]`.
///
/// ## Numerical behavior
///
/// Constant work using rationalized distances, `log1p`, and scaled large-input
/// limits for finite values. This preserves small branch components and avoids
/// intermediate overflow. Floating-point rounding and subnormal underflow remain.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::atan(Complex { re: 0.5, im: 0.25 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
pub fn atan<T: Float>(z: Complex<T>) -> Complex<T> {
    let a = atanh(Complex {
        re: -z.im,
        im: z.re,
    });
    Complex {
        re: a.im,
        im: -a.re,
    }
}

/// Computes the principal inverse `asinh(z) = ln(z + sqrt(1 + z²))`.
///
/// ## Parameters
///
/// `z` is dimensionless; angle outputs use radians.
///
/// ## Domain
///
/// Branch cuts: imaginary `(-∞i, −i)` and `(i, ∞i)`. Signed zeros select the side of each cut.
/// Every finite complex value is accepted. Nonfinite inputs follow backend IEEE
/// behavior; componentwise NumPy compatibility is not guaranteed.
///
/// ## Returns
///
/// Principal value with imaginary part in `[-π/2, π/2]`.
///
/// ## Numerical behavior
///
/// Constant work using rationalized distances, `log1p`, and scaled large-input
/// limits for finite values. This preserves small branch components and avoids
/// intermediate overflow. Floating-point rounding and subnormal underflow remain.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::asinh(Complex { re: 0.5, im: 0.25 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
pub fn asinh<T: Float>(z: Complex<T>) -> Complex<T> {
    let a = asin(Complex {
        re: -z.im,
        im: z.re,
    });
    Complex {
        re: a.im,
        im: -a.re,
    }
}

/// Computes the principal inverse `acosh(z) = 2 ln(sqrt((z + 1)/2) + sqrt((z − 1)/2))`.
///
/// ## Parameters
///
/// `z` is dimensionless; angle outputs use radians.
///
/// ## Domain
///
/// Branch cuts: real `(-∞, 1)`. Signed zeros select the side of each cut.
/// Every finite complex value is accepted. Nonfinite inputs follow backend IEEE
/// behavior; componentwise NumPy compatibility is not guaranteed.
///
/// ## Returns
///
/// Principal value with nonnegative real part and imaginary part in `[-π, π]`.
///
/// ## Numerical behavior
///
/// Constant work using rationalized distances, `log1p`, and scaled large-input
/// limits for finite values. This preserves small branch components and avoids
/// intermediate overflow. Floating-point rounding and subnormal underflow remain.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::acosh(Complex { re: 0.5, im: 0.25 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn acosh<T: Float>(z: Complex<T>) -> Complex<T> {
    if z.im == 0.0 {
        if z.re >= 1.0 {
            return Complex {
                re: real_acosh(z.re),
                im: z.im,
            };
        }
        if z.re >= -1.0 {
            return Complex {
                re: 0.0,
                im: z.re.acos().copysign(z.im),
            };
        }
        return Complex {
            re: real_acosh(-z.re),
            im: T::from(std::f64::consts::PI).unwrap().copysign(z.im),
        };
    }
    if !z.re.is_finite() || !z.im.is_finite() {
        return owned(native(z).acosh());
    }
    let a = acos(z);
    Complex {
        re: a.im.abs(),
        im: a.re.copysign(z.im),
    }
}

/// Computes the principal inverse `atanh(z) = (ln(1 + z) − ln(1 − z))/2`.
///
/// ## Parameters
///
/// `z` is dimensionless; angle outputs use radians.
///
/// ## Domain
///
/// Branch cuts: real `(-∞, −1]` and `[1, ∞)`. Signed zeros select the side of each cut.
/// At `z = ±1`, the real result is signed infinity.
///
/// ## Returns
///
/// Principal value with imaginary part in `[-π/2, π/2]`.
///
/// ## Numerical behavior
///
/// Constant work using rationalized distances, `log1p`, and scaled large-input
/// limits for finite values. This preserves small branch components and avoids
/// intermediate overflow. Floating-point rounding and subnormal underflow remain.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::atanh(Complex { re: 0.5, im: 0.25 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn atanh<T: Float>(z: Complex<T>) -> Complex<T> {
    if z.im == 0.0 {
        let re = if z.re.abs() <= 1.0 {
            z.re.atanh()
        } else {
            (1.0 / z.re).atanh()
        };
        let im = if z.re.abs() <= 1.0 {
            z.im
        } else {
            T::from(std::f64::consts::FRAC_PI_2).unwrap().copysign(z.im)
        };
        return Complex { re, im };
    }
    if !z.re.is_finite() || !z.im.is_finite() {
        return owned(native(z).atanh());
    }
    let x = z.re.abs();
    let y = z.im.abs();
    let scale = x.max(y);
    let (re, im) = if scale > (T::max_value().sqrt() * 0.01) {
        // atanh(z) ~ 1/z + sign(im) i pi/2; scale before squaring.
        let a = x / scale;
        let b = y / scale;
        (
            (a / (a * a + b * b)) / scale,
            T::from(std::f64::consts::FRAC_PI_2).unwrap(),
        )
    } else {
        let radius = (1.0 - x).hypot(y);
        let ratio = (4.0 * x / radius) / radius;
        let re = if ratio.is_finite() {
            0.25 * ratio.ln_1p()
        } else {
            0.5 * ((1.0 + x).hypot(y).ln() - radius.ln())
        };
        let im = 0.5 * (2.0 * y).atan2((1.0 - x) * (1.0 + x) - y * y);
        (re, im)
    };
    Complex {
        re: re.copysign(z.re),
        im: im.copysign(z.im),
    }
}

/// Computes `z^p = |z|^p exp(i p arg(z))`.
///
/// ## Parameters
///
/// `z` is the base; `exponent` is the power. Uses the principal logarithm, with a negative-real-axis branch cut.
///
/// ## Domain
///
/// Zero to zero is one. Zero to a positive finite power is zero; negative powers
/// of zero return two NaNs. Nonfinite inputs follow backend IEEE behavior.
///
/// ## Returns
///
/// The complex power.
///
/// ## Numerical behavior
///
/// Finite nonzero bases scale overflowing magnitudes before applying real powers.
/// An overflowing radial power uses the scaled logarithm and exponential to
/// preserve representable components. Half powers reuse the principal square
/// root; a first-order near-positive-axis limit preserves tiny imaginary parts.
/// Exponent one returns the input exactly. Work is constant; phase reduction, rounding,
/// true overflow, and underflow still limit accuracy.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::powf(Complex { re: 1.0, im: 1.0 }, 0.5);
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn powf<T: Float>(z: Complex<T>, exponent: T) -> Complex<T> {
    if exponent == 0.0 {
        return Complex { re: 1.0, im: 0.0 };
    }
    if z.re == 0.0 && z.im == 0.0 && exponent < 0.0 {
        return Complex {
            re: T::nan(),
            im: T::nan(),
        };
    }
    if z.re.is_finite() && z.im.is_finite() && (z.re != 0.0 || z.im != 0.0) && exponent.is_finite()
    {
        if exponent == 1.0 {
            return z;
        }
        if exponent == 0.5 {
            return sqrt(z);
        }
        if exponent == -0.5 {
            return reciprocal(sqrt(z));
        }
        let magnitude = abs(z);
        let radius = if magnitude.is_finite() {
            magnitude.powf(exponent)
        } else {
            let scale = z.re.abs().max(z.im.abs());
            scale.powf(exponent) * (z.re / scale).hypot(z.im / scale).powf(exponent)
        };
        let mut result = if radius.is_finite() {
            let (s, c) = (exponent * arg(z)).sin_cos();
            Complex {
                re: radius * c,
                im: radius * s,
            }
        } else {
            let logarithm = ln(z);
            exp(Complex {
                re: exponent * logarithm.re,
                im: exponent * logarithm.im,
            })
        };
        if z.re > 0.0 && z.im.abs() <= T::epsilon() * z.re / exponent.abs().max(1.0) {
            // The phase may underflow even when the final component is finite.
            // Bound the powered phase too; the first-order error is O(eps²).
            let derivative = z.re.powf(exponent - 1.0);
            let numerator = exponent * z.im;
            if derivative.is_finite()
                && derivative > 0.0
                && numerator.is_finite()
                && (numerator != 0.0 || z.im == 0.0 || exponent == 0.0)
            {
                result.im = numerator * derivative;
            }
        }
        result
    } else {
        owned(native(z).powf(exponent))
    }
}

/// Computes `z^w = exp(w ln(z))`.
///
/// ## Parameters
///
/// `z` is the base; `exponent` is the power. Uses the principal logarithm, with a negative-real-axis branch cut.
///
/// ## Domain
///
/// Zero to zero is one; zero to a positive real power is zero. Other powers
/// of zero return two NaNs. Nonfinite values follow IEEE arithmetic.
///
/// ## Returns
///
/// The complex power.
///
/// ## Numerical behavior
///
/// Real exponents reuse [`powf`] and its scaled evaluation and axis limits.
/// Other exponents use the principal logarithm and scaled exponential.
/// Constant work; rounding, overflow, phase reduction, and underflow apply.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::powc(Complex { re: 1.0, im: 1.0 }, Complex { re: 0.5, im: 0.25 });
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn powc<T: Float>(z: Complex<T>, exponent: Complex<T>) -> Complex<T> {
    if exponent.im == 0.0 {
        return powf(z, exponent.re);
    }
    if exponent.re == 0.0 && exponent.im == 0.0 {
        return Complex { re: 1.0, im: 0.0 };
    }
    if z.re == 0.0 && z.im == 0.0 {
        return if exponent.im == 0.0 && exponent.re > 0.0 {
            Complex { re: 0.0, im: 0.0 }
        } else {
            Complex {
                re: T::nan(),
                im: T::nan(),
            }
        };
    }
    exp(multiply(exponent, ln(z)))
}

/// Computes `z^n = z ⋯ z (n factors) for n > 0, and 1/z^(−n) for n < 0`.
///
/// ## Parameters
///
/// `z` is the base; `exponent` is the power. Integer powers have no branch cut.
///
/// ## Domain
///
/// Zero to zero is one; negative powers of zero return NaNs. Every `i32`
/// exponent, including `i32::MIN`, is accepted. IEEE overflow/underflow apply.
///
/// ## Returns
///
/// The complex power.
///
/// ## Numerical behavior
///
/// Exponentiation by squaring takes at most 32 iterations and constant memory.
///
/// # Examples
///
/// ```
/// use sci_complex::{self as complex, Complex};
/// let _ = complex::powi(Complex { re: 1.0, im: 1.0 }, 3);
/// ```
///
/// # References
///
/// [NIST DLMF §4](https://dlmf.nist.gov/4) and
/// [num-complex contracts](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).
pub fn powi<T: Float>(z: Complex<T>, exponent: i32) -> Complex<T> {
    powu(
        if exponent < 0 { reciprocal(z) } else { z },
        exponent.unsigned_abs(),
    )
}

/// Computes z^n for an unsigned integer exponent.
///
/// ## Parameters and conventions
/// `z` is Cartesian complex; `exponent` is an integer in `0..=u32::MAX`.
///
/// ## Domain and special values
/// All complex components are accepted with IEEE arithmetic. Any value raised
/// to zero, including zero and NaN, returns `1 + 0i`.
///
/// ## Returns
/// The complex integer power. There is no branch cut.
///
/// ## Numerical behavior
/// Uses Sci's multiplication with exponentiation by squaring, at most 32
/// steps and 63 multiplications. Overflow, underflow, and rounding apply.
///
/// ## Examples
/// ```
/// use sci_complex::{powu, Complex};
/// assert_eq!(powu(Complex { re: 0.0, im: 1.0 }, 4), Complex { re: 1.0, im: 0.0 });
/// ```
///
/// ## References
/// [num-complex integer powers](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html#method.powu).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn powu<T: Float>(z: Complex<T>, exponent: u32) -> Complex<T> {
    let mut n = exponent;
    let mut base = z;
    let mut value = Complex { re: 1.0, im: 0.0 };
    while n != 0 {
        if n & 1 != 0 {
            value = multiply(value, base);
        }
        n >>= 1;
        if n != 0 {
            base = multiply(base, base);
        }
    }
    value
}

/// Computes the principal complex cube root, |z|^(1/3) exp(i arg(z)/3).
///
/// ## Parameters and conventions
/// `z` is Cartesian complex. Arguments are radians with arg(z) in [−π, π].
///
/// ## Domain and special values
/// All components are accepted. The branch cut lies on the negative real axis;
/// signed imaginary zero selects its side. The root of −8 is 1 ± i√3.
/// Zero returns zero. Nonfinite inputs follow the private num-complex backend.
///
/// ## Returns
/// The principal root with argument in [−π/3, π/3].
///
/// ## Numerical behavior
/// Constant work. Finite off-axis magnitudes are scaled before taking a cube
/// root to avoid intermediate overflow and underflow. Axis cases use the
/// private backend's elementary formulas; rounding still limits accuracy.
///
/// ## Examples
/// ```
/// use sci_complex::{cbrt, Complex};
/// assert_eq!(cbrt(Complex { re: 8.0, im: 0.0 }), Complex { re: 2.0, im: 0.0 });
/// ```
///
/// ## References
/// [NIST DLMF §4.2](https://dlmf.nist.gov/4.2),
/// [num-complex cube root](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html#method.cbrt).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn cbrt<T: Float>(z: Complex<T>) -> Complex<T> {
    if !z.re.is_finite() || !z.im.is_finite() || z.re == 0.0 || z.im == 0.0 {
        return owned(native(z).cbrt());
    }
    let scale = z.re.abs().max(z.im.abs());
    let radius = scale.cbrt() * (z.re / scale).hypot(z.im / scale).cbrt();
    let (sin, cos) = (arg(z) / 3.0).sin_cos();
    Complex {
        re: radius * cos,
        im: radius * sin,
    }
}
