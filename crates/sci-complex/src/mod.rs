use crate::{Error, Result};

/// A complex value `re + i im`, shared across Sci modules.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    /// Real component.
    pub re: f64,
    /// Imaginary component.
    pub im: f64,
}

pub(crate) fn native(z: Complex) -> num_complex::Complex64 {
    num_complex::Complex64::new(z.re, z.im)
}

pub(crate) fn owned(z: num_complex::Complex64) -> Complex {
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
/// Constant work using `f64` arithmetic; rounding, overflow, and underflow apply.
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
pub fn add(a: Complex, b: Complex) -> Complex {
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
/// Constant work using `f64` arithmetic; rounding, overflow, and underflow apply.
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
pub fn subtract(a: Complex, b: Complex) -> Complex {
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
pub fn multiply(a: Complex, b: Complex) -> Complex {
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
fn binary_parts(value: f64) -> (f64, i32) {
    if value == 0.0 {
        return (value, 0);
    }
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    if exponent == 0 {
        let (mantissa, exponent) = binary_parts(value * (1_u64 << 54) as f64);
        return (mantissa, exponent - 54);
    }
    let mantissa = f64::from_bits((bits & 0x800f_ffff_ffff_ffff) | (1023_u64 << 52));
    (mantissa, exponent - 1023)
}

fn restore_binary_scale(mut value: f64, mut exponent: i32) -> f64 {
    while exponent > 1023 {
        value *= 2.0_f64.powi(1023);
        exponent -= 1023;
    }
    while exponent < -1022 {
        value *= f64::MIN_POSITIVE;
        exponent += 1022;
    }
    value * 2.0_f64.powi(exponent)
}

// Two products are combined before restoring their shared exponent. FMA
// residuals preserve cancellation that separate rounded products would lose.
fn product_sum(a: f64, b: f64, c: f64, d: f64) -> f64 {
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
    a *= 2.0_f64.powi(ae + be - exponent);
    c *= 2.0_f64.powi(ce + de - exponent);
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
pub fn divide(a: Complex, b: Complex) -> Complex {
    let scale = b.re.abs().max(b.im.abs());
    if scale == 0.0 {
        return Complex {
            re: f64::NAN,
            im: f64::NAN,
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
/// Constant work using `f64` arithmetic; rounding, overflow, and underflow apply.
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
pub fn negate(z: Complex) -> Complex {
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
/// Constant work using `f64` arithmetic; rounding, overflow, and underflow apply.
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
pub fn conjugate(z: Complex) -> Complex {
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
/// Constant work using `f64` arithmetic; rounding, overflow, and underflow apply.
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
pub fn reciprocal(z: Complex) -> Complex {
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
pub fn real(z: Complex) -> f64 {
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
pub fn imag(z: Complex) -> f64 {
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
pub fn abs(z: Complex) -> f64 {
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
/// zero. Binary64 rounding remains; correct rounding is not guaranteed.
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
pub fn norm_sqr(z: Complex) -> f64 {
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
pub fn arg(z: Complex) -> f64 {
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
/// Constant work using `f64` arithmetic; rounding, overflow, and underflow apply.
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
pub fn to_polar(z: Complex) -> (f64, f64) {
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
pub fn from_polar(radius: f64, phase: f64) -> Result<Complex> {
    if !radius.is_finite() || radius < 0.0 || !phase.is_finite() {
        return Err(Error::InvalidInput(
            "polar radius must be finite and nonnegative; phase must be finite",
        ));
    }
    Ok(owned(num_complex::Complex64::from_polar(radius, phase)))
}

// Multiply a bounded trigonometric coefficient by exp(x), optionally halved,
// without overflowing the exponential first. Powers of exp(512) have exact
// exponent offsets; multiply the small coefficient before restoring the scale.
fn exponential_component(x: f64, coefficient: f64, half: bool) -> f64 {
    if coefficient == 0.0 {
        return coefficient;
    }
    if x > 1500.0 {
        // Even the smallest nonzero binary64 coefficient must overflow here.
        return f64::INFINITY.copysign(coefficient);
    }
    let chunk = 512.0_f64.exp();
    let mut value = coefficient * chunk;
    if half {
        value *= 0.5;
    }
    if x <= 1024.0 {
        value * (x - 512.0).exp()
    } else {
        (value * (x - 1024.0).exp()) * chunk
    }
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
pub fn exp(z: Complex) -> Complex {
    if z.re.is_finite() && z.im.is_finite() && z.re > 700.0 {
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
pub fn ln(z: Complex) -> Complex {
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
/// Binary64 rounding and subnormal underflow remain.
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
pub fn sqrt(z: Complex) -> Complex {
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
pub fn sin(z: Complex) -> Complex {
    if z.re.is_finite() && z.im.is_finite() && z.im.abs() > 700.0 {
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
pub fn cos(z: Complex) -> Complex {
    if z.re.is_finite() && z.im.is_finite() && z.im.abs() > 700.0 {
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
pub fn sinh(z: Complex) -> Complex {
    if z.re.is_finite() && z.im.is_finite() && z.re.abs() > 700.0 {
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
pub fn cosh(z: Complex) -> Complex {
    if z.re.is_finite() && z.im.is_finite() && z.re.abs() > 700.0 {
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
pub fn tan(z: Complex) -> Complex {
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
pub fn tanh(z: Complex) -> Complex {
    let t = tan(Complex {
        re: -z.im,
        im: z.re,
    });
    Complex {
        re: t.im,
        im: -t.re,
    }
}

// First-quadrant inverse geometry. Rationalize hypot differences before
// evaluating A-1 and A-x; direct subtraction loses the small branch component.
fn inverse_geometry(x: f64, y: f64) -> (f64, f64) {
    if x.max(y) > 1e150 {
        let scale = x.max(y);
        let height = scale.ln() + (x / scale).hypot(y / scale).ln() + std::f64::consts::LN_2;
        return (y, height);
    }
    if y == 0.0 {
        return if x <= 1.0 {
            (((1.0 - x) * (1.0 + x)).sqrt(), 0.0)
        } else {
            (0.0, x.acosh())
        };
    }
    // Away from the branch point, first-order limits preserve subnormals.
    if y < 1e-150 && x != 1.0 {
        let root = (x - 1.0).abs().sqrt() * (x + 1.0).sqrt();
        return if x < 1.0 {
            (root, y / root)
        } else {
            ((y / root) * x, x.acosh())
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
/// intermediate overflow. Binary64 rounding and subnormal underflow remain.
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
pub fn asin(z: Complex) -> Complex {
    if z.im == 0.0 {
        if z.re.abs() <= 1.0 {
            return Complex {
                re: z.re.asin(),
                im: z.im,
            };
        }
        return Complex {
            re: std::f64::consts::FRAC_PI_2.copysign(z.re),
            im: z.re.abs().acosh().copysign(z.im),
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
/// intermediate overflow. Binary64 rounding and subnormal underflow remain.
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
pub fn acos(z: Complex) -> Complex {
    if !z.re.is_finite() || !z.im.is_finite() {
        return owned(native(z).acos());
    }
    let (distance, height) = inverse_geometry(z.re.abs(), z.im.abs());
    let angle = distance.atan2(z.re.abs());
    Complex {
        re: if z.re < 0.0 {
            std::f64::consts::PI - angle
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
/// intermediate overflow. Binary64 rounding and subnormal underflow remain.
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
pub fn atan(z: Complex) -> Complex {
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
/// intermediate overflow. Binary64 rounding and subnormal underflow remain.
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
pub fn asinh(z: Complex) -> Complex {
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
/// intermediate overflow. Binary64 rounding and subnormal underflow remain.
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
pub fn acosh(z: Complex) -> Complex {
    if z.im == 0.0 {
        if z.re >= 1.0 {
            return Complex {
                re: z.re.acosh(),
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
            re: (-z.re).acosh(),
            im: std::f64::consts::PI.copysign(z.im),
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
/// intermediate overflow. Binary64 rounding and subnormal underflow remain.
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
pub fn atanh(z: Complex) -> Complex {
    if z.im == 0.0 {
        let re = if z.re.abs() <= 1.0 {
            z.re.atanh()
        } else {
            (1.0 / z.re).atanh()
        };
        let im = if z.re.abs() <= 1.0 {
            z.im
        } else {
            std::f64::consts::FRAC_PI_2.copysign(z.im)
        };
        return Complex { re, im };
    }
    if !z.re.is_finite() || !z.im.is_finite() {
        return owned(native(z).atanh());
    }
    let x = z.re.abs();
    let y = z.im.abs();
    let scale = x.max(y);
    let (re, im) = if scale > 1e150 {
        // atanh(z) ~ 1/z + sign(im) i pi/2; scale before squaring.
        let a = x / scale;
        let b = y / scale;
        ((a / (a * a + b * b)) / scale, std::f64::consts::FRAC_PI_2)
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
pub fn powf(z: Complex, exponent: f64) -> Complex {
    if exponent == 0.0 {
        return Complex { re: 1.0, im: 0.0 };
    }
    if z.re == 0.0 && z.im == 0.0 && exponent < 0.0 {
        return Complex {
            re: f64::NAN,
            im: f64::NAN,
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
        if z.re > 0.0 && z.im.abs() <= f64::EPSILON * z.re / exponent.abs().max(1.0) {
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
pub fn powc(z: Complex, exponent: Complex) -> Complex {
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
                re: f64::NAN,
                im: f64::NAN,
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
pub fn powi(z: Complex, exponent: i32) -> Complex {
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
pub fn powu(z: Complex, exponent: u32) -> Complex {
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
pub fn cbrt(z: Complex) -> Complex {
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
