//! Kelvin functions expressed through modified Bessel functions at rotated arguments.
use crate::backend::{StrErr, bessel::backend_error, complex::finish};
use core::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_4};
use num_complex::{Complex, Complex64};
#[allow(unused_imports)]
use sci_core::Float;

/// Kelvin values (ber + i bei, ker + i kei, ber′ + i bei′, ker′ + i kei′).
pub type Kelvin<T> = (Complex<T>, Complex<T>, Complex<T>, Complex<T>);

// Evaluate both the ordinary and harmonic-weighted I_0 series. Keeping the
// derivative series separate preserves its leading term when x*x underflows.
fn small(x: f64) -> Kelvin<f64> {
    if x < 1e-150 {
        let logarithm = x.ln() - core::f64::consts::LN_2 + 0.5772156649015328606;
        return (
            Complex64::new(1.0, (x * 0.5) * (x * 0.5)),
            Complex64::new(-logarithm, -FRAC_PI_4),
            Complex64::new(-((x * 0.25) * x) * (x * 0.25), x * 0.5),
            Complex64::new(-1.0 / x, x * (0.25 - 0.5 * logarithm)),
        );
    }
    let q = Complex64::new(0.0, (x * 0.5) * (x * 0.5));
    let mut term = Complex64::new(1.0, 0.0);
    let mut derivative = Complex64::new(0.0, x * 0.5);
    let mut be = term;
    let mut bep = derivative;
    let mut harmonic = 0.0;
    let mut weighted = Complex64::new(0.0, 0.0);
    let mut weighted_derivative = derivative;
    for k in 1..=32 {
        let k = k as f64;
        harmonic += 1.0 / k;
        term *= q / (k * k);
        be += term;
        weighted += term * harmonic;
        derivative *= q / (k * (k + 1.0));
        bep += derivative;
        weighted_derivative += derivative * (harmonic + 1.0 / (k + 1.0));
        if term.norm() <= f64::EPSILON * be.norm() && derivative.norm() <= f64::EPSILON * bep.norm()
        {
            break;
        }
    }
    let logarithm = Complex64::new(
        x.ln() - core::f64::consts::LN_2 + 0.5772156649015328606,
        FRAC_PI_4,
    );
    let ke = weighted - logarithm * be;
    let kep = weighted_derivative - logarithm * bep - Complex64::new(be.re / x, be.im / x);
    (be, ke, bep, kep)
}

fn argument<T: Float>(x: T) -> Result<f64, StrErr> {
    let x = x.to_f64().ok_or("kelvin: Invalid argument.")?;
    if !x.is_finite() {
        return Err("kelvin: Argument must be finite.");
    }
    Ok(x)
}

/// Which of the value and derivative a caller needs from [`pair`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum Need {
    Both,
    /// Both, from two single-order calls that avoid heap allocation.
    BothInPlace,
    Value,
    Derivative,
}

/// Order-zero and order-one AMOS values at `z`, computing only what `need`
/// requires (unused entries are NaN). Both orders come from one call.
fn orders(z: Complex64, second: bool, need: Need) -> Result<(Complex64, Complex64), StrErr> {
    let nan = Complex64::new(f64::NAN, f64::NAN);
    let scaling = complex_bessel::Scaling::Exponential;
    let single = |order: f64| {
        if second {
            complex_bessel::besselk_scaled(order, z)
        } else {
            complex_bessel::besseli_scaled(order, z)
        }
        .map_err(backend_error)
    };
    Ok(match need {
        Need::Both => {
            let values = if second {
                complex_bessel::besselk_seq(0.0, z, 2, scaling)
            } else {
                complex_bessel::besseli_seq(0.0, z, 2, scaling)
            }
            .map_err(backend_error)?
            .values;
            (values[0], values[1])
        }
        Need::BothInPlace => (single(0.0)?, single(1.0)?),
        Need::Value => (single(0.0)?, nan),
        Need::Derivative => (nan, single(1.0)?),
    })
}

fn pair(x: f64, second: bool, scaled: bool) -> Result<(Complex64, Complex64), StrErr> {
    pair_needed(x, second, scaled, Need::Both)
}

fn pair_needed(
    x: f64,
    second: bool,
    scaled: bool,
    need: Need,
) -> Result<(Complex64, Complex64), StrErr> {
    if second && x < 0.0 {
        return Err("kelvin: ker and kei require x >= 0.");
    }
    let a = x.abs();
    let (value, mut derivative) = if a == 0.0 {
        if second {
            (
                Complex64::new(f64::INFINITY, -FRAC_PI_4),
                Complex64::new(f64::NEG_INFINITY, 0.0),
            )
        } else {
            (Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0))
        }
    } else if a < 0.5 {
        let (be, ke, bep, kep) = small(a);
        if second { (ke, kep) } else { (be, bep) }
    } else {
        let rotation = Complex64::new(FRAC_1_SQRT_2, FRAC_1_SQRT_2);
        let z = rotation * a;
        if second {
            let scale = Complex64::new(if scaled { 0.0 } else { -z.re }, -z.im).exp();
            let (k0, k1) = orders(z, true, need)?;
            let value = k0 * scale;
            let derivative = -rotation * k1 * scale;
            (value, derivative)
        } else {
            let half_scale = if scaled { 1.0 } else { (z.re * 0.5).exp() };
            let restore = |z: Complex64| {
                Complex64::new(
                    (z.re * half_scale) * half_scale,
                    (z.im * half_scale) * half_scale,
                )
            };
            let (i0, i1) = orders(z, false, need)?;
            let value = restore(i0);
            let derivative = restore(rotation * i1);
            (value, derivative)
        }
    };
    if !second && x.is_sign_negative() {
        derivative = -derivative;
    }
    Ok((value, derivative))
}

/// All four complex Kelvin pairs. Requires finite x >= 0.
///
/// At zero, ker and ker′ return their infinite limits. Backend precision
/// failures are reported as errors, as for the complex Bessel functions.
pub fn kelvin<T: Float>(x: T) -> Result<Kelvin<T>, StrErr> {
    let x = argument(x)?;
    let (be, bep) = pair(x, false, false)?;
    let (ke, kep) = pair(x, true, false)?;
    Ok((finish(be)?, finish(ke)?, finish(bep)?, finish(kep)?))
}

macro_rules! kelvin_component {
    ($($name:ident, $second:literal, $index:tt, $component:ident, $doc:literal;)+) => {$(
        #[doc = $doc]
        pub fn $name<T: Float>(x: T) -> Result<T, StrErr> {
            let need = if $index == 0 { Need::Value } else { Need::Derivative };
            let value = pair_needed(argument(x)?, $second, false, need)?.$index.$component;
            crate::backend::utils::finish(value, concat!(stringify!($name), ": Undefined result."))
        }
    )+};
}
kelvin_component! {
    ber, false, 0, re, "Kelvin ber(x), even in x; requires a finite argument.";
    bei, false, 0, im, "Kelvin bei(x), even in x; requires a finite argument.";
    berp, false, 1, re, "Derivative of Kelvin ber(x), odd in x; requires a finite argument.";
    beip, false, 1, im, "Derivative of Kelvin bei(x), odd in x; requires a finite argument.";
    ker, true, 0, re, "Kelvin ker(x), for finite x >= 0; ker(0) = infinity.";
    kei, true, 0, im, "Kelvin kei(x), for finite x >= 0; kei(0) = -pi/4.";
    kerp, true, 1, re, "Derivative of Kelvin ker(x), for finite x >= 0; ker′(0) = -infinity.";
    keip, true, 1, im, "Derivative of Kelvin kei(x), for finite x >= 0; kei′(0) = 0.";
}

/// Select a Kelvin function or its first derivative for zero finding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KelvinFunction {
    /// ber(x).
    Ber,
    /// bei(x).
    Bei,
    /// ker(x).
    Ker,
    /// kei(x).
    Kei,
    /// ber′(x).
    BerPrime,
    /// bei′(x).
    BeiPrime,
    /// ker′(x).
    KerPrime,
    /// kei′(x).
    KeiPrime,
}

impl KelvinFunction {
    /// Scaled value alone, for bracketing; one Bessel evaluation.
    fn value(self, x: f64) -> Result<f64, StrErr> {
        use KelvinFunction::*;
        let second = matches!(self, Ker | Kei | KerPrime | KeiPrime);
        let prime = matches!(self, BerPrime | BeiPrime | KerPrime | KeiPrime);
        let imaginary = matches!(self, Bei | Kei | BeiPrime | KeiPrime);
        let need = if prime { Need::Derivative } else { Need::Value };
        let (value, derivative) = pair_needed(x, second, true, need)?;
        let value = if prime { derivative } else { value };
        Ok(if imaginary { value.im } else { value.re })
    }

    fn value_and_slope(self, x: f64) -> Result<(f64, f64), StrErr> {
        use KelvinFunction::*;
        let second = matches!(self, Ker | Kei | KerPrime | KeiPrime);
        let prime = matches!(self, BerPrime | BeiPrime | KerPrime | KeiPrime);
        let imaginary = matches!(self, Bei | Kei | BeiPrime | KeiPrime);
        // Scale value and actual derivative equally, retaining their Newton
        // ratio even beyond the unscaled overflow/underflow thresholds.
        let (value, derivative) = pair_needed(x, second, true, Need::BothInPlace)?;
        let (value, derivative) = if prime {
            (
                derivative,
                Complex64::new(-value.im, value.re) - derivative / x,
            )
        } else {
            (value, derivative)
        };
        Ok(if imaginary {
            (value.im, derivative.im)
        } else {
            (value.re, derivative.re)
        })
    }
}

/// Fill a caller-owned slice with successive positive Kelvin zeros.
///
/// The origin is excluded. Uses a bracketed Newton iteration and scaled Bessel
/// kernels without allocating. On failure, earlier entries remain populated.
/// An empty slice succeeds without evaluating any functions.
pub fn kelvin_zeros<T: Float>(function: KelvinFunction, zeros: &mut [T]) -> Result<(), StrErr> {
    let mut left = 0.5;
    for zero in zeros {
        let mut fl = function.value(left)?;
        let mut right = left + 1.0;
        let mut fr = f64::NAN;
        let mut bracketed = false;
        for _ in 0..32 {
            fr = function.value(right)?;
            if fl.is_sign_negative() != fr.is_sign_negative() || fr == 0.0 {
                bracketed = true;
                break;
            }
            left = right;
            fl = fr;
            right += 1.0;
        }
        if !bracketed {
            return Err("kelvin_zeros: Could not bracket a root.");
        }
        let x = crate::backend::utils::bracketed_root_from(left, fl, right, fr, |x| {
            function.value_and_slope(x)
        })?;
        *zero = T::from(x).ok_or("kelvin_zeros: Cannot represent root.")?;
        // Consecutive positive zeros of every Kelvin function and derivative
        // are at least 4.3 apart (asymptotically pi*sqrt(2)), so the next
        // scan can start 3.5 beyond this zero without skipping one.
        left = x + 3.5;
    }
    Ok(())
}
