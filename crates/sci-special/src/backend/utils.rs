use crate::backend::StrErr;
use sci_core::Float;

pub(crate) fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let (mantissa, exponent, sign) = Float::integer_decode(x);
    let shift = 64 - mantissa.leading_zeros() as i32;
    (
        sign as f64 * mantissa as f64 / (2.0_f64).powi(shift),
        exponent as i32 + shift,
    )
}

pub(crate) fn finish<T: Float>(value: f64, error: StrErr) -> Result<T, StrErr> {
    if value.is_nan() {
        Err(error)
    } else {
        T::from(value).ok_or("Unsupported floating-point representation.")
    }
}

/// Define checked real APIs over double-precision kernels, matching xsf's float promotion.
macro_rules! real_kernel {
    ($(#[$meta:meta])* $name:ident($($arg:ident),+) $body:block) => {
        $(#[$meta])*
        pub fn $name<T: sci_core::Float>($($arg: T),+) -> Result<T, crate::backend::StrErr> {
            $(if $arg.is_nan() { return Err(concat!(stringify!($name), ": Arguments cannot be NAN.")); }
            let $arg = $arg.to_f64().ok_or(concat!(stringify!($name), ": Cannot represent argument as f64."))?;)+
            let value: f64 = $body;
            crate::backend::utils::finish(value, concat!(stringify!($name), ": Result is undefined in the real domain."))
        }
    };
}
pub(crate) use real_kernel;

/// Safeguarded Newton iteration for a known sign-changing bracket.
pub(crate) fn bracketed_root(
    mut left: f64,
    mut right: f64,
    mut evaluate: impl FnMut(f64) -> Result<(f64, f64), StrErr>,
) -> Result<f64, StrErr> {
    let mut fl = evaluate(left)?.0;
    let fr = evaluate(right)?.0;
    if fl == 0.0 {
        return Ok(left);
    }
    if fr == 0.0 {
        return Ok(right);
    }
    if !fl.is_finite() || !fr.is_finite() || fl.is_sign_negative() == fr.is_sign_negative() {
        return Err("Zero finder requires a finite sign-changing bracket.");
    }
    let mut x = 0.5 * (left + right);
    for _ in 0..64 {
        let (value, slope) = evaluate(x)?;
        if !value.is_finite() {
            return Err("Zero finder encountered a nonfinite value.");
        }
        if value == 0.0 || right - left <= 4.0 * f64::EPSILON * x.abs().max(1.0) {
            return Ok(x);
        }
        if fl.is_sign_negative() != value.is_sign_negative() {
            right = x;
        } else {
            left = x;
            fl = value;
        }
        let next = x - value / slope;
        x = if next > left && next < right {
            next
        } else {
            0.5 * (left + right)
        };
    }
    Err("Zero finder did not converge.")
}
