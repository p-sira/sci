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

/// Brent's method (Brent 1973; Numerical Recipes `zbrent`) for a root of `f`
/// in `[a, b]`, given the residuals `fa = f(a)` and `fb = f(b)` of opposite
/// sign (or zero). Converges to within a few ulps of the sign change.
/// Returns `None` if a residual is not finite or 200 iterations do not suffice.
pub(crate) fn brent_root(
    mut a: f64,
    mut b: f64,
    mut fa: f64,
    mut fb: f64,
    mut f: impl FnMut(f64) -> f64,
) -> Option<f64> {
    if !fa.is_finite() || !fb.is_finite() || ((fa > 0.0) == (fb > 0.0) && fa != 0.0 && fb != 0.0) {
        return None;
    }
    if fa == 0.0 {
        return Some(a);
    }
    let (mut c, mut fc) = (b, fb);
    let (mut d, mut e) = (b - a, b - a);
    for _ in 0..200 {
        if (fb > 0.0) == (fc > 0.0) {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol = 2.0 * f64::EPSILON * b.abs() + f64::MIN_POSITIVE;
        let xm = 0.5 * (c - b);
        if xm.abs() <= tol || fb == 0.0 {
            return Some(b);
        }
        if e.abs() >= tol && fa.abs() > fb.abs() {
            // Inverse quadratic interpolation, or secant when a == c.
            let s = fb / fa;
            let (mut p, mut q) = if a == c {
                (2.0 * xm * s, 1.0 - s)
            } else {
                let q = fa / fc;
                let r = fb / fc;
                (
                    s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0)),
                    (q - 1.0) * (r - 1.0) * (s - 1.0),
                )
            };
            if p > 0.0 {
                q = -q;
            }
            p = p.abs();
            if 2.0 * p < (3.0 * xm * q - (tol * q).abs()).min((e * q).abs()) {
                e = d;
                d = p / q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }
        a = b;
        fa = fb;
        b += if d.abs() > tol { d } else { tol.copysign(xm) };
        fb = f(b);
        if !fb.is_finite() {
            return None;
        }
    }
    None
}
