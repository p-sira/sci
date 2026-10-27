//! Stable elementary functions and transformations.
use crate::backend::StrErr;
use sci_core::Float;

/// Logistic sigmoid 1 / (1 + exp(-x)).
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn expit<T: Float>(x: T) -> Result<T, StrErr> {
    if x.is_nan() {
        return Err("expit: Arguments cannot be NAN.");
    }
    if x < 0.0 {
        let e = x.exp();
        Ok(e / (1.0 + e))
    } else {
        Ok(1.0 / (1.0 + (-x).exp()))
    }
}

/// Logit log(p/(1-p)), for 0 <= p <= 1; accurate near p = 1/2.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn logit<T: Float>(p: T) -> Result<T, StrErr> {
    if p.is_nan() {
        return Err("logit: Arguments cannot be NAN.");
    }
    if p < 0.0 || p > 1.0 {
        return Err("logit: Requires 0 <= p <= 1.");
    }
    if p < 0.3 || p > 0.65 {
        Ok((p / (1.0 - p)).ln())
    } else {
        let s = 2.0 * (p - 0.5);
        Ok(s.ln_1p() - (-s).ln_1p())
    }
}

/// Natural logarithm of the logistic sigmoid, accurate in both tails.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn log_expit<T: Float>(x: T) -> Result<T, StrErr> {
    if x.is_nan() {
        return Err("log_expit: Arguments cannot be NAN.");
    }
    if x < 0.0 {
        Ok(x - x.exp().ln_1p())
    } else {
        Ok(-(-x).exp().ln_1p())
    }
}

/// Relative exponential (exp(x)-1)/x, with value 1 at x = 0.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn exprel<T: Float>(x: T) -> Result<T, StrErr> {
    if x.is_nan() {
        return Err("exprel: Arguments cannot be NAN.");
    }
    if x.abs() < T::epsilon() {
        return Ok(1.0);
    }
    if x == T::infinity() {
        return Ok(x);
    }
    // Division after exponentiation would overflow prematurely.
    if x > 1.0 && x > T::max_value().ln() {
        return Ok((x - x.ln()).exp() - x.recip());
    }
    Ok(x.exp_m1() / x)
}

/// log(1-exp(x)), for x <= 0, accurate near zero.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn log1mexp<T: Float>(x: T) -> Result<T, StrErr> {
    if x.is_nan() || x > 0.0 {
        return Err("log1mexp: Requires x <= 0.");
    }
    if x < -1.0 {
        Ok((-x.exp()).ln_1p())
    } else {
        Ok((-x.exp_m1()).ln())
    }
}

/// x*log(y), with value zero when x = 0 and y is not NaN.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn xlogy<T: Float>(x: T, y: T) -> Result<T, StrErr> {
    if x.is_nan() || y.is_nan() {
        return Err("xlogy: Arguments cannot be NAN.");
    }
    if x == 0.0 {
        return Ok(0.0);
    }
    if y < 0.0 {
        return Err("xlogy: Requires y >= 0 when x != 0.");
    }
    Ok(x * y.ln())
}

/// x*log(1+y), with value zero when x = 0 and y is not NaN.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn xlog1py<T: Float>(x: T, y: T) -> Result<T, StrErr> {
    if x.is_nan() || y.is_nan() {
        return Err("xlog1py: Arguments cannot be NAN.");
    }
    if x == 0.0 {
        return Ok(0.0);
    }
    if y < -1.0 {
        return Err("xlog1py: Requires y >= -1 when x != 0.");
    }
    Ok(x * y.ln_1p())
}

/// Arithmetic-geometric mean; both arguments must have the same sign.
/// Returns zero if either finite argument is zero.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn agm<T: Float>(a: T, b: T) -> Result<T, StrErr> {
    if a.is_nan() || b.is_nan() {
        return Err("agm: Arguments cannot be NAN.");
    }
    if (a < 0.0 && b > 0.0) || (a > 0.0 && b < 0.0) {
        return Err("agm: Arguments must have the same sign.");
    }
    if (a.is_infinite() && b == 0.0) || (b.is_infinite() && a == 0.0) {
        return Err("agm: Zero and infinity have no defined mean.");
    }
    if a == b {
        return Ok(a);
    }
    if a == 0.0 || b == 0.0 {
        return Ok(0.0);
    }
    if a.is_infinite() || b.is_infinite() {
        return Ok(T::infinity().copysign(a));
    }
    let sign = T::one().copysign(a);
    let (mut a, mut b) = (a.abs(), b.abs());
    for _ in 0..64 {
        let mean = 0.5 * a + 0.5 * b;
        if mean == a || mean == b {
            return Ok(sign * mean);
        }
        b = a.sqrt() * b.sqrt();
        a = mean;
    }
    Err("agm: Iteration did not converge.")
}

/// sin(pi*x), with argument reduction before multiplication by pi.
/// Integer arguments are exact zeros; infinity is outside the domain.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn sinpi<T: Float>(x: T) -> Result<T, StrErr> {
    if !x.is_finite() {
        return Err("sinpi: Argument must be finite.");
    }
    let r = x.abs() % 2.0;
    let pi = T::from(core::f64::consts::PI).unwrap();
    let s = T::one().copysign(x);
    if r < 0.5 {
        Ok(s * (pi * r).sin())
    } else if r > 1.5 {
        Ok(s * (pi * (r - 2.0)).sin())
    } else {
        Ok(-s * (pi * (r - 1.0)).sin())
    }
}

/// cos(pi*x), with exact zeros at representable half-integers.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn cospi<T: Float>(x: T) -> Result<T, StrErr> {
    if !x.is_finite() {
        return Err("cospi: Argument must be finite.");
    }
    let r = x.abs() % 2.0;
    let pi = T::from(core::f64::consts::PI).unwrap();
    if r == 0.5 {
        Ok(0.0)
    } else if r < 1.0 {
        Ok(-(pi * (r - 0.5)).sin())
    } else {
        Ok((pi * (r - 1.5)).sin())
    }
}

/// Sine of an angle in degrees, with exact zeros at multiples of 180.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn sindg<T: Float>(x: T) -> Result<T, StrErr> {
    sinpi((x % 360.0) / 180.0)
}

/// Cosine of an angle in degrees, with exact zeros at odd multiples of 90.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn cosdg<T: Float>(x: T) -> Result<T, StrErr> {
    cospi((x % 360.0) / 180.0)
}

/// Tangent of an angle in degrees. Poles at 90 + 180*k return infinity.
pub fn tandg<T: Float>(x: T) -> Result<T, StrErr> {
    Ok(sindg(x)? / cosdg(x)?)
}

/// Cotangent of an angle in degrees. Integer multiples of 180 are poles.
pub fn cotdg<T: Float>(x: T) -> Result<T, StrErr> {
    Ok(cosdg(x)? / sindg(x)?)
}

/// Convert degrees, minutes, and seconds to radians.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn radian<T: Float>(degrees: T, minutes: T, seconds: T) -> Result<T, StrErr> {
    if degrees.is_nan() || minutes.is_nan() || seconds.is_nan() {
        return Err("radian: Arguments cannot be NAN.");
    }
    let value = (degrees + minutes / 60.0 + seconds / 3600.0)
        * (T::from(core::f64::consts::PI).unwrap() / 180.0);
    if value.is_nan() {
        Err("radian: Undefined result.")
    } else {
        Ok(value)
    }
}

/// cos(x)-1, computed accurately near zero.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn cosm1<T: Float>(x: T) -> Result<T, StrErr> {
    if !x.is_finite() {
        return Err("cosm1: Argument must be finite.");
    }
    if x.abs() < 1.0 {
        let s = (x / 2.0).sin();
        Ok(-2.0 * s * s)
    } else {
        Ok(x.cos() - 1.0)
    }
}

macro_rules! elementary_unary {
    ($($name:ident, $method:ident, $doc:literal;)+) => {$(
        #[doc = $doc]
        pub fn $name<T: Float>(x: T) -> Result<T, StrErr> {
            if x.is_nan() { return Err(concat!(stringify!($name), ": Arguments cannot be NAN.")); }
            let value = x.$method();
            if value.is_nan() { Err(concat!(stringify!($name), ": Argument outside the real domain.")) } else { Ok(value) }
        }
    )+};
}
elementary_unary! {
    log1p, ln_1p, "Natural logarithm of 1+x, for x >= -1, accurate near zero.";
    expm1, exp_m1, "exp(x)-1, accurate near zero.";
    exp2, exp2, "Base-two exponential.";
    cbrt, cbrt, "Real cube root.";
}

/// Base-ten exponential 10^x.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn exp10<T: Float>(x: T) -> Result<T, StrErr> {
    if x.is_nan() {
        return Err("exp10: Arguments cannot be NAN.");
    }
    Ok(T::from(10).unwrap().powf(x))
}

/// Round to the nearest integer, with halfway cases rounded away from zero.
pub fn round<T: Float>(x: T) -> Result<T, StrErr> {
    if x.is_nan() {
        return Err("round: Arguments cannot be NAN.");
    }
    Ok(x.round())
}
