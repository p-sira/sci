//! Entropy and convex loss functions, ported from xsf/convex_analysis.h.
use crate::backend::StrErr;
use sci_core::Float;

/// Entropy -x*log(x). Returns 0 at zero and -infinity for negative x.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn entr<T: Float>(x: T) -> Result<T, StrErr> {
    if x.is_nan() {
        return Err("entr: Arguments cannot be NAN.");
    }
    if x > 0.0 {
        Ok(-x * x.ln())
    } else if x == 0.0 {
        Ok(0.0)
    } else {
        Ok(T::neg_infinity())
    }
}

/// Relative entropy x*log(x/y). Returns infinity outside x,y > 0,
/// except that x = 0 and y >= 0 gives zero.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn rel_entr<T: Float>(x: T, y: T) -> Result<T, StrErr> {
    if x.is_nan() || y.is_nan() {
        return Err("rel_entr: Arguments cannot be NAN.");
    }
    if x <= 0.0 || y <= 0.0 {
        return Ok(if x == 0.0 && y >= 0.0 {
            0.0
        } else {
            T::infinity()
        });
    }
    let ratio = x / y;
    if ratio > 0.5 && ratio < 2.0 {
        Ok(x * ((x - y) / y).ln_1p())
    } else if ratio > T::min_positive_value() && ratio.is_finite() {
        Ok(x * ratio.ln())
    } else {
        let r = x * (x.ln() - y.ln());
        if r.is_nan() {
            Err("rel_entr: Undefined result.")
        } else {
            Ok(r)
        }
    }
}

/// Kullback-Leibler divergence x*log(x/y) - x + y.
/// Returns infinity outside the convex function's effective domain.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn kl_div<T: Float>(x: T, y: T) -> Result<T, StrErr> {
    if x.is_nan() || y.is_nan() {
        return Err("kl_div: Arguments cannot be NAN.");
    }
    if x > 0.0 && y > 0.0 {
        let r = rel_entr(x, y)? - x + y;
        if r.is_nan() {
            Err("kl_div: Undefined result.")
        } else {
            Ok(r)
        }
    } else if x == 0.0 && y >= 0.0 {
        Ok(y)
    } else {
        Ok(T::infinity())
    }
}

/// Huber loss, quadratic for |r| <= delta and linear outside.
/// Negative delta gives infinity.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn huber<T: Float>(delta: T, r: T) -> Result<T, StrErr> {
    if delta.is_nan() || r.is_nan() {
        return Err("huber: Arguments cannot be NAN.");
    }
    let value = if delta < 0.0 {
        T::infinity()
    } else if r.abs() <= delta {
        0.5 * r * r
    } else {
        delta * (r.abs() - 0.5 * delta)
    };
    if value.is_nan() {
        Err("huber: Undefined result.")
    } else {
        Ok(value)
    }
}

/// Smooth pseudo-Huber loss delta²*(sqrt(1+(r/delta)²)-1).
/// Uses a rationalized expression to retain small residuals and avoid overflow.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn pseudo_huber<T: Float>(delta: T, r: T) -> Result<T, StrErr> {
    if delta.is_nan() || r.is_nan() {
        return Err("pseudo_huber: Arguments cannot be NAN.");
    }
    if delta < 0.0 {
        return Ok(T::infinity());
    }
    if delta == 0.0 || r == 0.0 {
        return Ok(0.0);
    }
    if delta.is_infinite() && r.is_infinite() {
        return Err("pseudo_huber: Undefined result.");
    }
    let r = r.abs();
    if r <= delta {
        let v = r / delta;
        Ok(r * (r / ((1.0 + v * v).sqrt() + 1.0)))
    } else {
        Ok(delta * (delta.hypot(r) - delta))
    }
}
