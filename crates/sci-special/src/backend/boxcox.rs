//! Box-Cox transformations, ported from xsf/boxcox.h.
use crate::backend::StrErr;
use sci_core::Float;

#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
fn transform<T: Float>(log_x: T, lambda: T) -> T {
    if lambda.abs() < 1e-19 {
        return log_x;
    }
    let power = lambda * log_x;
    if power < T::max_value().ln() {
        power.exp_m1() / lambda
    } else {
        T::one().copysign(lambda) * (power - lambda.abs().ln()).exp() - lambda.recip()
    }
}

/// (x^lambda-1)/lambda, with log(x) at lambda = 0. Requires x >= 0.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn boxcox<T: Float>(x: T, lambda: T) -> Result<T, StrErr> {
    if x.is_nan() || lambda.is_nan() || x < 0.0 {
        return Err("boxcox: Requires x >= 0 and non-NaN arguments.");
    }
    if x == 0.0 {
        return Ok(if lambda > 0.0 {
            -lambda.recip()
        } else {
            T::neg_infinity()
        });
    }
    let value = transform(x.ln(), lambda);
    if value.is_nan() {
        Err("boxcox: Undefined result.")
    } else {
        Ok(value)
    }
}

/// Box-Cox transformation of 1+x. Requires x >= -1; accurate near zero.
#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
pub fn boxcox1p<T: Float>(x: T, lambda: T) -> Result<T, StrErr> {
    if x.is_nan() || lambda.is_nan() || x < -1.0 {
        return Err("boxcox1p: Requires x >= -1 and non-NaN arguments.");
    }
    if x == -1.0 {
        return Ok(if lambda > 0.0 {
            -lambda.recip()
        } else {
            T::neg_infinity()
        });
    }
    let log_x = x.ln_1p();
    if log_x.abs() < 1e-289 && lambda.abs() < 1e273 {
        return Ok(log_x);
    }
    let value = transform(log_x, lambda);
    if value.is_nan() {
        Err("boxcox1p: Undefined result.")
    } else {
        Ok(value)
    }
}

#[numeric_literals::replace_float_literals(T::from(literal).unwrap())]
fn inverse_log<T: Float>(x: T, lambda: T) -> Result<T, StrErr> {
    if x.is_nan() || lambda.is_nan() {
        return Err("inv_boxcox: Arguments cannot be NAN.");
    }
    if lambda == 0.0 {
        return Ok(x);
    }
    let product = lambda * x;
    if product < -1.0 {
        return Err("inv_boxcox: Requires 1 + lambda*x >= 0.");
    }
    if product.is_finite() {
        Ok(product.ln_1p() / lambda)
    } else {
        let value =
            ((T::one().copysign(lambda) * (x + lambda.recip())).ln() + lambda.abs().ln()) / lambda;
        if value.is_nan() {
            Err("inv_boxcox: Undefined result.")
        } else {
            Ok(value)
        }
    }
}

/// Inverse Box-Cox transformation. Requires 1 + lambda*x >= 0.
pub fn inv_boxcox<T: Float>(x: T, lambda: T) -> Result<T, StrErr> {
    Ok(inverse_log(x, lambda)?.exp())
}

/// Inverse Box-Cox transformation minus one, accurate near zero.
pub fn inv_boxcox1p<T: Float>(x: T, lambda: T) -> Result<T, StrErr> {
    Ok(inverse_log(x, lambda)?.exp_m1())
}
