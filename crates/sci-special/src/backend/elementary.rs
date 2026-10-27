//! Stable elementary functions and transformations.
use crate::backend::StrErr;
use sci_core::Float;

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
