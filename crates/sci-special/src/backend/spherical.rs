//! Spherical Bessel functions and derivatives, with xsf naming.
use crate::backend::{StrErr, utils::finish};
use num_complex::Complex64;
use sci_core::Float;

fn series(n: i32, x: f64, modified: bool) -> f64 {
    let mut leading = 1.0;
    // Early underflow terminates this product even for very large orders.
    for k in 1..=n {
        leading *= x / (2.0 * k as f64 + 1.0);
        if leading == 0.0 {
            return 0.0;
        }
    }
    let mut sum = 1.0;
    let mut term = 1.0;
    let square = if modified { x * x } else { -x * x };
    for k in 1..10000 {
        term *= square / (2.0 * k as f64 * (2.0 * n as f64 + 2.0 * k as f64 + 1.0));
        sum += term;
        if term.abs() <= f64::EPSILON * sum.abs() {
            return leading * sum;
        }
    }
    f64::NAN
}
fn real(n: i32, x: f64, kind: u8) -> Result<f64, StrErr> {
    if n < 0 || x.is_nan() {
        return Err("sph_bessel: Requires n >= 0 and a non-NaN argument.");
    }
    if kind == 3 && x < 0.0 {
        if x == f64::NEG_INFINITY {
            return Ok(f64::NEG_INFINITY);
        }
        if n == 1 && x == -1.0 {
            return Ok(0.0);
        }
        return Ok(-(core::f64::consts::PI * real(n, -x, 2)?
            + if n % 2 == 0 {
                real(n, -x, 3)?
            } else {
                -real(n, -x, 3)?
            }));
    }
    if x == 0.0 {
        return Ok(match kind {
            0 | 2 => {
                if n == 0 {
                    1.0
                } else {
                    0.0
                }
            }
            1 => f64::NEG_INFINITY,
            _ => f64::INFINITY,
        });
    }
    if x.is_infinite() {
        return Ok(if kind == 2 {
            if x.is_sign_negative() && n % 2 != 0 {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            }
        } else if x.is_sign_negative() && ((kind == 1 && n % 2 == 0) || (kind == 0 && n % 2 == 1)) {
            -0.0
        } else {
            0.0
        });
    }
    let sign = if x < 0.0 && ((kind == 1 && n % 2 == 0) || (kind != 1 && n % 2 != 0)) {
        -1.0
    } else {
        1.0
    };
    let x = x.abs();
    if (kind == 0 || kind == 2) && x < 0.5 {
        return Ok(sign * series(n, x, kind == 2));
    }
    if n == 0 {
        return Ok(sign
            * match kind {
                0 => x.sin() / x,
                1 => -x.cos() / x,
                2 => 0.5 * ((x - x.ln()).exp() - (-x - x.ln()).exp()),
                _ => core::f64::consts::FRAC_PI_2 * (-x).exp() / x,
            });
    }
    // Elementary recurrences avoid the general half-integer AMOS kernels for
    // the common moderate-order cases, as xsf's spherical kernels do.
    match kind {
        0 if x > n as f64 => return Ok(sign * upward(n, x, false)),
        1 => return Ok(sign * upward(n, x, true)),
        0 | 2 if n <= 256 && x <= 700.0 => {
            if let Some(value) = downward(n, x, kind == 2) {
                return Ok(sign * value);
            }
        }
        _ => {}
    }
    let order = n as f64 + 0.5;
    let z = Complex64::new(x, 0.0);
    let result = match kind {
        0 => complex_bessel::besselj(order, z),
        1 => complex_bessel::bessely(order, z),
        2 => {
            let value = crate::backend::cyl_bessel_ie(order, x)?;
            let prefactor = (core::f64::consts::FRAC_PI_2 / x).sqrt();
            let e = (0.5 * x).exp();
            return Ok(sign * (prefactor * value * e) * e);
        }
        _ => complex_bessel::besselk(order, z),
    }
    .map_err(crate::backend::bessel::backend_error)?;
    Ok(sign * (core::f64::consts::FRAC_PI_2 / x).sqrt() * result.re)
}
/// Upward recurrence from the elementary j_0, j_1 or y_0, y_1 values. It is
/// stable for y_n at every x > 0 and for j_n when x > n.
fn upward(n: i32, x: f64, second: bool) -> f64 {
    let (sin, cos) = x.sin_cos();
    let mut s0 = if second { -cos / x } else { sin / x };
    let mut s1 = if second {
        (s0 - sin) / x
    } else {
        (s0 - cos) / x
    };
    for k in 1..n {
        let next = (2.0 * k as f64 + 1.0) * s1 / x - s0;
        s0 = s1;
        s1 = next;
        if next.is_infinite() {
            break;
        }
    }
    s1
}

/// Continued fraction for j_(n-1)/j_n (or i_(n-1)/i_n), followed by Miller's
/// downward recurrence normalized by the elementary order-zero/one values.
/// Returns `None` if the fraction does not converge within its work limit.
fn downward(n: i32, x: f64, modified: bool) -> Option<f64> {
    let s = if modified { 1.0 } else { -1.0 };
    let b = |k: i32| (2.0 * k as f64 + 1.0) / x;
    const TINY: f64 = 1e-300;
    let mut g = b(n);
    if g == 0.0 {
        g = TINY;
    }
    let mut c = g;
    let mut d = 0.0;
    let mut converged = false;
    for j in 1..=20_000 {
        let bj = b(n + j);
        d = bj + s * d;
        if d == 0.0 {
            d = TINY;
        }
        c = bj + s / c;
        if c == 0.0 {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = c * d;
        g *= delta;
        if (delta - 1.0).abs() <= f64::EPSILON {
            converged = true;
            break;
        }
    }
    if !converged || !g.is_finite() {
        return None;
    }
    // f_k is proportional to j_k (or i_k), with f_n = reference.
    let mut reference = 1.0;
    let mut upper = 1.0; // f_(k+1)
    let mut current = g; // f_k
    for k in (1..n).rev() {
        let next = b(k) * current + s * upper;
        upper = current;
        current = next;
        if current.abs() > 1e250 {
            reference *= 1e-250;
            upper *= 1e-250;
            current *= 1e-250;
        }
    }
    // Now current = f_0 and upper = f_1.
    let (sin, cos) = x.sin_cos();
    let scale = if modified {
        let i0 = 0.5 * ((x - x.ln()).exp() - (-x - x.ln()).exp());
        i0 / current
    } else {
        let j0 = sin / x;
        let j1 = (j0 - cos) / x;
        if j0.abs() > j1.abs() {
            j0 / current
        } else {
            j1 / upper
        }
    };
    let value = scale * reference;
    value.is_finite().then_some(value)
}

macro_rules! spherical {
    ($($name:ident,$jac:ident,$kind:literal,$doc:literal;)+)=>{$(
        #[doc=$doc]
        pub fn $name<T:Float>(n:i32,x:T)->Result<T,StrErr> {
            let x=x.to_f64().ok_or("sph_bessel: Invalid argument.")?;
            finish(real(n,x,$kind)?,"sph_bessel: Undefined result.")
        }
        #[doc=concat!("Derivative with respect to x of [`",stringify!($name),"`].")]
        pub fn $jac<T:Float>(n:i32,x:T)->Result<T,StrErr> {
            let x=x.to_f64().ok_or("sph_bessel: Invalid argument.")?;
            if n<0 || x.is_nan() { return Err("sph_bessel: Requires n >= 0 and a non-NaN argument."); }
            let value=if x==0.0 {
                match $kind { 0|2=>if n==1 {1.0/3.0}else if n==0 && $kind==0 {-x}else{0.0},1=>f64::INFINITY,_=>f64::NEG_INFINITY }
            } else if ($kind==0 || $kind==2) && x.abs()<0.5 { series_derivative(n,x,$kind==2) }
            else if n==0 { if $kind==2 {real(1,x,$kind)?}else{-real(1,x,$kind)?} }
            else {
                let previous=real(n-1,x,$kind)?;
                let current=real(n,x,$kind)?;
                if $kind==3 {-previous-(n as f64+1.0)/x*current}
                else {previous-(n as f64+1.0)/x*current}
            };
            finish(value,"sph_bessel: Undefined derivative.")
        }
    )+};
}
spherical! {
    sph_bessel_j,sph_bessel_j_jac,0,"Spherical Bessel j_n(x), for integer n >= 0.";
    sph_bessel_i,sph_bessel_i_jac,2,"Modified spherical Bessel i_n(x), for integer n >= 0.";
    sph_bessel_k,sph_bessel_k_jac,3,"Modified spherical Bessel k_n(x) = sqrt(pi/(2x))*K_(n+1/2)(x), continued to negative real x by the spherical reflection identity.";
}

/// Spherical Bessel y_n(x), for integer n >= 0.
pub fn sph_bessel_y<T: Float>(n: i32, x: T) -> Result<T, StrErr> {
    let x = x.to_f64().ok_or("sph_bessel: Invalid argument.")?;
    finish(real(n, x, 1)?, "sph_bessel: Undefined result.")
}

// Retain adjacent y values from one upward recurrence. The scalar derivative
// identity y_n' = y_(n-1) - (n+1)y_n/x then needs only one sin_cos and one
// recurrence, instead of calling the complete scalar kernel twice.
// Return None if the recurrence overflows: the existing scalar fallback
// preserves its exceptional-value and pole conventions.
fn upward_y_neighbors(n: i32, x: f64) -> Option<(f64, f64)> {
    let (sin, cos) = x.sin_cos();
    let mut previous = -cos / x;
    let mut current = (previous - sin) / x;
    if !previous.is_finite() || !current.is_finite() {
        return None;
    }
    for k in 1..n {
        let next = (2.0 * k as f64 + 1.0) * current / x - previous;
        if !next.is_finite() {
            return None;
        }
        previous = current;
        current = next;
    }
    Some((previous, current))
}

/// Derivative with respect to x of [`sph_bessel_y`].
pub fn sph_bessel_y_jac<T: Float>(n: i32, x: T) -> Result<T, StrErr> {
    let x = x.to_f64().ok_or("sph_bessel: Invalid argument.")?;
    if n < 0 || x.is_nan() {
        return Err("sph_bessel: Requires n >= 0 and a non-NaN argument.");
    }
    let value = if x == 0.0 {
        f64::INFINITY
    } else if n == 0 {
        -real(1, x, 1)?
    } else if x.is_finite() {
        let xa = x.abs();
        if let Some((previous, current)) = upward_y_neighbors(n, xa) {
            let positive_derivative = previous - (n as f64 + 1.0) / xa * current;
            // y_n(-x)=(-1)^(n+1)y_n(x), so y_n'(-x)=(-1)^n y_n'(x).
            if x < 0.0 && n % 2 != 0 {
                -positive_derivative
            } else {
                positive_derivative
            }
        } else {
            real(n - 1, x, 1)? - (n as f64 + 1.0) / x * real(n, x, 1)?
        }
    } else {
        real(n - 1, x, 1)? - (n as f64 + 1.0) / x * real(n, x, 1)?
    };
    finish(value, "sph_bessel: Undefined derivative.")
}

fn series_derivative(n: i32, x: f64, modified: bool) -> f64 {
    let mut term = if n == 0 {
        if modified { x / 3.0 } else { -x / 3.0 }
    } else {
        let mut base = 1.0 / (2.0 * n as f64 + 1.0);
        for k in 1..n {
            base *= x / (2.0 * k as f64 + 1.0);
            if base == 0.0 {
                return 0.0;
            }
        }
        base * n as f64
    };
    let mut sum = term;
    let square = if modified { x * x } else { -x * x };
    let start = if n == 0 { 2 } else { 1 };
    for k in start..10000 {
        let k = k as f64;
        term *= square / (2.0 * k * (2.0 * n as f64 + 2.0 * k + 1.0))
            * ((n as f64 + 2.0 * k) / (n as f64 + 2.0 * k - 2.0));
        sum += term;
        if term.abs() <= f64::EPSILON * sum.abs() {
            return sum;
        }
    }
    f64::NAN
}

#[cfg(test)]
mod spherical_y_recurrence_tests {
    use super::*;

    fn close(actual: f64, expected: f64, relative: f64, absolute: f64) {
        assert!(
            actual.is_finite() && (actual - expected).abs() <= relative * expected.abs() + absolute,
            "actual={actual:.17e}, expected={expected:.17e}, relative={relative:e}, absolute={absolute:e}"
        );
    }

    // Independent elementary expressions obtained by differentiating
    // y_3(x) = (6/x²-15/x⁴)cos(x) + (1/x-15/x³)sin(x).
    fn elementary_y3_and_derivative(x: f64) -> (f64, f64) {
        let (sin, cos) = x.sin_cos();
        let inverse = 1.0 / x;
        let inverse2 = inverse * inverse;
        let inverse3 = inverse2 * inverse;
        let inverse4 = inverse2 * inverse2;
        let inverse5 = inverse4 * inverse;
        (
            (6.0 * inverse2 - 15.0 * inverse4) * cos + (inverse - 15.0 * inverse3) * sin,
            (inverse - 27.0 * inverse3 + 60.0 * inverse5) * cos
                + (-7.0 * inverse2 + 60.0 * inverse4) * sin,
        )
    }

    #[test]
    fn y3_and_derivative_match_independent_analytic_form_on_exact_benchmark_grid() {
        // The Divan benchmark invokes n=3 for both value and derivative at
        // these 128 evenly spaced points. Check the actual input expression,
        // not only an artificial random test distribution.
        for i in 0..128 {
            let x = 0.5 + (20.0 - 0.5) * i as f64 / 127.0;
            let (expected_y3, expected_jac) = elementary_y3_and_derivative(x);
            close(
                sph_bessel_y::<f64>(3, x).unwrap(),
                expected_y3,
                2e-12,
                2e-13,
            );
            close(
                sph_bessel_y_jac::<f64>(3, x).unwrap(),
                expected_jac,
                2e-12,
                2e-13,
            );
        }
    }

    #[test]
    fn negative_arguments_parity_and_y0_closed_form_are_preserved() {
        for x in [0.25_f64, 0.5, 1.0, 3.0, 10.0, 20.0] {
            let (sin, cos) = x.sin_cos();
            close(sph_bessel_y::<f64>(0, x).unwrap(), -cos / x, 1e-14, 1e-14);
            close(
                sph_bessel_y_jac::<f64>(0, x).unwrap(),
                sin / x + cos / (x * x),
                2e-14,
                1e-14,
            );
            for n in [1, 2, 3, 4, 8] {
                let positive = sph_bessel_y::<f64>(n, x).unwrap();
                let negative = sph_bessel_y::<f64>(n, -x).unwrap();
                let jac = sph_bessel_y_jac::<f64>(n, x).unwrap();
                let negative_jac = sph_bessel_y_jac::<f64>(n, -x).unwrap();
                let odd = n % 2 != 0;
                close(
                    negative,
                    if odd { positive } else { -positive },
                    5e-14,
                    1e-14,
                );
                close(negative_jac, if odd { -jac } else { jac }, 5e-14, 1e-14);
            }
        }
    }

    #[test]
    fn derivative_boundary_and_domain_contracts_match_existing_spherical_conventions() {
        for x in [0.0_f64, -0.0] {
            for n in [0, 1, 3, 4096] {
                assert_eq!(sph_bessel_y_jac::<f64>(n, x).unwrap(), f64::INFINITY);
            }
        }
        assert!(sph_bessel_y_jac::<f64>(-1, 1.0).is_err());
        assert!(sph_bessel_y_jac::<f64>(2, f64::NAN).is_err());

        // At large n and tiny x, upward recurrence overflows early. For
        // this case and the infinite-argument limits, preserve the prior
        // scalar formula, including an error for a NaN arithmetic result
        // and signed zero/infinity when representable.
        for (n, x) in [
            (4096_i32, 0.5_f64),
            (1, f64::INFINITY),
            (2, f64::INFINITY),
            (3, f64::NEG_INFINITY),
        ] {
            let old_expression =
                real(n - 1, x, 1).unwrap() - (n as f64 + 1.0) / x * real(n, x, 1).unwrap();
            let current = sph_bessel_y_jac::<f64>(n, x);
            if old_expression.is_nan() {
                assert!(current.is_err(), "n={n}, x={x}");
            } else {
                assert_eq!(
                    current.unwrap().to_bits(),
                    old_expression.to_bits(),
                    "n={n}, x={x}"
                );
            }
        }
    }
}
