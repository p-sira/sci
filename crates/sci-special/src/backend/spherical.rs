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
    sph_bessel_y,sph_bessel_y_jac,1,"Spherical Bessel y_n(x), for integer n >= 0.";
    sph_bessel_i,sph_bessel_i_jac,2,"Modified spherical Bessel i_n(x), for integer n >= 0.";
    sph_bessel_k,sph_bessel_k_jac,3,"Modified spherical Bessel k_n(x) = sqrt(pi/(2x))*K_(n+1/2)(x), continued to negative real x by the spherical reflection identity.";
}

pub(crate) fn complex_value(n: i32, z: Complex64, kind: u8) -> Result<Complex64, StrErr> {
    if n < 0 {
        return Err("sph_bessel: Requires n >= 0.");
    }
    if z.im == 0.0 {
        return Ok(Complex64::new(real(n, z.re, kind)?, 0.0));
    }
    if z == Complex64::new(0.0, 0.0) && (kind == 0 || kind == 2) {
        return Ok(Complex64::new(if n == 0 { 1.0 } else { 0.0 }, 0.0));
    }
    // Tiny arguments need a series to avoid underflow in the cylindrical factor.
    if z.norm() < 0.5 && (kind == 0 || kind == 2) {
        let mut leading = Complex64::new(1.0, 0.0);
        for k in 1..=n {
            leading *= z / (2.0 * k as f64 + 1.0);
            if leading == Complex64::new(0.0, 0.0) {
                return Ok(leading);
            }
        }
        let mut sum = Complex64::new(1.0, 0.0);
        let mut term = sum;
        let square = if kind == 2 { z * z } else { -z * z };
        for k in 1..10000 {
            term *= square / (2.0 * k as f64 * (2.0 * n as f64 + 2.0 * k as f64 + 1.0));
            sum += term;
            if term.norm() <= f64::EPSILON * sum.norm() {
                return Ok(leading * sum);
            }
        }
        return Err("sph_bessel: Series did not converge.");
    }
    let order = n as f64 + 0.5;
    let value = match kind {
        0 => complex_bessel::besselj(order, z),
        1 => complex_bessel::bessely(order, z),
        2 => complex_bessel::besseli(order, z),
        _ => complex_bessel::besselk(order, z),
    }
    .map_err(crate::backend::bessel::backend_error)?;
    Ok((core::f64::consts::FRAC_PI_2.sqrt() / z.sqrt()) * value)
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

pub(crate) fn complex_derivative(
    n: i32,
    z: Complex64,
    modified: bool,
) -> Result<Complex64, StrErr> {
    let mut term = if n == 0 {
        if modified { z / 3.0 } else { -z / 3.0 }
    } else {
        let mut base = Complex64::new(1.0 / (2.0 * n as f64 + 1.0), 0.0);
        for k in 1..n {
            base *= z / (2.0 * k as f64 + 1.0);
            if base == Complex64::new(0.0, 0.0) {
                return Ok(base);
            }
        }
        base * n as f64
    };
    let mut sum = term;
    let square = if modified { z * z } else { -z * z };
    let start = if n == 0 { 2 } else { 1 };
    for k in start..10000 {
        let k = k as f64;
        term *= square / (2.0 * k * (2.0 * n as f64 + 2.0 * k + 1.0))
            * ((n as f64 + 2.0 * k) / (n as f64 + 2.0 * k - 2.0));
        sum += term;
        if term.norm() <= f64::EPSILON * sum.norm() {
            return Ok(sum);
        }
    }
    Err("sph_bessel: Derivative series did not converge.")
}
