// Rust translation adapted from spec_math 0.1.6 (MIT); see LICENSES.
/*                                                     unity.c
*
* Relative error approximations for function arguments near
* unity.
*
*    log1p(x) = log(1+x)
*    expm1(x) = exp(x) - 1
*    cosm1(x) = cos(x) - 1
*    lgam1p(x) = lgam(1+x)
*
*/

/* Scipy changes:
* - 06-10-2016: added lgam1p
*/

#![allow(clippy::excessive_precision)]

#[allow(unused_imports)]
use sci_core::Float;

use super::consts::{EULER, M_PI_4, M_SQRT1_2, M_SQRT2, MACHEP};
use super::gamma::lgam;
use super::polevl::{p1evl, polevl};
use super::zeta::zeta;

const MAXITER: isize = 500;

/* log1p(x) = log(1 + x)  */

/* Coefficients for log(1+x) = x - x**2/2 + x**3 P(x)/Q(x)
* 1/sqrt(2) <= x < sqrt(2)
* Theoretical peak relative error = 2.32e-20
*/
const LP: [f64; 7] = [
    4.5270000862445199635215E-5,
    4.9854102823193375972212E-1,
    6.5787325942061044846969E0,
    2.9911919328553073277375E1,
    6.0949667980987787057556E1,
    5.7112963590585538103336E1,
    2.0039553499201281259648E1,
];

const LQ: [f64; 6] = [
    /* 1.0000000000000000000000E0, */
    1.5062909083469192043167E1,
    8.3047565967967209469434E1,
    2.2176239823732856465394E2,
    3.0909872225312059774938E2,
    2.1642788614495947685003E2,
    6.0118660497603843919306E1,
];

pub fn log1p(x: f64) -> f64 {
    let mut z = 1.0 + x;
    if (M_SQRT1_2..=M_SQRT2).contains(&z) {
        z = x * x;
        z = -0.5 * z + x * (z * polevl(x, &LP, 6) / p1evl(x, &LQ, 6));
        x + z
    } else {
        z.ln()
    }
}

/* log(1 + x) - x */
pub fn log1pmx(x: f64) -> f64 {
    if x.abs() < 0.5 {
        let mut xfac = x;
        let mut res: f64 = 0.0;

        for n in 2..MAXITER {
            xfac *= -x;
            let term = xfac / n as f64;
            res += term;
            if term.abs() < MACHEP * res.abs() {
                break;
            }
        }

        res
    } else {
        log1p(x) - x
    }
}

/* expm1(x) = exp(x) - 1  */

/*  e^x =  1 + 2x P(x^2)/( Q(x^2) - P(x^2) )
* -0.5 <= x <= 0.5
*/

const EP: [f64; 3] = [
    1.2617719307481059087798E-4,
    3.0299440770744196129956E-2,
    9.9999999999999999991025E-1,
];

const EQ: [f64; 4] = [
    3.0019850513866445504159E-6,
    2.5244834034968410419224E-3,
    2.2726554820815502876593E-1,
    2.0000000000000000000897E0,
];

pub fn expm1(x: f64) -> f64 {
    if x.is_infinite() {
        if x.is_nan() || x > 0.0 { x } else { -1.0 }
    } else if (-0.5..=0.5).contains(&x) {
        let xx = x * x;
        let r = x * polevl(xx, &EP, 2);
        let r = r / (polevl(xx, &EQ, 3) - r);
        r + r
    } else {
        x.exp() - 1.0
    }
}

/* Compute lgam(x + 1) around x = 0 using its Taylor series. */
fn lgam1p_taylor(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let mut res = -EULER * x;
    let mut xfac = -x;
    let mut coeff: f64;
    for n in 2..42 {
        xfac *= -x;
        coeff = zeta(n as f64, 1.0) * xfac / n as f64;
        res += coeff;
        if coeff.abs() < MACHEP * res.abs() {
            break;
        }
    }

    res
}

/* Compute lgam(x + 1). */
pub fn lgam1p(x: f64) -> f64 {
    if x.abs() <= 0.5 {
        lgam1p_taylor(x)
    } else if (x - 1.0).abs() < 0.5 {
        x.ln() + lgam1p_taylor(x - 1.0)
    } else {
        lgam(x + 1.0)
    }
}
