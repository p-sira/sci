// Rust translation adapted from spec_math 0.1.6 (MIT); see LICENSES.
/*                                                     beta.c
*
*     Beta function
*
*
*
* SYNOPSIS:
*
* double a, b, y, beta();
*
* y = beta( a, b );
*
*
*
* DESCRIPTION:
*
*                   -     -
*                  | (a) | (b)
* beta( a, b )  =  -----------.
*                     -
*                    | (a+b)
*
* For large arguments the logarithm of the function is
* evaluated using lgam(), then exponentiated.
*
*
*
* ACCURACY:
*
*                      Relative error:
* arithmetic   domain     # trials      peak         rms
*    IEEE       0,30       30000       8.1e-14     1.1e-14
*
* ERROR MESSAGES:
*
*   message         condition          value returned
* beta overflow    log(beta) > MAXLOG       0.0
*                  a or b <0 integer        0.0
*
*/

/*
* Cephes Math Library Release 2.0:  April, 1987
* Copyright 1984, 1987 by Stephen L. Moshier
* Direct inquiries to 30 Frost Street, Cambridge, MA 02140
*/

#![allow(clippy::excessive_precision)]

#[allow(unused_imports)]
use sci_core::Float;

const MAXGAM: f64 = 171.624376956302725;

use crate::backend::cephes64::consts::MAXLOG;
use crate::backend::cephes64::gamma::{gamma, lgam_sgn};

const ASYMP_FACTOR: f64 = 1e6;

// static double lbeta_asymp(double a, double b, int *sgn);
// static double lbeta_negint(int a, double b);
// static double beta_negint(int a, double b);

// $$\mathrm{B}(a, b) = \frac{\Gamma(a)\,\Gamma(b)}{\Gamma(a + b)}$$

pub fn beta(a: f64, b: f64) -> f64 {
    let mut sign: isize = 1;

    if a <= 0.0 && a == a.floor() {
        if a == (a as isize) as f64 {
            return beta_negint(a as isize, b);
        } else {
            //sf_error("beta", SF_ERROR_OVERFLOW, NULL);
            return sign as f64 * f64::INFINITY;
        }
    }

    if b <= 0.0 && b == b.floor() {
        if b == (b as isize) as f64 {
            return beta_negint(b as isize, a);
        } else {
            //sf_error("beta", SF_ERROR_OVERFLOW, NULL);
            return sign as f64 * f64::INFINITY;
        }
    }

    let mut a = a;
    let mut b = b;

    if a.abs() < b.abs() {
        core::mem::swap(&mut a, &mut b)
    }

    if a.abs() > ASYMP_FACTOR * b.abs() && a > ASYMP_FACTOR {
        /* Avoid loss of precision in lgam(a + b) - lgam(a) */
        let y = lbeta_asymp(a, b, &mut sign);
        return sign as f64 * y.exp();
    }

    let mut y = a + b;
    if y.abs() > MAXGAM || a.abs() > MAXGAM || b.abs() > MAXGAM {
        let mut sgngam: isize = 0; // Value doesn't matter
        y = lgam_sgn(y, &mut sgngam);
        sign *= sgngam; /* keep track of the sign */
        y = lgam_sgn(b, &mut sgngam) - y;
        sign *= sgngam;
        y += lgam_sgn(a, &mut sgngam);
        sign *= sgngam;
        if y > MAXLOG {
            //sf_error("beta", SF_ERROR_OVERFLOW, NULL);
            return sign as f64 * f64::INFINITY;
        }
        return sign as f64 * y.exp();
    }

    y = crate::backend::cephes64::rgamma(y);
    a = gamma(a);
    b = gamma(b);
    if y.is_infinite() {
        //sf_error("beta", SF_ERROR_OVERFLOW, NULL);
        return sign as f64 * f64::INFINITY;
    }

    if ((a * y).abs() - 1.0).abs() > ((b * y).abs() - 1.0).abs() {
        y *= b;
        y * a
    } else {
        y *= a;
        y * b
    }

    //overflow:
    //sf_error("beta", SF_ERROR_OVERFLOW, NULL);
    //return (sign as f64 * f64::INFINITY);
}

/* Natural log of |beta|. */

pub fn lbeta(a: f64, b: f64) -> f64 {
    let mut sign: isize = 1;

    if a <= 0.0 && a == a.floor() {
        if a == (a as isize) as f64 {
            return lbeta_negint(a as isize, b);
        } else {
            // sf_error("lbeta", SF_ERROR_OVERFLOW, NULL);
            return sign as f64 * f64::INFINITY;
        }
    }

    if b <= 0.0 && b == b.floor() {
        if b == (b as isize) as f64 {
            return lbeta_negint(b as isize, a);
        } else {
            // sf_error("lbeta", SF_ERROR_OVERFLOW, NULL);
            return sign as f64 * f64::INFINITY;
        }
    }

    let mut a = a;
    let mut b = b;

    if a.abs() < b.abs() {
        core::mem::swap(&mut a, &mut b)
    }

    if a.abs() > ASYMP_FACTOR * b.abs() && a > ASYMP_FACTOR {
        /* Avoid loss of precision in lgam(a + b) - lgam(a) */
        return lbeta_asymp(a, b, &mut sign);
    }

    let mut y = a + b;
    if y.abs() > MAXGAM || a.abs() > MAXGAM || b.abs() > MAXGAM {
        let mut sgngam: isize = 0; // Value doesn't matter
        y = lgam_sgn(y, &mut sgngam);
        //sign *= sgngam;		/* keep track of the sign */
        y = lgam_sgn(b, &mut sgngam) - y;
        //sign *= sgngam;
        y += lgam_sgn(a, &mut sgngam);
        //sign *= sgngam;
        return y;
    }

    y = crate::backend::cephes64::rgamma(y);
    a = gamma(a);
    b = gamma(b);
    if y.is_infinite() {
        // over:
        // sf_error("lbeta", SF_ERROR_OVERFLOW, NULL);
        return sign as f64 * f64::INFINITY;
    }

    if ((a * y).abs() - 1.0).abs() > ((b * y).abs() - 1.0).abs() {
        y *= b;
        y *= a;
    } else {
        y *= a;
        y *= b;
    }

    y.abs().ln()
}

/*
* Asymptotic expansion for  ln(|B(a, b)|) for a > ASYMP_FACTOR*max(|b|, 1).
*/
fn lbeta_asymp(a: f64, b: f64, sgn: &mut isize) -> f64 {
    let mut r = lgam_sgn(b, sgn);
    r -= b * a.ln();

    r += b * (1.0 - b) / (2.0 * a);
    r += b * (1.0 - b) * (1.0 - 2.0 * b) / (12.0 * a * a);
    r + -b * b * (1.0 - b) * (1.0 - b) / (12.0 * a * a * a)
}

/*
* Special case for a negative integer argument
*/

fn beta_negint(a: isize, b: f64) -> f64 {
    if b == (b as isize) as f64 && (1 - a) as f64 - b > 0.0 {
        if b as isize % 2 == 0 {
            beta((1 - a) as f64 - b, b)
        } else {
            -beta((1 - a) as f64 - b, b)
        }
    } else {
        //sf_error("lbeta", SF_ERROR_OVERFLOW, NULL);
        f64::INFINITY
    }
}

fn lbeta_negint(a: isize, b: f64) -> f64 {
    if b == (b as isize) as f64 && (1 - a) as f64 - b > 0.0 {
        lbeta((1 - a) as f64 - b, b)
    } else {
        //sf_error("lbeta", SF_ERROR_OVERFLOW, NULL);
        f64::INFINITY
    }
}
