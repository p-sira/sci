// Rust translation adapted from spec_math 0.1.6 (MIT); see LICENSES.
/*                                                     psi.c
*
*     Psi (digamma) function
*
*
* SYNOPSIS:
*
* double x, y, psi();
*
* y = psi( x );
*
*
* DESCRIPTION:
*
*              d      -
*   psi(x)  =  -- ln | (x)
*              dx
*
* is the logarithmic derivative of the gamma function.
* For integer x,
*                   n-1
*                    -
* psi(n) = -EUL  +   >  1/k.
*                    -
*                   k=1
*
* This formula is used for 0 < n <= 10.  If x is negative, it
* is transformed to a positive argument by the reflection
* formula  psi(1-x) = psi(x) + pi cot(pi x).
* For general positive x, the argument is made greater than 10
* using the recurrence  psi(x+1) = psi(x) + 1/x.
* Then the following asymptotic expansion is applied:
*
*                           inf.   B
*                            -      2k
* psi(x) = log(x) - 1/2x -   >   -------
*                            -        2k
*                           k=1   2k x
*
* where the B2k are Bernoulli numbers.
*
* ACCURACY:
*    Relative error (except absolute when |psi| < 1):
* arithmetic   domain     # trials      peak         rms
*    IEEE      0,30        30000       1.3e-15     1.4e-16
*    IEEE      -30,0       40000       1.5e-15     2.2e-16
*
* ERROR MESSAGES:
*     message         condition      value returned
* psi singularity    x integer <=0      INFINITY
*/

/*
* Cephes Math Library Release 2.8:  June, 2000
* Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier
*/

/*
* Code for the rational approximation on [1, 2] is:
*
* (C) Copyright John Maddock 2006.
* Use, modification and distribution are subject to the
* Boost Software License, Version 1.0. (See accompanying file
* LICENSE_1_0.txt or copy at https://www.boost.org/LICENSE_1_0.txt)
*/

#![allow(clippy::excessive_precision)]

#[allow(unused_imports)]
use sci_core::Float;

use crate::backend::cephes64::consts::{EULER, M_PI};
use crate::backend::cephes64::polevl::polevl;

const A: [f64; 7] = [
    8.33333333333333333333E-2,
    -2.10927960927960927961E-2,
    7.57575757575757575758E-3,
    -4.16666666666666666667E-3,
    3.96825396825396825397E-3,
    -8.33333333333333333333E-3,
    8.33333333333333333333E-2,
];

const P: [f64; 6] = [
    -0.0020713321167745952,
    -0.045251321448739056,
    -0.28919126444774784,
    -0.65031853770896507,
    -0.32555031186804491,
    0.25479851061131551,
];
const Q: [f64; 7] = [
    -0.55789841321675513e-6,
    0.0021284987017821144,
    0.054151797245674225,
    0.43593529692665969,
    1.4606242909763515,
    2.0767117023730469,
    1.0,
];

const Y: f64 = 0.99558162689208984;

const ROOT1: f64 = 1569415565.0 / 1073741824.0;
const ROOT2: f64 = (381566830.0 / 1073741824.0) / 1073741824.0;
const ROOT3: f64 = 0.9016312093258695918615325266959189453125e-19;

fn digamma_imp_1_2(x: f64) -> f64 {
    /*
     * Rational approximation on [1, 2] taken from Boost.
     *
     * Now for the approximation, we use the form:
     *
     * digamma(x) = (x - root) * (Y + R(x-1))
     *
     * Where root is the location of the positive root of digamma,
     * Y is a constant, and R is optimised for low absolute error
     * compared to Y.
     *
     * Maximum Deviation Found:               1.466e-18
     * At double precision, max error found:  2.452e-17
     */

    let mut g = x - ROOT1;
    g -= ROOT2;
    g -= ROOT3;
    let r = polevl(x - 1.0, &P, 5) / polevl(x - 1.0, &Q, 6);

    g * Y + g * r
}

fn psi_asy(x: f64) -> f64 {
    //double y, z;

    let y = if x < 1.0e17 {
        let z = 1.0 / (x * x);
        z * polevl(z, &A, 6)
    } else {
        0.0
    };

    x.ln() - 0.5 / x - y
}

// $$\mathrm{psi}(x) = \frac{d}{dx}\,\ln(\Gamma(x))$$
// $$\mathrm{psi}(x) = -\mathrm{EUL} + \sum_{k=1}^{n-1}{\frac{1}{k}}$$
// $$\mathrm{psi}(x) = \ln(x) - \frac{1}{2}\,x +
// \sum_{k=1}^{\infty}{\frac{B_{2\,k}}{2\,k\,x^{2\,k}}}$$

pub fn psi(x: f64) -> f64 {
    let mut y: f64 = 0.0;
    let mut x = x;

    if x.is_nan() || x == f64::INFINITY {
        return x;
    } else if x == -f64::INFINITY {
        return f64::NAN;
    } else if x == 0.0 {
        //sf_error("psi", SF_ERROR_SINGULAR, NULL);
        return f64::INFINITY.copysign(-x);
    } else if x < 0.0 {
        /* argument reduction before evaluating tan(pi * x) */
        let r = x.fract();
        if r == 0.0 {
            //sf_error("psi", SF_ERROR_SINGULAR, NULL);
            return f64::NAN;
        }
        y = -M_PI / (M_PI * r).tan();
        x = 1.0 - x;
    }

    /* check for positive integer up to 10 */
    if x <= 10.0 && x == x.floor() {
        let n: isize = x as isize;
        for i in 1..n {
            y += 1.0 / i as f64;
        }
        y - EULER
    } else {
        /* use the recurrence relation to move x into [1, 2] */
        if x < 1.0 {
            y -= 1.0 / x;
            x += 1.0;
        } else if x < 10.0 {
            while x > 2.0 {
                x -= 1.0;
                y += 1.0 / x;
            }
        }
        if (1.0..=2.0).contains(&x) {
            y + digamma_imp_1_2(x)
        } else {
            /* x is large, use the asymptotic series */
            y + psi_asy(x)
        }
    }
}
