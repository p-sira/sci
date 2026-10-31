// Rust translation adapted from spec_math 0.1.6 (MIT); see LICENSES.
/*                                                     ndtri.c
*
*     Inverse of Normal distribution function
*
*
*
* SYNOPSIS:
*
* double x, y, ndtri();
*
* x = ndtri( y );
*
*
*
* DESCRIPTION:
*
* Returns the argument, x, for which the area under the
* Gaussian probability density function (integrated from
* minus infinity to x) is equal to y.
*
*
* For small arguments 0 < y < exp(-2), the program computes
* z = sqrt( -2.0 * log(y) );  then the approximation is
* x = z - log(z)/z  - (1/z) P(1/z) / Q(1/z).
* There are two rational functions P/Q, one for 0 < y < exp(-32)
* and the other for y up to exp(-2).  For larger arguments,
* w = y - 0.5, and  x/sqrt(2pi) = w + w**3 R(w**2)/S(w**2)).
*
*
* ACCURACY:
*
*                      Relative error:
* arithmetic   domain        # trials      peak         rms
*    IEEE     0.125, 1        20000       7.2e-16     1.3e-16
*    IEEE     3e-308, 0.135   50000       4.6e-16     9.8e-17
*
*
* ERROR MESSAGES:
*
*   message         condition    value returned
* ndtri domain       x < 0        NAN
* ndtri domain       x > 1        NAN
*
*/

/*
* Cephes Math Library Release 2.1:  January, 1989
* Copyright 1984, 1987, 1989 by Stephen L. Moshier
* Direct inquiries to 30 Frost Street, Cambridge, MA 02140
*/

#![allow(clippy::excessive_precision)]

#[allow(unused_imports)]
use sci_core::Float;

use crate::backend::cephes64::polevl::{p1evl, polevl};

/* sqrt(2pi) */
const S2PI: f64 = 2.50662827463100050242E0;

/* approximation for 0 <= |y - 0.5| <= 3/8 */
const P0: [f64; 5] = [
    -5.99633501014107895267E1,
    9.80010754185999661536E1,
    -5.66762857469070293439E1,
    1.39312609387279679503E1,
    -1.23916583867381258016E0,
];

const Q0: [f64; 8] = [
    /* 1.00000000000000000000E0, */
    1.95448858338141759834E0,
    4.67627912898881538453E0,
    8.63602421390890590575E1,
    -2.25462687854119370527E2,
    2.00260212380060660359E2,
    -8.20372256168333339912E1,
    1.59056225126211695515E1,
    -1.18331621121330003142E0,
];

/* Approximation for interval z = sqrt(-2 log y ) between 2 and 8
* i.e., y between exp(-2) = .135 and exp(-32) = 1.27e-14.
*/
const P1: [f64; 9] = [
    4.05544892305962419923E0,
    3.15251094599893866154E1,
    5.71628192246421288162E1,
    4.40805073893200834700E1,
    1.46849561928858024014E1,
    2.18663306850790267539E0,
    -1.40256079171354495875E-1,
    -3.50424626827848203418E-2,
    -8.57456785154685413611E-4,
];

const Q1: [f64; 8] = [
    /*  1.00000000000000000000E0, */
    1.57799883256466749731E1,
    4.53907635128879210584E1,
    4.13172038254672030440E1,
    1.50425385692907503408E1,
    2.50464946208309415979E0,
    -1.42182922854787788574E-1,
    -3.80806407691578277194E-2,
    -9.33259480895457427372E-4,
];

/* Approximation for interval z = sqrt(-2 log y ) between 8 and 64
* i.e., y between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890.
*/

const P2: [f64; 9] = [
    3.23774891776946035970E0,
    6.91522889068984211695E0,
    3.93881025292474443415E0,
    1.33303460815807542389E0,
    2.01485389549179081538E-1,
    1.23716634817820021358E-2,
    3.01581553508235416007E-4,
    2.65806974686737550832E-6,
    6.23974539184983293730E-9,
];

const Q2: [f64; 8] = [
    /*  1.00000000000000000000E0, */
    6.02427039364742014255E0,
    3.67983563856160859403E0,
    1.37702099489081330271E0,
    2.16236993594496635890E-1,
    1.34204006088543189037E-2,
    3.28014464682127739104E-4,
    2.89247864745380683936E-6,
    6.79019408009981274425E-9,
];

// The overwhelmingly common ndtri_exp tail (sqrt(-2*y) < 8) has fixed
// coefficients. Evaluate its numerator/denominator directly instead of
// passing a runtime degree and slices through the shared Cephes helpers.
// Keep the exact multiply-then-add order of polevl/p1evl (no FMA).
#[inline(always)]
fn ndtri_exp_p1(z: f64) -> f64 {
    let mut ans = P1[0];
    ans = ans * z + P1[1];
    ans = ans * z + P1[2];
    ans = ans * z + P1[3];
    ans = ans * z + P1[4];
    ans = ans * z + P1[5];
    ans = ans * z + P1[6];
    ans = ans * z + P1[7];
    ans * z + P1[8]
}

#[inline(always)]
fn ndtri_exp_q1(z: f64) -> f64 {
    let mut ans = z + Q1[0];
    ans = ans * z + Q1[1];
    ans = ans * z + Q1[2];
    ans = ans * z + Q1[3];
    ans = ans * z + Q1[4];
    ans = ans * z + Q1[5];
    ans = ans * z + Q1[6];
    ans * z + Q1[7]
}

pub fn ndtri(y0: f64) -> f64 {
    if y0 == 0.0 {
        return -f64::INFINITY;
    } else if y0 == 1.0 {
        return f64::INFINITY;
    } else if !(0.0..=1.0).contains(&y0) {
        //sf_error("ndtri", SF_ERROR_DOMAIN, NULL);
        return f64::NAN;
    }
    let mut code = true;
    let mut y = y0;
    if y > 1.0 - 0.13533528323661269189 {
        /* 0.135... = exp(-2) */
        y = 1.0 - y;
        code = false;
    } else if y > 0.13533528323661269189 {
        y -= 0.5;
        let y2 = y * y;
        let x = y + y * (y2 * polevl(y2, &P0, 4) / p1evl(y2, &Q0, 8));
        return x * S2PI;
    }

    let x = (-2.0 * y.ln()).sqrt();
    let x0 = x - x.ln() / x;

    let z = 1.0 / x;
    let x1 = if x < 8.0 {
        /* y > exp(-32) = 1.2664165549e-14 */
        z * polevl(z, &P1, 8) / p1evl(z, &Q1, 8)
    } else {
        z * polevl(z, &P2, 8) / p1evl(z, &Q2, 8)
    };

    if code { x1 - x0 } else { x0 - x1 }
}

// Port of xsf/ndtri_exp.h, reusing the ndtri tail rational approximations.
pub(crate) fn ndtri_exp(y: f64) -> f64 {
    if y == f64::NEG_INFINITY {
        return y;
    }
    if y < -2.0 {
        let x = if y >= -f64::MAX * 0.5 {
            (-2.0 * y).sqrt()
        } else {
            core::f64::consts::SQRT_2 * (-y).sqrt()
        };
        let x0 = x - x.ln() / x;
        let z = 1.0 / x;
        let correction = if x < 8.0 {
            z * ndtri_exp_p1(z) / ndtri_exp_q1(z)
        } else {
            z * polevl(z, &P2, 8) / p1evl(z, &Q2, 8)
        };
        correction - x0
    } else if y > -0.14541345786885906 {
        -ndtri(-y.exp_m1())
    } else {
        ndtri(y.exp())
    }
}

#[cfg(test)]
mod ndtri_exp_horner_tests {
    use super::*;

    #[test]
    fn local_tail_horner_preserves_cephes_polynomial_order() {
        for z in [0.125, 0.14, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0] {
            assert_eq!(ndtri_exp_p1(z).to_bits(), polevl(z, &P1, 8).to_bits());
            assert_eq!(ndtri_exp_q1(z).to_bits(), p1evl(z, &Q1, 8).to_bits());
        }
    }
}
