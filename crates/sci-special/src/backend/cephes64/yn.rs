#[allow(unused_imports)]
use sci_core::Float;
// Rust translation adapted from spec_math 0.1.6 (MIT); see LICENSES.
/*                                                     yn.c
*
*     Bessel function of second kind of integer order
*
*
*
* SYNOPSIS:
*
* double x, y, yn();
* int n;
*
* y = yn( n, x );
*
*
*
* DESCRIPTION:
*
* Returns Bessel function of order n, where n is a
* (possibly negative) integer.
*
* The function is evaluated by forward recurrence on
* n, starting with values computed by the routines
* y0() and y1().
*
* If n = 0 or 1 the routine for y0 or y1 is called
* directly.
*
*
*
* ACCURACY:
*
*
*                      Absolute error, except relative
*                      when y > 1:
* arithmetic   domain     # trials      peak         rms
*    IEEE      0, 30       30000       3.4e-15     4.3e-16
*
*
* ERROR MESSAGES:
*
*   message         condition      value returned
* yn singularity   x = 0              INFINITY
* yn overflow                         INFINITY
*
* Spot checked against tables for x, n between 0 and 100.
*
*/

/*
* Cephes Math Library Release 2.8:  June, 2000
* Copyright 1984, 1987, 2000 by Stephen L. Moshier
*/

use crate::backend::cephes64::j0::y0;
use crate::backend::cephes64::j1::y1;

pub fn yn(n: isize, x: f64) -> f64 {
    let mut n = n;
    let sign = if n < 0 {
        n = -n;
        if (n & 1) == 0 {
            /* -1**n */
            1.0
        } else {
            -1.0
        }
    } else {
        1.0
    };

    if n == 0 {
        sign * y0(x)
    } else if n == 1 {
        sign * y1(x)
    } else if x == 0.0 {
        /* test for overflow */
        //sf_error("yn", SF_ERROR_SINGULAR, NULL);
        -f64::INFINITY * sign
    } else if x < 0.0 {
        //sf_error("yn", SF_ERROR_DOMAIN, NULL);
        f64::NAN
    } else {
        /* forward recurrence on n */

        let mut anm2 = y0(x);
        let mut anm1 = y1(x);
        let mut r = 2.0;
        let mut an = 0.0; // Value doesn't matter, just need to declare it
        for _ in 1..n {
            if !anm1.is_finite() {
                return sign * anm1;
            }
            an = r * anm1 / x - anm2;
            anm2 = anm1;
            anm1 = an;
            r += 2.0;
        }
        sign * an
    }
}
