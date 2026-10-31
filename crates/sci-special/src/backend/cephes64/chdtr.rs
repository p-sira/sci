#[allow(unused_imports)]
use sci_core::Float;
// Rust translation adapted from spec_math 0.1.6 (MIT); see LICENSES.
/*                                                     chdtr.c
*
*     Chi-square distribution
*
*
*
* SYNOPSIS:
*
* double df, x, y, chdtr();
*
* y = chdtr( df, x );
*
*
*
* DESCRIPTION:
*
* Returns the area under the left hand tail (from 0 to x)
* of the Chi square probability density function with
* v degrees of freedom.
*
*
*                                  inf.
*                                    -
*                        1          | |  v/2-1  -t/2
*  P( x | v )   =   -----------     |   t      e     dt
*                    v/2  -       | |
*                   2    | (v/2)   -
*                                   x
*
* where x is the Chi-square variable.
*
* The incomplete Gamma integral is used, according to the
* formula
*
*     y = chdtr( v, x ) = igam( v/2.0, x/2.0 ).
*
*
* The arguments must both be positive.
*
*
*
* ACCURACY:
*
* See igam().
*
* ERROR MESSAGES:
*
*   message         condition      value returned
* chdtr domain   x < 0 or v < 1        0.0
*/
/*							chdtrc()
 *
 *	Complemented Chi-square distribution
 *
 *
 *
 * SYNOPSIS:
 *
 * double v, x, y, chdtrc();
 *
 * y = chdtrc( v, x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns the area under the right hand tail (from x to
 * infinity) of the Chi square probability density function
 * with v degrees of freedom:
 *
 *
 *                                  inf.
 *                                    -
 *                        1          | |  v/2-1  -t/2
 *  P( x | v )   =   -----------     |   t      e     dt
 *                    v/2  -       | |
 *                   2    | (v/2)   -
 *                                   x
 *
 * where x is the Chi-square variable.
 *
 * The incomplete Gamma integral is used, according to the
 * formula
 *
 *	y = chdtr( v, x ) = igamc( v/2.0, x/2.0 ).
 *
 *
 * The arguments must both be positive.
 *
 *
 *
 * ACCURACY:
 *
 * See igamc().
 *
 * ERROR MESSAGES:
 *
 *   message         condition      value returned
 * chdtrc domain  x < 0 or v < 1        0.0
 */
/*							chdtri()
 *
 *	Inverse of complemented Chi-square distribution
 *
 *
 *
 * SYNOPSIS:
 *
 * double df, x, y, chdtri();
 *
 * x = chdtri( df, y );
 *
 *
 *
 *
 * DESCRIPTION:
 *
 * Finds the Chi-square argument x such that the integral
 * from x to infinity of the Chi-square density is equal
 * to the given cumulative probability y.
 *
 * This is accomplished using the inverse Gamma integral
 * function and the relation
 *
 *    x/2 = igamci( df/2, y );
 *
 *
 *
 *
 * ACCURACY:
 *
 * See igami.c.
 *
 * ERROR MESSAGES:
 *
 *   message         condition      value returned
 * chdtri domain   y < 0 or y > 1        0.0
 *                     v < 1
 *
 */

/*                                                             chdtr() */

/*
* Cephes Math Library Release 2.0:  April, 1987
* Copyright 1984, 1987 by Stephen L. Moshier
* Direct inquiries to 30 Frost Street, Cambridge, MA 02140
*/

use crate::backend::cephes64::igam::{igam, igamc};
use crate::backend::cephes64::igami::igamci;

pub fn chdtrc(df: f64, x: f64) -> f64 {
    if x < 0.0 {
        1.0 /* modified by T. Oliphant */
    } else {
        igamc(df / 2.0, x / 2.0)
    }
}

// $$\mathrm{P}(x, v) = \frac{1}{2^{v/2}\,\Gamma(v/2)}\,
// \int_x^{\infty}{t^{v/2-1}\,e^{-t/2}\,dt}$$

// TODO: I think the equation in the documentation is wrong...
pub fn chdtr(df: f64, x: f64) -> f64 {
    if x < 0.0 {
        /* || (df < 1.0) ) */
        //sf_error("chdtr", SF_ERROR_DOMAIN, NULL);
        f64::NAN
    } else {
        igam(df / 2.0, x / 2.0)
    }
}

pub fn chdtri(df: f64, y: f64) -> f64 {
    if !(0.0..=1.0).contains(&y) {
        /* || (df < 1.0) ) */
        //sf_error("chdtri", SF_ERROR_DOMAIN, NULL);
        f64::NAN
    } else {
        2.0 * igamci(0.5 * df, y)
    }
}

// Very few tests due to simplicity of implementation
