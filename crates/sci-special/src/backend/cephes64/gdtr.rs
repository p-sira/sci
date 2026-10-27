#[allow(unused_imports)]
use sci_core::Float;
// Rust translation adapted from spec_math 0.1.6 (MIT); see LICENSES.
/*                                                     gdtr.c
*
*     Gamma distribution function
*
*
*
* SYNOPSIS:
*
* double a, b, x, y, gdtr();
*
* y = gdtr( a, b, x );
*
*
*
* DESCRIPTION:
*
* Returns the integral from zero to x of the Gamma probability
* density function:
*
*
*                x
*        b       -
*       a       | |   b-1  -at
* y =  -----    |    t    e    dt
*       -     | |
*      | (b)   -
*               0
*
*  The incomplete Gamma integral is used, according to the
* relation
*
* y = igam( b, ax ).
*
*
* ACCURACY:
*
* See igam().
*
* ERROR MESSAGES:
*
*   message         condition      value returned
* gdtr domain         x < 0            0.0
*
*/
/*							gdtrc.c
 *
 *	Complemented Gamma distribution function
 *
 *
 *
 * SYNOPSIS:
 *
 * double a, b, x, y, gdtrc();
 *
 * y = gdtrc( a, b, x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns the integral from x to infinity of the Gamma
 * probability density function:
 *
 *
 *               inf.
 *        b       -
 *       a       | |   b-1  -at
 * y =  -----    |    t    e    dt
 *       -     | |
 *      | (b)   -
 *               x
 *
 *  The incomplete Gamma integral is used, according to the
 * relation
 *
 * y = igamc( b, ax ).
 *
 *
 * ACCURACY:
 *
 * See igamc().
 *
 * ERROR MESSAGES:
 *
 *   message         condition      value returned
 * gdtrc domain         x < 0            0.0
 *
 */

/*                                                     gdtr()  */

/*
* Cephes Math Library Release 2.3:  March,1995
* Copyright 1984, 1987, 1995 by Stephen L. Moshier
*/

use crate::backend::cephes64::igam::{igam, igamc};
use crate::backend::cephes64::igami::igamci;

// $$y = \frac{a^b}{\Gamma(b)}\,\int_0^x{t^{b-1}\,e^{-a\,t}\,dt}$$

pub fn gdtr(a: f64, b: f64, x: f64) -> f64 {
    if x < 0.0 {
        //sf_error("gdtr", SF_ERROR_DOMAIN, NULL);
        f64::NAN
    } else {
        igam(b, a * x)
    }
}

// $$y = \frac{a^b}{\Gamma(b)}\,\int_x^{\infty}{t^{b-1}\,e^{-a\,t}\,dt}$$

pub fn gdtrc(a: f64, b: f64, x: f64) -> f64 {
    if x < 0.0 {
        //sf_error("gdtrc", SF_ERROR_DOMAIN, NULL);
        f64::NAN
    } else {
        igamc(b, a * x)
    }
}

// Very few tests due to simplicity of implementation
