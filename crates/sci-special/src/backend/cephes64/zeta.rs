// Rust translation adapted from spec_math 0.1.6 (MIT); see LICENSES.
/*                                                     zeta.c
*
*     Riemann zeta function of two arguments
*
*
*
* SYNOPSIS:
*
* double x, q, y, zeta();
*
* y = zeta( x, q );
*
*
*
* DESCRIPTION:
*
*
*
*                 inf.
*                  -        -x
*   zeta(x,q)  =   >   (k+q)
*                  -
*                 k=0
*
* where x > 1 and q is not a negative integer or zero.
* The Euler-Maclaurin summation formula is used to obtain
* the expansion
*
*                n
*                -       -x
* zeta(x,q)  =   >  (k+q)
*                -
*               k=1
*
*           1-x                 inf.  B   x(x+1)...(x+2j)
*      (n+q)           1         -     2j
*  +  ---------  -  -------  +   >    --------------------
*        x-1              x      -                   x+2j+1
*                   2(n+q)      j=1       (2j)! (n+q)
*
* where the B2j are Bernoulli numbers.  Note that (see zetac.c)
* zeta(x,1) = zetac(x) + 1.
*
*
*
* ACCURACY:
*
*
*
* REFERENCE:
*
* Gradshteyn, I. S., and I. M. Ryzhik, Tables of Integrals,
* Series, and Products, p. 1073; Academic Press, 1980.
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

use crate::backend::cephes64::consts::MACHEP;

/* Expansion coefficients
* for Euler-Maclaurin summation formula
* (2k)! / B2k
* where B2k are Bernoulli numbers
*/
const A: [f64; 12] = [
    12.0,
    -720.0,
    30240.0,
    -1209600.0,
    47900160.0,
    -1.8924375803183791606e9, /*1.307674368e12/691 */
    7.47242496e10,
    -2.950130727918164224e12,  /*1.067062284288e16/3617 */
    1.1646782814350067249e14,  /*5.109094217170944e18/43867 */
    -4.5979787224074726105e15, /*8.028576626982912e20/174611 */
    1.8152105401943546773e17,  /*1.5511210043330985984e23/854513 */
    -7.1661652561756670113e18, /*1.6938241367317436694528e27/236364091 */
];

/* 30 Nov 86 -- error in third coefficient fixed */

// $$\mathrm{zeta}(x, q) = \sum_{k = 0}^{\infty}{(k + q)^{-x}}$$
//
// $$\mathrm{zeta}(x, q) = \sum_{k = 1}^{n}{(k + q)^{-x}} +
// \frac{(n + q)^{1 - x}}{x - 1}-\frac{1}{2\,(n + q)^x} +
// \sum_{j=1}^{\infty}{\frac{B_{2j}\,x\,(x + 1)\,...\,(x + 2\,j)}
// {(2\,j)!\,(n + q)^{x + 2\,j + 1}}}$$

pub fn zeta(x: f64, q: f64) -> f64 {
    if x == 1.0 {
        return f64::INFINITY;
    } else if x < 1.0 {
        //sf_error("zeta", SF_ERROR_DOMAIN, NULL);
        return f64::NAN;
    } else if q <= 0.0 {
        if q == q.floor() {
            //sf_error("zeta", SF_ERROR_SINGULAR, NULL);
            return f64::INFINITY;
        } else if x != x.floor() {
            return f64::NAN; /* because q^-x not defined */
        }
    }

    /* Asymptotic expansion
     * https://dlmf.nist.gov/25.11#E43
     */
    if q > 1e8 {
        return (1.0 / (x - 1.0) + 1.0 / (2.0 * q)) * q.powf(1.0 - x);
    }

    /* Euler-Maclaurin summation formula */

    /* Permit negative q but continue sum until n+q > +9 .
     * This case should be handled by a reflection formula.
     * If q<0 and x is an integer, there is a relation to
     * the polyGamma function.
     */
    let mut s = q.powf(-x);
    let mut a = q;
    let mut i = 0;
    let mut b = 0.0;
    while (i < 9) || (a <= 9.0) {
        i += 1;
        a += 1.0;
        b = a.powf(-x);
        s += b;
        if (b / s).abs() < MACHEP {
            return s;
        }
    }

    let w = a;
    s += b * w / (x - 1.0);
    s -= 0.5 * b;
    a = 1.0;
    let mut k = 0.0;
    for a_i in &A {
        a *= x + k;
        b /= w;
        let t = a * b / a_i;
        s += t;
        if (t / s).abs() < MACHEP {
            return s;
        }
        k += 1.0;
        a *= x + k;
        b /= w;
        k += 1.0;
    }

    // TODO: Find a way to test this case
    s
}
