// Rust translation adapted from spec_math 0.1.6 (MIT); see LICENSES.
/*                                                     incbet.c
*
*     Incomplete beta integral
*
*
* SYNOPSIS:
*
* double a, b, x, y, incbet();
*
* y = incbet( a, b, x );
*
*
* DESCRIPTION:
*
* Returns incomplete beta integral of the arguments, evaluated
* from zero to x.  The function is defined as
*
*                  x
*     -            -
*    | (a+b)      | |  a-1     b-1
*  -----------    |   t   (1-t)   dt.
*   -     -     | |
*  | (a) | (b)   -
*                 0
*
* The domain of definition is 0 <= x <= 1.  In this
* implementation a and b are restricted to positive values.
* The integral from x to 1 may be obtained by the symmetry
* relation
*
*    1 - incbet( a, b, x )  =  incbet( b, a, 1-x ).
*
* The integral is evaluated by a continued fraction expansion
* or, when b*x is small, by a power series.
*
* ACCURACY:
*
* Tested at uniformly distributed random points (a,b,x) with a and b
* in "domain" and x between 0 and 1.
*                                        Relative error
* arithmetic   domain     # trials      peak         rms
*    IEEE      0,5         10000       6.9e-15     4.5e-16
*    IEEE      0,85       250000       2.2e-13     1.7e-14
*    IEEE      0,1000      30000       5.3e-12     6.3e-13
*    IEEE      0,10000    250000       9.3e-11     7.1e-12
*    IEEE      0,100000    10000       8.7e-10     4.8e-11
* Outputs smaller than the IEEE gradual underflow threshold
* were excluded from these statistics.
*
* ERROR MESSAGES:
*   message         condition      value returned
* incbet domain      x<0, x>1          0.0
* incbet underflow                     0.0
*/

/*
* Cephes Math Library, Release 2.3:  March, 1995
* Copyright 1984, 1995 by Stephen L. Moshier
*/

#![allow(clippy::excessive_precision)]

#[allow(unused_imports)]
use sci_core::Float;

const MAXGAM: f64 = 171.624376956302725;

use crate::backend::cephes64::beta::{beta, lbeta};
use crate::backend::cephes64::consts::{MACHEP, MAXLOG, MINLOG};

const BIG: f64 = 4.503599627370496e15;
const BIGINV: f64 = 2.22044604925031308085e-16;

/* Power series for incomplete beta integral.
* Use when b*x is small and x not too close to 1.  */

fn pseries(a: f64, b: f64, x: f64) -> f64 {
    //double s, t, u, v, n, t1, z, ai;

    let ai = 1.0 / a;
    let mut u = (1.0 - b) * x;
    let mut v = u / (a + 1.0);
    let t1 = v;
    let mut t = u;
    let mut n = 2.0;
    let mut s = 0.0;
    let z = MACHEP * ai;
    while v.abs() > z {
        u = (n - b) * x / n;
        t *= u;
        v = t / (a + n);
        s += v;
        n += 1.0;
    }
    s += t1;
    s += ai;

    u = a * x.ln();
    if a + b < MAXGAM && u.abs() < MAXLOG {
        t = 1.0 / beta(a, b);
        s * t * x.powf(a)
    } else {
        t = -lbeta(a, b) + u + s.ln();
        if t < MINLOG { 0.0 } else { t.exp() }
    }
}

/* Continued fraction expansion #1 for incomplete beta integral */

fn incbcf(a: f64, b: f64, x: f64) -> f64 {
    let mut k1 = a;
    let mut k2 = a + b;
    let mut k3 = a;
    let mut k4 = a + 1.0;
    let mut k5: f64 = 1.0;
    let mut k6 = b - 1.0;
    let mut k7 = k4;
    let mut k8 = a + 2.0;

    let mut pkm2: f64 = 0.0;
    let mut qkm2: f64 = 1.0;
    let mut pkm1: f64 = 1.0;
    let mut qkm1: f64 = 1.0;
    let mut ans: f64 = 1.0;
    let mut r: f64 = 1.0;
    let thresh = 3.0 * MACHEP;
    for _ in 0..300 {
        let xk = -(x * k1 * k2) / (k3 * k4);
        let pk = pkm1 + pkm2 * xk;
        let qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        let xk = (x * k5 * k6) / (k7 * k8);
        let pk = pkm1 + pkm2 * xk;
        let qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        if qk != 0.0 {
            r = pk / qk;
        }
        let t: f64;
        if r != 0.0 {
            t = ((ans - r) / r).abs();
            ans = r;
        } else {
            t = 1.0;
        }

        if t < thresh {
            break;
        }

        k1 += 1.0;
        k2 += 1.0;
        k3 += 2.0;
        k4 += 2.0;
        k5 += 1.0;
        k6 -= 1.0;
        k7 += 2.0;
        k8 += 2.0;

        if qk.abs() + pk.abs() > BIG {
            pkm2 *= BIGINV;
            pkm1 *= BIGINV;
            qkm2 *= BIGINV;
            qkm1 *= BIGINV;
        }
        if qk.abs() < BIGINV || pk.abs() < BIGINV {
            pkm2 *= BIG;
            pkm1 *= BIG;
            qkm2 *= BIG;
            qkm1 *= BIG;
        }
    }

    ans
}

/* Continued fraction expansion #2 for incomplete beta integral */

fn incbd(a: f64, b: f64, x: f64) -> f64 {
    let mut k1 = a;
    let mut k2 = b - 1.0;
    let mut k3 = a;
    let mut k4 = a + 1.0;
    let mut k5: f64 = 1.0;
    let mut k6 = a + b;
    let mut k7 = a + 1.0;
    let mut k8 = a + 2.0;

    let mut pkm2: f64 = 0.0;
    let mut qkm2: f64 = 1.0;
    let mut pkm1: f64 = 1.0;
    let mut qkm1: f64 = 1.0;
    let z = x / (1.0 - x);
    let mut ans: f64 = 1.0;
    let mut r: f64 = 1.0;
    let thresh = 3.0 * MACHEP;
    for _ in 0..300 {
        let xk = -(z * k1 * k2) / (k3 * k4);
        let pk = pkm1 + pkm2 * xk;
        let qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        let xk = (z * k5 * k6) / (k7 * k8);
        let pk = pkm1 + pkm2 * xk;
        let qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        if qk != 0.0 {
            r = pk / qk;
        }
        let t: f64;
        if r != 0.0 {
            t = ((ans - r) / r).abs();
            ans = r;
        } else {
            t = 1.0;
        }

        if t < thresh {
            break;
        }

        k1 += 1.0;
        k2 -= 1.0;
        k3 += 2.0;
        k4 += 2.0;
        k5 += 1.0;
        k6 += 1.0;
        k7 += 2.0;
        k8 += 2.0;

        if qk.abs() + pk.abs() > BIG {
            pkm2 *= BIGINV;
            pkm1 *= BIGINV;
            qkm2 *= BIGINV;
            qkm1 *= BIGINV;
        }
        if qk.abs() < BIGINV || pk.abs() < BIGINV {
            pkm2 *= BIG;
            pkm1 *= BIG;
            qkm2 *= BIG;
            qkm1 *= BIG;
        }
    }

    ans
}

// $$\frac{\Gamma(a + b)}{\Gamma(a)\,\Gamma(b)}\,
// \int_{0}^{x}{t^{a-1}\,(1-t)^{b-1}\,dt}$$

pub fn incbet(aa: f64, bb: f64, xx: f64) -> f64 {
    if aa <= 0.0 || bb <= 0.0 {
        //sf_error("incbet", SF_ERROR_DOMAIN, NULL);
        return f64::NAN;
    } else if xx <= 0.0 || xx >= 1.0 {
        if xx == 0.0 {
            return 0.0;
        } else if xx == 1.0 {
            return 1.0;
        } else {
            //sf_error("incbet", SF_ERROR_DOMAIN, NULL);
            return f64::NAN;
        }
    }

    let mut flag: isize = 0;
    if bb * xx <= 1.0 && xx <= 0.95 {
        let t = pseries(aa, bb, xx);
        return t;
    }

    let mut w = 1.0 - xx;

    /* Reverse a and b if x is greater than the mean. */
    let (a, b, xc, x) = if xx > aa / (aa + bb) {
        flag = 1;
        (bb, aa, xx, w)
    } else {
        (aa, bb, w, xx)
    };

    let mut t: f64;
    if flag == 1 && b * x <= 1.0 && x <= 0.95 {
        t = pseries(a, b, x);
    } else {
        /* Choose expansion for better convergence. */
        let mut y = x * (a + b - 2.0) - (a - 1.0);
        if y < 0.0 {
            w = incbcf(a, b, x);
        } else {
            // TODO: Make more tests for this case
            w = incbd(a, b, x) / xc;
        }

        /* Multiply w by the factor
         * a      b   _             _     _
         * x  (1-x)   | (a+b) / ( a | (a) | (b) ) .   */

        y = a * x.ln();
        t = b * xc.ln();
        if a + b < MAXGAM && y.abs() < MAXLOG && t.abs() < MAXLOG {
            t = xc.powf(b);
            t *= x.powf(a);
            t /= a;
            t *= w;
            t *= 1.0 / beta(a, b);
        } else {
            /* Resort to logarithms.  */
            y += t - lbeta(a, b);
            y += (w / a).ln();
            if y < MINLOG {
                t = 0.0;
            } else {
                t = y.exp();
            }
        }
    }

    if flag == 1 {
        if t <= MACHEP { 1.0 - MACHEP } else { 1.0 - t }
    } else {
        t
    }
}
