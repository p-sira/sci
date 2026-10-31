// Rust translation adapted from spec_math 0.1.6 (MIT); see LICENSES.
/* Cephes Math Library Release 1.1:  March, 1985
* Copyright 1985 by Stephen L. Moshier
* Direct inquiries to 30 Frost Street, Cambridge, MA 02140 */

/* Sources
* [1] NIST, "The Digital Library of Mathematical Functions", dlmf.nist.gov
*/

/* Scipy changes:
* - 09-10-2016: improved asymptotic expansion for large n
*/

#![allow(clippy::excessive_precision)]

#[allow(unused_imports)]
use sci_core::Float;

use crate::backend::cephes64::consts::{MACHEP, MAXLOG};
use crate::backend::cephes64::gamma::gamma;
use crate::backend::cephes64::polevl::polevl;

const N_A: usize = 13;

const A0: [f64; 1] = [1.00000000000000000];

const A1: [f64; 1] = [1.00000000000000000];

const A2: [f64; 2] = [-2.00000000000000000, 1.00000000000000000];

const A3: [f64; 3] = [
    6.00000000000000000,
    -8.00000000000000000,
    1.00000000000000000,
];

const A4: [f64; 4] = [
    -24.0000000000000000,
    58.0000000000000000,
    -22.0000000000000000,
    1.00000000000000000,
];

const A5: [f64; 5] = [
    120.000000000000000,
    -444.000000000000000,
    328.000000000000000,
    -52.0000000000000000,
    1.00000000000000000,
];

const A6: [f64; 6] = [
    -720.000000000000000,
    3708.00000000000000,
    -4400.00000000000000,
    1452.00000000000000,
    -114.000000000000000,
    1.00000000000000000,
];

const A7: [f64; 7] = [
    5040.00000000000000,
    -33984.0000000000000,
    58140.0000000000000,
    -32120.0000000000000,
    5610.00000000000000,
    -240.000000000000000,
    1.00000000000000000,
];

const A8: [f64; 8] = [
    -40320.0000000000000,
    341136.000000000000,
    -785304.000000000000,
    644020.000000000000,
    -195800.000000000000,
    19950.0000000000000,
    -494.000000000000000,
    1.00000000000000000,
];

const A9: [f64; 9] = [
    362880.000000000000,
    -3733920.00000000000,
    11026296.0000000000,
    -12440064.0000000000,
    5765500.00000000000,
    -1062500.00000000000,
    67260.0000000000000,
    -1004.00000000000000,
    1.00000000000000000,
];

const A10: [f64; 10] = [
    -3628800.00000000000,
    44339040.0000000000,
    -162186912.000000000,
    238904904.000000000,
    -155357384.000000000,
    44765000.0000000000,
    -5326160.00000000000,
    218848.000000000000,
    -2026.00000000000000,
    1.00000000000000000,
];

const A11: [f64; 11] = [
    39916800.0000000000,
    -568356480.000000000,
    2507481216.00000000,
    -4642163952.00000000,
    4002695088.00000000,
    -1648384304.00000000,
    314369720.000000000,
    -25243904.0000000000,
    695038.000000000000,
    -4072.00000000000000,
    1.00000000000000000,
];

const A12: [f64; 12] = [
    -479001600.000000000,
    7827719040.00000000,
    -40788301824.0000000,
    92199790224.0000000,
    -101180433024.000000,
    56041398784.0000000,
    -15548960784.0000000,
    2051482776.00000000,
    -114876376.000000000,
    2170626.00000000000,
    -8166.00000000000000,
    1.00000000000000000,
];

const EUL: f64 = 0.57721566490153286060;
const BIG: f64 = 1.44115188075855872E+17;

pub fn expn(n: i32, x: f64) -> f64 {
    if x.is_nan() || n < 0 || x < 0.0 {
        //sf_error("expn", SF_ERROR_DOMAIN, NULL);
        f64::NAN
    } else if x > MAXLOG {
        0.0
    } else if x == 0.0 {
        if n < 2 {
            //sf_error("expn", SF_ERROR_SINGULAR, NULL);
            f64::INFINITY
        } else {
            1.0 / (n as f64 - 1.0)
        }
    } else if n == 0 {
        (-x).exp() / x
    } else if n > 50 {
        /* Asymptotic expansion for large n, DLMF 8.20(ii) */
        expn_large_n(n, x)
    } else if x > 1.0 {
        /* Continued fraction, DLMF 8.19.17 */
        let mut k: i32 = 1;
        let mut pkm2: f64 = 1.0;
        let mut qkm2 = x;
        let mut pkm1: f64 = 1.0;
        let mut qkm1 = x + n as f64;
        let mut ans = pkm1 / qkm1;
        let mut t: f64;
        let mut yk: f64;
        let mut xk: f64;

        loop {
            k += 1;
            if k & 1 != 0 {
                yk = 1.0;
                xk = (n + (k - 1) / 2) as f64;
            } else {
                yk = x;
                xk = (k / 2) as f64;
            }
            let pk = pkm1 * yk + pkm2 * xk;
            let qk = qkm1 * yk + qkm2 * xk;
            if qk != 0.0 {
                let r = pk / qk;
                t = ((ans - r) / r).abs();
                ans = r;
            } else {
                t = 1.0;
            }
            pkm2 = pkm1;
            pkm1 = pk;
            qkm2 = qkm1;
            qkm1 = qk;
            if pk.abs() > BIG {
                pkm2 /= BIG;
                pkm1 /= BIG;
                qkm2 /= BIG;
                qkm1 /= BIG;
            }

            if t.partial_cmp(&MACHEP) != Some(core::cmp::Ordering::Greater) {
                break;
            }
        }

        ans * (-x).exp()
    } else {
        /* Power series expansion, DLMF 8.19.8 */
        let mut psi = -EUL - x.ln();
        for i in 1..n {
            psi += 1.0 / i as f64;
        }

        let z = -x;
        let mut xk: f64 = 0.0;
        let mut yk: f64 = 1.0;
        let mut pk = 1.0 - n as f64;
        let mut ans = if n == 1 { 0.0 } else { 1.0 / pk };

        loop {
            xk += 1.0;
            yk *= z / xk;
            pk += 1.0;
            if pk != 0.0 {
                ans += yk / pk;
            }
            let t = if ans != 0.0 { (yk / ans).abs() } else { 1.0 };
            if t.partial_cmp(&MACHEP) != Some(core::cmp::Ordering::Greater) {
                break;
            }
        }

        //let k = xk;
        let t = n;
        let r = n - 1;
        z.powi(r) * psi / gamma(t as f64) - ans
    }
}

/* Asymptotic expansion for large n, DLMF 8.20(ii) */
fn expn_large_n(n: i32, x: f64) -> f64 {
    let p = n as f64;
    let lambda = x / p;
    let multiplier = 1.0 / p / (lambda + 1.0) / (lambda + 1.0);
    let mut fac: f64 = 1.0;
    let mut res: f64 = 1.0; /* A[0] = 1 */

    let expfac = (-lambda * p).exp() / (lambda + 1.0) / p;
    if expfac == 0.0 {
        //sf_error("expn", SF_ERROR_UNDERFLOW, NULL);
        return 0.0;
    }

    /* Do the k = 1 term outside the loop since A[1] = 1 */
    fac *= multiplier;
    res += fac;

    for k in 2..N_A {
        fac *= multiplier;
        let term = match k {
            0 => fac * polevl(lambda, &A0, 0),
            1 => fac * polevl(lambda, &A1, 0),
            2 => fac * polevl(lambda, &A2, 1),
            3 => fac * polevl(lambda, &A3, 2),
            4 => fac * polevl(lambda, &A4, 3),
            5 => fac * polevl(lambda, &A5, 4),
            6 => fac * polevl(lambda, &A6, 5),
            7 => fac * polevl(lambda, &A7, 6),
            8 => fac * polevl(lambda, &A8, 7),
            9 => fac * polevl(lambda, &A9, 8),
            10 => fac * polevl(lambda, &A10, 9),
            11 => fac * polevl(lambda, &A11, 10),
            12 => fac * polevl(lambda, &A12, 11),
            _ => panic!("Impossible State"),
        };
        //fac * polevl(lambda, &A[k], Adegs[k]);

        res += term;
        if term.abs() < MACHEP * res.abs() {
            break;
        }
    }

    expfac * res
}
