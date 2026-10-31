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

// Use the asymptotic expansion after shifting positive arguments to x > 8.
// The eight Bernoulli terms resolve the full double-precision range there.
// This avoids the rational approximation and its downward recurrence for
// 0 < x <= 8, and leaves the Cephes reflection path for x <= 0 unchanged.
#[inline]
fn psi_positive(mut x: f64) -> f64 {
    let mut correction = 0.0;
    while x <= 8.0 {
        correction -= 1.0 / x;
        x += 1.0;
    }
    let inverse = 1.0 / x;
    let square = inverse * inverse;
    let bernoulli = [
        -3617.0 / 8160.0,
        1.0 / 12.0,
        -691.0 / 32760.0,
        5.0 / 660.0,
        -1.0 / 240.0,
        1.0 / 252.0,
        -1.0 / 120.0,
        1.0 / 12.0,
    ];
    correction + x.ln() - 0.5 * inverse - square * polevl(square, &bernoulli, 7)
}

// $$\mathrm{psi}(x) = \frac{d}{dx}\,\ln(\Gamma(x))$$
// $$\mathrm{psi}(x) = -\mathrm{EUL} + \sum_{k=1}^{n-1}{\frac{1}{k}}$$
// $$\mathrm{psi}(x) = \ln(x) - \frac{1}{2}\,x +
// \sum_{k=1}^{\infty}{\frac{B_{2\,k}}{2\,k\,x^{2\,k}}}$$

pub fn psi(x: f64) -> f64 {
    if x > 0.0 && x.is_finite() {
        return psi_positive(x);
    }
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

#[cfg(test)]
mod tests {
    use super::psi;

    #[test]
    fn positive_digamma_matches_scipy_at_every_benchmark_input() {
        // scipy.special.digamma, SciPy 1.17.0; matches the 128 points used in
        // benches/special.rs (0.1 + 49.9 * index / 127).
        const EXPECTED: [f64; 128] = [
            -10.423754940411076,
            -1.9989094972276544,
            -0.7825018296411963,
            -0.19357968451148685,
            0.18581640633801341,
            0.46360493084148663,
            0.68209035175129584,
            0.86190566248429434,
            1.0145799373470585,
            1.1471828882398594,
            1.2643527418561025,
            1.3692926612806242,
            1.4643056019364808,
            1.5511018422421248,
            1.6309858119816896,
            1.7049748255418955,
            1.773877430267581,
            1.8383467787477115,
            1.8989179937308651,
            1.9560349536414929,
            2.0100698965192025,
            2.0613380329888153,
            2.1101086177270236,
            2.1566134608717533,
            2.2010535577616963,
            2.2436043146901166,
            2.2844197127185826,
            2.3236356582315061,
            2.3613727035632404,
            2.3977382745856244,
            2.4328285086700898,
            2.4667297819993053,
            2.4995199871463272,
            2.531269608349334,
            2.562042631728136,
            2.591897319929557,
            2.620886874722983,
            2.6490600064417311,
            2.6764614255511505,
            2.7031322687786608,
            2.729110469984982,
            2.7544310841556592,
            2.7791265714466191,
            2.8032270470501013,
            2.8267605016992476,
            2.8497529968555702,
            2.8722288379883474,
            2.8942107288313244,
            2.9157199090683519,
            2.9367762775387964,
            2.9573985027521745,
            2.9776041222487235,
            2.9974096321298491,
            3.0168305679026703,
            3.0358815776304331,
            3.0545764882509845,
            3.0729283658148288,
            3.0909495702996104,
            3.1086518055765495,
            3.1260461650343632,
            3.1431431733057558,
            3.1599528244892441,
            3.1764846172136707,
            3.1927475868532591,
            3.2087503351665942,
            3.2245010576028128,
            3.2400075684918885,
            3.2552773243127398,
            3.2703174452125032,
            3.2851347349323565,
            3.2997356992793976,
            3.3141265632700483,
            3.3283132870579908,
            3.3423015807486101,
            3.3560969181920353,
            3.3697045498381453,
            3.3831295147290557,
            3.3963766516976031,
            3.4094506098340838,
            3.4223558582778781,
            3.4350966953855511,
            3.4476772573224657,
            3.4601015261208721,
            3.4723733372437331,
            3.4844963866902257,
            3.4964742376758497,
            3.5083103269173259,
            3.5200079705500285,
            3.5315703697034215,
            3.5430006157579559,
            3.5543016953050075,
            3.5654764948297659,
            3.5765278051354437,
            3.5874583255257493,
            3.5982706677613194,
            3.6089673598046006,
            3.6195508493666115,
            3.6300235072680387,
            3.640387630626194,
            3.6506454458785691,
            3.6607991116529117,
            3.670850721493101,
            3.6808023064494195,
            3.6906558375412346,
            3.7004132280995612,
            3.7100763359964715,
            3.7196469657678346,
            3.7291268706354681,
            3.7385177544343571,
            3.747821273450243,
            3.7570390381725365,
            3.7661726149671928,
            3.7752235276738988,
            3.7841932591316398,
            3.7930832526364688,
            3.801894913335063,
            3.8106296095574201,
            3.8192886740918803,
            3.827873405405414,
            3.836385068811996,
            3.8448248975916788,
            3.8531940940628426,
            3.8614938306099611,
            3.8697252506690716,
            3.8778894696730202,
            3.8859875759584459,
            3.8940206316363337,
            3.9019896734278925,
        ];
        for (index, &expected) in EXPECTED.iter().enumerate() {
            let x = 0.1 + 49.9 * index as f64 / 127.0;
            let actual = psi(x);
            assert!(
                (actual - expected).abs() <= 2e-14 * expected.abs().max(1.0),
                "psi({x}) = {actual}, SciPy = {expected}"
            );
        }
    }

    #[test]
    fn digamma_reflection_poles_and_positive_boundaries() {
        // Independently calculated with scipy.special.digamma (1.17.0).
        for &(x, expected) in &[
            (1e-8, -100000000.57721564),
            (0.5, -1.9635100260214235),
            (1.0, -0.5772156649015329),
            (1.4616321449683622, -9.241265521729427e-17),
            (2.0, 0.42278433509846713),
            (8.0, 2.0156414779556098),
            (9.0, 2.1406414779556098),
            (10.0, 2.251752589066721),
            (100.0, 4.600161852738088),
            (1e20, 46.051701859880914),
            (-0.5, 0.03648997397857651),
            (-2.3, 3.3173231575618236),
        ] {
            let actual = psi(x);
            assert!(
                (actual - expected).abs() <= 3e-14 * expected.abs().max(1.0),
                "psi({x}) = {actual}, SciPy = {expected}"
            );
        }
        assert_eq!(psi(0.0), f64::NEG_INFINITY);
        assert_eq!(psi(-0.0), f64::INFINITY);
        for x in [-1.0, -2.0, -100.0] {
            assert!(psi(x).is_nan());
        }
        assert!(psi(f64::NEG_INFINITY).is_nan());
        assert_eq!(psi(f64::INFINITY), f64::INFINITY);
        assert!(psi(f64::NAN).is_nan());
    }
}
