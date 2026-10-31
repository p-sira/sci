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

// The moderate-positive beta path does not need Gamma's pole/reflection,
// Stirling, or infinite-argument handling. Keep its 2..3 Cephes rational
// approximation local so the three evaluations inline into the caller.
#[inline(always)]
fn gamma_moderate_positive(mut x: f64) -> f64 {
    let mut scale = 1.0;
    while x >= 3.0 {
        x -= 1.0;
        scale *= x;
    }
    while x < 2.0 {
        scale /= x;
        x += 1.0;
    }
    if x == 2.0 {
        return scale;
    }
    let t = x - 2.0;
    let p = 1.60119522476751861407e-4_f64
        .mul_add(t, 1.19135147006586384913e-3)
        .mul_add(t, 1.04213797561761569935e-2)
        .mul_add(t, 4.76367800457137231464e-2)
        .mul_add(t, 2.07448227648435975150e-1)
        .mul_add(t, 4.94214826801497100753e-1)
        .mul_add(t, 9.99999999999999996796e-1);
    let q = (-2.31581873324120129819e-5_f64)
        .mul_add(t, 5.39605580493303397842e-4)
        .mul_add(t, -4.45641913851797240494e-3)
        .mul_add(t, 1.18139785222060435552e-2)
        .mul_add(t, 3.58236398605498653373e-2)
        .mul_add(t, -2.34591795718243348568e-1)
        .mul_add(t, 7.14304917030273074085e-2)
        .mul_add(t, 1.00000000000000000320);
    scale * (p / q)
}

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

    // For moderate positive arguments, all three Gamma values and their
    // product fit comfortably in binary64. Specialized inlined Gamma avoids
    // both the longer reciprocal-Gamma Chebyshev expansion and the generic
    // Gamma pole/reflection/Stirling dispatch.
    // Leave the exceptional, highly unbalanced and large-argument cases on
    // the original Cephes paths below.
    if a >= 0.1 && b >= 0.1 && a + b <= 32.0 {
        return (gamma_moderate_positive(a) * gamma_moderate_positive(b)
            / gamma_moderate_positive(a + b))
        .ln();
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

#[cfg(test)]
mod tests {
    use super::{beta, lbeta};

    #[test]
    fn moderate_positive_logbeta_matches_high_precision_references() {
        // Independently evaluated with 80-digit mpmath log(beta(a,b)).
        // Includes exact CodSpeed input pairs from positive_values(128,1/2),
        // the small-shape edge, and the upper boundary of the fast path.
        let cases = [
            (0.1, 0.1, 2.9813614810376274),
            (0.25, 9.7, 0.7297356961285787),
            (5.1, 5.1, -6.594738407285101),
            (8.9, 1.37, -3.139775074596382),
            (10.1, 9.9, -13.735177544346377),
            (15.0, 14.99, -20.86749393422148),
            (16.0, 16.0, -22.293680785633527),
            (0.1, 31.9, 1.9078685837064263),
            (5.194074428837206, 9.271161254706481, -9.107256962634744),
            (6.583593939634305, 7.013954653016276, -9.092255165988822),
            (3.928633905082601, 3.7451057732121957, -4.6908536302730615),
            (8.497261096476889, 2.066586641881247, -4.514737247441051),
            (6.316081841493, 0.6387656150871024, -0.8180420809505264),
            (8.69635623326492, 5.884713080910398, -9.524858388782952),
            (1.6859440249728328, 3.925613346611085, -2.5383231548682404),
        ];
        for (a, b, reference) in cases {
            for (a, b) in [(a, b), (b, a)] {
                let result = lbeta(a, b);
                assert!(
                    (result - reference).abs() <= 7e-14 * reference.abs().max(1.0),
                    "lbeta({a:.17e},{b:.17e})={result:.17e}, 80-digit reference={reference:.17e}"
                );
            }
        }
    }

    #[test]
    fn logbeta_exceptional_arguments_remain_on_original_paths() {
        // Lower than the positive fast-path threshold: independent 80-digit
        // mpmath values, including a very small positive argument.
        for (a, b, reference) in [
            (0.099999999, 9.7, 2.030202196448405),
            (1e-8, 2.5, 18.420680731148644),
            (1e9, 0.5, -9.789267975423505),
            (-0.5, 2.75, 1.6158550755071859),
        ] {
            let result = lbeta(a, b);
            assert!(
                (result - reference).abs() <= 3e-11 * reference.abs().max(1.0),
                "lbeta({a},{b})={result}, reference={reference}"
            );
        }
        // Analytic continuation of B(a,b) through the negative-integer
        // recurrence, including the sign of beta (lost by log absolute beta).
        assert_eq!(beta(-2.0, 1.0), -0.5);
        assert_eq!(beta(-2.0, 2.0), 0.5);
        assert_eq!(lbeta(-2.0, 1.0), -2.0_f64.ln());
        assert_eq!(lbeta(-2.0, 2.0), -2.0_f64.ln());
        assert_eq!(lbeta(-2.0, 3.0), f64::INFINITY);
        assert_eq!(beta(-2.0, 3.0), f64::INFINITY);
    }
}
