// Rust translation adapted from spec_math 0.1.6 (MIT); see LICENSES.
/*
* (C) Copyright John Maddock 2006.
* Use, modification and distribution are subject to the
* Boost Software License, Version 1.0. (See accompanying file
*  LICENSE_1_0.txt or copy at https://www.boost.org/LICENSE_1_0.txt)
*/

#![allow(clippy::excessive_precision)]

#[allow(unused_imports)]
use sci_core::Float;

use crate::backend::cephes64::consts::EULER;
use crate::backend::cephes64::consts::MACHEP;
use crate::backend::cephes64::gamma::{gamma, lgam};
use crate::backend::cephes64::igam::{igam_with_fac, igamc_with_fac};
use crate::backend::cephes64::polevl::polevl;
use crate::backend::cephes64::unity::log1p;

const A: [f64; 4] = [
    0.213623493715853,
    4.28342155967104,
    11.6616720288968,
    3.31125922108741,
];
const B: [f64; 5] = [
    0.3611708101884203e-1,
    1.27364489782223,
    6.40691597760039,
    6.61053765625462,
    1.0,
];

fn find_inverse_s(p: f64, q: f64) -> f64 {
    /*
     * Computation of the Incomplete Gamma Function Ratios and their Inverse
     * ARMIDO R. DIDONATO and ALFRED H. MORRIS, JR.
     * ACM Transactions on Mathematical Software, Vol. 12, No. 4,
     * December 1986, Pages 377-393.
     *
     * See equation 32.
     */

    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * q.ln()).sqrt()
    };
    let s = t - polevl(t, &A, 3) / polevl(t, &B, 4);
    if p < 0.5 { -s } else { s }
}

fn didonato_sn(a: f64, x: f64, n: usize, tolerance: f64) -> f64 {
    /*
     * Computation of the Incomplete Gamma Function Ratios and their Inverse
     * ARMIDO R. DIDONATO and ALFRED H. MORRIS, JR.
     * ACM Transactions on Mathematical Software, Vol. 12, No. 4,
     * December 1986, Pages 377-393.
     *
     * See equation 34.
     */
    let mut sum: f64 = 1.0;

    if n >= 1 {
        let mut partial: f64 = x / (a + 1.0);

        sum += partial;
        for i in 2..=n {
            partial *= x / (a + i as f64);
            sum += partial;
            if partial < tolerance {
                break;
            }
        }
    }

    sum
}

fn find_inverse_gamma(a: f64, p: f64, q: f64) -> f64 {
    /*
     * In order to understand what's going on here, you will
     * need to refer to:
     *
     * Computation of the Incomplete Gamma Function Ratios and their Inverse
     * ARMIDO R. DIDONATO and ALFRED H. MORRIS, JR.
     * ACM Transactions on Mathematical Software, Vol. 12, No. 4,
     * December 1986, Pages 377-393.
     */

    if a == 1.0 {
        if q > 0.9 { -log1p(-p) } else { -q.ln() }
    } else if a < 1.0 {
        let g: f64 = gamma(a);
        let b: f64 = q * g;

        if (b > 0.6) || ((b >= 0.45) && (a >= 0.3)) {
            /* DiDonato & Morris Eq 21:
             *
             * There is a slight variation from DiDonato and Morris here:
             * the first form given here is unstable when p is close to 1,
             * making it impossible to compute the inverse of Q(a,x) for small
             * q. Fortunately the second form works perfectly well in this case.
             */
            let u: f64 = if (b * q > 1e-8) && (q > 1e-5) {
                (p * g * a).powf(1.0 / a)
            } else {
                ((-q / a) - EULER).exp()
            };
            u / (1.0 - (u / (a + 1.0)))
        } else if (a < 0.3) && (b >= 0.35) {
            /* DiDonato & Morris Eq 22: */
            let t = (-EULER - b).exp();
            let u = t * t.exp();
            t * u.exp()
        } else if (b > 0.15) || (a >= 0.3) {
            /* DiDonato & Morris Eq 23: */
            let y = -b.ln();
            let u = y - (1.0 - a) * y.ln();
            y - (1.0 - a) * u.ln() - (1.0 + (1.0 - a) / (1.0 + u)).ln()
        } else if b > 0.1 {
            // TODO: This case may be impossible to reach
            /* DiDonato & Morris Eq 24: */
            let y = -b.ln();
            let u = y - (1.0 - a) * y.ln();
            y - (1.0 - a) * u.ln()
                - ((u * u + 2.0 * (3.0 - a) * u + (2.0 - a) * (3.0 - a))
                    / (u * u + (5.0 - a) * u + 2.0))
                    .ln()
        } else {
            /* DiDonato & Morris Eq 25: */
            let y = -b.ln();
            let c1 = (a - 1.0) * y.ln();
            let c1_2 = c1 * c1;
            let c1_3 = c1_2 * c1;
            let c1_4 = c1_2 * c1_2;
            let a_2 = a * a;
            let a_3 = a_2 * a;

            let c2 = (a - 1.0) * (1.0 + c1);
            let c3 = (a - 1.0) * (-(c1_2 / 2.0) + (a - 2.0) * c1 + (3.0 * a - 5.0) / 2.0);
            let c4 = (a - 1.0)
                * ((c1_3 / 3.0) - (3.0 * a - 5.0) * c1_2 / 2.0
                    + (a_2 - 6.0 * a + 7.0) * c1
                    + (11.0 * a_2 - 46.0 * a + 47.0) / 6.0);
            let c5 = (a - 1.0)
                * (-(c1_4 / 4.0)
                    + (11.0 * a - 17.0) * c1_3 / 6.0
                    + (-3.0 * a_2 + 13.0 * a - 13.0) * c1_2
                    + (2.0 * a_3 - 25.0 * a_2 + 72.0 * a - 61.0) * c1 / 2.0
                    + (25.0 * a_3 - 195.0 * a_2 + 477.0 * a - 379.0) / 12.0);

            let y_2 = y * y;
            let y_3 = y_2 * y;
            let y_4 = y_2 * y_2;
            y + c1 + (c2 / y) + (c3 / y_2) + (c4 / y_3) + (c5 / y_4)
        }
    } else {
        /* DiDonato and Morris Eq 31: */
        let s = find_inverse_s(p, q);

        let s_2 = s * s;
        let s_3 = s_2 * s;
        let s_4 = s_2 * s_2;
        let s_5 = s_4 * s;
        let ra = a.sqrt();

        let mut w = a + s * ra + (s_2 - 1.0) / 3.0;
        w += (s_3 - 7.0 * s) / (36.0 * ra);
        w -= (3.0 * s_4 + 7.0 * s_2 - 16.0) / (810.0 * a);
        w += (9.0 * s_5 + 256.0 * s_3 - 433.0 * s) / (38880.0 * a * ra);

        if (a >= 500.0) && ((1.0 - w / a).abs() < 1e-6) {
            w
        } else if p > 0.5 {
            if w < 3.0 * a {
                w
            } else {
                let d = 2.0_f64.max(a * (a - 1.0));
                let lg = lgam(a);
                let lb = q.ln() + lg;
                if lb < -d * 2.3 {
                    /* DiDonato and Morris Eq 25: */
                    let y = -lb;
                    let c1 = (a - 1.0) * y.ln();
                    let c1_2 = c1 * c1;
                    let c1_3 = c1_2 * c1;
                    let c1_4 = c1_2 * c1_2;
                    let a_2 = a * a;
                    let a_3 = a_2 * a;

                    let c2 = (a - 1.0) * (1.0 + c1);
                    let c3 = (a - 1.0) * (-(c1_2 / 2.0) + (a - 2.0) * c1 + (3.0 * a - 5.0) / 2.0);
                    let c4 = (a - 1.0)
                        * ((c1_3 / 3.0) - (3.0 * a - 5.0) * c1_2 / 2.0
                            + (a_2 - 6.0 * a + 7.0) * c1
                            + (11.0 * a_2 - 46.0 * a + 47.0) / 6.0);
                    let c5 = (a - 1.0)
                        * (-(c1_4 / 4.0)
                            + (11.0 * a - 17.0) * c1_3 / 6.0
                            + (-3.0 * a_2 + 13.0 * a - 13.0) * c1_2
                            + (2.0 * a_3 - 25.0 * a_2 + 72.0 * a - 61.0) * c1 / 2.0
                            + (25.0 * a_3 - 195.0 * a_2 + 477.0 * a - 379.0) / 12.0);

                    let y_2 = y * y;
                    let y_3 = y_2 * y;
                    let y_4 = y_2 * y_2;
                    y + c1 + (c2 / y) + (c3 / y_2) + (c4 / y_3) + (c5 / y_4)
                } else {
                    /* DiDonato and Morris Eq 33: */
                    let u = -lb + (a - 1.0) * w.ln() - (1.0 + (1.0 - a) / (1.0 + w)).ln();
                    -lb + (a - 1.0) * u.ln() - (1.0 + (1.0 - a) / (1.0 + u)).ln()
                }
            }
        } else {
            let mut z = w;
            let ap1 = a + 1.0;
            let ap2 = a + 2.0;
            if w < 0.15 * ap1 {
                /* DiDonato and Morris Eq 35: */
                let v = p.ln() + lgam(ap1);
                z = ((v + w) / a).exp();
                let mut s = log1p(z / ap1 * (1.0 + z / ap2));
                z = ((v + z - s) / a).exp();
                s = log1p(z / ap1 * (1.0 + z / ap2));
                z = ((v + z - s) / a).exp();
                s = log1p(z / ap1 * (1.0 + z / ap2 * (1.0 + z / (a + 3.0))));
                z = ((v + z - s) / a).exp();
            }

            if (z <= 0.01 * ap1) || (z > 0.7 * ap1) {
                z
            } else {
                /* DiDonato and Morris Eq 36: */
                let ls = didonato_sn(a, z, 100, 1e-4).ln();
                let v = p.ln() + lgam(ap1);
                let z = ((v + z - ls) / a).exp();
                z * (1.0 - (a * z.ln() - z - v + ls) / (a - z))
            }
        }
    }
}

pub fn igami(a: f64, p: f64) -> f64 {
    if a.is_nan() || p.is_nan() || (a < 0.0) || !(0.0..=1.0).contains(&p) {
        //sf_error("gammaincinv", SF_ERROR_DOMAIN, NULL);
        return f64::NAN;
    } else if p == 0.0 {
        return 0.0;
    } else if p == 1.0 {
        return f64::INFINITY;
    } else if p > 0.9 {
        return igamci(a, 1.0 - p);
    }

    let x = find_inverse_gamma(a, p, 1.0 - p);
    halley_refine(a, x, |x| {
        let (value, fac) = igam_with_fac(a, x);
        (value - p, fac)
    })
}

pub fn igamci(a: f64, q: f64) -> f64 {
    // int i;
    // double x, fac, f_fp, fpp_fp;

    if a.is_nan() || q.is_nan() {
        return f64::NAN;
    } else if (a < 0.0) || !(0.0..=1.0).contains(&q) {
        //sf_error("gammainccinv", SF_ERROR_DOMAIN, NULL);
    } else if q == 0.0 {
        return f64::INFINITY;
    } else if q == 1.0 {
        return 0.0;
    } else if q > 0.9 {
        return igami(a, 1.0 - q);
    }

    let x = find_inverse_gamma(a, 1.0 - q, q);
    halley_refine(a, x, |x| {
        let (value, fac) = igamc_with_fac(a, x);
        (q - value, fac)
    })
}

/// Up to three Halley steps on `P(a, x) - p` from the initial estimate `x`.
///
/// `residual(x)` returns `(P(a, x) - p, x^a e^-x / Γ(a))`, so the derivative
/// is `fac / x` and `f''/f' = (a - 1)/x - 1`. This is the Cephes iteration,
/// but it stops early once the predicted error of the last step is far below
/// rounding (see [`halley_converged`]) instead of always taking three steps.
fn halley_refine(a: f64, mut x: f64, residual: impl Fn(f64) -> (f64, f64)) -> f64 {
    for _ in 0..3 {
        let (f, fac) = residual(x);
        if fac == 0.0 {
            break;
        }
        let f_fp = f * x / fac;
        /* The ratio of the first and second derivatives simplifies */
        let fpp_fp = -1.0 + (a - 1.0) / x;
        if fpp_fp.is_infinite() {
            /* Resort to Newton's method in the case of overflow */
            x -= f_fp;
        } else {
            let step = f_fp / (1.0 - 0.5 * f_fp * fpp_fp);
            x -= step;
            if halley_converged(a, x, step) {
                break;
            }
        }
    }

    x
}

/// Whether a Halley step `step` on `P(a, x)` that ended at `x` leaves a
/// predicted error below a quarter of an ulp of `x`.
///
/// The leading error term of a Halley step is
/// `(c^2/4 - (c' + c^2)/6) step^3` with `c = f''/f' = (a-1)/x - 1` and
/// `c' = -(a-1)/x^2`. Bounding it in units of `x` gives
/// `k = (|a-1| + (cx)^2)/6 + (cx)^2/4` with `cx = a - 1 - x`. The bound only
/// holds in the asymptotic regime, so the step must also be small relative
/// to `x`.
fn halley_converged(a: f64, x: f64, step: f64) -> bool {
    let r = step / x;
    if r.is_nan() || r.abs() >= 1e-3 {
        return false;
    }
    let cx = a - 1.0 - x;
    let k = ((a - 1.0).abs() + cx * cx) / 6.0 + 0.25 * cx * cx;
    k * (r * r * r).abs() < 0.25 * MACHEP
}
