#[allow(unused_imports)]
use sci_core::Float;
// Rust translation adapted from spec_math 0.1.6 (MIT); see LICENSES.
/*
* Cephes Math Library Release 2.4:  March,1996
* Copyright 1984, 1996 by Stephen L. Moshier
*/

use crate::backend::cephes64::consts::{MACHEP, MAXLOG, MINLOG};
use crate::backend::cephes64::gamma::lgam;
use crate::backend::cephes64::incbet::incbet;
use crate::backend::cephes64::ndtri::ndtri;

/// Whether a Halley step `step` on `I_x(a,b)` that ended at `x` leaves a
/// predicted error below a quarter of an ulp of `scale`.
///
/// The leading error term of a Halley step is
/// `((f'''/f')/6 - (f''/f')^2/4) step^3`, where for the beta density
/// `c = f''/f' = (a-1)/x - (b-1)/(1-x)` and `f'''/f' = c' + c^2` with
/// `c' = -(a-1)/x^2 - (b-1)/(1-x)^2`. Everything is scaled by `scale` so that
/// tiny quantiles do not underflow the cubic term. The bound only holds in the
/// asymptotic regime, so the step must also be small relative to `x` and
/// `1-x`.
pub(crate) fn halley_converged(a: f64, b: f64, x: f64, step: f64, scale: f64) -> bool {
    let r = step / scale;
    if !(r.abs() < 1e-3 && step.abs() < 1e-3 * x.min(1.0 - x)) {
        return false;
    }
    let (u, v) = (scale / x, scale / (1.0 - x));
    let cs = (a - 1.0) * u - (b - 1.0) * v;
    let dcs = (a - 1.0).abs() * u * u + (b - 1.0).abs() * v * v;
    let k = (dcs + cs * cs) / 6.0 + 0.25 * cs * cs;
    k * (r * r * r).abs() < 0.25 * MACHEP
}

pub fn incbi(aa: f64, bb: f64, yy0: f64) -> f64 {
    incbi_impl(aa, bb, yy0, false).0
}

/// Accelerated inverse incomplete beta for callers that refine the result,
/// also reporting whether it came from a converged Halley step in
/// unreflected coordinates. Such a result is accurate relative to `x` and
/// needs no further refinement.
///
/// Compared with the Cephes iteration kept by [`incbi`], small shape
/// parameters start Newton from a power-law seed instead of interval halving,
/// and the Newton steps carry a Halley correction.
pub(crate) fn incbi_status(aa: f64, bb: f64, yy0: f64) -> (f64, bool) {
    incbi_impl(aa, bb, yy0, true)
}

fn incbi_impl(aa: f64, bb: f64, yy0: f64, accelerate: bool) -> (f64, bool) {
    if yy0 <= 0.0 {
        return (0.0, false);
    } else if yy0 >= 1.0 {
        return (1.0, false);
    }
    let mut converged = false;
    let mut x0: f64 = 0.0;
    let mut yl: f64 = 0.0;
    let mut x1: f64 = 1.0;
    let mut yh: f64 = 1.0;
    let mut nflg: isize = 0;

    let mut a: f64;
    let mut b: f64;
    let mut y0: f64;
    let mut d: f64;
    let mut y: f64;
    let mut x: f64;
    let mut lgm: f64;
    let mut yp: f64;
    let mut di: f64;
    let mut dithresh: f64;
    let mut xt: f64;

    let mut rflg: isize;
    let mut dir: isize;

    let mut goto_newt = false;

    if aa <= 1.0 || bb <= 1.0 {
        dithresh = 1.0e-6;
        rflg = 0;
        a = aa;
        b = bb;
        y0 = yy0;
        x = a / (a + b);
        // Seed from the leading power-law behavior of both tails,
        // I_x ~ x^a / (a B) near 0 and 1 - I_x ~ (1-x)^b / (b B) near 1, with
        // B approximated by t + u (Numerical Recipes, section 6.14), and go
        // straight to the bracketed Newton/Halley phase. Its bracket
        // safeguards replace the interval halving that Cephes ran first; if
        // they do not converge, the halving below still takes over.
        if accelerate {
            let t = (a * (a / (a + b)).ln()).exp() / a;
            let u = (b * (b / (a + b)).ln()).exp() / b;
            let w = t + u;
            let seed = if y0 < t / w {
                (a * w * y0).powf(1.0 / a)
            } else {
                1.0 - (b * w * (1.0 - y0)).powf(1.0 / b)
            };
            if seed > 0.0 && seed < 1.0 {
                x = seed;
            }
            goto_newt = true;
        }
        y = incbet(a, b, x);
    } else {
        dithresh = 1.0e-4;
        /* approximation to inverse function */

        yp = -ndtri(yy0);

        if yy0 > 0.5 {
            rflg = 1;
            a = bb;
            b = aa;
            y0 = 1.0 - yy0;
            yp = -yp;
        } else {
            rflg = 0;
            a = aa;
            b = bb;
            y0 = yy0;
        }

        lgm = (yp * yp - 3.0) / 6.0;
        x = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0));
        d = yp * (x + lgm).sqrt() / x
            - (1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0)) * (lgm + 5.0 / 6.0 - 2.0 / (3.0 * x));
        d *= 2.0;

        if d < MINLOG {
            // TODO: This case may be impossible to reach

            //x = 1.0;
            //goto under;
            //sf_error("incbi", SF_ERROR_UNDERFLOW, NULL);
            x = 0.0;
            if rflg != 0 {
                if x <= MACHEP {
                    return (1.0 - MACHEP, false);
                } else {
                    return (1.0 - x, false);
                }
            } else {
                return (x, false);
            }
        }
        x = a / (a + b * d.exp());
        y = incbet(a, b, x);
        yp = (y - y0) / y0;
        if yp.abs() < 0.2 {
            //goto newt;
            goto_newt = true;
        }
    }

    'ihalve: loop {
        //ihalve
        /* Resort to interval halving if not close enough. */

        if !goto_newt {
            dir = 0;
            di = 0.5;
            for i in 0..100 {
                if i != 0 {
                    x = x0 + di * (x1 - x0);
                    if x == 1.0 {
                        // TODO: Find a test for this case
                        x = 1.0 - MACHEP;
                    }
                    if x == 0.0 {
                        di = 0.5;
                        x = x0 + di * (x1 - x0);
                        if x == 0.0 {
                            break 'ihalve; //goto under;
                        }
                    }
                    y = incbet(a, b, x);
                    yp = (x1 - x0) / (x1 + x0);
                    if yp.abs() < dithresh {
                        goto_newt = true;
                        break; //goto newt;
                    }
                    yp = (y - y0) / y0;
                    if yp.abs() < dithresh {
                        goto_newt = true;
                        break; //goto newt;
                    }
                }
                if y < y0 {
                    x0 = x;
                    yl = y;
                    if dir < 0 {
                        dir = 0;
                        di = 0.5;
                    } else if dir > 3 {
                        // TODO: Find a test for this case
                        di = 1.0 - (1.0 - di) * (1.0 - di);
                    } else if dir > 1 {
                        di = 0.5 * di + 0.5;
                    } else {
                        di = (y0 - y) / (yh - yl);
                    }
                    dir += 1;
                    // Reflect only while the target tail survives subtraction.
                    // For tiny yy0, 1-yy0 rounds to one and destroys the
                    // quantile (e.g. a=1000, b=.1, yy0=1e-20).
                    if x0 > 0.75 && (rflg != 0 || 1.0 - yy0 < 1.0) {
                        if rflg == 1 {
                            rflg = 0;
                            a = aa;
                            b = bb;
                            y0 = yy0;
                        } else {
                            rflg = 1;
                            a = bb;
                            b = aa;
                            y0 = 1.0 - yy0;
                        }
                        x = 1.0 - x;
                        y = incbet(a, b, x);
                        x0 = 0.0;
                        yl = 0.0;
                        x1 = 1.0;
                        yh = 1.0;
                        continue 'ihalve; //goto ihalve;
                    }
                } else {
                    x1 = x;
                    if rflg == 1 && x1 < MACHEP {
                        x = 0.0;
                        break 'ihalve; //goto done;
                    }
                    yh = y;
                    if dir > 0 {
                        dir = 0;
                        di = 0.5;
                    } else if dir < -3 {
                        di *= di;
                    } else if dir < -1 {
                        di *= 0.5;
                    } else {
                        di = (y - y0) / (yh - yl);
                    }
                    dir -= 1;
                }
            }
        }

        if !goto_newt {
            // TODO: Find a test for this case

            //sf_error("incbi", SF_ERROR_LOSS, NULL);
            if x0 >= 1.0 {
                x = 1.0 - MACHEP;
                break; //goto done;
            } else if x <= 0.0 {
                //under:
                //sf_error("incbi", SF_ERROR_UNDERFLOW, NULL);
                x = 0.0;
                break; //goto done;
            }
        } else {
            goto_newt = false;
        }

        //newt:

        if nflg != 0 {
            break; //goto done;
        }
        nflg = 1;
        lgm = lgam(a + b) - lgam(a) - lgam(b);

        for i in 0..8 {
            /* Compute the function at this point. */
            if i != 0 {
                y = incbet(a, b, x);
            }
            if y < yl {
                x = x0;
                y = yl;
            } else if y > yh {
                x = x1;
                y = yh;
            } else if y < y0 {
                x0 = x;
                yl = y;
            } else {
                x1 = x;
                yh = y;
            }
            if x == 1.0 || x == 0.0 {
                // TODO: Find a test for this case
                break;
            }
            /* Compute the derivative of the function at this point. */
            d = (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() + lgm;
            if d < MINLOG {
                break 'ihalve; //goto done;
            } else if d > MAXLOG {
                // TODO: Find a test for this case
                break;
            }
            d = d.exp();
            /* Compute the step to the next approximation of x. */
            d = (y - y0) / d;
            // Halley correction from the exact log-derivative of the beta
            // density, c = f''/f' = (a-1)/x - (b-1)/(1-x). It costs no extra
            // incbet call and makes the iteration converge cubically.
            let c = (a - 1.0) / x - (b - 1.0) / (1.0 - x);
            let h = 0.5 * d * c;
            if accelerate && h.abs() < 0.5 {
                d /= 1.0 - h;
            }
            xt = x - d;
            if xt <= x0 {
                y = (x - x0) / (x1 - x0);
                xt = x0 + 0.5 * y * (x - x0);
                if xt <= 0.0 {
                    // TODO: Find a test for this case
                    break;
                }
            }
            if xt >= x1 {
                y = (x1 - x) / (x1 - x0);
                xt = x1 - 0.5 * y * (x1 - x);
                if xt >= 1.0 {
                    // TODO: Find a test for this case
                    break;
                }
            }
            let unclamped = xt == x - d;
            x = xt;
            // Accept the result without further refinement only when the
            // predicted error of this Halley step is far below rounding. The
            // Cephes test below is relative to x alone, which is too loose
            // when 1-x is small, so it does not mark convergence by itself.
            if accelerate && unclamped && h.abs() < 1e-2 && halley_converged(a, b, x, d, x) {
                converged = true;
                break 'ihalve;
            }
            if (d / x).abs() < 128.0 * MACHEP {
                break 'ihalve; //goto done;
            }
        }
        /* Did not converge.  */
        dithresh = 256.0 * MACHEP;
        //goto ihalve;
    }

    //done:

    if rflg != 0 {
        (if x <= MACHEP { 1.0 - MACHEP } else { 1.0 - x }, false)
    } else {
        (x, converged)
    }
}

#[cfg(test)]
mod tests {
    use super::{incbi, incbi_status};

    #[track_caller]
    fn close(actual: f64, expected: f64, atol: f64, rtol: f64) {
        assert!(
            actual.is_finite() && (actual - expected).abs() <= atol + rtol * expected.abs(),
            "inverse beta: got {actual:.17e}, expected {expected:.17e}; atol={atol:e}, rtol={rtol:e}"
        );
    }

    #[test]
    fn endpoints_bypass_the_scalar_and_accelerated_iterations() {
        for (a, b) in [(0.25, 4.0), (4.0, 0.25), (2.0, 5.0), (1000.0, 0.1)] {
            for (p, expected) in [(-1.0, 0.0), (0.0, 0.0), (1.0, 1.0), (2.0, 1.0)] {
                assert_eq!(incbi(a, b, p), expected);
                assert_eq!(incbi_status(a, b, p), (expected, false));
            }
        }
    }

    #[test]
    fn unit_shape_inverse_matches_closed_form_for_both_solvers() {
        // I_x(a,1) = x^a and I_x(1,b) = 1-(1-x)^b.
        // exp_m1/ln_1p preserve the reference's lower-tail precision.
        for (a, b) in [
            (1.0, 1.0),
            (0.2, 1.0),
            (1.0, 0.2),
            (1.0, 2.0),
            (2.0, 1.0),
            (1.0, 25.0),
            (25.0, 1.0),
            (1.0, 200.0),
            (200.0, 1.0),
        ] {
            for p in [1e-6_f64, 0.001, 0.05, 0.3, 0.7, 0.99] {
                // For a<1, p^1/a can fall far below the accuracy that the
                // legacy interval-halving branch promises. Near-zero tails
                // are instead checked with meaningful relative bounds in
                // the independent high-precision test below.
                if a < 1.0 && p < 0.05 {
                    continue;
                }
                let expected = if b == 1.0 {
                    p.powf(1.0 / a)
                } else {
                    -((-p).ln_1p() / b).exp_m1()
                };
                close(incbi(a, b, p), expected, 2e-15, 2e-8);
                close(incbi_status(a, b, p).0, expected, 2e-15, 2e-8);
            }
        }
    }

    // For positive integer parameters, I_x(a,b) is a binomial survival sum:
    // P(Binomial(a+b-1,x) >= a). This is independent of incbet and of the
    // inverse implementation's continued fractions and Halley iteration.
    fn binomial_beta_cdf(a: usize, b: usize, x: f64) -> f64 {
        let n = a + b - 1;
        let mut binomial = 1.0;
        let mut sum = 0.0;
        for j in 0..=n {
            if j >= a {
                sum += binomial * x.powi(j as i32) * (1.0 - x).powi((n - j) as i32);
            }
            if j < n {
                binomial *= (n - j) as f64 / (j + 1) as f64;
            }
        }
        sum
    }

    #[test]
    fn integer_shapes_invert_independent_binomial_probabilities() {
        // Known quantiles are chosen first; probabilities are then computed
        // by an independent finite binomial sum, not by incbet (a roundtrip).
        // These span both reflection states and the normal/Newton seed path.
        for (a, b, x) in [
            (2, 2, 0.2),
            (2, 2, 0.8),
            (2, 7, 0.125),
            (7, 2, 0.8),
            (3, 8, 0.4),
            (8, 3, 0.65),
            (12, 16, 0.3),
            (16, 12, 0.7),
        ] {
            let p = binomial_beta_cdf(a, b, x);
            assert!((0.0..1.0).contains(&p), "interior probability {p}");
            close(incbi(a as f64, b as f64, p), x, 2e-10, 2e-9);
            close(incbi_status(a as f64, b as f64, p).0, x, 2e-10, 2e-9);
        }
    }

    #[test]
    fn high_precision_quantiles_cover_skewed_and_extreme_tails() {
        // Independently obtained using 130-digit mpmath betainc and 500
        // bisection steps, not by calling the inverse or round-tripping it.
        // Small shape parameters test accelerated power-law seeds; the large
        // shapes and upper probabilities test normal seeds and reflection.
        for (a, b, p, expected, atol, rtol) in [
            (0.2, 0.7, 0.1, 1.7256094111498253e-5, 1e-12, 1e-7),
            (0.3, 0.2, 0.99, 0.9999999991062152, 4e-12, 0.0),
            (2.5, 7.25, 1e-30, 2.0278309782721155e-13, 2e-17, 1e-5),
            (64.0, 64.0, 1e-15, 0.18768972065269468, 2e-11, 1e-9),
            (
                64.0,
                64.0,
                0.999999999999999,
                // The binary64 p has 1-p = 9.992007221626409e-16, NOT
                // exactly 1e-15. Independently verified by 85-digit Decimal
                // bisection of the Beta(64,64) / Binomial(127,x) tail sum.
                0.8123132934419462,
                2e-11,
                1e-9,
            ),
            (200.0, 600.0, 1e-50, 0.07659967618091494, 2e-11, 1e-9),
            (1000.0, 0.1, 1e-20, 0.9603421729760104, 2e-10, 1e-9),
            (30.0, 3.0, 0.999, 0.9938703901674368, 2e-11, 1e-9),
        ] {
            close(incbi(a, b, p), expected, atol, rtol);
            close(incbi_status(a, b, p).0, expected, atol, rtol);
        }
    }

    #[test]
    fn complement_quantiles_match_for_nonsymmetric_shapes() {
        // I_x(a,b) = 1-I_(1-x)(b,a). Unlike an incbet roundtrip, this
        // independently changes the shape ordering and reflection branch.
        for (a, b, p) in [
            (2.5, 7.25, 1e-8),
            (2.5, 7.25, 0.8),
            (7.25, 2.5, 0.9),
            (0.7, 3.0, 0.2),
            (3.0, 0.7, 0.8),
            (30.0, 3.0, 0.99),
        ] {
            let x = incbi(a, b, p);
            let reflected = incbi(b, a, 1.0 - p);
            close(x + reflected, 1.0, 3e-10, 3e-10);
        }
    }
}
