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
