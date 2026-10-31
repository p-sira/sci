// Rust translation adapted from spec_math 0.1.6 (MIT); see LICENSES.
/*
* Cephes Math Library Release 2.0:  April, 1987
* Copyright 1985, 1987 by Stephen L. Moshier
* Direct inquiries to 30 Frost Street, Cambridge, MA 02140
*/

#![allow(clippy::excessive_precision)]

#[allow(unused_imports)]
use sci_core::Float;

/* Chebyshev coefficients for reciprocal Gamma function
* in interval 0 to 1.  Function is 1/(x Gamma(x)) - 1
*/

const R: [f64; 16] = [
    3.13173458231230000000E-17,
    -6.70718606477908000000E-16,
    2.20039078172259550000E-15,
    2.47691630348254132600E-13,
    -6.60074100411295197440E-12,
    5.13850186324226978840E-11,
    1.08965386454418662084E-9,
    -3.33964630686836942556E-8,
    2.68975996440595483619E-7,
    2.96001177518801696639E-6,
    -8.04814124978471142852E-5,
    4.16609138709688864714E-4,
    5.06579864028608725080E-3,
    -6.41925436109158228810E-2,
    -4.98558728684003594785E-3,
    1.27546015610523951063E-1,
];

use crate::backend::cephes64::chbevl::chbevl;
use crate::backend::cephes64::consts::{M_PI, MAXLOG};
use crate::backend::cephes64::gamma::lgam;
use crate::backend::cephes64::sinpi::sinpi;

pub fn rgamma(x: f64) -> f64 {
    if x > 34.84425627277176174 {
        (-lgam(x)).exp()
    } else if x < -34.034 {
        let w = -x;
        let mut z = sinpi(w);
        if z == 0.0 {
            return 0.0;
        }

        let sign: f64 = if z < 0.0 {
            z = -z;
            1.0
        } else {
            -1.0
        };

        let y = (w * z).ln() - M_PI.ln() + lgam(w);
        if y < -MAXLOG {
            //sf_error(name, SF_ERROR_UNDERFLOW, NULL);
            sign * 0.0
        } else if y > MAXLOG {
            //sf_error(name, SF_ERROR_OVERFLOW, NULL);
            sign * f64::INFINITY
        } else {
            sign * y.exp()
        }
    } else {
        let mut z = 1.0;
        let mut w = x;

        while w > 1.0 {
            /* Downward recurrence */
            w -= 1.0;
            z *= w;
        }

        while w < 0.0 {
            /* Upward recurrence */
            z /= w;
            w += 1.0;
        }

        if w == 0.0 {
            /* Nonpositive integer */
            0.0
        } else if w == 1.0 {
            /* Other integer */
            1.0 / z
        } else {
            w * (1.0 + chbevl(4.0 * w - 2.0, &R, 16)) / z
        }
    }
}
