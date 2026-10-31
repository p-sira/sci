#[allow(unused_imports)]
use sci_core::Float;
// Rust translation adapted from spec_math 0.1.6 (MIT); see LICENSES.
/*
* Cephes Math Library Release 2.8: June, 2000
* Copyright 1984, 1987, 2000 by Stephen L. Moshier
*/

use crate::backend::cephes64::consts::M_PI;
use crate::backend::cephes64::jv::jv;
use crate::backend::cephes64::yn::yn;

pub fn yv(v: f64, x: f64) -> f64 {
    let n = v as isize;
    if n as f64 == v {
        yn(n, x)
    } else if v == v.floor() {
        /* Zero in denominator. */
        //sf_error("yv", SF_ERROR_DOMAIN, NULL);
        f64::NAN
    } else {
        let c = crate::backend::cephes64::cospi(v);
        let s = crate::backend::cephes64::sinpi(v);
        let first = if c == 0.0 { 0.0 } else { c * jv(v, x) };
        let y = (first - jv(-v, x)) / s;

        if y.is_infinite() {
            if v > 0.0 {
                //sf_error("yv", SF_ERROR_OVERFLOW, NULL);
                -f64::INFINITY
            } else if v < -1e10 {
                /* Whether it's +inf or -inf is numerically ill-defined. */
                //sf_error("yv", SF_ERROR_DOMAIN, NULL);
                f64::NAN
            } else {
                y
            }
        } else {
            y
        }
    }
}
