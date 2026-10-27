#[allow(unused_imports)]
use sci_core::Float;
// Rust translation adapted from spec_math 0.1.6 (MIT); see LICENSES.
/*  pdtri()
*
*  Inverse Poisson distribution
*
*
*
* SYNOPSIS:
*
* int k;
* double m, y, pdtr();
*
* m = pdtri( k, y );
*
*
*
*
* DESCRIPTION:
*
* Finds the Poisson variable x such that the integral
* from 0 to x of the Poisson density is equal to the
* given probability y.
*
* This is accomplished using the inverse Gamma integral
* function and the relation
*
*    m = igamci( k+1, y ).
*
*
*
*
* ACCURACY:
*
* See igami.c.
*
* ERROR MESSAGES:
*
*   message         condition      value returned
* pdtri domain    y < 0 or y >= 1       0.0
*                     k < 0
*
*/

/*
* Cephes Math Library Release 2.3:  March, 1995
* Copyright 1984, 1987, 1995 by Stephen L. Moshier
*/

use crate::backend::cephes64::igam::{igam, igamc};
use crate::backend::cephes64::igami::igamci;

pub fn pdtrc(k: f64, m: f64) -> f64 {
    if k < 0.0 || m < 0.0 {
        //sf_error("pdtrc", SF_ERROR_DOMAIN, NULL);
        f64::NAN
    } else if m == 0.0 {
        0.0
    } else {
        let v = k.floor() + 1.0;
        igam(v, m)
    }
}

// $$\sum_{j=0}^k{e^{-m}\,\frac{m^j}{j!}}$$

pub fn pdtr(k: f64, m: f64) -> f64 {
    if k < 0.0 || m < 0.0 {
        //sf_error("pdtr", SF_ERROR_DOMAIN, NULL);
        f64::NAN
    } else if m == 0.0 {
        1.0
    } else {
        let v = k.floor() + 1.0;
        igamc(v, m)
    }
}

pub fn pdtri(k: isize, y: f64) -> f64 {
    if k < 0 || !(0.0..1.0).contains(&y) {
        //sf_error("pdtri", SF_ERROR_DOMAIN, NULL);
        f64::NAN
    } else {
        let v = k as f64 + 1.0;
        igamci(v, y)
    }
}

// Very few tests due to simplicity of implementation
