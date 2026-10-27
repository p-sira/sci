// Rust translation adapted from spec_math 0.1.6 (MIT); see LICENSES.
/*
* Cephes Math Library Release 2.0:  April, 1987
* Copyright 1984, 1987 by Stephen L. Moshier
* Direct inquiries to 30 Frost Street, Cambridge, MA 02140
*/

/* Scipy changes:
* - 07-18-2016: improve evaluation of dn near quarter periods
*/

#![allow(clippy::excessive_precision)]

#[allow(unused_imports)]
use sci_core::Float;

use super::consts::{M_PI_2, MACHEP};

pub fn ellpj(u: f64, m: f64) -> (f64, f64, f64, f64) // sn, cn, dn, ph
{
    /* Check for special cases */
    if !(0.0..=1.0).contains(&m) || m.is_nan() {
        //sf_error("ellpj", SF_ERROR_DOMAIN, NULL);
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    } else if m < 1.0e-9 {
        let t = u.sin();
        let b = u.cos();
        let ai = 0.25 * m * (u - t * b);
        (t - ai * b, b + ai * t, 1.0 - 0.5 * m * t * t, u - ai)
    } else if m >= 0.9999999999 {
        // TODO: values become NaN as phi becomes large (> 100)
        let mut ai = 0.25 * (1.0 - m);
        let b = u.cosh();
        let t = u.tanh();
        let phi = 1.0 / b;
        let twon = b * u.sinh();
        let sn = t + ai * (twon - u) / (b * b);
        let ph = 2.0 * u.exp().atan() - M_PI_2 + ai * (twon - u) / b;
        ai *= t * phi;
        let cn = phi - ai * (twon - u);
        let dn = phi + ai * (twon + u);
        (sn, cn, dn, ph)
    } else {
        /* A. G. M. scale. See DLMF 22.20(ii) */
        let mut a: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut b = (1.0 - m).sqrt();
        let mut c: [f64; 9] = [m.sqrt(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut twon = 1.0;
        let mut i = 0;

        while (c[i] / a[i]).abs() > MACHEP {
            if i > 7 {
                //sf_error("ellpj", SF_ERROR_OVERFLOW, NULL);
                break;
            }
            let ai = a[i];
            i += 1;
            c[i] = (ai - b) / 2.0;
            let t = (ai * b).sqrt();
            a[i] = (ai + b) / 2.0;
            b = t;
            twon *= 2.0;
        }

        /* backward recurrence */
        let mut phi = twon * a[i] * u;
        while i >= 1 {
            let t = c[i] * phi.sin() / a[i];
            b = phi;
            phi = (t.asin() + phi) / 2.0;
            i -= 1;
        }

        let sn = phi.sin();
        let t = phi.cos();
        let cn = t;
        let dnfac = (phi - b).cos();
        /* See discussion after DLMF 22.20.5 */
        let dn = if dnfac.abs() < 0.1 {
            (1.0 - m * sn * sn).sqrt()
        } else {
            t / dnfac
        };
        let ph = phi;
        (sn, cn, dn, ph)
    }
}
