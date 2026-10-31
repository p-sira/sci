// Rust translation adapted from spec_math 0.1.6 (MIT); see LICENSES.
/*
* Cephes Math Library Release 2.1:  January, 1989
* Copyright 1985, 1987, 1989 by Stephen L. Moshier
* Direct inquiries to 30 Frost Street, Cambridge, MA 02140
*/

#![allow(clippy::excessive_precision)]

#[allow(unused_imports)]
use sci_core::Float;

const A: [f64; 8] = [
    4.65128586073990045278E-5,
    7.31589045238094711071E-3,
    1.33847639578309018650E-1,
    8.79691311754530315341E-1,
    2.71149851196553469920E0,
    4.25697156008121755724E0,
    3.29771340985225106936E0,
    1.00000000000000000126E0,
];

const B: [f64; 8] = [
    6.90990488912553276999E-4,
    2.54043763932544379113E-2,
    2.82974860602568089943E-1,
    1.41172597751831069617E0,
    3.63800533345137075418E0,
    5.03278880143316990390E0,
    3.54771340985225096217E0,
    9.99999999999999998740E-1,
];

use crate::backend::cephes64::consts::M_PI;
use crate::backend::cephes64::polevl::polevl;

pub fn spence(x: f64) -> f64 {
    if x < 0.0 {
        //sf_error("spence", SF_ERROR_DOMAIN, NULL);
        f64::NAN
    } else if x == 1.0 {
        0.0
    } else if x == 0.0 {
        M_PI * M_PI / 6.0
    } else {
        let mut flag = 0;
        let mut x = x;

        if x > 2.0 {
            x = 1.0 / x;
            flag |= 2;
        }

        let w = if x > 1.5 {
            flag |= 2;
            1.0 / x - 1.0
        } else if x < 0.5 {
            flag |= 1;
            -x
        } else {
            x - 1.0
        };

        let mut y = -w * polevl(w, &A, 7) / polevl(w, &B, 7);

        if flag & 1 != 0 {
            y = (M_PI * M_PI) / 6.0 - x.ln() * (1.0 - x).ln() - y;
        }

        if flag & 2 != 0 {
            let z = x.ln();
            y = -0.5 * z * z - y;
        }

        y
    }
}
