// Rust translation adapted from spec_math 0.1.6 (MIT); see LICENSES.
/*
* Cephes Math Library Release 2.2:  July, 1992
* Copyright 1984, 1987, 1989, 1992 by Stephen L. Moshier
* Direct inquiries to 30 Frost Street, Cambridge, MA 02140
*/

#![allow(clippy::excessive_precision)]

#[allow(unused_imports)]
use sci_core::Float;

const P: [f64; 7] = [
    1.60119522476751861407E-4,
    1.19135147006586384913E-3,
    1.04213797561761569935E-2,
    4.76367800457137231464E-2,
    2.07448227648435975150E-1,
    4.94214826801497100753E-1,
    9.99999999999999996796E-1,
];

const Q: [f64; 8] = [
    -2.31581873324120129819E-5,
    5.39605580493303397842E-4,
    -4.45641913851797240494E-3,
    1.18139785222060435552E-2,
    3.58236398605498653373E-2,
    -2.34591795718243348568E-1,
    7.14304917030273074085E-2,
    1.00000000000000000320E0,
];

const MAXGAM: f64 = 171.624376956302725;
const LOGPI: f64 = 1.14472988584940017414;

/* Stirling's formula for the Gamma function */
const STIR: [f64; 5] = [
    7.87311395793093628397E-4,
    -2.29549961613378126380E-4,
    -2.68132617805781232825E-3,
    3.47222221605458667310E-3,
    8.33333333333482257126E-2,
];

const MAXSTIR: f64 = 143.01608;
const SQTPI: f64 = 2.50662827463100050242E0;

use crate::backend::cephes64::consts::M_PI;
use crate::backend::cephes64::polevl::{p1evl, polevl};

/* Gamma function computed by Stirling's formula.
* The polynomial STIR is valid for 33 <= x <= 172.
*/
fn stirf(x: f64) -> f64 {
    if x >= MAXGAM {
        return f64::INFINITY;
    }
    let w = 1.0 / x;
    let w = 1.0 + w * polevl(w, &STIR, 4);
    let y = x.exp();
    let y = if x > MAXSTIR {
        /* Avoid overflow in pow() */
        let v = x.powf(0.5 * x - 0.25);
        v * (v / y)
    } else {
        x.powf(x - 0.5) / y
    };
    SQTPI * y * w
}

pub fn gamma(x: f64) -> f64 {
    let mut sgngam = 1;

    if x.is_infinite() {
        //return x;
        // M. Romanowicz: Should return NAN for negative infinity
        if x > 0.0 {
            return x;
        } else {
            return f64::NAN;
        }
    }
    let q = x.abs();

    if q > 33.0 {
        let z = if x < 0.0 {
            let mut p = q.floor();
            if p == q {
                // gamnan:
                // sf_error("Gamma", SF_ERROR_OVERFLOW, NULL);
                //return f64::INFINITY;
                // M. Romanowicz: Should return NAN for negative integers
                return f64::NAN;
            }
            if (p as isize & 1) == 0 {
                sgngam = -1;
            }
            let mut z = q - p;
            if z > 0.5 {
                p += 1.0;
                z = q - p;
            }
            z = q * (M_PI * z).sin();
            if z == 0.0 {
                return sgngam as f64 * f64::INFINITY;
            }
            z = z.abs();
            M_PI / (z * stirf(q))
        } else {
            stirf(x)
        };
        return sgngam as f64 * z;
    }

    let mut z = 1.0;
    let mut x = x;
    while x >= 3.0 {
        x -= 1.0;
        z *= x;
    }

    while x < 0.0 {
        if x > -1.0E-9 {
            if x == 0.0 {
                //return f64::INFINITY;
                // M. Romanowicz: Should return NAN for negative integers
                return f64::NAN;
            } else {
                return z / ((1.0 + 0.5772156649015329 * x) * x);
            }
        }
        z /= x;
        x += 1.0;
    }

    while x < 2.0 {
        if x < 1.0e-9 {
            if x == 0.0 {
                //return f64::INFINITY;
                // M. Romanowicz: Should return NAN for negative integers
                return f64::NAN;
            } else {
                return z / ((1.0 + 0.5772156649015329 * x) * x);
            }
        }
        z /= x;
        x += 1.0;
    }

    if x == 2.0 {
        return z;
    }

    x -= 2.0;
    let p = polevl(x, &P, 6);
    let q = polevl(x, &Q, 7);

    z * p / q
}

/* A[]: Stirling's formula expansion of log Gamma
* B[], C[]: log Gamma function between 2 and 3
*/
const A: [f64; 5] = [
    8.11614167470508450300E-4,
    -5.95061904284301438324E-4,
    7.93650340457716943945E-4,
    -2.77777777730099687205E-3,
    8.33333333333331927722E-2,
];

const B: [f64; 6] = [
    -1.37825152569120859100E3,
    -3.88016315134637840924E4,
    -3.31612992738871184744E5,
    -1.16237097492762307383E6,
    -1.72173700820839662146E6,
    -8.53555664245765465627E5,
];

const C: [f64; 6] = [
    /* 1.00000000000000000000E0, */
    -3.51815701436523470549E2,
    -1.70642106651881159223E4,
    -2.20528590553854454839E5,
    -1.13933444367982507207E6,
    -2.53252307177582951285E6,
    -2.01889141433532773231E6,
];

/* log( sqrt( 2*pi ) ) */
const LS2PI: f64 = 0.91893853320467274178;

const MAXLGM: f64 = 2.556348e305;

pub fn lgam(x: f64) -> f64 {
    let mut sign = 0;
    lgam_sgn(x, &mut sign)
}

pub fn lgam_sgn(x: f64, sign: &mut isize) -> f64 {
    *sign = 1;

    if x.is_infinite() {
        return x;
    }

    if x < -34.0 {
        let q = -x;
        let w: f64 = lgam_sgn(q, sign);
        let mut p = q.floor();
        if p == q {
            //lgsing:
            //sf_error("lgam", SF_ERROR_SINGULAR, NULL);
            return f64::INFINITY;
        }
        let i = p as isize;
        if (i & 1) == 0 {
            *sign = -1;
        } else {
            *sign = 1;
        }
        let mut z = q - p;
        if z > 0.5 {
            p += 1.0;
            z = p - q;
        }
        z = q * (M_PI * z).sin();
        if z == 0.0 {
            return f64::INFINITY;
        }
        /*     z = log(M_PI) - log( z ) - w; */
        z = LOGPI - z.ln() - w;
        return z;
    }
    let mut x = x;
    if x < 13.0 {
        let mut z = 1.0;
        let mut p = 0.0;
        let mut u = x;
        while u >= 3.0 {
            p -= 1.0;
            u = x + p;
            z *= u;
        }
        while u < 2.0 {
            if u == 0.0 {
                return f64::INFINITY;
            }
            z /= u;
            p += 1.0;
            u = x + p;
        }
        if z < 0.0 {
            *sign = -1;
            z = -z;
        } else {
            *sign = 1;
        }
        if u == 2.0 {
            return z.ln();
        }
        p -= 2.0;
        x += p;
        p = x * polevl(x, &B, 5) / p1evl(x, &C, 6);
        return z.ln() + p;
    }

    if x > MAXLGM {
        return (*sign as f64) * f64::INFINITY;
    }

    let mut q = (x - 0.5) * x.ln() - x + LS2PI;
    if x > 1.0e8 {
        return q;
    }

    let p = 1.0 / (x * x);
    if x >= 1000.0 {
        q += ((7.9365079365079365079365e-4 * p - 2.7777777777777777777778e-3) * p
            + 0.0833333333333333333333)
            / x;
    } else {
        q += polevl(p, &A, 4) / x;
    }

    q
}
