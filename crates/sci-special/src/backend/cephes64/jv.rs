// Rust translation adapted from spec_math 0.1.6 (MIT); see LICENSES.
/*
* Cephes Math Library Release 2.8:  June, 2000
* Copyright 1984, 1987, 1989, 1992, 2000 by Stephen L. Moshier
*/

#![allow(clippy::excessive_precision)]

#[allow(unused_imports)]
use sci_core::Float;

use crate::backend::cephes64::airy::airy;
use crate::backend::cephes64::cbrt;
use crate::backend::cephes64::consts::{M_PI, MACHEP, MAXLOG};
use crate::backend::cephes64::gamma::{gamma, lgam_sgn};
use crate::backend::cephes64::j0::j0;
use crate::backend::cephes64::j1::j1;
use crate::backend::cephes64::polevl::polevl;
use crate::backend::utils::frexp;

const MAXGAM: f64 = 171.624376956302725;
const BIG: f64 = 1.44115188075855872E+17;

pub fn jv(n: f64, x: f64) -> f64 {
    let mut n = n;
    let mut x = x;
    let mut nint = 0; /* Flag for integer n */
    let mut sign = 1; /* Flag for sign inversion */
    let an = n.abs();
    let mut y = an.floor();
    if y == an {
        nint = 1;
        let i = (an - 16384.0 * (an / 16384.0).floor()) as isize;
        if n < 0.0 {
            if i & 1 != 0 {
                sign = -sign;
            }
            n = an;
        }
        if x < 0.0 {
            if i & 1 != 0 {
                sign = -sign;
            }
            x = -x;
        }
        if n == 0.0 {
            // case 1
            return j0(x);
        } else if n == 1.0 {
            // case 2
            return sign as f64 * j1(x);
        }
        // case 3
    }
    // else {
    // case 4
    //}

    if (x < 0.0) && (y != an) {
        // case 5
        //sf_error("Jv", SF_ERROR_DOMAIN, NULL);
        y = f64::NAN;
        return sign as f64 * y;
    }

    if x == 0.0 && n < 0.0 && nint == 0 {
        // case 6
        //sf_error("Jv", SF_ERROR_OVERFLOW, NULL);
        return f64::INFINITY / gamma(n + 1.0);
    }

    y = x.abs();

    if y * y < (n + 1.0).abs() * MACHEP {
        // case 7
        return (0.5 * x).powf(n) / gamma(n + 1.0);
    }

    let mut k = 3.6 * y.sqrt();
    let mut t = 3.6 * an.sqrt();
    if (y < t) && (an > 21.0) {
        // case 8
        return sign as f64 * jvs(n, x);
    }
    if (an < k) && (y > 21.0) {
        // case 9
        return sign as f64 * hankel(n, x);
    }

    if an < 500.0 {
        let mut q;
        /* Note: if x is too large, the continued fraction will fail; but then the
         * Hankel expansion can be used. */
        if nint != 0 {
            k = 0.0;
            q = recur(&mut n, x, &mut k, 1);
            if k == 0.0 {
                // case 10
                y = j0(x) / q;
                return sign as f64 * y;
            } else if k == 1.0 {
                // case 11
                y = j1(x) / q;
                return sign as f64 * y;
            }
            // case 12
        }
        // else {
        // case 13
        //}

        if (an > 2.0 * y) || ((0.0..20.0).contains(&n) && (y > 6.0) && (y < 20.0)) {
            // case 14
            /* Recur backwards from a larger value of n */

            // TODO: This case appears to give bad results
            k = n;

            y = y + an + 1.0;
            if y < 30.0 {
                y = 30.0;
            }
            y = n + (y - n).floor();
            q = recur(&mut y, x, &mut k, 0);
            y = jvs(y, x) * q;
            return sign as f64 * y;
        }

        if k <= 30.0 {
            // case 15
            k = 2.0;
        } else if k < 90.0 {
            // case 16
            k = (3.0 * k) / 4.0;
        }
        // else {
        // case 17
        //}

        if an > (k + 3.0) {
            if n < 0.0 {
                // case 18
                k = -k;
            }
            // else {
            // case 19
            //}
            q = n - n.floor();
            k = k.floor() + q;
            if n > 0.0 {
                // case 20
                q = recur(&mut n, x, &mut k, 1);
            } else {
                // case 21
                t = k;
                k = n;
                q = recur(&mut t, x, &mut k, 1);
                k = t;
            }
            if q == 0.0 {
                // case 22
                y = 0.0;
                return sign as f64 * y;
            }

            // case 23
        } else {
            // case 24
            k = n;
            q = 1.0;
        }

        /* boundary between convergence of
         * power series and Hankel expansion
         */
        y = k.abs();
        if y < 26.0 {
            // case 25
            t = (0.0083 * y + 0.09) * y + 12.9;
        } else {
            // case 26
            t = 0.9 * y;
        }

        if x > t {
            // case 27
            y = hankel(k, x);
        } else {
            // case 28
            y = jvs(k, x);
        }

        if n > 0.0 {
            // case 29
            y /= q;
        } else {
            // case 30
            y *= q;
        }
    } else {
        /* For large n, use the uniform expansion or the transitional expansion.
         * But if x is of the order of n**2, these may blow up, whereas the
         * Hankel expansion will then work.
         */
        if n < 0.0 {
            // case 31
            //sf_error("Jv", SF_ERROR_LOSS, NULL);
            y = f64::NAN;
            return sign as f64 * y;
        }
        t = x / n;
        t /= n;
        if t > 0.3 {
            // case 32
            y = hankel(n, x);
        } else {
            // case 33
            y = jnx(n, x);
        }
    }

    // case 34
    sign as f64 * y
}

/* Reduce the order by backward recurrence.
* AMS55 #9.1.27 and 9.1.73.
*/

fn recur(n: &mut f64, x: f64, newn: &mut f64, cancel: isize) -> f64 {
    /* Continued fraction for Jn(x)/Jn-1(x)
     * AMS 9.1.73
     *
     *    x       -x^2      -x^2
     * ------  ---------  ---------   ...
     * 2 n +   2(n+1) +   2(n+2) +
     *
     * Compute it with the simplest possible algorithm.
     *
     * This continued fraction starts to converge when (|n| + m) > |x|.
     * Hence, at least |x|-|n| iterations are necessary before convergence is
     * achieved. There is a hard limit set below, m <= 30000, which is chosen
     * so that no branch in `jv` requires more iterations to converge.
     * The exact maximum number is (500/3.6)^2 - 500 ~ 19000
     */

    let maxiter = 22000;
    let mut miniter = (x.abs() - n.abs()) as i32;
    if miniter < 1 {
        miniter = 1;
    }

    let mut nflag = if *n < 0.0 { 1 } else { 0 };

    let mut ans: f64;

    loop {
        let mut pkm2 = 0.0;
        let mut qkm2 = 1.0;
        let mut pkm1 = x;
        let mut qkm1 = *n + *n;
        let xk = -x * x;
        let mut yk = qkm1;
        ans = 0.0; /* ans=0.0 ensures that t=1.0 in the first iteration */
        let mut ctr = 0;
        loop {
            yk += 2.0;
            let pk = pkm1 * yk + pkm2 * xk;
            let qk = qkm1 * yk + qkm2 * xk;
            pkm2 = pkm1;
            pkm1 = pk;
            qkm2 = qkm1;
            qkm1 = qk;

            /* check convergence */
            let r = if qk != 0.0 && ctr > miniter {
                pk / qk
            } else {
                0.0
            };

            let t;
            if r != 0.0 {
                t = ((ans - r) / r).abs();
                ans = r;
            } else {
                t = 1.0;
            }

            ctr += 1;
            if ctr > maxiter {
                //sf_error("jv", SF_ERROR_UNDERFLOW, NULL);
                break;
            }
            if t < MACHEP {
                break;
            }

            /* renormalize coefficients */
            if pk.abs() > BIG {
                pkm2 /= BIG;
                pkm1 /= BIG;
                qkm2 /= BIG;
                qkm1 /= BIG;
            }
            if t <= MACHEP {
                break;
            }
        }

        if ans == 0.0 {
            ans = 1.0;
        }

        /* Change n to n-1 if n < 0 and the continued fraction is small */
        if nflag > 0 && ans.abs() < 0.125 {
            nflag = -1;
            *n -= 1.0;
            continue;
        }
        break;
    }

    let kf = *newn;

    /* backward recurrence
     *              2k
     *  J   (x)  =  --- J (x)  -  J   (x)
     *   k-1         x   k         k+1
     */

    let mut pk = 1.0;
    let mut pkm1 = 1.0 / ans;
    let mut k = *n - 1.0;
    let mut r = 2.0 * k;
    let mut pkm2: f64;
    loop {
        pkm2 = (pkm1 * r - pk * x) / x;
        pk = pkm1;
        pkm1 = pkm2;
        r -= 2.0;
        k -= 1.0;
        if k <= kf + 0.5 {
            break;
        }
    }

    /* Take the larger of the last two iterates
     * on the theory that it may have less cancellation error.
     */

    if cancel != 0 && (kf >= 0.0) && (pk.abs() > pkm1.abs()) {
        k += 1.0;
        pkm2 = pk;
    }
    *newn = k;

    pkm2
}

/* Ascending power series for Jv(x).
* AMS55 #9.1.10.
*/

fn jvs(n: f64, x: f64) -> f64 {
    let mut sgngam: isize = 0; // Value doesn't matter, will be overwritten

    let z = -x * x / 4.0;
    let mut u = 1.0;
    let mut y = u;
    let mut k = 1.0;
    let mut t = 1.0;

    while t > MACHEP {
        u *= z / (k * (n + k));
        y += u;
        k += 1.0;
        if y != 0.0 {
            t = (u / y).abs();
        }
    }

    let (_, mut ex) = frexp(0.5 * x);
    ex = (ex as f64 * n) as i32;
    if (ex > -1023) && (ex < 1023) && (n > 0.0) && (n < (MAXGAM - 1.0)) {
        let t = (0.5 * x).powf(n) / gamma(n + 1.0);
        y * t
    } else {
        let mut t = n * (0.5 * x).ln() - lgam_sgn(n + 1.0, &mut sgngam);
        if y < 0.0 {
            sgngam = -sgngam;
            y = -y;
        }
        t += y.ln();
        if t < -MAXLOG {
            return 0.0;
        } else if t > MAXLOG {
            //sf_error("Jv", SF_ERROR_OVERFLOW, NULL);
            return f64::INFINITY;
        }
        sgngam as f64 * t.exp()
    }
}

/* Hankel's asymptotic expansion
* for large x.
* AMS55 #9.2.5.
*/

pub fn hankel(n: f64, x: f64) -> f64 {
    let m = 4.0 * n * n;
    let mut j = 1.0;
    let z = 8.0 * x;
    let mut k = 1.0;
    let mut p = 1.0;
    let mut u = (m - 1.0) / z;
    let mut q = u;
    let mut sign = 1.0;
    let mut conv = 1.0;
    let mut flag = 0;
    let mut t = 1.0;
    let mut pp = 1.0e38;
    let mut qq = 1.0e38;

    while t > MACHEP {
        k += 2.0;
        j += 1.0;
        sign = -sign;
        u *= (m - k * k) / (j * z);
        p += sign * u;
        k += 2.0;
        j += 1.0;
        u *= (m - k * k) / (j * z);
        q += sign * u;
        t = (u / p).abs();
        if t < conv {
            conv = t;
            qq = q;
            pp = p;
            flag = 1;
        }
        /* stop if the terms start getting larger */
        if (flag != 0) && (t > conv) {
            break;
        }
    }

    //hank1:
    let c = crate::backend::cephes64::cospi(0.5 * n + 0.25);
    let s = crate::backend::cephes64::sinpi(0.5 * n + 0.25);
    (2.0 / (M_PI * x)).sqrt() * ((pp * c + qq * s) * x.cos() + (pp * s - qq * c) * x.sin())
}

/* Asymptotic expansion for large n.
* AMS55 #9.3.35.
*/

const LAMBDA: [f64; 11] = [
    1.0,
    1.041666666666666666666667E-1,
    8.355034722222222222222222E-2,
    1.282265745563271604938272E-1,
    2.918490264641404642489712E-1,
    8.816272674437576524187671E-1,
    3.321408281862767544702647E+0,
    1.499576298686255465867237E+1,
    7.892301301158651813848139E+1,
    4.744515388682643231611949E+2,
    3.207490090890661934704328E+3,
];

const MU: [f64; 11] = [
    1.0,
    -1.458333333333333333333333E-1,
    -9.874131944444444444444444E-2,
    -1.433120539158950617283951E-1,
    -3.172272026784135480967078E-1,
    -9.424291479571202491373028E-1,
    -3.511203040826354261542798E+0,
    -1.572726362036804512982712E+1,
    -8.228143909718594444224656E+1,
    -4.923553705236705240352022E+2,
    -3.316218568547972508762102E+3,
];

const P1: [f64; 2] = [
    -2.083333333333333333333333E-1,
    1.250000000000000000000000E-1,
];

const P2: [f64; 3] = [
    3.342013888888888888888889E-1,
    -4.010416666666666666666667E-1,
    7.031250000000000000000000E-2,
];

const P3: [f64; 4] = [
    -1.025812596450617283950617E+0,
    1.846462673611111111111111E+0,
    -8.912109375000000000000000E-1,
    7.324218750000000000000000E-2,
];

const P4: [f64; 5] = [
    4.669584423426247427983539E+0,
    -1.120700261622299382716049E+1,
    8.789123535156250000000000E+0,
    -2.364086914062500000000000E+0,
    1.121520996093750000000000E-1,
];

const P5: [f64; 6] = [
    -2.8212072558200244877E1,
    8.4636217674600734632E1,
    -9.1818241543240017361E1,
    4.2534998745388454861E1,
    -7.3687943594796316964E0,
    2.27108001708984375E-1,
];

const P6: [f64; 7] = [
    2.1257013003921712286E2,
    -7.6525246814118164230E2,
    1.0599904525279998779E3,
    -6.9957962737613254123E2,
    2.1819051174421159048E2,
    -2.6491430486951555525E1,
    5.7250142097473144531E-1,
];

const P7: [f64; 8] = [
    -1.9194576623184069963E3,
    8.0617221817373093845E3,
    -1.3586550006434137439E4,
    1.1655393336864533248E4,
    -5.3056469786134031084E3,
    1.2009029132163524628E3,
    -1.0809091978839465550E2,
    1.7277275025844573975E0,
];

fn jnx(n: f64, x: f64) -> f64 {
    /* Test for x very close to n. Use expansion for transition region if so. */
    let cbn = cbrt(n);
    let mut z = (x - n) / cbn;
    if z.abs() <= 0.7 {
        return jnt(n, x);
    }

    z = x / n;
    let zz = 1.0 - z * z;
    if zz == 0.0 {
        return 0.0;
    }

    let sz: f64;
    let mut t: f64;
    let zeta: f64;
    let nflg: isize;

    if zz > 0.0 {
        sz = zz.sqrt();
        t = 1.5 * (((1.0 + sz) / z).ln() - sz); /* zeta ** 3/2          */
        zeta = cbrt(t * t);
        nflg = 1;
    } else {
        sz = (-zz).sqrt();
        t = 1.5 * (sz - (1.0 / z).acos());
        zeta = -cbrt(t * t);
        nflg = -1;
    }
    let z32i = (1.0 / t).abs();
    let sqz = cbrt(t);

    /* Airy function */
    let n23 = cbrt(n * n);
    t = n23 * zeta;

    let (ai, aip, _, _) = airy(t);

    let mut u = [0.0; 8];
    /* polynomials in expansion */
    u[0] = 1.0;
    let zzi = 1.0 / zz;
    u[1] = polevl(zzi, &P1, 1) / sz;
    u[2] = polevl(zzi, &P2, 2) / zz;
    u[3] = polevl(zzi, &P3, 3) / (sz * zz);
    let mut pp = zz * zz;
    u[4] = polevl(zzi, &P4, 4) / pp;
    u[5] = polevl(zzi, &P5, 5) / (pp * sz);
    pp *= zz;
    u[6] = polevl(zzi, &P6, 6) / pp;
    u[7] = polevl(zzi, &P7, 7) / (pp * sz);

    // #if CEPHES_DEBUG
    //     for (k = 0; k <= 7; k++)
    //     printf("u[%d] = %.5E\n", k, u[k]);
    // #endif

    let mut pp = 0.0;
    let mut qq = 0.0;
    let mut np = 1.0;
    /* flags to stop when terms get larger */
    let mut doa = 1;
    let mut dob = 1;
    let mut akl = f64::INFINITY;
    let mut bkl = f64::INFINITY;

    for k in 0..=3 {
        let tk = 2 * k;
        let tkp1 = tk + 1;
        let mut zp = 1.0;
        let mut ak: f64 = 0.0;
        let mut bk = 0.0;
        for s in 0..=tk {
            if doa != 0 {
                let sign = if (s & 3) > 1 { nflg } else { 1 };
                ak += sign as f64 * MU[s] * zp * u[tk - s];
            }

            if dob != 0 {
                let m = tkp1 - s;
                let sign = if ((m + 1) & 3) > 1 { nflg } else { 1 };
                bk += sign as f64 * LAMBDA[s] * zp * u[m];
            }
            zp *= z32i;
        }

        if doa != 0 {
            ak *= np;
            t = ak.abs();
            if t < akl {
                akl = t;
                pp += ak;
            } else {
                doa = 0;
            }
        }

        if dob != 0 {
            bk += LAMBDA[tkp1] * zp * u[0];
            bk *= -np / sqz;
            t = bk.abs();
            if t < bkl {
                bkl = t;
                qq += bk;
            } else {
                dob = 0;
            }
        }

        if np < MACHEP {
            break;
        }
        np /= n * n;
    }

    /* normalizing factor ( 4*zeta/(1 - z**2) )**1/4    */
    t = 4.0 * zeta / zz;
    t = t.sqrt().sqrt();

    t * (ai * pp / cbrt(n) + aip * qq / (n23 * n))
}

/* Asymptotic expansion for transition region,
* n large and x close to n.
* AMS55 #9.3.23.
*/

const PF2: [f64; 2] = [-9.0000000000000000000e-2, 8.5714285714285714286e-2];

const PF3: [f64; 3] = [
    1.3671428571428571429e-1,
    -5.4920634920634920635e-2,
    -4.4444444444444444444e-3,
];

const PF4: [f64; 4] = [
    1.3500000000000000000e-3,
    -1.6036054421768707483e-1,
    4.2590187590187590188e-2,
    2.7330447330447330447e-3,
];

const PG1: [f64; 2] = [-2.4285714285714285714e-1, 1.4285714285714285714e-2];

const PG2: [f64; 3] = [
    -9.0000000000000000000e-3,
    1.9396825396825396825e-1,
    -1.1746031746031746032e-2,
];

const PG3: [f64; 3] = [
    1.9607142857142857143e-2,
    -1.5983694083694083694e-1,
    6.3838383838383838384e-3,
];

fn jnt(n: f64, x: f64) -> f64 {
    let cbn = cbrt(n);
    let z = (x - n) / cbn;
    let cbtwo = cbrt(2.0);

    /* Airy function */
    let zz = -cbtwo * z;
    let (ai, aip, _, _) = airy(zz);

    /* polynomials in expansion */
    let zz = z * z;
    let z3 = zz * z;
    let f: [f64; 5] = [
        1.0,
        -z / 5.0,
        polevl(z3, &PF2, 1) * zz,
        polevl(z3, &PF3, 2),
        polevl(z3, &PF4, 3) * z,
    ];
    let g: [f64; 4] = [
        0.3 * zz,
        polevl(z3, &PG1, 1),
        polevl(z3, &PG2, 2) * z,
        polevl(z3, &PG3, 2) * zz,
    ];

    let mut pp = 0.0;
    let mut qq = 0.0;
    let mut nk = 1.0;
    let n23 = cbrt(n * n);

    for k in 0..=4 {
        let fk = f[k] * nk;
        pp += fk;
        if k != 4 {
            let gk = g[k] * nk;
            qq += gk;
        }
        nk /= n23;
    }

    cbtwo * ai * pp / cbn + cbrt(4.0) * aip * qq / n
}
