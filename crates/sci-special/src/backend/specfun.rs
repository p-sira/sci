//! Safe Rust translations of the Zhang–Jin real kernels used by Sci.
//! Source: SciPy xsf/cpu/specfun, par_cyl, airy, bessel, and struve headers.
//! See LICENSES/scipy-specfun.txt, xsf-BSD-3-Clause.txt, and xsf-bundled.txt.
//! Recurrence indices and evaluation order follow the source algorithms. The
//! local lint allowances retain their textbook notation during maintenance.
#![allow(
    unused_assignments,
    unused_mut,
    unused_parens,
    non_snake_case,
    clippy::all
)]
use super::cephes64::{gamma as Gamma, psi, rgamma};
use super::legacy::{fcoef, gamma_ratio, lgam_sgn};
const HYP2F1_EPS: f64 = 1e-13;
const HYP2F1_ETHRESH: f64 = 1e-12;
const HYP2F1_MAXITER: i32 = 10000;
const M_PI: f64 = core::f64::consts::PI;
const MACHEP: f64 = f64::EPSILON * 0.5;

/// `(-1)^k` for an integer `k`, without a `pow` call.
#[inline]
fn neg_one_pow(k: i32) -> f64 {
    if k % 2 == 0 { 1.0 } else { -1.0 }
}

/// Length of the zeroed work arrays of the spheroidal kernels below. They
/// previously used fixed 512-entry arrays. Every index they touch is below
/// `2 nm + 2 m + 8`, where `nm <= 40 + (n - m)/2 + c` is the largest
/// truncation order (`aswfa`). Some kernels read entries past the prefix they
/// write and expect zeros, so the arrays stay zero-initialized; the length is
/// capped at the previous fixed size.
fn spheroidal_work_len(m: i32, n: i32, c: f64) -> usize {
    let nm = 40.0 + f64::from((n - m).max(0) / 2) + c.max(0.0);
    let len = 2.0 * nm + 2.0 * f64::from(m.max(0)) + 16.0;
    if len < 512.0 {
        len.ceil() as usize
    } else {
        512
    }
}
pub(super) fn hyp2f1(mut a: f64, mut b: f64, mut c: f64, mut x: f64) -> f64 {
    let mut d: f64 = 0.0;
    let mut d1: f64 = 0.0;
    let mut d2: f64 = 0.0;
    let mut e: f64 = 0.0;
    let mut p: f64 = 0.0;
    let mut q: f64 = 0.0;
    let mut r: f64 = 0.0;
    let mut s: f64 = 0.0;
    let mut y: f64 = 0.0;
    let mut ax: f64 = 0.0;
    let mut ia: f64 = 0.0;
    let mut ib: f64 = 0.0;
    let mut ic: f64 = 0.0;
    let mut id: f64 = 0.0;
    let mut err: f64 = 0.0;
    let mut t1: f64 = 0.0;
    let mut i: i32 = 0;
    let mut aid: i32 = 0;
    let mut neg_int_a: i32 = 0;
    let mut neg_int_b: i32 = 0;
    let mut neg_int_ca_or_cb: i32 = 0;
    err = 0.0f64;
    ax = (x).abs();
    s = (1.0f64 - x);
    ia = (a).round();
    ib = (b).round();
    if (x == 0.0f64) {
        {
            return 1.0f64;
        }
    }
    d = ((c - a) - b);
    id = (d).round();
    if (((a == ((0) as f64)) || (b == ((0) as f64))) && (c != ((0) as f64))) {
        {
            return 1.0f64;
        }
    }
    if ((a == ((-1) as f64)) && (c != ((0) as f64))) {
        {
            let mut result: f64 = (((1) as f64) - ((b / c) * x));
            if ((result).abs() < 1e-15f64) {
                {
                    return 0.0f64;
                }
            }
            return result;
        }
    }
    if ((a <= ((0) as f64)) && ((a - ia).abs() < HYP2F1_EPS)) {
        {
            neg_int_a = 1;
        }
    }
    if ((b <= ((0) as f64)) && ((b - ib).abs() < HYP2F1_EPS)) {
        {
            neg_int_b = 1;
        }
    }
    if (((d <= ((-1) as f64)) && !(((d - id).abs() > HYP2F1_EPS) && (s < ((0) as f64))))
        && !((neg_int_a != 0) || (neg_int_b != 0)))
    {
        {
            return ((s).powf(d) * hyp2f1((c - a), (c - b), c, x));
        }
    }
    if (((d <= ((0) as f64)) && (x == ((1) as f64))) && !((neg_int_a != 0) || (neg_int_b != 0))) {
        return f64::INFINITY;
    }
    if ((ax < 1.0f64) || (x == (-1.0f64))) {
        {
            if ((b - c).abs() < HYP2F1_EPS) {
                {
                    if (neg_int_b != 0) {
                        {
                            y = hyp2f1_neg_c_equal_bc(a, b, x);
                        }
                    } else {
                        {
                            y = (s).powf((-a));
                        }
                    }
                    return y;
                }
            }
            if ((a - c).abs() < HYP2F1_EPS) {
                {
                    y = (s).powf((-b));
                    return y;
                }
            }
        }
    }
    if (c <= 0.0f64) {
        {
            ic = (c).round();
            if ((c - ic).abs() < HYP2F1_EPS) {
                {
                    if ((neg_int_a != 0) && (ia > ic)) {
                        return hyt2f1(a, b, c, x, std::slice::from_mut(&mut err));
                    }
                    if ((neg_int_b != 0) && (ib > ic)) {
                        return hyt2f1(a, b, c, x, std::slice::from_mut(&mut err));
                    }
                    return f64::INFINITY;
                }
            }
        }
    }
    if ((neg_int_a != 0) || (neg_int_b != 0)) {
        return hyt2f1(a, b, c, x, std::slice::from_mut(&mut err));
    }
    t1 = (b - a).abs();
    if ((x < (-2.0f64)) && ((t1 - (t1).round()).abs() > HYP2F1_EPS)) {
        {
            p = hyp2f1(
                a,
                ((((1) as f64) - c) + a),
                ((((1) as f64) - b) + a),
                (1.0f64 / x),
            );
            q = hyp2f1(
                b,
                ((((1) as f64) - c) + b),
                ((((1) as f64) - a) + b),
                (1.0f64 / x),
            );
            p *= (-x).powf((-a));
            q *= (-x).powf((-b));
            t1 = Gamma(c);
            s = ((t1 * Gamma((b - a))) * (rgamma(b) * rgamma((c - a))));
            y = ((t1 * Gamma((a - b))) * (rgamma(a) * rgamma((c - b))));
            return ((s * p) + (y * q));
        }
    } else {
        if (x < (-1.0f64)) {
            {
                if ((a).abs() < (b).abs()) {
                    {
                        return ((s).powf((-a)) * hyp2f1(a, (c - b), c, (x / (x - ((1) as f64)))));
                    }
                } else {
                    {
                        return ((s).powf((-b)) * hyp2f1(b, (c - a), c, (x / (x - ((1) as f64)))));
                    }
                }
            }
        }
    }
    if (ax > 1.0f64) {
        return f64::INFINITY;
    }
    p = (c - a);
    ia = (p).round();
    if ((ia <= 0.0f64) && ((p - ia).abs() < HYP2F1_EPS)) {
        neg_int_ca_or_cb = 1;
    }
    r = (c - b);
    ib = (r).round();
    if ((ib <= 0.0f64) && ((r - ib).abs() < HYP2F1_EPS)) {
        neg_int_ca_or_cb = 1;
    }
    id = (d).round();
    q = (d - id).abs();
    if ((ax - 1.0f64).abs() < HYP2F1_EPS) {
        {
            if (x > 0.0f64) {
                {
                    if (neg_int_ca_or_cb != 0) {
                        {
                            if (d >= 0.0f64) {
                                return ((s).powf(d)
                                    * hys2f1(
                                        (c - a),
                                        (c - b),
                                        c,
                                        x,
                                        std::slice::from_mut(&mut err),
                                    ));
                            } else {
                                return f64::INFINITY;
                            }
                        }
                    }
                    if (d <= 0.0f64) {
                        return f64::INFINITY;
                    }
                    y = ((Gamma(c) * Gamma(d)) * (rgamma(p) * rgamma(r)));
                    return y;
                }
            }
            if (d <= (-1.0f64)) {
                return f64::INFINITY;
            }
        }
    }
    if (d < 0.0f64) {
        {
            y = hyt2f1(a, b, c, x, std::slice::from_mut(&mut err));
            if (err < HYP2F1_ETHRESH) {
                return y;
            }
            err = 0.0f64;
            aid = ((((2) as f64) - id) as i32);
            e = (c + ((aid) as f64));
            d2 = hyp2f1(a, b, e, x);
            d1 = hyp2f1(a, b, (e + 1.0f64), x);
            q = ((a + b) + 1.0f64);
            i = 0;
            while (i < aid) {
                {
                    r = (e - 1.0f64);
                    y = ((((e * (r - (((2.0f64 * e) - q) * x))) * d2)
                        + ((((e - a) * (e - b)) * x) * d1))
                        / ((e * r) * s));
                    e = r;
                    d1 = d2;
                    d2 = y;
                }
                {
                    i += 1;
                    i
                };
            }
            return y;
        }
    }
    if (neg_int_ca_or_cb != 0) {
        {
            return ((s).powf(d) * hys2f1((c - a), (c - b), c, x, std::slice::from_mut(&mut err)));
        }
    }
    y = hyt2f1(a, b, c, x, std::slice::from_mut(&mut err));
    if (err > HYP2F1_ETHRESH) {
        {}
    }
    return y;
}

pub(super) fn hyp2f1_neg_c_equal_bc(mut a: f64, mut b: f64, mut x: f64) -> f64 {
    let mut k: f64 = 0.0;
    let mut collector: f64 = ((1) as f64);
    let mut sum: f64 = ((1) as f64);
    let mut collector_max: f64 = ((1) as f64);
    if !((b).abs() < 1e5f64) {
        {
            return f64::NAN;
        }
    }
    k = ((1) as f64);
    while (k <= (-b)) {
        {
            collector *= ((((a + k) - ((1) as f64)) * x) / k);
            collector_max = ((collector).abs()).max(collector_max);
            sum += collector;
        }
        {
            k += 1.0;
            k
        };
    }
    if ((1e-16f64 * (((1) as f64) + (collector_max / (sum).abs()))) > 1e-7f64) {
        {
            return f64::NAN;
        }
    }
    return sum;
}

pub(super) fn hyt2f1(mut a: f64, mut b: f64, mut c: f64, mut x: f64, loss: &mut [f64]) -> f64 {
    let mut p: f64 = 0.0;
    let mut q: f64 = 0.0;
    let mut r: f64 = 0.0;
    let mut s: f64 = 0.0;
    let mut t: f64 = 0.0;
    let mut y: f64 = 0.0;
    let mut w: f64 = 0.0;
    let mut d: f64 = 0.0;
    let mut err: f64 = 0.0;
    let mut err1: f64 = 0.0;
    let mut ax: f64 = 0.0;
    let mut id: f64 = 0.0;
    let mut d1: f64 = 0.0;
    let mut d2: f64 = 0.0;
    let mut e: f64 = 0.0;
    let mut y1: f64 = 0.0;
    let mut i: i32 = 0;
    let mut aid: i32 = 0;
    let mut sign: i32 = 0;
    let mut ia: i32 = 0;
    let mut ib: i32 = 0;
    let mut neg_int_a: i32 = 0;
    let mut neg_int_b: i32 = 0;
    ia = (((a).round()) as i32);
    ib = (((b).round()) as i32);
    if ((a <= ((0) as f64)) && ((a - ((ia) as f64)).abs() < HYP2F1_EPS)) {
        {
            neg_int_a = 1;
        }
    }
    if ((b <= ((0) as f64)) && ((b - ((ib) as f64)).abs() < HYP2F1_EPS)) {
        {
            neg_int_b = 1;
        }
    }
    err = 0.0f64;
    s = (1.0f64 - x);
    if ((x < (-0.5f64)) && !((neg_int_a != 0) || (neg_int_b != 0))) {
        {
            if (b > a) {
                y = ((s).powf((-a))
                    * hys2f1(a, (c - b), c, ((-x) / s), std::slice::from_mut(&mut err)));
            } else {
                y = ((s).powf((-b))
                    * hys2f1((c - a), b, c, ((-x) / s), std::slice::from_mut(&mut err)));
            }
            {
                loss[0] = err;
                return y;
            }
        }
    }
    d = ((c - a) - b);
    id = (d).round();
    if ((x > 0.9f64) && !((neg_int_a != 0) || (neg_int_b != 0))) {
        {
            if ((d - id).abs() > MACHEP) {
                {
                    let mut sgngam: i32 = 0;
                    y = hys2f1(a, b, c, x, std::slice::from_mut(&mut err));
                    if (err < HYP2F1_ETHRESH) {
                        {
                            {
                                loss[0] = err;
                                return y;
                            }
                        }
                    }
                    q = hys2f1(a, b, (1.0f64 - d), s, std::slice::from_mut(&mut err));
                    sign = 1;
                    w = lgam_sgn(d, std::slice::from_mut(&mut sgngam));
                    sign *= sgngam;
                    w -= lgam_sgn((c - a), std::slice::from_mut(&mut sgngam));
                    sign *= sgngam;
                    w -= lgam_sgn((c - b), std::slice::from_mut(&mut sgngam));
                    sign *= sgngam;
                    q *= (((sign) as f64) * (w).exp());
                    r = ((s).powf(d)
                        * hys2f1(
                            (c - a),
                            (c - b),
                            (d + 1.0f64),
                            s,
                            std::slice::from_mut(&mut err1),
                        ));
                    sign = 1;
                    w = lgam_sgn((-d), std::slice::from_mut(&mut sgngam));
                    sign *= sgngam;
                    w -= lgam_sgn(a, std::slice::from_mut(&mut sgngam));
                    sign *= sgngam;
                    w -= lgam_sgn(b, std::slice::from_mut(&mut sgngam));
                    sign *= sgngam;
                    r *= (((sign) as f64) * (w).exp());
                    y = (q + r);
                    q = (q).abs();
                    r = (r).abs();
                    if (q > r) {
                        {
                            r = q;
                        }
                    }
                    err += (err1 + ((MACHEP * r) / y));
                    y *= Gamma(c);
                    {
                        loss[0] = err;
                        return y;
                    }
                }
            } else {
                {
                    if (id >= 0.0f64) {
                        {
                            e = d;
                            d1 = d;
                            d2 = 0.0f64;
                            aid = ((id) as i32);
                        }
                    } else {
                        {
                            e = (-d);
                            d1 = 0.0f64;
                            d2 = d;
                            aid = ((-id) as i32);
                        }
                    }
                    ax = (s).ln();
                    y = ((((psi(1.0f64) + psi((1.0f64 + e))) - psi((a + d1))) - psi((b + d1)))
                        - ax);
                    y *= rgamma((e + 1.0f64));
                    p = ((((a + d1) * (b + d1)) * s) * rgamma((e + 2.0f64)));
                    t = 1.0f64;
                    loop {
                        {
                            r = ((((psi((1.0f64 + t)) + psi(((1.0f64 + t) + e)))
                                - psi(((a + t) + d1)))
                                - psi(((b + t) + d1)))
                                - ax);
                            q = (p * r);
                            y += q;
                            p *= ((s * ((a + t) + d1)) / (t + 1.0f64));
                            p *= (((b + t) + d1) / ((t + 1.0f64) + e));
                            t += 1.0f64;
                            if (t > ((HYP2F1_MAXITER) as f64)) {
                                {
                                    loss[0] = 1.0f64;
                                    return f64::NAN;
                                }
                            }
                        }
                        if !((y == ((0) as f64)) || ((q / y).abs() > HYP2F1_EPS)) {
                            break;
                        }
                    }
                    if (id == 0.0f64) {
                        {
                            y *= (Gamma(c) / (Gamma(a) * Gamma(b)));
                            {
                                loss[0] = err;
                                return y;
                            }
                        }
                    }
                    y1 = 1.0f64;
                    t = 0.0f64;
                    p = 1.0f64;
                    i = 1;
                    while (i < aid) {
                        {
                            r = ((1.0f64 - e) + t);
                            p *= (((s * ((a + t) + d2)) * ((b + t) + d2)) / r);
                            t += 1.0f64;
                            p /= t;
                            y1 += p;
                        }
                        {
                            i += 1;
                            i
                        };
                    }
                    p = Gamma(c);
                    y1 *= ((Gamma(e) * p) * (rgamma((a + d1)) * rgamma((b + d1))));
                    y *= (p * (rgamma((a + d2)) * rgamma((b + d2))));
                    if ((aid & 1) != 0) {
                        y = (-y);
                    }
                    q = (s).powf(id);
                    if (id > 0.0f64) {
                        y *= q;
                    } else {
                        y1 *= q;
                    }
                    y += y1;
                    {
                        loss[0] = err;
                        return y;
                    }
                }
            }
        }
    }
    y = hys2f1(a, b, c, x, std::slice::from_mut(&mut err));
    loss[0] = err;
    return y;
}

pub(super) fn hys2f1(mut a: f64, mut b: f64, mut c: f64, mut x: f64, loss: &mut [f64]) -> f64 {
    let mut f: f64 = 0.0;
    let mut g: f64 = 0.0;
    let mut h: f64 = 0.0;
    let mut k: f64 = 0.0;
    let mut m: f64 = 0.0;
    let mut s: f64 = 0.0;
    let mut u: f64 = 0.0;
    let mut umax: f64 = 0.0;
    let mut i: i32 = 0;
    let mut ib: i32 = 0;
    let mut intflag: i32 = 0;
    if ((b).abs() > (a).abs()) {
        {
            f = b;
            b = a;
            a = f;
        }
    }
    ib = (((b).round()) as i32);
    if ((((b - ((ib) as f64)).abs() < HYP2F1_EPS) && (ib <= 0)) && ((b).abs() < (a).abs())) {
        {
            f = b;
            b = a;
            a = f;
            intflag = 1;
        }
    }
    if (((((a).abs() > ((c).abs() + ((1) as f64))) || (intflag != 0))
        && ((c - a).abs() > ((2) as f64)))
        && ((a).abs() > ((2) as f64)))
    {
        {
            return hyp2f1ra(a, b, c, x, loss);
        }
    }
    i = 0;
    umax = 0.0f64;
    f = a;
    g = b;
    h = c;
    s = 1.0f64;
    u = 1.0f64;
    k = 0.0f64;
    loop {
        {
            if ((h).abs() < HYP2F1_EPS) {
                {
                    loss[0] = 1.0f64;
                    return f64::INFINITY;
                }
            }
            m = (k + 1.0f64);
            u = (u * ((((f + k) * (g + k)) * x) / ((h + k) * m)));
            s += u;
            k = (u).abs();
            if (k > umax) {
                umax = k;
            }
            k = m;
            if ({
                i += 1;
                i
            } > HYP2F1_MAXITER)
            {
                {
                    loss[0] = 1.0f64;
                    return s;
                }
            }
        }
        if !((s == ((0) as f64)) || ((u / s).abs() > MACHEP)) {
            break;
        }
    }
    loss[0] = (((MACHEP * umax) / (s).abs()) + (MACHEP * ((i) as f64)));
    return s;
}

pub(super) fn hyp2f1ra(mut a: f64, mut b: f64, mut c: f64, mut x: f64, loss: &mut [f64]) -> f64 {
    let mut f2: f64 = 0.0;
    let mut f1: f64 = 0.0;
    let mut f0: f64 = 0.0;
    let mut n: i32 = 0;
    let mut t: f64 = 0.0;
    let mut err: f64 = 0.0;
    let mut da: f64 = 0.0;
    if (((c < ((0) as f64)) && (a <= c)) || ((c >= ((0) as f64)) && (a >= c))) {
        {
            da = (a - c).round();
        }
    } else {
        {
            da = (a).round();
        }
    }
    t = (a - da);
    loss[0] = ((0) as f64);
    debug_assert!((da != ((0) as f64)));
    if ((da).abs() > ((HYP2F1_MAXITER) as f64)) {
        {
            loss[0] = 1.0f64;
            return f64::NAN;
        }
    }
    if (da < ((0) as f64)) {
        {
            f2 = ((0) as f64);
            f1 = hys2f1(t, b, c, x, std::slice::from_mut(&mut err));
            loss[0] += err;
            f0 = hys2f1((t - ((1) as f64)), b, c, x, std::slice::from_mut(&mut err));
            loss[0] += err;
            t -= ((1) as f64);
            n = 1;
            while (((n) as f64) < (-da)) {
                {
                    f2 = f1;
                    f1 = f0;
                    f0 = ((((-((((((2) as f64) * t) - c) - (t * x)) + (b * x))) / (c - t)) * f1)
                        - (((t * (x - ((1) as f64))) / (c - t)) * f2));
                    t -= ((1) as f64);
                }
                {
                    n += 1;
                    n
                };
            }
        }
    } else {
        {
            f2 = ((0) as f64);
            f1 = hys2f1(t, b, c, x, std::slice::from_mut(&mut err));
            loss[0] += err;
            f0 = hys2f1((t + ((1) as f64)), b, c, x, std::slice::from_mut(&mut err));
            loss[0] += err;
            t += ((1) as f64);
            n = 1;
            while (((n) as f64) < da) {
                {
                    f2 = f1;
                    f1 = f0;
                    f0 = ((-((((((((2) as f64) * t) - c) - (t * x)) + (b * x)) * f1)
                        + ((c - t) * f2)))
                        / (t * (x - ((1) as f64))));
                    t += ((1) as f64);
                }
                {
                    n += 1;
                    n
                };
            }
        }
    }
    return f0;
}

pub(super) fn aswfa(
    mut x: f64,
    mut m: i32,
    mut n: i32,
    mut c: f64,
    mut kd: i32,
    mut cv: f64,
    s1f: &mut [f64],
    s1d: &mut [f64],
) -> i32 {
    let mut ip: i32 = 0;
    let mut k: i32 = 0;
    let mut nm: i32 = 0;
    let mut nm2: i32 = 0;
    let mut a0: f64 = 0.0;
    let mut d0: f64 = 0.0;
    let mut d1: f64 = 0.0;
    let mut r: f64 = 0.0;
    let mut su1: f64 = 0.0;
    let mut su2: f64 = 0.0;
    let mut x0: f64 = 0.0;
    let mut x1: f64 = 0.0;
    let mut ck = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut df = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut eps: f64 = 1e-14f64;
    x0 = x;
    x = (x).abs();
    ip = if (((n - m) % 2) == 0) { 0 } else { 1 };
    nm = (40 + (((((n - m) / 2) as f64) + c) as i32));
    nm2 = ((nm / 2) - 2);
    if (sdmn(m, n, c, cv, kd, &mut df[..]) == 1) {
        {
            return 1;
        }
    }
    sckb(m, n, c, &mut df[..], &mut ck[..]);
    x1 = (1.0f64 - (x * x));
    if ((m == 0) && (x1 == 0.0f64)) {
        {
            a0 = 1.0f64;
        }
    } else {
        {
            a0 = (x1).powf((0.5f64 * ((m) as f64)));
        }
    }
    su1 = ck[(0) as usize];
    k = 1;
    while (k <= nm2) {
        {
            r = (ck[(k) as usize] * (x1).powf(((k) as f64)));
            su1 += r;
            if ((k >= 10) && ((r / su1).abs() < eps)) {
                {
                    break;
                }
            }
        }
        {
            k += 1;
            k
        };
    }
    s1f[0] = ((a0 * (x).powf(((ip) as f64))) * su1);
    if (x == 1.0f64) {
        {
            if (m == 0) {
                {
                    s1d[0] = ((((ip) as f64) * ck[(0) as usize]) - (2.0f64 * ck[(1) as usize]));
                }
            } else {
                if (m == 1) {
                    {
                        s1d[0] = (-1e100f64);
                    }
                } else {
                    if (m == 2) {
                        {
                            s1d[0] = ((-2.0f64) * ck[(0) as usize]);
                        }
                    } else {
                        if (m >= 3) {
                            {
                                s1d[0] = 0.0f64;
                            }
                        }
                    }
                }
            }
        }
    } else {
        {
            d0 = (((ip) as f64) - ((((m) as f64) / x1) * (x).powf((((ip) as f64) + 1.0f64))));
            d1 = (((-2.0f64) * a0) * (x).powf((((ip) as f64) + 1.0f64)));
            su2 = ck[(1) as usize];
            k = 2;
            while (k <= nm2) {
                {
                    r = ((((k) as f64) * ck[(k) as usize]) * (x1).powf((((k) as f64) - 1.0f64)));
                    su2 += r;
                    if ((k >= 10) && ((r / su2).abs() < eps)) {
                        {
                            break;
                        }
                    }
                }
                {
                    k += 1;
                    k
                };
            }
            s1d[0] = (((d0 * a0) * su1) + (d1 * su2));
        }
    }
    if ((x0 < 0.0f64) && (ip == 0)) {
        {
            s1d[0] = (-s1d[0]);
        }
    }
    if ((x0 < 0.0f64) && (ip == 1)) {
        {
            s1f[0] = (-s1f[0]);
        }
    }
    x = x0;
    return 0;
}

pub(super) fn sdmn(
    mut m: i32,
    mut n: i32,
    mut c: f64,
    mut cv: f64,
    mut kd: i32,
    df: &mut [f64],
) -> i32 {
    let mut nm: i32 = 0;
    let mut ip: i32 = 0;
    let mut k: i32 = 0;
    let mut kb: i32 = 0;
    let mut cs: f64 = 0.0;
    let mut dk0: f64 = 0.0;
    let mut dk1: f64 = 0.0;
    let mut dk2: f64 = 0.0;
    let mut d2k: f64 = 0.0;
    let mut f: f64 = 0.0;
    let mut fs: f64 = 0.0;
    let mut f1: f64 = 0.0;
    let mut f0: f64 = 0.0;
    let mut fl: f64 = 0.0;
    let mut f2: f64 = 0.0;
    let mut su1: f64 = 0.0;
    let mut su2: f64 = 0.0;
    let mut sw: f64 = 0.0;
    let mut r1: f64 = 0.0;
    let mut r3: f64 = 0.0;
    let mut r4: f64 = 0.0;
    let mut s0: f64 = 0.0;
    nm = (25 + (((0.5f64 * ((n - m) as f64)) + c) as i32));
    if (c < 1e-10f64) {
        {
            let mut i: i32 = 1;
            while (i <= nm) {
                {
                    df[(i - 1) as usize] = 0.0f64;
                }
                {
                    i += 1;
                    i
                };
            }
            df[((n - m) / 2) as usize] = 1.0f64;
            return 0;
        }
    }
    let mut a = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut d = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut g = vec![0.0; spheroidal_work_len(m, n, c)];
    cs = ((c * c) * ((kd) as f64));
    ip = ((n - m) % 2);
    let mut i: i32 = 1;
    while (i <= (nm + 2)) {
        {
            k = if (ip == 0) {
                (2 * (i - 1))
            } else {
                ((2 * i) - 1)
            };
            dk0 = ((m + k) as f64);
            dk1 = (((m + k) + 1) as f64);
            dk2 = ((2 * (m + k)) as f64);
            d2k = (((2 * m) + k) as f64);
            a[(i - 1) as usize] =
                ((((d2k + 2.0f64) * (d2k + 1.0f64)) / ((dk2 + 3.0f64) * (dk2 + 5.0f64))) * cs);
            d[(i - 1) as usize] = ((dk0 * dk1)
                + ((((((2.0f64 * dk0) * dk1) - ((2.0f64 * ((m) as f64)) * ((m) as f64)))
                    - 1.0f64)
                    / ((dk2 - 1.0f64) * (dk2 + 3.0f64)))
                    * cs));
            g[(i - 1) as usize] = (((((k) as f64) * (((k) as f64) - 1.0f64))
                / ((dk2 - 3.0f64) * (dk2 - 1.0f64)))
                * cs);
        }
        {
            i += 1;
            i
        };
    }
    fs = 1.0f64;
    f1 = 0.0f64;
    f0 = 1.0e-100f64;
    kb = 0;
    df[(nm) as usize] = 0.0f64;
    fl = 0.0f64;
    let mut k: i32 = nm;
    while (k >= 1) {
        {
            f = ((-(((d[(k) as usize] - cv) * f0) + (a[(k) as usize] * f1))) / g[(k) as usize]);
            if ((f).abs() > (df[(k) as usize]).abs()) {
                {
                    df[(k - 1) as usize] = f;
                    f1 = f0;
                    f0 = f;
                    if ((f).abs() > 1.0e+100f64) {
                        {
                            let mut k1: i32 = k;
                            while (k1 <= nm) {
                                df[(k1 - 1) as usize] *= 1.0e-100f64;
                                {
                                    k1 += 1;
                                    k1
                                };
                            }
                            f1 *= 1.0e-100f64;
                            f0 *= 1.0e-100f64;
                        }
                    }
                }
            } else {
                {
                    kb = k;
                    fl = df[(k) as usize];
                    f1 = 1.0e-100f64;
                    f2 = ((-((d[(0) as usize] - cv) / a[(0) as usize])) * f1);
                    df[(0) as usize] = f1;
                    if (kb == 1) {
                        {
                            fs = f2;
                        }
                    } else {
                        if (kb == 2) {
                            {
                                df[(1) as usize] = f2;
                                fs = ((-(((d[(1) as usize] - cv) * f2) + (g[(1) as usize] * f1)))
                                    / a[(1) as usize]);
                            }
                        } else {
                            {
                                df[(1) as usize] = f2;
                                let mut j: i32 = 3;
                                while (j <= (kb + 1)) {
                                    {
                                        f = ((-(((d[(j - 2) as usize] - cv) * f2)
                                            + (g[(j - 2) as usize] * f1)))
                                            / a[(j - 2) as usize]);
                                        if (j <= kb) {
                                            {
                                                df[(j - 1) as usize] = f;
                                            }
                                        }
                                        if ((f).abs() > 1.0e+100f64) {
                                            {
                                                let mut k1: i32 = 1;
                                                while (k1 <= j) {
                                                    {
                                                        df[(k1 - 1) as usize] *= 1.0e-100f64;
                                                    }
                                                    {
                                                        k1 += 1;
                                                        k1
                                                    };
                                                }
                                                f *= 1.0e-100f64;
                                                f2 *= 1.0e-100f64;
                                            }
                                        }
                                        f1 = f2;
                                        f2 = f;
                                    }
                                    {
                                        j += 1;
                                        j
                                    };
                                }
                                fs = f;
                            }
                        }
                    }
                    break;
                }
            }
        }
        {
            k -= 1;
            k
        };
    }
    su1 = 0.0f64;
    r1 = 1.0f64;
    let mut j: i32 = ((m + ip) + 1);
    while (j <= (2 * (m + ip))) {
        {
            r1 *= ((j) as f64);
        }
        {
            j += 1;
            j
        };
    }
    su1 = (df[(0) as usize] * r1);
    let mut k: i32 = 2;
    while (k <= kb) {
        {
            r1 = (((-r1) * ((((k + m) + ip) as f64) - 1.5f64)) / (((k) as f64) - 1.0f64));
            su1 += (r1 * df[(k - 1) as usize]);
        }
        {
            k += 1;
            k
        };
    }
    su2 = 0.0f64;
    sw = 0.0f64;
    let mut k: i32 = (kb + 1);
    while (k <= nm) {
        {
            if (k != 1) {
                {
                    r1 = (((-r1) * ((((k + m) + ip) as f64) - 1.5f64)) / (((k) as f64) - 1.0f64));
                }
            }
            su2 += (r1 * df[(k - 1) as usize]);
            if ((sw - su2).abs() < ((su2).abs() * 1.0e-14f64)) {
                {
                    break;
                }
            }
            sw = su2;
        }
        {
            k += 1;
            k
        };
    }
    r3 = 1.0f64;
    let mut j: i32 = 1;
    while (j <= (((m + n) + ip) / 2)) {
        {
            r3 *= (((j) as f64) + (0.5f64 * (((n + m) + ip) as f64)));
        }
        {
            j += 1;
            j
        };
    }
    r4 = 1.0f64;
    let mut j: i32 = 1;
    while (j <= (((n - m) - ip) / 2)) {
        {
            r4 *= ((-4.0f64) * ((j) as f64));
        }
        {
            j += 1;
            j
        };
    }
    s0 = ((r3 / ((fl * (su1 / fs)) + su2)) / r4);
    let mut k: i32 = 1;
    while (k <= kb) {
        {
            df[(k - 1) as usize] *= ((fl / fs) * s0);
        }
        {
            k += 1;
            k
        };
    }
    let mut k: i32 = (kb + 1);
    while (k <= nm) {
        {
            df[(k - 1) as usize] *= s0;
        }
        {
            k += 1;
            k
        };
    }
    return 0;
}

pub(super) fn sckb(mut m: i32, mut n: i32, mut c: f64, df: &mut [f64], ck: &mut [f64]) {
    let mut i: i32 = 0;
    let mut ip: i32 = 0;
    let mut i1: i32 = 0;
    let mut i2: i32 = 0;
    let mut k: i32 = 0;
    let mut nm: i32 = 0;
    let mut reg: f64 = 0.0;
    let mut fac: f64 = 0.0;
    let mut sw: f64 = 0.0;
    let mut r: f64 = 0.0;
    let mut d1: f64 = 0.0;
    let mut d2: f64 = 0.0;
    let mut d3: f64 = 0.0;
    let mut sum: f64 = 0.0;
    let mut r1: f64 = 0.0;
    if (c <= 1.0e-10f64) {
        {
            c = 1.0e-10f64;
        }
    }
    nm = (25 + (((0.5f64 * ((n - m) as f64)) + c) as i32));
    ip = ((n - m) % 2);
    reg = if ((m + nm) > 80) { 1.0e-200f64 } else { 1.0f64 };
    fac = (-(0.5f64).powf(((m) as f64)));
    sw = 0.0f64;
    k = 0;
    while (k < nm) {
        {
            fac = (-fac);
            i1 = (((2 * k) + ip) + 1);
            r = reg;
            i = i1;
            while (i <= ((i1 + (2 * m)) - 1)) {
                {
                    r *= ((i) as f64);
                }
                {
                    i += 1;
                    i
                };
            }
            i2 = ((k + m) + ip);
            i = i2;
            while (i <= ((i2 + k) - 1)) {
                {
                    r *= (((i) as f64) + 0.5f64);
                }
                {
                    i += 1;
                    i
                };
            }
            sum = (r * df[(k) as usize]);
            i = (k + 1);
            while (i <= nm) {
                {
                    d1 = ((2.0f64 * ((i) as f64)) + ((ip) as f64));
                    d2 = ((2.0f64 * ((m) as f64)) + d1);
                    d3 = ((((i + m) + ip) as f64) - 0.5f64);
                    r = (((((r * d2) * (d2 - 1.0f64)) * ((i) as f64)) * (d3 + ((k) as f64)))
                        / (((d1 * (d1 - 1.0f64)) * ((i - k) as f64)) * d3));
                    sum += (r * df[(i) as usize]);
                    if ((sw - sum).abs() < ((sum).abs() * 1.0e-14f64)) {
                        {
                            break;
                        }
                    }
                    sw = sum;
                }
                {
                    i += 1;
                    i
                };
            }
            r1 = reg;
            i = 2;
            while (i <= (m + k)) {
                {
                    r1 *= ((i) as f64);
                }
                {
                    i += 1;
                    i
                };
            }
            ck[(k) as usize] = ((fac * sum) / r1);
        }
        {
            k += 1;
            k
        };
    }
}

pub(super) fn rswfp(
    mut m: i32,
    mut n: i32,
    mut c: f64,
    mut x: f64,
    mut cv: f64,
    mut kf: i32,
    r1f: &mut [f64],
    r1d: &mut [f64],
    r2f: &mut [f64],
    r2d: &mut [f64],
) -> i32 {
    let mut df = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut id: i32 = 0;
    let mut kd: i32 = 1;
    if (sdmn(m, n, c, cv, kd, &mut df[..]) == 1) {
        {
            return 1;
        }
    }
    if (kf != 2) {
        {
            if (rmn1(m, n, c, x, kd, &mut df[..], r1f, r1d) == 1) {
                {
                    return 1;
                }
            }
        }
    }
    if (kf > 1) {
        {
            if (rmn2l(
                m,
                n,
                c,
                x,
                kd,
                &mut df[..],
                r2f,
                r2d,
                std::slice::from_mut(&mut id),
            ) == 1)
            {
                {
                    return 1;
                }
            }
            if (id > (-8)) {
                {
                    if (rmn2sp(m, n, c, x, cv, kd, &mut df[..], r2f, r2d) == 1) {
                        {
                            return 1;
                        }
                    }
                }
            }
        }
    }
    return 0;
}

pub(super) fn rmn1(
    mut m: i32,
    mut n: i32,
    mut c: f64,
    mut x: f64,
    mut kd: i32,
    df: &mut [f64],
    r1f: &mut [f64],
    r1d: &mut [f64],
) -> i32 {
    let mut a0: f64 = 0.0;
    let mut b0: f64 = 0.0;
    let mut cx: f64 = 0.0;
    let mut r: f64 = 0.0;
    let mut r0: f64 = 0.0;
    let mut r1: f64 = 0.0;
    let mut r2: f64 = 0.0;
    let mut r3: f64 = 0.0;
    let mut reg: f64 = 0.0;
    let mut sa0: f64 = 0.0;
    let mut suc: f64 = 0.0;
    let mut sud: f64 = 0.0;
    let mut sum: f64 = 0.0;
    let mut sw: f64 = 0.0;
    let mut sw1: f64 = 0.0;
    let mut ip: i32 = 0;
    let mut j: i32 = 0;
    let mut k: i32 = 0;
    let mut l: i32 = 0;
    let mut lg: i32 = 0;
    let mut nm: i32 = 0;
    let mut nm1: i32 = 0;
    let mut nm2: i32 = 0;
    let mut np: i32 = 0;
    let mut ck = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut dj = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut sj = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut eps: f64 = 1.0e-14f64;
    nm1 = ((n - m) / 2);
    ip = if ((n - m) == (2 * nm1)) { 0 } else { 1 };
    nm = ((25 + nm1) + ((c) as i32));
    reg = if ((m + nm) > 80) { 1.0e-200f64 } else { 1.0f64 };
    r0 = reg;
    j = 1;
    while (j <= ((2 * m) + ip)) {
        {
            r0 *= ((j) as f64);
        }
        {
            j += 1;
            j
        };
    }
    r = r0;
    suc = (r * df[(0) as usize]);
    sw = 0.0f64;
    k = 2;
    while (k <= nm) {
        {
            r = ((((r * (((m + k) as f64) - 1.0f64)) * ((((m + k) + ip) as f64) - 1.5f64))
                / (((k) as f64) - 1.0f64))
                / (((k + ip) as f64) - 1.5f64));
            suc += (r * df[(k - 1) as usize]);
            if ((k > nm1) && ((suc - sw).abs() < ((suc).abs() * eps))) {
                {
                    break;
                }
            }
            sw = suc;
        }
        {
            k += 1;
            k
        };
    }
    if (x == 0.0f64) {
        {
            sckb(m, n, c, df, &mut ck[..]);
            sum = 0.0f64;
            sw1 = 0.0f64;
            j = 1;
            while (j <= nm) {
                {
                    sum += ck[(j - 1) as usize];
                    if ((sum - sw1).abs() < ((sum).abs() * eps)) {
                        {
                            break;
                        }
                    }
                    sw1 = sum;
                }
                {
                    j += 1;
                    j
                };
            }
            r1 = 1.0f64;
            j = 1;
            while (j <= (((n + m) + ip) / 2)) {
                {
                    r1 = (r1 * (((j) as f64) + (0.5f64 * (((n + m) + ip) as f64))));
                }
                {
                    j += 1;
                    j
                };
            }
            r2 = 1.0f64;
            j = 1;
            while (j <= m) {
                {
                    r2 *= ((2.0f64 * c) * ((j) as f64));
                }
                {
                    j += 1;
                    j
                };
            }
            r3 = 1.0f64;
            j = 1;
            while (j <= (((n - m) - ip) / 2)) {
                {
                    r3 *= ((j) as f64);
                }
                {
                    j += 1;
                    j
                };
            }
            sa0 = ((((2.0f64 * ((m + ip) as f64)) + 1.0f64) * r1)
                / ((((2.0f64).powf(((n) as f64)) * (c).powf(((ip) as f64))) * r2) * r3));
            if (ip == 0) {
                {
                    r1f[0] = (((sum / (sa0 * suc)) * df[(0) as usize]) * reg);
                    r1d[0] = 0.0f64;
                }
            } else {
                if (ip == 1) {
                    {
                        r1f[0] = 0.0f64;
                        r1d[0] = (((sum / (sa0 * suc)) * df[(0) as usize]) * reg);
                    }
                }
            }
            return 0;
        }
    }
    cx = (c * x);
    nm2 = ((2 * nm) + m);
    sphj(
        cx,
        nm2,
        std::slice::from_mut(&mut nm2),
        &mut sj[..],
        &mut dj[..],
    );
    a0 = ((1.0f64 - (((kd) as f64) / (x * x))).powf((0.5f64 * ((m) as f64))) / suc);
    r1f[0] = 0.0f64;
    sw = 0.0f64;
    lg = 0;
    k = 1;
    while (k <= nm) {
        {
            l = (((((2 * k) + m) - n) - 2) + ip);
            lg = if ((l % 4) == 0) { 1 } else { (-1) };
            if (k == 1) {
                {
                    r = r0;
                }
            } else {
                {
                    r = ((((r * (((m + k) as f64) - 1.0f64))
                        * ((((m + k) + ip) as f64) - 1.5f64))
                        / (((k) as f64) - 1.0f64))
                        / (((k + ip) as f64) - 1.5f64));
                }
            }
            np = (((m + (2 * k)) - 2) + ip);
            r1f[0] += (((((lg) as f64) * r) * df[(k - 1) as usize]) * sj[(np) as usize]);
            if ((k > nm1) && ((r1f[0] - sw).abs() < ((r1f[0]).abs() * eps))) {
                {
                    break;
                }
            }
            sw = r1f[0];
        }
        {
            k += 1;
            k
        };
    }
    r1f[0] *= a0;
    b0 = (((((kd * m) as f64) / (x).powf(3.0f64)) / (1.0f64 - (((kd) as f64) / (x * x)))) * r1f[0]);
    sud = 0.0f64;
    sw = 0.0f64;
    k = 1;
    while (k <= nm) {
        {
            l = (((((2 * k) + m) - n) - 2) + ip);
            lg = if ((l % 4) == 0) { 1 } else { (-1) };
            if (k == 1) {
                {
                    r = r0;
                }
            } else {
                {
                    r = ((((r * (((m + k) as f64) - 1.0f64))
                        * ((((m + k) + ip) as f64) - 1.5f64))
                        / (((k) as f64) - 1.0f64))
                        / (((k + ip) as f64) - 1.5f64));
                }
            }
            np = (((m + (2 * k)) - 2) + ip);
            sud = (sud + (((((lg) as f64) * r) * df[(k - 1) as usize]) * dj[(np) as usize]));
            if ((k > nm1) && ((sud - sw).abs() < ((sud).abs() * eps))) {
                {
                    break;
                }
            }
            sw = sud;
        }
        {
            k += 1;
            k
        };
    }
    r1d[0] = (b0 + ((a0 * c) * sud));
    return 0;
}

pub(super) fn sphj(mut x: f64, mut n: i32, nm: &mut [i32], sj: &mut [f64], dj: &mut [f64]) {
    let mut k: i32 = 0;
    let mut m: i32 = 0;
    let mut cs: f64 = 0.0;
    let mut f: f64 = 0.0;
    let mut f0: f64 = 0.0;
    let mut f1: f64 = 0.0;
    let mut sa: f64 = 0.0;
    let mut sb: f64 = 0.0;
    nm[0] = n;
    if ((x).abs() < 1e-100f64) {
        {
            k = 0;
            while (k <= n) {
                {
                    sj[(k) as usize] = 0.0f64;
                    dj[(k) as usize] = 0.0f64;
                }
                {
                    k += 1;
                    k
                };
            }
            sj[(0) as usize] = 1.0f64;
            if (n > 0) {
                {
                    dj[(1) as usize] = (1.0f64 / 3.0f64);
                }
            }
            return;
        }
    }
    sj[(0) as usize] = ((x).sin() / x);
    dj[(0) as usize] = (((x).cos() - ((x).sin() / x)) / x);
    if (n < 1) {
        {
            return;
        }
    }
    sj[(1) as usize] = ((sj[(0) as usize] - (x).cos()) / x);
    if (n >= 2) {
        {
            sa = sj[(0) as usize];
            sb = sj[(1) as usize];
            m = msta1(x, 200);
            if (m < n) {
                {
                    nm[0] = m;
                }
            } else {
                {
                    m = msta2(x, n, 15);
                }
            }
            f = 0.0f64;
            f0 = 0.0f64;
            f1 = 1e-100f64;
            k = m;
            while (k >= 0) {
                {
                    f = (((((2.0f64 * ((k) as f64)) + 3.0f64) * f1) / x) - f0);
                    if (k <= nm[0]) {
                        {
                            sj[(k) as usize] = f;
                        }
                    }
                    f0 = f1;
                    f1 = f;
                }
                {
                    k -= 1;
                    k
                };
            }
            cs = if ((sa).abs() > (sb).abs()) {
                (sa / f)
            } else {
                (sb / f0)
            };
            k = 0;
            while (k <= nm[0]) {
                {
                    sj[(k) as usize] *= cs;
                }
                {
                    k += 1;
                    k
                };
            }
        }
    }
    k = 1;
    while (k <= nm[0]) {
        {
            dj[(k) as usize] =
                (sj[(k - 1) as usize] - (((((k) as f64) + 1.0f64) * sj[(k) as usize]) / x));
        }
        {
            k += 1;
            k
        };
    }
    return;
}

pub(super) fn msta1(mut x: f64, mut mp: i32) -> i32 {
    let mut it: i32 = 0;
    let mut nn: i32 = 0;
    let mut n0: i32 = 0;
    let mut n1: i32 = 0;
    let mut a0: f64 = 0.0;
    let mut f: f64 = 0.0;
    let mut f0: f64 = 0.0;
    let mut f1: f64 = 0.0;
    a0 = (x).abs();
    n0 = (((1.1f64 * a0) as i32) + 1);
    f0 = (((0.5f64 * (6.28f64 * ((n0) as f64)).log10())
        - (((n0) as f64) * ((1.36f64 * a0) / ((n0) as f64)).log10()))
        - ((mp) as f64));
    n1 = (n0 + 5);
    f1 = (((0.5f64 * (6.28f64 * ((n1) as f64)).log10())
        - (((n1) as f64) * ((1.36f64 * a0) / ((n1) as f64)).log10()))
        - ((mp) as f64));
    it = 1;
    while (it <= 20) {
        {
            nn = ((((n1) as f64) - (((n1 - n0) as f64) / (1.0f64 - (f0 / f1)))) as i32);
            f = (((0.5f64 * (6.28f64 * ((nn) as f64)).log10())
                - (((nn) as f64) * ((1.36f64 * a0) / ((nn) as f64)).log10()))
                - ((mp) as f64));
            if (((nn - n1) as f64).abs() < ((1) as f64)) {
                {
                    break;
                }
            }
            n0 = n1;
            f0 = f1;
            n1 = nn;
            f1 = f;
        }
        {
            it += 1;
            it
        };
    }
    return nn;
}

pub(super) fn msta2(mut x: f64, mut n: i32, mut mp: i32) -> i32 {
    let mut it: i32 = 0;
    let mut n0: i32 = 0;
    let mut n1: i32 = 0;
    let mut nn: i32 = 0;
    let mut a0: f64 = 0.0;
    let mut hmp: f64 = 0.0;
    let mut ejn: f64 = 0.0;
    let mut obj: f64 = 0.0;
    let mut f: f64 = 0.0;
    let mut f0: f64 = 0.0;
    let mut f1: f64 = 0.0;
    a0 = (x).abs();
    hmp = (0.5f64 * ((mp) as f64));
    ejn = ((0.5f64 * (6.28f64 * ((n) as f64)).log10())
        - (((n) as f64) * ((1.36f64 * a0) / ((n) as f64)).log10()));
    if (ejn <= hmp) {
        {
            obj = ((mp) as f64);
            n0 = (((1.1f64 * a0) as i32) + 1);
        }
    } else {
        {
            obj = (hmp + ejn);
            n0 = n;
        }
    }
    f0 = (((0.5f64 * (6.28f64 * ((n0) as f64)).log10())
        - (((n0) as f64) * ((1.36f64 * a0) / ((n0) as f64)).log10()))
        - obj);
    n1 = (n0 + 5);
    f1 = (((0.5f64 * (6.28f64 * ((n1) as f64)).log10())
        - (((n1) as f64) * ((1.36f64 * a0) / ((n1) as f64)).log10()))
        - obj);
    it = 1;
    while (it <= 20) {
        {
            nn = ((((n1) as f64) - (((n1 - n0) as f64) / (1.0f64 - (f0 / f1)))) as i32);
            f = (((0.5f64 * (6.28f64 * ((nn) as f64)).log10())
                - (((nn) as f64) * ((1.36f64 * a0) / ((nn) as f64)).log10()))
                - obj);
            if (((nn - n1) as f64).abs() < ((1) as f64)) {
                {
                    break;
                }
            }
            n0 = n1;
            f0 = f1;
            n1 = nn;
            f1 = f;
        }
        {
            it += 1;
            it
        };
    }
    return (nn + 10);
}

pub(super) fn rmn2l(
    mut m: i32,
    mut n: i32,
    mut c: f64,
    mut x: f64,
    mut Kd: i32,
    Df: &mut [f64],
    R2f: &mut [f64],
    R2d: &mut [f64],
    Id: &mut [i32],
) -> i32 {
    let mut ip: i32 = 0;
    let mut nm1: i32 = 0;
    let mut nm: i32 = 0;
    let mut nm2: i32 = 0;
    let mut np: i32 = 0;
    let mut j: i32 = 0;
    let mut k: i32 = 0;
    let mut l: i32 = 0;
    let mut lg: i32 = 0;
    let mut id1: i32 = 0;
    let mut id2: i32 = 0;
    let mut a0: f64 = 0.0;
    let mut b0: f64 = 0.0;
    let mut cx: f64 = 0.0;
    let mut reg: f64 = 0.0;
    let mut r0: f64 = 0.0;
    let mut r: f64 = 0.0;
    let mut suc: f64 = 0.0;
    let mut sud: f64 = 0.0;
    let mut sw: f64 = 0.0;
    let mut eps1: f64 = 0.0;
    let mut eps2: f64 = 0.0;
    let mut eps: f64 = 1.0e-14f64;
    let mut sy = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut dy = vec![0.0; spheroidal_work_len(m, n, c)];
    ip = 1;
    nm1 = ((n - m) / 2);
    if ((n - m) == (2 * nm1)) {
        {
            ip = 0;
        }
    }
    nm = ((25 + nm1) + ((c) as i32));
    reg = 1.0f64;
    if ((m + nm) > 80) {
        {
            reg = 1.0e-200f64;
        }
    }
    nm2 = ((2 * nm) + m);
    cx = (c * x);
    sphy(
        cx,
        nm2,
        std::slice::from_mut(&mut nm2),
        &mut sy[..],
        &mut dy[..],
    );
    r0 = reg;
    j = 1;
    while (j <= ((2 * m) + ip)) {
        {
            r0 *= ((j) as f64);
        }
        {
            j += 1;
            j
        };
    }
    r = r0;
    suc = (r * Df[(0) as usize]);
    sw = 0.0f64;
    k = 2;
    while (k <= nm) {
        {
            r = ((((r * (((m + k) as f64) - 1.0f64)) * ((((m + k) + ip) as f64) - 1.5f64))
                / (((k) as f64) - 1.0f64))
                / (((k + ip) as f64) - 1.5f64));
            suc += (r * Df[(k - 1) as usize]);
            if ((k > nm1) && ((suc - sw).abs() < ((suc).abs() * eps))) {
                {
                    break;
                }
            }
            sw = suc;
        }
        {
            k += 1;
            k
        };
    }
    a0 = ((1.0f64 - (((Kd) as f64) / (x * x))).powf((0.5f64 * ((m) as f64))) / suc);
    R2f[0] = 0.0f64;
    eps1 = 0.0f64;
    np = 0;
    k = 1;
    while (k <= nm) {
        {
            l = (((((2 * k) + m) - n) - 2) + ip);
            lg = if ((l % 4) == 0) { 1 } else { (-1) };
            if (k == 1) {
                {
                    r = r0;
                }
            } else {
                {
                    r = ((((r * (((m + k) as f64) - 1.0f64))
                        * ((((m + k) + ip) as f64) - 1.5f64))
                        / (((k) as f64) - 1.0f64))
                        / (((k + ip) as f64) - 1.5f64));
                }
            }
            np = (((m + (2 * k)) - 2) + ip);
            R2f[0] += ((((lg) as f64) * r) * (Df[(k - 1) as usize] * sy[(np) as usize]));
            eps1 = (R2f[0] - sw).abs();
            if ((k > nm1) && (eps1 < ((R2f[0]).abs() * eps))) {
                {
                    break;
                }
            }
            sw = R2f[0];
        }
        {
            k += 1;
            k
        };
    }
    id1 = ((((eps1 / (R2f[0]).abs()) + eps).log10()) as i32);
    R2f[0] *= a0;
    if (np >= nm2) {
        {
            Id[0] = 10;
            return 2;
        }
    }
    b0 = (((((Kd * m) as f64) / (x).powf(((3) as f64))) / (1.0f64 - (((Kd) as f64) / (x * x))))
        * R2f[0]);
    sud = 0.0f64;
    eps2 = 0.0f64;
    k = 1;
    while (k <= nm) {
        {
            l = (((((2 * k) + m) - n) - 2) + ip);
            lg = if ((l % 4) == 0) { 1 } else { (-1) };
            if (k == 1) {
                {
                    r = r0;
                }
            } else {
                {
                    r = ((((r * (((m + k) as f64) - 1.0f64))
                        * ((((m + k) + ip) as f64) - 1.5f64))
                        / (((k) as f64) - 1.0f64))
                        / (((k + ip) as f64) - 1.5f64));
                }
            }
            np = (((m + (2 * k)) - 2) + ip);
            sud += ((((lg) as f64) * r) * (Df[(k - 1) as usize] * dy[(np) as usize]));
            eps2 = (sud - sw).abs();
            if ((k > nm1) && (eps2 < ((sud).abs() * eps))) {
                {
                    break;
                }
            }
            sw = sud;
        }
        {
            k += 1;
            k
        };
    }
    R2d[0] = (b0 + ((a0 * c) * sud));
    id2 = ((((eps2 / (sud).abs()) + eps).log10()) as i32);
    Id[0] = if (id1 > id2) { id1 } else { id2 };
    return 0;
}

pub(super) fn sphy(mut x: f64, mut n: i32, nm: &mut [i32], sy: &mut [f64], dy: &mut [f64]) {
    let mut f: f64 = 0.0;
    let mut f0: f64 = 0.0;
    let mut f1: f64 = 0.0;
    if (x < 1.0e-60f64) {
        {
            let mut k: i32 = 0;
            while (k <= n) {
                {
                    sy[(k) as usize] = (-1.0e300f64);
                    dy[(k) as usize] = 1.0e300f64;
                }
                {
                    k += 1;
                    k
                };
            }
            nm[0] = n;
            return;
        }
    }
    sy[(0) as usize] = ((-(x).cos()) / x);
    f0 = sy[(0) as usize];
    dy[(0) as usize] = (((x).sin() + ((x).cos() / x)) / x);
    if (n < 1) {
        {
            nm[0] = n;
            return;
        }
    }
    sy[(1) as usize] = ((sy[(0) as usize] - (x).sin()) / x);
    f1 = sy[(1) as usize];
    let mut k: i32 = 2;
    while (k <= n) {
        {
            f = (((((2.0f64 * ((k) as f64)) - 1.0f64) * f1) / x) - f0);
            sy[(k) as usize] = f;
            if ((f).abs() >= 1.0e300f64) {
                {
                    nm[0] = (k - 1);
                    return;
                }
            }
            f0 = f1;
            f1 = f;
        }
        {
            k += 1;
            k
        };
    }
    nm[0] = (n - 1);
    let mut k: i32 = 1;
    while (k <= nm[0]) {
        {
            dy[(k) as usize] =
                (sy[(k - 1) as usize] - (((((k) as f64) + 1.0f64) * sy[(k) as usize]) / x));
        }
        {
            k += 1;
            k
        };
    }
    return;
}

pub(super) fn rmn2sp(
    mut m: i32,
    mut n: i32,
    mut c: f64,
    mut x: f64,
    mut cv: f64,
    mut kd: i32,
    df: &mut [f64],
    r2f: &mut [f64],
    r2d: &mut [f64],
) -> i32 {
    let mut k: i32 = 0;
    let mut j: i32 = 0;
    let mut j1: i32 = 0;
    let mut j2: i32 = 0;
    let mut l1: i32 = 0;
    let mut ki: i32 = 0;
    let mut nm3: i32 = 0;
    let mut ip: f64 = 0.0;
    let mut nm1: f64 = 0.0;
    let mut nm: f64 = 0.0;
    let mut nm2: f64 = 0.0;
    let mut su0: f64 = 0.0;
    let mut sw: f64 = 0.0;
    let mut sd0: f64 = 0.0;
    let mut su1: f64 = 0.0;
    let mut sd1: f64 = 0.0;
    let mut sd2: f64 = 0.0;
    let mut ga: f64 = 0.0;
    let mut r1: f64 = 0.0;
    let mut r2: f64 = 0.0;
    let mut r3: f64 = 0.0;
    let mut sf: f64 = 0.0;
    let mut gb: f64 = 0.0;
    let mut spl: f64 = 0.0;
    let mut gc: f64 = 0.0;
    let mut sd: f64 = 0.0;
    let mut r4: f64 = 0.0;
    let mut spd1: f64 = 0.0;
    let mut spd2: f64 = 0.0;
    let mut su2: f64 = 0.0;
    let mut ck1: f64 = 0.0;
    let mut ck2: f64 = 0.0;
    let mut sum: f64 = 0.0;
    let mut sdm: f64 = 0.0;
    let mut pm = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut pd = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut qm = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut qd = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut dn = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut eps: f64 = 1.0e-14f64;
    nm1 = (((n - m) / 2) as f64);
    nm = ((25.0f64 + nm1) + c);
    nm2 = ((((2) as f64) * nm) + ((m) as f64));
    ip = (((n - m) % 2) as f64);
    if (kmn(
        m,
        n,
        c,
        cv,
        kd,
        df,
        &mut dn[..],
        std::slice::from_mut(&mut ck1),
        std::slice::from_mut(&mut ck2),
    ) == 1)
    {
        {
            return 1;
        }
    }
    lpmns(m, ((nm2) as i32), x, &mut pm[..], &mut pd[..]);
    lqmns(m, ((nm2) as i32), x, &mut qm[..], &mut qd[..]);
    su0 = 0.0f64;
    sw = 0.0f64;
    k = 1;
    while (((k) as f64) <= nm) {
        {
            j = ((((((2 * k) - 2) + m) as f64) + ip) as i32);
            su0 += (df[(k - 1) as usize] * qm[(j) as usize]);
            if ((((k) as f64) > nm1) && ((su0 - sw).abs() < ((su0).abs() * eps))) {
                {
                    break;
                }
            }
            sw = su0;
        }
        {
            k += 1;
            k
        };
    }
    sd0 = 0.0f64;
    k = 1;
    while (((k) as f64) <= nm) {
        {
            j = ((((((2 * k) - 2) + m) as f64) + ip) as i32);
            sd0 += (df[(k - 1) as usize] * qd[(j) as usize]);
            if ((((k) as f64) > nm1) && ((sd0 - sw).abs() < ((sd0).abs() * eps))) {
                {
                    break;
                }
            }
            sw = sd0;
        }
        {
            k += 1;
            k
        };
    }
    su1 = 0.0f64;
    sd1 = 0.0f64;
    k = 1;
    while (k <= m) {
        {
            j = ((((m - (2 * k)) as f64) + ip) as i32);
            if (j < 0) {
                {
                    j = ((-j) - 1);
                }
            }
            su1 += (dn[(k - 1) as usize] * qm[(j) as usize]);
            sd1 += (dn[(k - 1) as usize] * qd[(j) as usize]);
        }
        {
            k += 1;
            k
        };
    }
    ga = ((x - 1.0f64) / (x + 1.0f64)).powf((0.5f64 * ((m) as f64)));
    k = 1;
    while (k <= m) {
        {
            j = ((((m - (2 * k)) as f64) + ip) as i32);
            if (j >= 0) {
                {
                    {
                        k += 1;
                        k
                    };
                    continue;
                }
            }
            if (j < 0) {
                {
                    j = ((-j) - 1);
                }
            }
            r1 = 1.0f64;
            j1 = 0;
            while (j1 < j) {
                {
                    r1 *= ((m + j1) as f64);
                }
                {
                    j1 += 1;
                    j1
                };
            }
            r2 = 1.0f64;
            j2 = 1;
            while (j2 <= ((m - j) - 2)) {
                {
                    r2 *= ((j2) as f64);
                }
                {
                    j2 += 1;
                    j2
                };
            }
            r3 = 1.0f64;
            sf = 1.0f64;
            l1 = 1;
            while (l1 <= j) {
                {
                    r3 = (((((0.5f64 * r3) * ((((-j) + l1) as f64) - 1.0f64))
                        * ((j + l1) as f64))
                        / (((m + l1) * l1) as f64))
                        * (1.0f64 - x));
                    sf += r3;
                }
                {
                    l1 += 1;
                    l1
                };
            }
            if ((m - j) >= 2) {
                {
                    gb = ((((m - j) as f64) - 1.0f64) * r2);
                }
            }
            if ((m - j) <= 1) {
                {
                    gb = 1.0f64;
                }
            }
            spl = (((r1 * ga) * gb) * sf);
            su1 += ((neg_one_pow(j + m) * dn[(k - 1) as usize]) * spl);
            spd1 = ((((m) as f64) / ((x * x) - 1.0f64)) * spl);
            gc = (((0.5f64 * ((j) as f64)) * (((j) as f64) + 1.0f64)) / (((m) as f64) + 1.0f64));
            sd = 1.0f64;
            r4 = 1.0f64;
            l1 = 1;
            while (l1 <= (j - 1)) {
                {
                    r4 = (((((0.5f64 * r4) * (((-j) + l1) as f64))
                        * (((j + l1) as f64) + 1.0f64))
                        / ((((m + l1) as f64) + 1.0f64) * ((l1) as f64)))
                        * (1.0f64 - x));
                    sd += r4;
                }
                {
                    l1 += 1;
                    l1
                };
            }
            spd2 = ((((r1 * ga) * gb) * gc) * sd);
            sd1 += ((neg_one_pow(j + m) * dn[(k - 1) as usize]) * (spd1 + spd2));
        }
        {
            k += 1;
            k
        };
    }
    su2 = 0.0f64;
    ki = ((((((2 * m) + 1) as f64) + ip) / ((2) as f64)) as i32);
    ki = (1).max(ki);
    debug_assert!(((ki - 1) >= 0));
    nm3 = ((nm + ((ki) as f64)) as i32);
    k = ki;
    while (k <= nm3) {
        {
            j = ((((((2 * k) - 1) - m) as f64) - ip) as i32);
            su2 += (dn[(k - 1) as usize] * pm[(j) as usize]);
            if ((j > m) && ((su2 - sw).abs() < ((su2).abs() * eps))) {
                {
                    break;
                }
            }
            sw = su2;
        }
        {
            k += 1;
            k
        };
    }
    sd2 = 0.0f64;
    k = ki;
    while (k < nm3) {
        {
            j = ((((((2 * k) - 1) - m) as f64) - ip) as i32);
            sd2 += (dn[(k - 1) as usize] * pd[(j) as usize]);
            if ((j > m) && ((sd2 - sw).abs() < ((sd2).abs() * eps))) {
                {
                    break;
                }
            }
            sw = sd2;
        }
        {
            k += 1;
            k
        };
    }
    sum = ((su0 + su1) + su2);
    sdm = ((sd0 + sd1) + sd2);
    r2f[0] = (sum / ck2);
    r2d[0] = (sdm / ck2);
    return 0;
}

pub(super) fn kmn(
    mut m: i32,
    mut n: i32,
    mut c: f64,
    mut cv: f64,
    mut kd: i32,
    df: &mut [f64],
    dn: &mut [f64],
    ck1: &mut [f64],
    ck2: &mut [f64],
) -> i32 {
    let mut nm: i32 = 0;
    let mut nn: i32 = 0;
    let mut ip: i32 = 0;
    let mut k: i32 = 0;
    let mut i: i32 = 0;
    let mut l: i32 = 0;
    let mut j: i32 = 0;
    let mut cs: f64 = 0.0;
    let mut gk0: f64 = 0.0;
    let mut gk1: f64 = 0.0;
    let mut gk2: f64 = 0.0;
    let mut gk3: f64 = 0.0;
    let mut t: f64 = 0.0;
    let mut r: f64 = 0.0;
    let mut dnp: f64 = 0.0;
    let mut su0: f64 = 0.0;
    let mut sw: f64 = 0.0;
    let mut r1: f64 = 0.0;
    let mut r2: f64 = 0.0;
    let mut r3: f64 = 0.0;
    let mut sa0: f64 = 0.0;
    let mut r4: f64 = 0.0;
    let mut r5: f64 = 0.0;
    let mut g0: f64 = 0.0;
    let mut sb0: f64 = 0.0;
    nm = (25 + (((0.5f64 * ((n - m) as f64)) + c) as i32));
    nn = (nm + m);
    let mut u = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut v = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut w = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut tp = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut rk = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut eps: f64 = 1.0e-14f64;
    cs = ((c * c) * ((kd) as f64));
    ck1[0] = 0.0f64;
    ck2[0] = 0.0f64;
    ip = if (((n - m) % 2) == 0) { 0 } else { 1 };
    k = 0;
    i = 1;
    while (i <= (nn + 3)) {
        {
            k = if (ip == 0) {
                ((-2) * (i - 1))
            } else {
                (-((2 * i) - 3))
            };
            gk0 = ((2.0f64 * ((m) as f64)) + ((k) as f64));
            gk1 = (((m + k) as f64) * (((m + k) as f64) + 1.0f64));
            gk2 = ((2.0f64 * ((m + k) as f64)) - 1.0f64);
            gk3 = ((2.0f64 * ((m + k) as f64)) + 3.0f64);
            u[(i - 1) as usize] = (((gk0 * (gk0 - 1.0f64)) * cs) / (gk2 * (gk2 + 2.0f64)));
            v[(i - 1) as usize] = ((gk1 - cv)
                + ((((2.0f64 * (gk1 - ((m * m) as f64))) - 1.0f64) * cs) / (gk2 * gk3)));
            w[(i - 1) as usize] = ((((((k) as f64) + 1.0f64) * (((k) as f64) + 2.0f64)) * cs)
                / ((gk2 + 2.0f64) * gk3));
        }
        {
            i += 1;
            i
        };
    }
    k = 1;
    while (k <= m) {
        {
            t = v[(m) as usize];
            l = 0;
            while (l <= ((m - k) - 1)) {
                t = (v[((m - l) - 1) as usize]
                    - ((w[(m - l) as usize] * u[((m - l) - 1) as usize]) / t));
                {
                    l += 1;
                    l
                };
            }
            rk[(k - 1) as usize] = ((-u[(k - 1) as usize]) / t);
        }
        {
            k += 1;
            k
        };
    }
    r = 1.0f64;
    k = 1;
    while (k <= m) {
        {
            r = (r * rk[(k - 1) as usize]);
            dn[(k - 1) as usize] = (df[(0) as usize] * r);
        }
        {
            k += 1;
            k
        };
    }
    tp[(nn - 1) as usize] = v[(nn) as usize];
    k = (nn - 1);
    while (k >= (m + 1)) {
        {
            tp[(k - 1) as usize] =
                (v[(k) as usize] - ((w[(k + 1) as usize] * u[(k) as usize]) / tp[(k) as usize]));
            if (k > (m + 1)) {
                rk[(k - 1) as usize] = ((-u[(k - 1) as usize]) / tp[(k - 1) as usize]);
            }
        }
        {
            k -= 1;
            k
        };
    }
    dnp = if (m == 0) {
        df[(0) as usize]
    } else {
        dn[(m - 1) as usize]
    };
    dn[(m) as usize] = (((neg_one_pow(ip) * dnp) * cs)
        / ((((2.0f64 * ((m) as f64)) - 1.0f64)
            * (((2.0f64 * ((m) as f64)) + 1.0f64) - (4.0f64 * ((ip) as f64))))
            * tp[(m) as usize]));
    k = (m + 2);
    while (k <= nn) {
        dn[(k - 1) as usize] = (rk[(k - 1) as usize] * dn[(k - 2) as usize]);
        {
            k += 1;
            k
        };
    }
    r1 = 1.0f64;
    j = 1;
    while (j <= (((n + m) + ip) / 2)) {
        {
            r1 = (r1 * (((j) as f64) + (0.5f64 * (((n + m) + ip) as f64))));
        }
        {
            j += 1;
            j
        };
    }
    r = 1.0f64;
    j = 1;
    while (j <= ((2 * m) + ip)) {
        {
            r *= ((j) as f64);
        }
        {
            j += 1;
            j
        };
    }
    su0 = (r * df[(0) as usize]);
    sw = 0.0f64;
    k = 2;
    while (k <= nm) {
        {
            r = ((((r * (((m + k) as f64) - 1.0f64)) * ((((m + k) + ip) as f64) - 1.5f64))
                / (((k) as f64) - 1.0f64))
                / (((k + ip) as f64) - 1.5f64));
            su0 = (su0 + (r * df[(k - 1) as usize]));
            if ((k > ((n - m) / 2)) && (((su0 - sw) / su0).abs() < eps)) {
                {
                    break;
                }
            }
            sw = su0;
        }
        {
            k += 1;
            k
        };
    }
    if (kd != 1) {
        {
            r2 = 1.0f64;
            j = 1;
            while (j <= m) {
                r2 = (((2.0f64 * c) * r2) * ((j) as f64));
                {
                    j += 1;
                    j
                };
            }
            r3 = 1.0f64;
            j = 1;
            while (j <= (((n - m) - ip) / 2)) {
                r3 = (r3 * ((j) as f64));
                {
                    j += 1;
                    j
                };
            }
            sa0 = ((((2.0f64 * ((m + ip) as f64)) + 1.0f64) * r1)
                / (((((2.0f64).powf(((n) as f64)) * (c).powf(((ip) as f64))) * r2) * r3)
                    * df[(0) as usize]));
            ck1[0] = (sa0 * su0);
            if (kd == (-1)) {
                {
                    return 0;
                }
            }
        }
    }
    r4 = 1.0f64;
    j = 1;
    while (j <= (((n - m) - ip) / 2)) {
        {
            r4 *= (4.0f64 * ((j) as f64));
        }
        {
            j += 1;
            j
        };
    }
    r5 = 1.0f64;
    j = 1;
    while (j <= m) {
        r5 = ((r5 * ((j + m) as f64)) / c);
        {
            j += 1;
            j
        };
    }
    if (m == 0) {
        g0 = df[(0) as usize];
    } else {
        g0 = dn[(m - 1) as usize];
    }
    sb0 = ((((((ip) as f64) + 1.0f64) * (c).powf(((ip + 1) as f64)))
        / (((2.0f64 * ((ip) as f64)) * (((m) as f64) - 2.0f64)) + 1.0f64))
        / ((2.0f64 * ((m) as f64)) - 1.0f64));
    ck2[0] = ((((((neg_one_pow(ip) * sb0) * r4) * r5) * g0) / r1) * su0);
    return 0;
}

pub(super) fn lpmns(mut m: i32, mut n: i32, mut x: f64, pm: &mut [f64], pd: &mut [f64]) {
    let mut k: i32 = 0;
    let mut coef: f64 = 0.0;
    let mut x0: f64 = 0.0;
    let mut pm0: f64 = 0.0;
    let mut pm1: f64 = 0.0;
    let mut pm2: f64 = 0.0;
    let mut pmk: f64 = 0.0;
    k = 0;
    while (k <= n) {
        {
            pm[(k) as usize] = 0.0f64;
            pd[(k) as usize] = 0.0f64;
        }
        {
            k += 1;
            k
        };
    }
    if ((x).abs() == 1.0f64) {
        {
            k = 0;
            while (k <= n) {
                {
                    if (m == 0) {
                        {
                            pm[(k) as usize] = 1.0f64;
                            pd[(k) as usize] = ((0.5f64 * ((k) as f64)) * (((k) as f64) + 1.0f64));
                            if (x < 0.0f64) {
                                {
                                    pm[(k) as usize] *=
                                        ((if ((k % 2) == 0) { 1 } else { (-1) }) as f64);
                                    pd[(k) as usize] *=
                                        ((if (((k + 1) % 2) == 0) { 1 } else { (-1) }) as f64);
                                }
                            }
                        }
                    } else {
                        if (m == 1) {
                            {
                                pd[(k) as usize] = 1e300f64;
                            }
                        } else {
                            if (m == 2) {
                                {
                                    pd[(k) as usize] = (((((-0.25f64) * (((k) as f64) + 2.0f64))
                                        * (((k) as f64) + 1.0f64))
                                        * ((k) as f64))
                                        * (((k) as f64) - 1.0f64));
                                    if (x < 0.0f64) {
                                        pd[(k) as usize] *=
                                            ((if (((k + 1) % 2) == 0) { 1 } else { (-1) }) as f64);
                                    }
                                }
                            }
                        }
                    }
                }
                {
                    k += 1;
                    k
                };
            }
            return;
        }
    }
    x0 = (1.0f64 - (x * x)).abs();
    pm0 = 1.0f64;
    pmk = pm0;
    k = 1;
    while (k <= m) {
        {
            pmk = ((((2.0f64 * ((k) as f64)) - 1.0f64) * (x0).sqrt()) * pm0);
            pm0 = pmk;
        }
        {
            k += 1;
            k
        };
    }
    pm1 = ((((2.0f64 * ((m) as f64)) + 1.0f64) * x) * pm0);
    pm[(m) as usize] = pmk;
    pm[(m + 1) as usize] = pm1;
    k = (m + 2);
    while (k <= n) {
        {
            pm2 = ((((((2.0f64 * ((k) as f64)) - 1.0f64) * x) * pm1)
                - ((((k + m) as f64) - 1.0f64) * pmk))
                / ((k - m) as f64));
            pm[(k) as usize] = pm2;
            pmk = pm1;
            pm1 = pm2;
        }
        {
            k += 1;
            k
        };
    }
    pd[(0) as usize] = ((((1.0f64 - ((m) as f64)) * pm[(1) as usize]) - (x * pm[(0) as usize]))
        / ((x * x) - 1.0f64));
    k = 1;
    while (k <= n) {
        {
            pd[(k) as usize] = ((((((k) as f64) * x) * pm[(k) as usize])
                - (((k + m) as f64) * pm[(k - 1) as usize]))
                / ((x * x) - 1.0f64));
        }
        {
            k += 1;
            k
        };
    }
    coef = ((if ((m % 2) == 0) { 1 } else { (-1) }) as f64);
    k = 1;
    while (k <= n) {
        {
            pm[(k) as usize] *= coef;
            pd[(k) as usize] *= coef;
        }
        {
            k += 1;
            k
        };
    }
    return;
}

pub(super) fn lqmns(mut m: i32, mut n: i32, mut x: f64, qm: &mut [f64], qd: &mut [f64]) {
    let mut l: i32 = 0;
    let mut ls: i32 = 0;
    let mut k: i32 = 0;
    let mut km: i32 = 0;
    let mut xq: f64 = 0.0;
    let mut q0: f64 = 0.0;
    let mut q00: f64 = 0.0;
    let mut q10: f64 = 0.0;
    let mut q01: f64 = 0.0;
    let mut q11: f64 = 0.0;
    let mut qf0: f64 = 0.0;
    let mut qf1: f64 = 0.0;
    let mut qm0: f64 = 0.0;
    let mut qm1: f64 = 0.0;
    let mut qg0: f64 = 0.0;
    let mut qg1: f64 = 0.0;
    let mut qh0: f64 = 0.0;
    let mut qh1: f64 = 0.0;
    let mut qh2: f64 = 0.0;
    let mut qmk: f64 = 0.0;
    let mut q0l: f64 = 0.0;
    let mut q1l: f64 = 0.0;
    let mut qf2: f64 = 0.0;
    let mut val: f64 = 0.0;
    val = 0.0f64;
    if ((x).abs() == 1.0f64) {
        {
            val = 1e300f64;
        }
    }
    k = 0;
    while (k <= n) {
        {
            qm[(k) as usize] = val;
            qd[(k) as usize] = val;
        }
        {
            k += 1;
            k
        };
    }
    if ((x).abs() == 1.0f64) {
        {
            return;
        }
    }
    ls = if ((x).abs() > 1.0f64) { (-1) } else { 1 };
    xq = (((ls) as f64) * (1.0f64 - (x * x))).sqrt();
    q0 = (0.5f64 * (((x + 1.0f64) / (x - 1.0f64)).abs()).ln());
    q00 = q0;
    q10 = ((-1.0f64) / xq);
    q01 = ((x * q0) - 1.0f64);
    q11 = ((((-ls) as f64) * xq) * (q0 + (x / (1.0f64 - (x * x)))));
    qf0 = q00;
    qf1 = q10;
    qm0 = 0.0f64;
    qm1 = 0.0f64;
    k = 2;
    while (k <= m) {
        {
            qm0 = ((((((-2.0f64) * (((k) as f64) - 1.0f64)) / xq) * x) * qf1)
                - (((((ls) as f64) * (((k) as f64) - 1.0f64)) * (2.0f64 - ((k) as f64))) * qf0));
            qf0 = qf1;
            qf1 = qm0;
        }
        {
            k += 1;
            k
        };
    }
    if (m == 0) {
        {
            qm0 = q00;
        }
    }
    if (m == 1) {
        {
            qm0 = q10;
        }
    }
    qm[(0) as usize] = qm0;
    if ((x).abs() < 1.0001f64) {
        {
            if ((m == 0) && (n > 0)) {
                {
                    qf0 = q00;
                    qf1 = q01;
                    k = 2;
                    while (k <= n) {
                        {
                            qf2 = ((((((2.0f64 * ((k) as f64)) - 1.0f64) * x) * qf1)
                                - ((((k) as f64) - 1.0f64) * qf0))
                                / ((k) as f64));
                            qm[(k) as usize] = qf2;
                            qf0 = qf1;
                            qf1 = qf2;
                        }
                        {
                            k += 1;
                            k
                        };
                    }
                }
            }
            qg0 = q01;
            qg1 = q11;
            k = 2;
            while (k <= m) {
                {
                    qm1 = ((((((-2.0f64) * (((k) as f64) - 1.0f64)) / xq) * x) * qg1)
                        - ((((ls * k) as f64) * (3.0f64 - ((k) as f64))) * qg0));
                    qg0 = qg1;
                    qg1 = qm1;
                }
                {
                    k += 1;
                    k
                };
            }
            if (m == 0) {
                {
                    qm1 = q01;
                }
            }
            if (m == 1) {
                {
                    qm1 = q11;
                }
            }
            qm[(1) as usize] = qm1;
            if ((m == 1) && (n > 1)) {
                {
                    qh0 = q10;
                    qh1 = q11;
                    k = 2;
                    while (k <= n) {
                        {
                            qh2 = ((((((2.0f64 * ((k) as f64)) - 1.0f64) * x) * qh1)
                                - (((k) as f64) * qh0))
                                / (((k) as f64) - 1.0f64));
                            qm[(k) as usize] = qh2;
                            qh0 = qh1;
                            qh1 = qh2;
                        }
                        {
                            k += 1;
                            k
                        };
                    }
                }
            } else {
                if (m >= 2) {
                    {
                        qg0 = q00;
                        qg1 = q01;
                        qh0 = q10;
                        qh1 = q11;
                        qmk = 0.0f64;
                        l = 2;
                        while (l <= n) {
                            {
                                q0l = ((((((2.0f64 * ((l) as f64)) - 1.0f64) * x) * qg1)
                                    - ((((l) as f64) - 1.0f64) * qg0))
                                    / ((l) as f64));
                                q1l = ((((((2.0f64 * ((l) as f64)) - 1.0f64) * x) * qh1)
                                    - (((l) as f64) * qh0))
                                    / (((l) as f64) - 1.0f64));
                                qf0 = q0l;
                                qf1 = q1l;
                                k = 2;
                                while (k <= m) {
                                    {
                                        qmk = ((((((-2.0f64) * (((k) as f64) - 1.0f64)) / xq)
                                            * x)
                                            * qf1)
                                            - (((((ls) as f64) * (((k + l) as f64) - 1.0f64))
                                                * ((((l) as f64) + 2.0f64) - ((k) as f64)))
                                                * qf0));
                                        qf0 = qf1;
                                        qf1 = qmk;
                                    }
                                    {
                                        k += 1;
                                        k
                                    };
                                }
                                qm[(l) as usize] = qmk;
                                qg0 = qg1;
                                qg1 = q0l;
                                qh0 = qh1;
                                qh1 = q1l;
                            }
                            {
                                l += 1;
                                l
                            };
                        }
                    }
                }
            }
        }
    } else {
        {
            if ((x).abs() > 1.1f64) {
                {
                    km = ((40 + m) + n);
                }
            } else {
                {
                    km = (((40 + m) + n) * (((-1.0f64) - (1.8f64 * (x - 1.0f64).ln())) as i32));
                }
            }
            qf2 = 0.0f64;
            qf1 = 1.0f64;
            k = km;
            while (k >= 0) {
                {
                    qf0 = ((((((2.0f64 * ((k) as f64)) + 3.0f64) * x) * qf1)
                        - (((((k) as f64) + 2.0f64) - ((m) as f64)) * qf2))
                        / (((k + m) as f64) + 1.0f64));
                    if (k <= n) {
                        {
                            qm[(k) as usize] = qf0;
                        }
                    }
                    qf2 = qf1;
                    qf1 = qf0;
                }
                {
                    k -= 1;
                    k
                };
            }
            k = 0;
            while (k <= n) {
                {
                    qm[(k) as usize] = ((qm[(k) as usize] * qm0) / qf0);
                }
                {
                    k += 1;
                    k
                };
            }
        }
    }
    if ((x).abs() < 1.0f64) {
        {
            k = 0;
            while (k <= n) {
                {
                    qm[(k) as usize] = (neg_one_pow(m) * qm[(k) as usize]);
                }
                {
                    k += 1;
                    k
                };
            }
        }
    }
    qd[(0) as usize] = ((((1.0f64 - ((m) as f64)) * qm[(1) as usize]) - (x * qm[(0) as usize]))
        / ((x * x) - 1.0f64));
    k = 1;
    while (k <= n) {
        {
            qd[(k) as usize] = ((((((k) as f64) * x) * qm[(k) as usize])
                - (((k + m) as f64) * qm[(k - 1) as usize]))
                / ((x * x) - 1.0f64));
        }
        {
            k += 1;
            k
        };
    }
    return;
}

pub(super) fn rswfo(
    mut m: i32,
    mut n: i32,
    mut c: f64,
    mut x: f64,
    mut cv: f64,
    mut kf: i32,
    r1f: &mut [f64],
    r1d: &mut [f64],
    r2f: &mut [f64],
    r2d: &mut [f64],
) -> i32 {
    let mut df = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut id: i32 = 0;
    let mut kd: i32 = (-1);
    if (sdmn(m, n, c, cv, kd, &mut df[..]) == 1) {
        {
            return 1;
        }
    }
    if (kf != 2) {
        {
            if (rmn1(m, n, c, x, kd, &mut df[..], r1f, r1d) == 1) {
                {
                    return 1;
                }
            }
        }
    }
    if (kf > 1) {
        {
            id = 10;
            if (x > 1e-8f64) {
                {
                    if (rmn2l(
                        m,
                        n,
                        c,
                        x,
                        kd,
                        &mut df[..],
                        r2f,
                        r2d,
                        std::slice::from_mut(&mut id),
                    ) == 1)
                    {
                        {
                            return 1;
                        }
                    }
                }
            }
            if (id > (-1)) {
                {
                    if (rmn2so(m, n, c, x, cv, kd, &mut df[..], r2f, r2d) == 1) {
                        {
                            return 1;
                        }
                    }
                }
            }
        }
    }
    return 0;
}

pub(super) fn rmn2so(
    mut m: i32,
    mut n: i32,
    mut c: f64,
    mut x: f64,
    mut cv: f64,
    mut kd: i32,
    df: &mut [f64],
    r2f: &mut [f64],
    r2d: &mut [f64],
) -> i32 {
    let mut nm: i32 = 0;
    let mut ip: i32 = 0;
    let mut j: i32 = 0;
    let mut ck1: f64 = 0.0;
    let mut ck2: f64 = 0.0;
    let mut r1f: f64 = 0.0;
    let mut r1d: f64 = 0.0;
    let mut qs: f64 = 0.0;
    let mut qt: f64 = 0.0;
    let mut sum: f64 = 0.0;
    let mut sw: f64 = 0.0;
    let mut gf: f64 = 0.0;
    let mut gd: f64 = 0.0;
    let mut h0: f64 = 0.0;
    let mut eps: f64 = 1.0e-14f64;
    let mut pi: f64 = 3.141592653589793f64;
    if ((df[(0) as usize]).abs() <= 1.0e-280f64) {
        {
            r2f[0] = 1.0e+300f64;
            r2d[0] = 1.0e+300f64;
            return 0;
        }
    }
    let mut bk = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut ck = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut dn = vec![0.0; spheroidal_work_len(m, n, c)];
    nm = (25 + (((((n - m) / 2) as f64) + c) as i32));
    ip = ((n - m) % 2);
    sckb(m, n, c, df, &mut ck[..]);
    if (kmn(
        m,
        n,
        c,
        cv,
        kd,
        df,
        &mut dn[..],
        std::slice::from_mut(&mut ck1),
        std::slice::from_mut(&mut ck2),
    ) == 1)
    {
        {
            return 1;
        }
    }
    if (qstar(
        m,
        n,
        c,
        ck1,
        &mut ck[..],
        std::slice::from_mut(&mut qs),
        std::slice::from_mut(&mut qt),
    ) == 1)
    {
        {
            return 1;
        }
    }
    if (cbk(m, n, c, cv, qt, &mut ck[..], &mut bk[..]) == 1) {
        {
            return 1;
        }
    }
    if (x == 0.0f64) {
        {
            sum = 0.0f64;
            sw = 0.0f64;
            j = 0;
            while (j < nm) {
                {
                    sum += ck[(j) as usize];
                    if ((sum - sw).abs() < ((sum).abs() * eps)) {
                        {
                            break;
                        }
                    }
                    sw = sum;
                }
                {
                    j += 1;
                    j
                };
            }
            if (ip == 0) {
                {
                    r1f = (sum / ck1);
                    r2f[0] = ((((-0.5f64) * pi) * qs) * r1f);
                    r2d[0] = ((qs * r1f) + bk[(0) as usize]);
                }
            } else {
                {
                    r1d = (sum / ck1);
                    r2f[0] = bk[(0) as usize];
                    r2d[0] = ((((-0.5f64) * pi) * qs) * r1d);
                }
            }
        }
    } else {
        {
            gmn(
                m,
                n,
                c,
                x,
                &mut bk[..],
                std::slice::from_mut(&mut gf),
                std::slice::from_mut(&mut gd),
            );
            if (rmn1(
                m,
                n,
                c,
                x,
                kd,
                df,
                std::slice::from_mut(&mut r1f),
                std::slice::from_mut(&mut r1d),
            ) == 1)
            {
                {
                    return 1;
                }
            }
            h0 = ((x).atan() - (0.5f64 * pi));
            r2f[0] = (((qs * r1f) * h0) + gf);
            r2d[0] = ((qs * ((r1d * h0) + (r1f / (1.0f64 + (x * x))))) + gd);
        }
    }
    return 0;
}

pub(super) fn qstar(
    mut m: i32,
    mut n: i32,
    mut c: f64,
    mut ck1: f64,
    ck: &mut [f64],
    qs: &mut [f64],
    qt: &mut [f64],
) -> i32 {
    let mut ip: i32 = 0;
    let mut i: i32 = 0;
    let mut l: i32 = 0;
    let mut k: i32 = 0;
    let mut r: f64 = 0.0;
    let mut s: f64 = 0.0;
    let mut sk: f64 = 0.0;
    let mut qs0: f64 = 0.0;
    let mut ap = vec![0.0; spheroidal_work_len(m, n, c)];
    ip = if ((n - m) == (2 * ((n - m) / 2))) {
        0
    } else {
        1
    };
    r = (1.0f64 / (ck[(0) as usize]).powf(((2) as f64)));
    ap[(0) as usize] = r;
    i = 1;
    while (i <= m) {
        {
            s = 0.0f64;
            l = 1;
            while (l <= i) {
                {
                    sk = 0.0f64;
                    k = 0;
                    while (k <= l) {
                        sk += (ck[(k) as usize] * ck[(l - k) as usize]);
                        {
                            k += 1;
                            k
                        };
                    }
                    s += (sk * ap[(i - l) as usize]);
                }
                {
                    l += 1;
                    l
                };
            }
            ap[(i) as usize] = ((-r) * s);
        }
        {
            i += 1;
            i
        };
    }
    // AP(M+1) and L=1..M in the original Fortran use one-based indexing.
    // Retain all M+1 convolution coefficients, including the m=0 case.
    qs0 = ap[m as usize];
    l = 1;
    while (l <= m) {
        {
            r = 1.0f64;
            k = 1;
            while (k <= l) {
                {
                    r = (((r * ((2.0f64 * ((k) as f64)) + ((ip) as f64)))
                        * (((2.0f64 * ((k) as f64)) - 1.0f64) + ((ip) as f64)))
                        / (2.0f64 * ((k) as f64)).powf(((2) as f64)));
                }
                {
                    k += 1;
                    k
                };
            }
            qs0 += (ap[(m - l) as usize] * r);
        }
        {
            l += 1;
            l
        };
    }
    qs[0] = (((neg_one_pow(ip) * ck1) * (ck1 * qs0)) / c);
    qt[0] = (((-2.0f64) / ck1) * qs[0]);
    return 0;
}

pub(super) fn cbk(
    mut m: i32,
    mut n: i32,
    mut c: f64,
    mut cv: f64,
    mut qt: f64,
    ck: &mut [f64],
    bk: &mut [f64],
) -> i32 {
    let mut eps: f64 = 1.0e-14f64;
    let mut i: i32 = 0;
    let mut i1: i32 = 0;
    let mut ip: i32 = 0;
    let mut j: i32 = 0;
    let mut k: i32 = 0;
    let mut n2: i32 = 0;
    let mut nm: i32 = 0;
    let mut r1: f64 = 0.0;
    let mut s1: f64 = 0.0;
    let mut sw: f64 = 0.0;
    let mut t: f64 = 0.0;
    ip = if (((n - m) % 2) == 0) { 0 } else { 1 };
    nm = (25 + (((0.5f64 * ((n - m) as f64)) + c) as i32));
    let mut u = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut v = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut w = vec![0.0; spheroidal_work_len(m, n, c)];
    u[(0) as usize] = 0.0f64;
    n2 = (nm - 2);
    j = 1;
    while (j < (n2 + 1)) {
        {
            u[(j) as usize] = (c * c);
        }
        {
            j += 1;
            j
        };
    }
    j = 1;
    while (j < (n2 + 1)) {
        {
            v[(j - 1) as usize] = ((((((2.0f64 * ((j) as f64)) - 1.0f64) - ((ip) as f64))
                * ((2.0f64 * ((j - m) as f64)) - ((ip) as f64)))
                + (((m) as f64) * (((m) as f64) - 1.0f64)))
                - cv);
        }
        {
            j += 1;
            j
        };
    }
    j = 1;
    while (j < nm) {
        w[(j - 1) as usize] = (((2.0f64 * ((j) as f64)) - ((ip) as f64))
            * (((2.0f64 * ((j) as f64)) + 1.0f64) - ((ip) as f64)));
        {
            j += 1;
            j
        };
    }
    if (ip == 0) {
        {
            sw = 0.0f64;
            k = 0;
            while (k < n2) {
                {
                    s1 = 0.0f64;
                    i1 = ((k - m) + 1);
                    i = i1;
                    while (i < (nm + 1)) {
                        {
                            if (i < 0) {
                                {
                                    {
                                        i += 1;
                                        i
                                    };
                                    continue;
                                }
                            }
                            r1 = 1.0f64;
                            j = 1;
                            while (j <= k) {
                                {
                                    r1 = ((r1 * (((i + m) - j) as f64)) / (1.0f64 * ((j) as f64)));
                                }
                                {
                                    j += 1;
                                    j
                                };
                            }
                            s1 += ((ck[(i) as usize] * ((2.0f64 * ((i) as f64)) + ((m) as f64)))
                                * r1);
                            if ((s1 - sw).abs() < ((s1).abs() * eps)) {
                                {
                                    break;
                                }
                            }
                            sw = s1;
                        }
                        {
                            i += 1;
                            i
                        };
                    }
                    bk[(k) as usize] = (qt * s1);
                }
                {
                    k += 1;
                    k
                };
            }
        }
    } else {
        if (ip == 1) {
            {
                sw = 0.0f64;
                k = 0;
                while (k < n2) {
                    {
                        s1 = 0.0f64;
                        i1 = ((k - m) + 1);
                        let mut i: i32 = i1;
                        while (i < (nm + 1)) {
                            {
                                if (i < 0) {
                                    {
                                        {
                                            i += 1;
                                            i
                                        };
                                        continue;
                                    }
                                }
                                r1 = 1.0f64;
                                j = 1;
                                while (j <= k) {
                                    {
                                        r1 = ((r1 * (((i + m) - j) as f64))
                                            / (1.0f64 * ((j) as f64)));
                                    }
                                    {
                                        j += 1;
                                        j
                                    };
                                }
                                if (i > 0) {
                                    {
                                        s1 += ((ck[(i - 1) as usize]
                                            * (((2.0f64 * ((i) as f64)) + ((m) as f64))
                                                - ((1) as f64)))
                                            * r1);
                                    }
                                }
                                s1 -= ((ck[(i) as usize]
                                    * ((2.0f64 * ((i) as f64)) + ((m) as f64)))
                                    * r1);
                                if ((s1 - sw).abs() < ((s1).abs() * eps)) {
                                    {
                                        break;
                                    }
                                }
                                sw = s1;
                            }
                            {
                                i += 1;
                                i
                            };
                        }
                        bk[(k) as usize] = (qt * s1);
                    }
                    {
                        k += 1;
                        k
                    };
                }
            }
        }
    }
    w[(0) as usize] /= v[(0) as usize];
    bk[(0) as usize] /= v[(0) as usize];
    k = 2;
    while (k <= n2) {
        {
            t = (v[(k - 1) as usize] - (w[(k - 2) as usize] * u[(k - 1) as usize]));
            w[(k - 1) as usize] /= t;
            bk[(k - 1) as usize] =
                ((bk[(k - 1) as usize] - (bk[(k - 2) as usize] * u[(k - 1) as usize])) / t);
        }
        {
            k += 1;
            k
        };
    }
    k = (n2 - 1);
    while (k >= 1) {
        {
            bk[(k - 1) as usize] -= (w[(k - 1) as usize] * bk[(k) as usize]);
        }
        {
            k -= 1;
            k
        };
    }
    return 0;
}

pub(super) fn gmn(
    mut m: i32,
    mut n: i32,
    mut c: f64,
    mut x: f64,
    bk: &mut [f64],
    gf: &mut [f64],
    gd: &mut [f64],
) {
    let mut ip: i32 = 0;
    let mut k: i32 = 0;
    let mut nm: i32 = 0;
    let mut xm: f64 = 0.0;
    let mut gf0: f64 = 0.0;
    let mut gw: f64 = 0.0;
    let mut gd0: f64 = 0.0;
    let mut gd1: f64 = 0.0;
    let mut eps: f64 = 1.0e-14f64;
    ip = if (((n - m) % 2) == 0) { 0 } else { 1 };
    nm = (25 + (((0.5f64 * ((n - m) as f64)) + c) as i32));
    xm = (1.0f64 + (x * x)).powf(((-0.5f64) * ((m) as f64)));
    gf0 = 0.0f64;
    gw = 0.0f64;
    k = 1;
    while (k <= nm) {
        {
            gf0 += (bk[(k - 1) as usize] * (x).powf(((2.0f64 * ((k) as f64)) - 2.0f64)));
            if ((((gf0 - gw) / gf0).abs() < eps) && (k >= 10)) {
                {
                    break;
                }
            }
            gw = gf0;
        }
        {
            k += 1;
            k
        };
    }
    gf[0] = ((xm * gf0) * (x).powf((((1) as f64) - ((ip) as f64))));
    gd1 = (((((-m) as f64) * x) / (1.0f64 + (x * x))) * gf[0]);
    gd0 = 0.0f64;
    k = 1;
    while (k < nm) {
        {
            if (ip == 0) {
                {
                    gd0 += ((((2.0f64 * ((k) as f64)) - 1.0f64) * bk[(k - 1) as usize])
                        * (x).powf(((2.0f64 * ((k) as f64)) - 2.0f64)));
                }
            } else {
                {
                    gd0 += (((2.0f64 * ((k) as f64)) * bk[k as usize])
                        * (x).powf(((2.0f64 * ((k) as f64)) - 1.0f64)));
                }
            }
            if ((((gd0 - gw) / gd0).abs() < eps) && (k >= 10)) {
                {
                    break;
                }
            }
            gw = gd0;
        }
        {
            k += 1;
            k
        };
    }
    gd[0] = (gd1 + (xm * gd0));
}

pub(super) fn segv(
    mut m: i32,
    mut n: i32,
    mut c: f64,
    mut kd: i32,
    cv: &mut [f64],
    eg: &mut [f64],
) -> i32 {
    let mut i: i32 = 0;
    let mut icm: i32 = 0;
    let mut j: i32 = 0;
    let mut k: i32 = 0;
    let mut k1: i32 = 0;
    let mut l: i32 = 0;
    let mut nm: i32 = 0;
    let mut nm1: i32 = 0;
    let mut cs: f64 = 0.0;
    let mut dk0: f64 = 0.0;
    let mut dk1: f64 = 0.0;
    let mut dk2: f64 = 0.0;
    let mut d2k: f64 = 0.0;
    let mut s: f64 = 0.0;
    let mut t: f64 = 0.0;
    let mut t1: f64 = 0.0;
    let mut x1: f64 = 0.0;
    let mut xa: f64 = 0.0;
    let mut xb: f64 = 0.0;
    if (c < 1e-10f64) {
        {
            i = 1;
            while (i <= ((n - m) + 1)) {
                {
                    eg[(i - 1) as usize] = (((i + m) * ((i + m) - 1)) as f64);
                }
                {
                    i += 1;
                    i
                };
            }
            cv[0] = eg[(n - m) as usize];
            return 0;
        }
    }
    let mut a = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut b = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut cv0 = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut d = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut e = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut f = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut g = vec![0.0; spheroidal_work_len(m, n, c)];
    let mut h = vec![0.0; spheroidal_work_len(m, n, c)];
    icm = (((n - m) + 2) / 2);
    nm = (10 + (((0.5f64 * ((n - m) as f64)) + c) as i32));
    cs = ((c * c) * ((kd) as f64));
    k = 0;
    l = 0;
    while (l <= 1) {
        {
            i = 1;
            while (i <= nm) {
                {
                    k = if (l == 0) {
                        (2 * (i - 1))
                    } else {
                        ((2 * i) - 1)
                    };
                    dk0 = ((m + k) as f64);
                    dk1 = (((m + k) + 1) as f64);
                    dk2 = ((2 * (m + k)) as f64);
                    d2k = (((2 * m) + k) as f64);
                    a[(i - 1) as usize] = ((((d2k + 2.0f64) * (d2k + 1.0f64))
                        / ((dk2 + 3.0f64) * (dk2 + 5.0f64)))
                        * cs);
                    d[(i - 1) as usize] = ((dk0 * dk1)
                        + ((((((2.0f64 * dk0) * dk1)
                            - ((2.0f64 * ((m) as f64)) * ((m) as f64)))
                            - 1.0f64)
                            / ((dk2 - 1.0f64) * (dk2 + 3.0f64)))
                            * cs));
                    g[(i - 1) as usize] = (((((k) as f64) * (((k) as f64) - 1.0f64))
                        / ((dk2 - 3.0f64) * (dk2 - 1.0f64)))
                        * cs);
                }
                {
                    i += 1;
                    i
                };
            }
            k = 2;
            while (k <= nm) {
                {
                    e[(k - 1) as usize] = (a[(k - 2) as usize] * g[(k - 1) as usize]).sqrt();
                    f[(k - 1) as usize] = (e[(k - 1) as usize] * e[(k - 1) as usize]);
                }
                {
                    k += 1;
                    k
                };
            }
            f[(0) as usize] = 0.0f64;
            e[(0) as usize] = 0.0f64;
            xa = (d[(nm - 1) as usize] + (e[(nm - 1) as usize]).abs());
            xb = (d[(nm - 1) as usize] - (e[(nm - 1) as usize]).abs());
            nm1 = (nm - 1);
            i = 1;
            while (i <= nm1) {
                {
                    t = ((e[(i - 1) as usize]).abs() + (e[(i) as usize]).abs());
                    t1 = (d[(i - 1) as usize] + t);
                    if (xa < t1) {
                        {
                            xa = t1;
                        }
                    }
                    t1 = (d[(i - 1) as usize] - t);
                    if (t1 < xb) {
                        {
                            xb = t1;
                        }
                    }
                }
                {
                    i += 1;
                    i
                };
            }
            i = 1;
            while (i <= icm) {
                {
                    b[(i - 1) as usize] = xa;
                    h[(i - 1) as usize] = xb;
                }
                {
                    i += 1;
                    i
                };
            }
            k = 1;
            while (k <= icm) {
                {
                    k1 = k;
                    while (k1 <= icm) {
                        {
                            if (b[(k1 - 1) as usize] < b[(k - 1) as usize]) {
                                {
                                    b[(k - 1) as usize] = b[(k1 - 1) as usize];
                                    break;
                                }
                            }
                        }
                        {
                            k1 += 1;
                            k1
                        };
                    }
                    if (k != 1) {
                        {
                            if (h[(k - 1) as usize] < h[(k - 2) as usize]) {
                                {
                                    h[(k - 1) as usize] = h[(k - 2) as usize];
                                }
                            }
                        }
                    }
                    for iteration in 0..128 {
                        if iteration == 127 {
                            return 2;
                        }
                        {
                            x1 = ((b[(k - 1) as usize] + h[(k - 1) as usize]) / 2.0f64);
                            cv0[(k - 1) as usize] = x1;
                            if (((b[(k - 1) as usize] - h[(k - 1) as usize]) / x1).abs() < 1e-14f64)
                            {
                                {
                                    break;
                                }
                            }
                            j = 0;
                            s = 1.0f64;
                            i = 1;
                            while (i <= nm) {
                                {
                                    if (s == 0.0f64) {
                                        {
                                            s += 1e-30f64;
                                        }
                                    }
                                    t = (f[(i - 1) as usize] / s);
                                    s = ((d[(i - 1) as usize] - t) - x1);
                                    if (s < 0.0f64) {
                                        {
                                            j += 1;
                                        }
                                    }
                                }
                                {
                                    i += 1;
                                    i
                                };
                            }
                            if (j < k) {
                                {
                                    h[(k - 1) as usize] = x1;
                                }
                            } else {
                                {
                                    b[(k - 1) as usize] = x1;
                                    if (j >= icm) {
                                        {
                                            b[(icm - 1) as usize] = x1;
                                        }
                                    } else {
                                        {
                                            if (h[(j) as usize] < x1) {
                                                {
                                                    h[(j) as usize] = x1;
                                                }
                                            }
                                            if (x1 < b[(j - 1) as usize]) {
                                                {
                                                    b[(j - 1) as usize] = x1;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    cv0[(k - 1) as usize] = x1;
                    let mut index: i32 = 0;
                    if (l == 0) {
                        index = ((2 * k) - 2);
                    }
                    if (l == 1) {
                        index = ((2 * k) - 1);
                    }
                    if (index >= ((n - m) + 1)) {
                        break;
                    }
                    eg[(index) as usize] = cv0[(k - 1) as usize];
                }
                {
                    k += 1;
                    k
                };
            }
        }
        {
            l += 1;
            l
        };
    }
    cv[0] = eg[(n - m) as usize];
    return 0;
}

pub(super) fn mtu12(
    mut kf: i32,
    mut kc: i32,
    mut m: i32,
    mut q: f64,
    mut x: f64,
    f1r: &mut [f64],
    d1r: &mut [f64],
    f2r: &mut [f64],
    d2r: &mut [f64],
) -> i32 {
    let mut eps: f64 = 1.0e-14f64;
    let mut a: f64 = 0.0;
    let mut qm: f64 = 0.0;
    let mut c1: f64 = 0.0;
    let mut c2: f64 = 0.0;
    let mut u1: f64 = 0.0;
    let mut u2: f64 = 0.0;
    let mut w1: f64 = 0.0;
    let mut w2: f64 = 0.0;
    let mut kd: i32 = 0;
    let mut km: i32 = 0;
    let mut ic: i32 = 0;
    let mut k: i32 = 0;
    let mut nm: i32 = 0;
    if ((kf == 1) && ((m % 2) == 0)) {
        {
            kd = 1;
        }
    }
    if ((kf == 1) && ((m % 2) != 0)) {
        {
            kd = 2;
        }
    }
    if ((kf == 2) && ((m % 2) != 0)) {
        {
            kd = 3;
        }
    }
    if ((kf == 2) && ((m % 2) == 0)) {
        {
            kd = 4;
        }
    }
    // `fcoef` computes the characteristic value together with the
    // coefficients; a separate `cva2` call would repeat the eigenvalue search.
    if (q <= 1.0f64) {
        {
            qm = (((7.5f64 + (56.1f64 * (q).sqrt())) - (134.7f64 * q))
                + ((90.7f64 * (q).sqrt()) * q));
        }
    } else {
        {
            qm = (((17.0f64 + (3.1f64 * (q).sqrt())) - (0.126f64 * q))
                + ((0.0037f64 * (q).sqrt()) * q));
        }
    }
    km = ((qm + (0.5f64 * ((m) as f64))) as i32);
    if (km >= 251) {
        {
            f1r[0] = f64::NAN;
            d1r[0] = f64::NAN;
            f2r[0] = f64::NAN;
            d2r[0] = f64::NAN;
            return 2;
        }
    }
    let mut fg = vec![0.0; (km.max(0) + 8) as usize];
    let mut bj1 = vec![0.0; (km.max(0) + 8) as usize];
    let mut dj1 = vec![0.0; (km.max(0) + 8) as usize];
    let mut bj2 = vec![0.0; (km.max(0) + 8) as usize];
    let mut dj2 = vec![0.0; if kc == 2 { 0 } else { (km.max(0) + 8) as usize }];
    // J is needed at both arguments; only kind two needs Y at u2.
    let y_len = if kc == 1 { 0 } else { (km.max(0) + 8) as usize };
    let mut by2 = vec![0.0; y_len];
    let mut dy2 = vec![0.0; y_len];
    fcoef(kd, m, q, a, &mut fg[..]);
    ic = ((m / 2) + 1);
    if (kd == 4) {
        {
            ic = (m / 2);
        }
    }
    c1 = (-x).exp();
    c2 = (x).exp();
    u1 = ((q).sqrt() * c1);
    u2 = ((q).sqrt() * c2);
    jynb(
        (km + 1),
        u1,
        std::slice::from_mut(&mut nm),
        &mut bj1[..],
        &mut dj1[..],
        &mut [],
        &mut [],
    );
    jynb(
        (km + 1),
        u2,
        std::slice::from_mut(&mut nm),
        &mut bj2[..],
        &mut dj2[..],
        &mut by2[..],
        &mut dy2[..],
    );
    w1 = 0.0f64;
    w2 = 0.0f64;
    let mut sign_kd: f64 = if ((kd % 2) == 0) {
        ((1) as f64)
    } else {
        ((-1) as f64)
    };
    if (kc != 2) {
        {
            f1r[0] = 0.0f64;
            k = 1;
            while (k <= km) {
                {
                    if (kd == 1) {
                        {
                            f1r[0] += (((neg_one_pow(ic + k) * fg[(k - 1) as usize])
                                * bj1[(k - 1) as usize])
                                * bj2[(k - 1) as usize]);
                        }
                    } else {
                        if ((kd == 2) || (kd == 3)) {
                            {
                                f1r[0] += ((neg_one_pow(ic + k) * fg[(k - 1) as usize])
                                    * ((bj1[(k - 1) as usize] * bj2[(k) as usize])
                                        + ((sign_kd * bj1[(k) as usize]) * bj2[(k - 1) as usize])));
                            }
                        } else {
                            {
                                f1r[0] += ((neg_one_pow(ic + k) * fg[(k - 1) as usize])
                                    * ((bj1[(k - 1) as usize] * bj2[(k + 1) as usize])
                                        - (bj1[(k + 1) as usize] * bj2[(k - 1) as usize])));
                            }
                        }
                    }
                    if ((k >= 5) && ((f1r[0] - w1).abs() < ((f1r[0]).abs() * eps))) {
                        {
                            break;
                        }
                    }
                    w1 = f1r[0];
                }
                {
                    k += 1;
                    k
                };
            }
            f1r[0] /= fg[(0) as usize];
            d1r[0] = 0.0f64;
            k = 1;
            while (k <= km) {
                {
                    if (kd == 1) {
                        {
                            d1r[0] += ((neg_one_pow(ic + k) * fg[(k - 1) as usize])
                                * (((c2 * bj1[(k - 1) as usize]) * dj2[(k - 1) as usize])
                                    - ((c1 * dj1[(k - 1) as usize]) * bj2[(k - 1) as usize])));
                        }
                    } else {
                        if ((kd == 2) || (kd == 3)) {
                            {
                                d1r[0] += ((neg_one_pow(ic + k) * fg[(k - 1) as usize])
                                    * ((c2
                                        * ((bj1[(k - 1) as usize] * dj2[(k) as usize])
                                            + ((sign_kd * bj1[(k) as usize])
                                                * dj2[(k - 1) as usize])))
                                        - (c1
                                            * ((dj1[(k - 1) as usize] * bj2[(k) as usize])
                                                + ((sign_kd * dj1[(k) as usize])
                                                    * bj2[(k - 1) as usize])))));
                            }
                        } else {
                            {
                                d1r[0] += ((neg_one_pow(ic + k) * fg[(k - 1) as usize])
                                    * ((c2
                                        * ((bj1[(k - 1) as usize] * dj2[(k + 1) as usize])
                                            - (bj1[(k + 1) as usize] * dj2[(k - 1) as usize])))
                                        - (c1
                                            * ((dj1[(k - 1) as usize] * bj2[(k + 1) as usize])
                                                - (dj1[(k + 1) as usize]
                                                    * bj2[(k - 1) as usize])))));
                            }
                        }
                    }
                    if ((k >= 5) && ((d1r[0] - w2).abs() < ((d1r[0]).abs() * eps))) {
                        {
                            break;
                        }
                    }
                    w2 = d1r[0];
                }
                {
                    k += 1;
                    k
                };
            }
            d1r[0] *= ((q).sqrt() / fg[(0) as usize]);
            if (kc == 1) {
                {
                    return 0;
                }
            }
        }
    }
    f2r[0] = 0.0f64;
    k = 1;
    while (k <= km) {
        {
            if (kd == 1) {
                {
                    f2r[0] += (((neg_one_pow(ic + k) * fg[(k - 1) as usize])
                        * bj1[(k - 1) as usize])
                        * by2[(k - 1) as usize]);
                }
            } else {
                if ((kd == 2) || (kd == 3)) {
                    {
                        f2r[0] += ((neg_one_pow(ic + k) * fg[(k - 1) as usize])
                            * ((bj1[(k - 1) as usize] * by2[(k) as usize])
                                + ((sign_kd * bj1[(k) as usize]) * by2[(k - 1) as usize])));
                    }
                } else {
                    {
                        f2r[0] += ((neg_one_pow(ic + k) * fg[(k - 1) as usize])
                            * ((bj1[(k - 1) as usize] * by2[(k + 1) as usize])
                                - (bj1[(k + 1) as usize] * by2[(k - 1) as usize])));
                    }
                }
            }
            if ((k >= 5) && ((f2r[0] - w1).abs() < ((f2r[0]).abs() * eps))) {
                {
                    break;
                }
            }
            w1 = f2r[0];
        }
        {
            k += 1;
            k
        };
    }
    f2r[0] /= fg[(0) as usize];
    d2r[0] = 0.0f64;
    k = 1;
    while (k <= km) {
        {
            if (kd == 1) {
                {
                    d2r[0] += ((neg_one_pow(ic + k) * fg[(k - 1) as usize])
                        * (((c2 * bj1[(k - 1) as usize]) * dy2[(k - 1) as usize])
                            - ((c1 * dj1[(k - 1) as usize]) * by2[(k - 1) as usize])));
                }
            } else {
                if ((kd == 2) || (kd == 3)) {
                    {
                        d2r[0] += ((neg_one_pow(ic + k) * fg[(k - 1) as usize])
                            * ((c2
                                * ((bj1[(k - 1) as usize] * dy2[(k) as usize])
                                    + ((sign_kd * bj1[(k) as usize]) * dy2[(k - 1) as usize])))
                                - (c1
                                    * ((dj1[(k - 1) as usize] * by2[(k) as usize])
                                        + ((sign_kd * dj1[(k) as usize])
                                            * by2[(k - 1) as usize])))));
                    }
                } else {
                    {
                        d2r[0] += ((neg_one_pow(ic + k) * fg[(k - 1) as usize])
                            * ((c2
                                * ((bj1[(k - 1) as usize] * dy2[(k + 1) as usize])
                                    - (bj1[(k + 1) as usize] * dy2[(k - 1) as usize])))
                                - (c1
                                    * ((dj1[(k - 1) as usize] * by2[(k + 1) as usize])
                                        - (dj1[(k + 1) as usize] * by2[(k - 1) as usize])))));
                    }
                }
            }
            if ((k >= 5) && ((d2r[0] - w2).abs() < ((d2r[0]).abs() * eps))) {
                {
                    break;
                }
            }
            w2 = d2r[0];
        }
        {
            k += 1;
            k
        };
    }
    d2r[0] = ((d2r[0] * (q).sqrt()) / fg[(0) as usize]);
    return 0;
}

// Empty dj omits J derivatives; empty by/dy omit Y and its derivatives.
pub(super) fn jynb(
    mut n: i32,
    mut x: f64,
    nm: &mut [i32],
    bj: &mut [f64],
    dj: &mut [f64],
    by: &mut [f64],
    dy: &mut [f64],
) {
    let mut k: i32 = 0;
    jynbh(n, 0, x, nm, bj, by);
    if (x < 1.0e-100f64) {
        {
            k = 0;
            while (k <= n) {
                {
                    if !dj.is_empty() {
                        dj[(k) as usize] = 0.0f64;
                    }
                    if !by.is_empty() {
                        dy[(k) as usize] = 1.0e+300f64;
                    }
                }
                {
                    k += 1;
                    k
                };
            }
            if !dj.is_empty() {
                dj[(1) as usize] = 0.5f64;
            }
        }
    } else {
        {
            if !dj.is_empty() {
                dj[(0) as usize] = (-bj[(1) as usize]);
                k = 1;
                while (k <= nm[0]) {
                    {
                        dj[(k) as usize] =
                            (bj[(k - 1) as usize] - ((((k) as f64) / x) * bj[(k) as usize]));
                    }
                    {
                        k += 1;
                        k
                    };
                }
            }
            if by.is_empty() {
                return;
            }
            dy[(0) as usize] = (-by[(1) as usize]);
            k = 1;
            while (k <= nm[0]) {
                {
                    dy[(k) as usize] =
                        (by[(k - 1) as usize] - ((((k) as f64) * by[(k) as usize]) / x));
                }
                {
                    k += 1;
                    k
                };
            }
        }
    }
    return;
}

pub(super) fn jynbh(
    mut n: i32,
    mut nmin: i32,
    mut x: f64,
    nm: &mut [i32],
    bj: &mut [f64],
    by: &mut [f64],
) {
    let mut k: i32 = 0;
    let mut m: i32 = 0;
    let mut ky: i32 = 0;
    let mut pi: f64 = 3.141592653589793f64;
    let mut r2p: f64 = 0.63661977236758f64;
    let mut bs: f64 = 0.0;
    let mut s0: f64 = 0.0;
    let mut su: f64 = 0.0;
    let mut sv: f64 = 0.0;
    let mut f2: f64 = 0.0;
    let mut f1: f64 = 0.0;
    let mut f: f64 = 0.0;
    let mut bj0: f64 = 0.0;
    let mut bj1: f64 = 0.0;
    let mut ec: f64 = 0.0;
    let mut by0: f64 = 0.0;
    let mut by1: f64 = 0.0;
    let mut bjk: f64 = 0.0;
    let mut byk: f64 = 0.0;
    let mut p0: f64 = 0.0;
    let mut q0: f64 = 0.0;
    let mut cu: f64 = 0.0;
    let mut t1: f64 = 0.0;
    let mut p1: f64 = 0.0;
    let mut q1: f64 = 0.0;
    let mut t2: f64 = 0.0;
    let mut a = [
        (-0.0703125f64),
        0.112152099609375f64,
        (-0.5725014209747314f64),
        0.6074042001273483e+01f64,
    ];
    let mut b = [
        0.0732421875f64,
        (-0.2271080017089844f64),
        0.1727727502584457e+01f64,
        (-0.2438052969955606e+02f64),
    ];
    let mut a1 = [
        0.1171875f64,
        (-0.144195556640625f64),
        0.6765925884246826f64,
        (-0.6883914268109947e+01f64),
    ];
    let mut b1 = [
        (-0.1025390625f64),
        0.2775764465332031f64,
        (-0.1993531733751297e+01f64),
        0.2724882731126854e+02f64,
    ];
    nm[0] = n;
    if (x < 1.0e-100f64) {
        {
            k = nmin;
            while (k <= n) {
                {
                    bj[(k - nmin) as usize] = 0.0f64;
                    if !by.is_empty() {
                        by[(k - nmin) as usize] = (-1.0e+300f64);
                    }
                }
                {
                    k += 1;
                    k
                };
            }
            if (nmin == 0) {
                {
                    bj[(0) as usize] = 1.0f64;
                }
            }
            return;
        }
    }
    if ((x <= 300.0f64) || (n > ((0.9f64 * x) as i32))) {
        {
            if (n == 0) {
                {
                    nm[0] = 1;
                }
            }
            // msta1 locates the order where J becomes smaller than 1e-200.
            // Beyond order 1.1*x its envelope increases monotonically.
            // Keep an extra order for the integer rounding in msta1. Below
            // that boundary, msta2 supplies the same recurrence start directly.
            let order = nm[0] as f64 + 1.0;
            let envelope = 0.5 * (6.28 * order).log10() - order * (1.36 * x / order).log10();
            m = if order <= 1.1 * x || envelope < 200.0 {
                nm[0]
            } else {
                msta1(x, 200)
            };
            if (m < nm[0]) {
                {
                    nm[0] = m;
                }
            } else {
                {
                    m = msta2(x, nm[0], 15);
                }
            }
            bs = 0.0f64;
            su = 0.0f64;
            sv = 0.0f64;
            f2 = 0.0f64;
            f1 = 1.0e-100f64;
            f = 0.0f64;
            k = m;
            while (k >= 0) {
                {
                    f = ((((2.0f64 * (((k) as f64) + 1.0f64)) / x) * f1) - f2);
                    if ((k <= nm[0]) && (k >= nmin)) {
                        {
                            bj[(k - nmin) as usize] = f;
                        }
                    }
                    if ((k == (2 * (k / 2))) && (k != 0)) {
                        {
                            bs += (2.0f64 * f);
                            if !by.is_empty() {
                                su += ((neg_one_pow(k / 2) * f) / ((k) as f64));
                            }
                        }
                    } else {
                        if (k > 1 && !by.is_empty()) {
                            {
                                sv += (((neg_one_pow(k / 2) * ((k) as f64))
                                    / (((k * k) as f64) - 1.0f64))
                                    * f);
                            }
                        }
                    }
                    f2 = f1;
                    f1 = f;
                }
                {
                    k -= 1;
                    k
                };
            }
            s0 = (bs + f);
            k = nmin;
            while (k <= nm[0]) {
                {
                    bj[(k - nmin) as usize] /= s0;
                }
                {
                    k += 1;
                    k
                };
            }
            if by.is_empty() {
                return;
            }
            bj0 = (f1 / s0);
            bj1 = (f2 / s0);
            ec = ((x / 2.0f64).ln() + 0.5772156649015329f64);
            by0 = (r2p * ((ec * bj0) - ((4.0f64 * su) / s0)));
            by1 = (r2p * ((((ec - 1.0f64) * bj1) - (bj0 / x)) - ((4.0f64 * sv) / s0)));
            if (0 >= nmin) {
                {
                    if !by.is_empty() {
                        by[(0 - nmin) as usize] = by0;
                    }
                }
            }
            if (1 >= nmin) {
                {
                    if !by.is_empty() {
                        by[(1 - nmin) as usize] = by1;
                    }
                }
            }
            ky = 2;
        }
    } else {
        {
            t1 = (x - (0.25f64 * pi));
            p0 = 1.0f64;
            q0 = ((-0.125f64) / x);
            k = 1;
            while (k <= 4) {
                {
                    p0 += (a[(k - 1) as usize] * (x).powf((((-2) * k) as f64)));
                    q0 += (b[(k - 1) as usize] * (x).powf(((((-2) * k) - 1) as f64)));
                }
                {
                    k += 1;
                    k
                };
            }
            cu = (r2p / x).sqrt();
            bj0 = (cu * ((p0 * (t1).cos()) - (q0 * (t1).sin())));
            by0 = (cu * ((p0 * (t1).sin()) + (q0 * (t1).cos())));
            if (0 >= nmin) {
                {
                    bj[(0 - nmin) as usize] = bj0;
                    if !by.is_empty() {
                        by[(0 - nmin) as usize] = by0;
                    }
                }
            }
            t2 = (x - (0.75f64 * pi));
            p1 = 1.0f64;
            q1 = (0.375f64 / x);
            k = 1;
            while (k <= 4) {
                {
                    p1 += (a1[(k - 1) as usize] * (x).powf((((-2) * k) as f64)));
                    q1 += (b1[(k - 1) as usize] * (x).powf(((((-2) * k) - 1) as f64)));
                }
                {
                    k += 1;
                    k
                };
            }
            bj1 = (cu * ((p1 * (t2).cos()) - (q1 * (t2).sin())));
            by1 = (cu * ((p1 * (t2).sin()) + (q1 * (t2).cos())));
            if (1 >= nmin) {
                {
                    bj[(1 - nmin) as usize] = bj1;
                    if !by.is_empty() {
                        by[(1 - nmin) as usize] = by1;
                    }
                }
            }
            k = 2;
            while (k <= nm[0]) {
                {
                    bjk = ((((2.0f64 * (((k) as f64) - 1.0f64)) / x) * bj1) - bj0);
                    if (k >= nmin) {
                        {
                            bj[(k - nmin) as usize] = bjk;
                        }
                    }
                    bj0 = bj1;
                    bj1 = bjk;
                }
                {
                    k += 1;
                    k
                };
            }
            ky = 2;
        }
    }
    if by.is_empty() {
        return;
    }
    k = ky;
    while (k <= nm[0]) {
        {
            byk = ((((2.0f64 * (((k) as f64) - 1.0f64)) * by1) / x) - by0);
            if (k >= nmin) {
                by[(k - nmin) as usize] = byk;
            }
            by0 = by1;
            by1 = byk;
        }
        {
            k += 1;
            k
        };
    }
}

pub(super) fn chgm(mut x: f64, mut a: f64, mut b: f64) -> f64 {
    let mut i: i32 = 0;
    let mut j: i32 = 0;
    let mut la: i32 = 0;
    let mut n: i32 = 0;
    let mut nl: i32 = 0;
    let mut a0: f64 = a;
    let mut a1: f64 = a;
    let mut x0: f64 = x;
    let mut y0: f64 = 0.0;
    let mut y1: f64 = 0.0;
    let mut hg1: f64 = 0.0;
    let mut hg2: f64 = 0.0;
    let mut r1: f64 = 0.0;
    let mut r2: f64 = 0.0;
    let mut rg: f64 = 0.0;
    let mut xg: f64 = 0.0;
    let mut sum1: f64 = 0.0;
    let mut sum2: f64 = 0.0;
    let mut pi: f64 = 3.141592653589793f64;
    let mut hg: f64 = 0.0f64;
    if (x < 0.0f64) {
        {
            a = (b - a);
            a0 = a;
            x = (x).abs();
        }
    }
    nl = 0;
    la = 0;
    if (a >= 2.0f64) {
        {
            nl = 1;
            la = ((a) as i32);
            a -= ((la + 1) as f64);
        }
    }
    y0 = 0.0f64;
    y1 = 0.0f64;
    n = 0;
    while (n < (nl + 1)) {
        {
            if (a0 >= 2.0f64) {
                {
                    a += 1.0f64;
                }
            }
            if ((x <= (30.0f64 + (b).abs())) || (a < 0.0f64)) {
                {
                    hg = 1.0f64;
                    rg = 1.0f64;
                    j = 1;
                    while (j < 501) {
                        {
                            rg = (((rg * ((a + ((j) as f64)) - 1.0f64))
                                / (((j) as f64) * ((b + ((j) as f64)) - 1.0f64)))
                                * x);
                            hg += rg;
                            if ((hg != 0.0f64) && ((rg / hg).abs() < 1e-15f64)) {
                                {
                                    if (x0 < 0.0f64) {
                                        {
                                            hg *= (x0).exp();
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                        {
                            j += 1;
                            j
                        };
                    }
                }
            } else {
                {
                    xg = (b - a);
                    sum1 = 1.0f64;
                    sum2 = 1.0f64;
                    r1 = 1.0f64;
                    r2 = 1.0f64;
                    i = 1;
                    while (i < 17) {
                        {
                            r1 = ((((-r1) * ((a + ((i) as f64)) - 1.0f64))
                                * ((a - b) + ((i) as f64)))
                                / (x * ((i) as f64)));
                            r2 = ((((-r2) * (((b - a) + ((i) as f64)) - 1.0f64))
                                * (a - ((i) as f64)))
                                / (x * ((i) as f64)));
                            sum1 += r1;
                            sum2 += r2;
                        }
                        {
                            i += 1;
                            i
                        };
                    }
                    if (x0 >= 0.0f64) {
                        {
                            hg1 = (((gamma_ratio(b, xg, 0.0f64) * (x).powf((-a)))
                                * (pi * a).cos())
                                * sum1);
                            hg2 = ((gamma_ratio(b, a, x) * (x).powf((a - b))) * sum2);
                        }
                    } else {
                        {
                            hg1 = (((gamma_ratio(b, xg, x0) * (x).powf((-a))) * (pi * a).cos())
                                * sum1);
                            hg2 = ((gamma_ratio(b, a, 0.0f64) * (x).powf((a - b))) * sum2);
                        }
                    }
                    hg = (hg1 + hg2);
                }
            }
            if (n == 0) {
                {
                    y0 = hg;
                }
            }
            if (n == 1) {
                {
                    y1 = hg;
                }
            }
        }
        {
            n += 1;
            n
        };
    }
    if (a0 >= 2.0f64) {
        {
            i = 1;
            while (i < la) {
                {
                    hg = ((((((2.0f64 * a) - b) + x) * y1) + ((b - a) * y0)) / a);
                    y0 = y1;
                    y1 = hg;
                    a += 1.0f64;
                }
                {
                    i += 1;
                    i
                };
            }
        }
    }
    a = a1;
    x = x0;
    return hg;
}

pub(super) fn chgu(mut x: f64, mut a: f64, mut b: f64, md: &mut [i32], isfer: &mut [i32]) -> f64 {
    let mut il1: i32 = 0;
    let mut il2: i32 = 0;
    let mut il3: i32 = 0;
    let mut bl1: i32 = 0;
    let mut bl2: i32 = 0;
    let mut bl3: i32 = 0;
    let mut bn: i32 = 0;
    let mut id1: i32 = 0;
    let mut id: i32 = 0;
    let mut aa: f64 = 0.0;
    let mut hu: f64 = 0.0f64;
    let mut hu1: f64 = 0.0;
    let mut b00: f64 = 0.0;
    aa = ((a - b) + 1.0f64);
    isfer[0] = 0;
    il1 = ((a == (((a) as i32) as f64)) && (a <= 0.0f64)) as i32;
    il2 = ((aa == (((aa) as i32) as f64)) && (aa <= 0.0f64)) as i32;
    il3 = (((a * ((a - b) + 1.0f64)).abs() / x) <= 2.0f64) as i32;
    bl1 = ((x <= 5.0f64) || ((x <= 10.0f64) && (a <= 2.0f64))) as i32;
    bl2 = (((x > 5.0f64) && (x <= 12.5f64)) && ((a >= 1.0f64) && (b >= (a + 4.0f64)))) as i32;
    bl3 = (((x > 12.5f64) && (a >= 5.0f64)) && (b >= (a + 5.0f64))) as i32;
    bn = ((b == (((b) as i32) as f64)) && (b != 0.0f64)) as i32;
    id = (-100);
    hu1 = 0.0f64;
    if (b != (((b) as i32) as f64)) {
        {
            hu = chgus(x, a, b, std::slice::from_mut(&mut id1));
            md[0] = 1;
            if (id1 >= 9) {
                {
                    return hu;
                }
            }
            hu1 = hu;
        }
    }
    if (((il1 != 0) || (il2 != 0)) || (il3 != 0)) {
        {
            hu = chgul(x, a, b, std::slice::from_mut(&mut id));
            md[0] = 2;
            if (id >= 9) {
                {
                    return hu;
                }
            }
            if (id1 > id) {
                {
                    md[0] = 1;
                    id = id1;
                    hu = hu1;
                }
            }
        }
    }
    if (a >= 1.0f64) {
        {
            if ((bn != 0) && (((bl1 != 0) || (bl2 != 0)) || (bl3 != 0))) {
                {
                    hu = chgubi(x, a, b, std::slice::from_mut(&mut id));
                    md[0] = 3;
                }
            } else {
                {
                    hu = chguit(x, a, b, std::slice::from_mut(&mut id));
                    md[0] = 4;
                }
            }
        }
    } else {
        {
            if (b <= a) {
                {
                    b00 = b;
                    a -= (b - 1.0f64);
                    b = (2.0f64 - b);
                    hu = chguit(x, a, b, std::slice::from_mut(&mut id));
                    hu *= (x).powf((1.0f64 - b00));
                    md[0] = 4;
                }
            } else {
                if ((bn != 0) && (il1 == 0)) {
                    {
                        hu = chgubi(x, a, b, std::slice::from_mut(&mut id));
                        md[0] = 3;
                    }
                }
            }
        }
    }
    if (id < 6) {
        {
            isfer[0] = 6;
        }
    }
    return hu;
}

pub(super) fn chgus(mut x: f64, mut a: f64, mut b: f64, id: &mut [i32]) -> f64 {
    let mut j: i32 = 0;
    let mut d1: f64 = 0.0;
    let mut d2: f64 = 0.0;
    let mut ga: f64 = 0.0;
    let mut gb: f64 = 0.0;
    let mut gab: f64 = 0.0;
    let mut gb2: f64 = 0.0;
    let mut h0: f64 = 0.0;
    let mut hmax: f64 = 0.0;
    let mut hmin: f64 = 0.0;
    let mut hu: f64 = 0.0;
    let mut hu0: f64 = 0.0;
    let mut hua: f64 = 0.0;
    let mut r1: f64 = 0.0;
    let mut r2: f64 = 0.0;
    let mut pi: f64 = 3.141592653589793f64;
    id[0] = 100;
    ga = gamma2(a);
    gb = gamma2(b);
    gab = gamma2(((1.0f64 + a) - b));
    gb2 = gamma2((2.0f64 - b));
    hu0 = (pi / (pi * b).sin());
    r1 = (hu0 / (gab * gb));
    r2 = ((hu0 * (x).powf((1.0f64 - b))) / (ga * gb2));
    hu = (r1 - r2);
    hmax = 0.0f64;
    hmin = 1e300f64;
    h0 = 0.0f64;
    j = 1;
    while (j < 151) {
        {
            r1 = (((r1 * ((a + ((j) as f64)) - 1.0f64))
                / (((j) as f64) * ((b + ((j) as f64)) - 1.0f64)))
                * x);
            r2 = (((r2 * ((a - b) + ((j) as f64)))
                / (((j) as f64) * ((1.0f64 - b) + ((j) as f64))))
                * x);
            hu += (r1 - r2);
            hua = (hu).abs();
            if (hua > hmax) {
                {
                    hmax = hua;
                }
            }
            if (hua < hmin) {
                {
                    hmin = hua;
                }
            }
            if ((hu - h0).abs() < ((hu).abs() * 1e-15f64)) {
                {
                    break;
                }
            }
            h0 = hu;
        }
        {
            j += 1;
            j
        };
    }
    d1 = (hmax).log10();
    d2 = 0.0f64;
    if (hmin != 0.0f64) {
        {
            d2 = (hmin).log10();
        }
    }
    id[0] = ((((15) as f64)
        - if ((d1 - d2) < ((0) as f64)) {
            (d2 - d1)
        } else {
            (d1 - d2)
        }) as i32);
    return hu;
}

pub(super) fn gamma2(mut x: f64) -> f64 {
    let mut ga: f64 = 0.0;
    let mut gr: f64 = 0.0;
    let mut r: f64 = 0.0;
    let mut z: f64 = 0.0;
    let mut k: i32 = 0;
    let mut m: i32 = 0;
    let mut pi: f64 = 3.141592653589793f64;
    let mut g = [
        1.0000000000000000e+00f64,
        0.5772156649015329e+00f64,
        (-0.6558780715202538e+00f64),
        (-0.4200263503409520e-01f64),
        0.1665386113822915e+00f64,
        (-0.4219773455554430e-01f64),
        (-0.9621971527877000e-02f64),
        0.7218943246663000e-02f64,
        (-0.1165167591859100e-02f64),
        (-0.2152416741149000e-03f64),
        0.1280502823882000e-03f64,
        (-0.2013485478070000e-04f64),
        (-0.1250493482100000e-05f64),
        0.1133027232000000e-05f64,
        (-0.2056338417000000e-06f64),
        0.6116095000000000e-08f64,
        0.5002007500000000e-08f64,
        (-0.1181274600000000e-08f64),
        0.1043427000000000e-09f64,
        0.7782300000000000e-11f64,
        (-0.3696800000000000e-11f64),
        0.5100000000000000e-12f64,
        (-0.2060000000000000e-13f64),
        (-0.5400000000000000e-14f64),
        0.1400000000000000e-14f64,
        0.1000000000000000e-15f64,
    ];
    if (x == (((x) as i32) as f64)) {
        {
            if (x > 0.0f64) {
                {
                    ga = 1.0f64;
                    m = ((x - ((1) as f64)) as i32);
                    k = 2;
                    while (k < (m + 1)) {
                        {
                            ga *= ((k) as f64);
                        }
                        {
                            k += 1;
                            k
                        };
                    }
                }
            } else {
                {
                    ga = 1e300f64;
                }
            }
        }
    } else {
        {
            r = 1.0f64;
            if ((x).abs() > 1.0f64) {
                {
                    z = (x).abs();
                    m = ((z) as i32);
                    k = 1;
                    while (k < (m + 1)) {
                        {
                            r *= (z - ((k) as f64));
                        }
                        {
                            k += 1;
                            k
                        };
                    }
                    z -= ((m) as f64);
                }
            } else {
                {
                    z = x;
                }
            }
            gr = g[(25) as usize];
            k = 25;
            while (k > 0) {
                {
                    gr *= z;
                    gr += g[(k - 1) as usize];
                }
                {
                    k -= 1;
                    k
                };
            }
            ga = (1.0f64 / (gr * z));
            if ((x).abs() > 1.0f64) {
                {
                    ga *= r;
                    if (x < 0.0f64) {
                        {
                            ga = ((-pi) / ((x * ga) * (pi * x).sin()));
                        }
                    }
                }
            }
        }
    }
    return ga;
}

pub(super) fn chgul(mut x: f64, mut a: f64, mut b: f64, id: &mut [i32]) -> f64 {
    let mut il1: i32 = 0;
    let mut il2: i32 = 0;
    let mut k: i32 = 0;
    let mut nm: i32 = 0;
    let mut aa: f64 = 0.0;
    let mut hu: f64 = 0.0;
    let mut r: f64 = 0.0;
    let mut r0: f64 = 0.0f64;
    let mut ra: f64 = 0.0f64;
    id[0] = (-100);
    aa = ((a - b) + 1.0f64);
    il1 = ((a == (((a) as i32) as f64)) && (a <= 0.0f64)) as i32;
    il2 = ((aa == (((aa) as i32) as f64)) && (aa <= 0.0f64)) as i32;
    nm = 0;
    if (il1 != 0) {
        {
            nm = (((a).abs()) as i32);
        }
    }
    if (il2 != 0) {
        {
            nm = (((aa).abs()) as i32);
        }
    }
    if ((il1 != 0) || (il2 != 0)) {
        {
            hu = 1.0f64;
            r = 1.0f64;
            k = 1;
            while (k <= nm) {
                {
                    r = ((((-r) * ((a + ((k) as f64)) - 1.0f64)) * ((a - b) + ((k) as f64)))
                        / (((k) as f64) * x));
                    hu += r;
                }
                {
                    k += 1;
                    k
                };
            }
            hu *= (x).powf((-a));
            id[0] = 10;
        }
    } else {
        {
            hu = 1.0f64;
            r = 1.0f64;
            k = 1;
            while (k <= 25) {
                {
                    r = ((((-r) * ((a + ((k) as f64)) - 1.0f64)) * ((a - b) + ((k) as f64)))
                        / (((k) as f64) * x));
                    ra = (r).abs();
                    if (((k > 5) && (ra >= r0)) || (ra < 1e-15f64)) {
                        {
                            break;
                        }
                    }
                    r0 = ra;
                    hu += r;
                }
                {
                    k += 1;
                    k
                };
            }
            id[0] = ((((ra).log10()).abs()) as i32);
            hu *= (x).powf((-a));
        }
    }
    return hu;
}

pub(super) fn chgubi(mut x: f64, mut a: f64, mut b: f64, id: &mut [i32]) -> f64 {
    let mut id1: i32 = 0;
    let mut id2: i32 = 0;
    let mut j: i32 = 0;
    let mut k: i32 = 0;
    let mut m: i32 = 0;
    let mut n: i32 = 0;
    let mut a0: f64 = 0.0;
    let mut a1: f64 = 0.0;
    let mut a2: f64 = 0.0;
    let mut da1: f64 = 0.0;
    let mut da2: f64 = 0.0;
    let mut db1: f64 = 0.0;
    let mut db2: f64 = 0.0;
    let mut ga: f64 = 0.0;
    let mut ga1: f64 = 0.0;
    let mut h0: f64 = 0.0;
    let mut hm1: f64 = 0.0;
    let mut hm2: f64 = 0.0;
    let mut hm3: f64 = 0.0;
    let mut hmax: f64 = 0.0;
    let mut hmin: f64 = 0.0;
    let mut hu: f64 = 0.0;
    let mut hu1: f64 = 0.0;
    let mut hu2: f64 = 0.0;
    let mut hw: f64 = 0.0;
    let mut ps: f64 = 0.0;
    let mut r: f64 = 0.0;
    let mut rn: f64 = 0.0;
    let mut rn1: f64 = 0.0;
    let mut s0: f64 = 0.0;
    let mut s1: f64 = 0.0;
    let mut s2: f64 = 0.0;
    let mut sa: f64 = 0.0;
    let mut sb: f64 = 0.0;
    let mut ua: f64 = 0.0;
    let mut ub: f64 = 0.0;
    let mut el: f64 = 0.5772156649015329f64;
    id[0] = (-100);
    n = (((b - ((1) as f64)).abs()) as i32);
    rn1 = 1.0f64;
    rn = 1.0f64;
    j = 1;
    while (j <= n) {
        {
            rn *= ((j) as f64);
            if (j == (n - 1)) {
                {
                    rn1 = rn;
                }
            }
        }
        {
            j += 1;
            j
        };
    }
    ps = psi_spec(a);
    ga = gamma2(a);
    if (b > 0.0f64) {
        {
            a0 = a;
            a1 = (a - ((n) as f64));
            a2 = a1;
            ga1 = gamma2(a1);
            ua = (neg_one_pow(n - 1) / (rn * ga1));
            ub = ((rn1 / ga) * (x).powf(((-n) as f64)));
        }
    } else {
        {
            a0 = (a + ((n) as f64));
            a1 = a0;
            a2 = a;
            ga1 = gamma2(a1);
            ua = ((neg_one_pow(n - 1) / (rn * ga)) * (x).powf(((n) as f64)));
            ub = (rn1 / ga1);
        }
    }
    hm1 = 1.0f64;
    r = 1.0f64;
    hmax = 0.0f64;
    hmin = 1e300f64;
    h0 = 0.0f64;
    k = 1;
    while (k <= 150) {
        {
            r = (((r * ((a0 + ((k) as f64)) - ((1) as f64))) * x) / (((n + k) * k) as f64));
            hm1 += r;
            hu1 = (hm1).abs();
            if (hu1 > hmax) {
                {
                    hmax = hu1;
                }
            }
            if (hu1 < hmin) {
                {
                    hmin = hu1;
                }
            }
            if ((hm1 - h0).abs() < ((hm1).abs() * 1.0e-15f64)) {
                {
                    break;
                }
            }
            h0 = hm1;
        }
        {
            k += 1;
            k
        };
    }
    da1 = (hmax).log10();
    da2 = ((0) as f64);
    if (hmin != ((0) as f64)) {
        {
            da2 = (hmin).log10();
        }
    }
    id[0] = (15 - (((da1 - da2).abs()) as i32));
    hm1 *= (x).ln();
    s0 = ((0) as f64);
    m = 1;
    while (m <= n) {
        {
            if (b >= ((0) as f64)) {
                {
                    s0 -= (1.0f64 / ((m) as f64));
                }
            }
            if (b < ((0) as f64)) {
                {
                    s0 += ((1.0f64 - a) / (((m) as f64) * ((a + ((m) as f64)) - ((1) as f64))));
                }
            }
        }
        {
            m += 1;
            m
        };
    }
    hm2 = ((ps + (((2) as f64) * el)) + s0);
    r = ((1) as f64);
    hmax = ((0) as f64);
    hmin = 1.0e+300f64;
    k = 1;
    while (k <= 150) {
        {
            s1 = ((0) as f64);
            s2 = ((0) as f64);
            if (b > ((0) as f64)) {
                {
                    m = 1;
                    while (m <= k) {
                        {
                            s1 -= (((((m) as f64) + (((2) as f64) * a)) - ((2) as f64))
                                / (((m) as f64) * ((((m) as f64) + a) - ((1) as f64))));
                        }
                        {
                            m += 1;
                            m
                        };
                    }
                    m = 1;
                    while (m <= n) {
                        {
                            s2 += (1.0f64 / ((k + m) as f64));
                        }
                        {
                            m += 1;
                            m
                        };
                    }
                }
            } else {
                {
                    m = 1;
                    while (m <= (k + n)) {
                        {
                            s1 += ((1.0f64 - a)
                                / (((m) as f64) * ((((m) as f64) + a) - ((1) as f64))));
                        }
                        {
                            m += 1;
                            m
                        };
                    }
                    m = 1;
                    while (m <= k) {
                        {
                            s2 += (1.0f64 / ((m) as f64));
                        }
                        {
                            m += 1;
                            m
                        };
                    }
                }
            }
            hw = ((((((2) as f64) * el) + ps) + s1) - s2);
            r = (((r * ((a0 + ((k) as f64)) - ((1) as f64))) * x) / (((n + k) * k) as f64));
            hm2 += (r * hw);
            hu2 = (hm2).abs();
            if (hu2 > hmax) {
                {
                    hmax = hu2;
                }
            }
            if (hu2 < hmin) {
                {
                    hmin = hu2;
                }
            }
            if (((hm2 - h0) / hm2).abs() < 1.0e-15f64) {
                {
                    break;
                }
            }
            h0 = hm2;
        }
        {
            k += 1;
            k
        };
    }
    db1 = (hmax).log10();
    db2 = 0.0f64;
    if (hmin != 0.0f64) {
        {
            db2 = (hmin).log10();
        }
    }
    id1 = (15 - (((db1 - db2).abs()) as i32));
    if (id1 < id[0]) {
        {
            id[0] = id1;
        }
    }
    hm3 = 1.0f64;
    if (n == 0) {
        {
            hm3 = 0.0f64;
        }
    }
    r = 1.0f64;
    k = 1;
    while (k < n) {
        {
            r = (((r * ((a2 + ((k) as f64)) - 1.0f64)) / (((k - n) * k) as f64)) * x);
            hm3 += r;
        }
        {
            k += 1;
            k
        };
    }
    sa = (ua * (hm1 + hm2));
    sb = (ub * hm3);
    hu = (sa + sb);
    id2 = 0;
    if (sa != 0.0f64) {
        {
            id1 = ((((sa).abs()).log10()) as i32);
        }
    }
    if (hu != 0.0f64) {
        {
            id2 = ((((hu).abs()).log10()) as i32);
        }
    }
    if ((sa * sb) < 0.0f64) {
        {
            id[0] -= ((((id1 - id2) as f64).abs()) as i32);
        }
    }
    return hu;
}

pub(super) fn psi_spec(mut x: f64) -> f64 {
    let mut k: i32 = 0;
    let mut n: i32 = 0;
    let mut ps: f64 = 0.0;
    let mut s: f64 = 0.0f64;
    let mut x2: f64 = 0.0;
    let mut xa: f64 = (x).abs();
    let mut pi: f64 = 3.141592653589793f64;
    let mut el: f64 = 0.5772156649015329f64;
    let mut a = [
        (-0.8333333333333e-01f64),
        0.83333333333333333e-02f64,
        (-0.39682539682539683e-02f64),
        0.41666666666666667e-02f64,
        (-0.75757575757575758e-02f64),
        0.21092796092796093e-01f64,
        (-0.83333333333333333e-01f64),
        0.4432598039215686f64,
    ];
    if ((x == (((x) as i32) as f64)) && (x <= 0.0f64)) {
        {
            return 1e300f64;
        }
    } else {
        if (xa == (((xa) as i32) as f64)) {
            {
                n = ((xa) as i32);
                k = 1;
                while (k < n) {
                    {
                        s += (1.0f64 / ((k) as f64));
                    }
                    {
                        k += 1;
                        k
                    };
                }
                ps = ((-el) + s);
            }
        } else {
            if ((xa + 0.5f64) == (((xa + 0.5f64) as i32) as f64)) {
                {
                    n = ((xa - 0.5f64) as i32);
                    k = 1;
                    while (k < (n + 1)) {
                        {
                            s += (1.0f64 / ((2.0f64 * ((k) as f64)) - 1.0f64));
                        }
                        {
                            k += 1;
                            k
                        };
                    }
                    ps = (((-el) + (2.0f64 * s)) - 1.386294361119891f64);
                }
            } else {
                {
                    if (xa < 10.0f64) {
                        {
                            n = ((10.0f64 - (((xa) as i32) as f64)) as i32);
                            k = 0;
                            while (k < n) {
                                {
                                    s += (1.0f64 / (xa + ((k) as f64)));
                                }
                                {
                                    k += 1;
                                    k
                                };
                            }
                            xa += ((n) as f64);
                        }
                    }
                    x2 = (1.0f64 / (xa * xa));
                    ps = ((xa).ln() - (0.5f64 / xa));
                    ps += (x2
                        * ((((((((((((((a[(7) as usize] * x2) + a[(6) as usize]) * x2)
                            + a[(5) as usize])
                            * x2)
                            + a[(4) as usize])
                            * x2)
                            + a[(3) as usize])
                            * x2)
                            + a[(2) as usize])
                            * x2)
                            + a[(1) as usize])
                            * x2)
                            + a[(0) as usize]));
                    ps -= s;
                }
            }
        }
    }
    if (x < 0.0f64) {
        {
            ps -= (((pi * (pi * x).cos()) / (pi * x).sin()) + (1.0f64 / x));
        }
    }
    return ps;
}

pub(super) fn chguit(mut x: f64, mut a: f64, mut b: f64, id: &mut [i32]) -> f64 {
    let mut k: i32 = 0;
    let mut j: i32 = 0;
    let mut m: i32 = 0;
    let mut a1: f64 = 0.0;
    let mut b1: f64 = 0.0;
    let mut c: f64 = 0.0;
    let mut d: f64 = 0.0;
    let mut f1: f64 = 0.0;
    let mut f2: f64 = 0.0;
    let mut g: f64 = 0.0;
    let mut ga: f64 = 0.0;
    let mut hu: f64 = 0.0;
    let mut hu0: f64 = 0.0;
    let mut hu1: f64 = 0.0;
    let mut hu2: f64 = 0.0;
    let mut s: f64 = 0.0;
    let mut t1: f64 = 0.0;
    let mut t2: f64 = 0.0;
    let mut t3: f64 = 0.0;
    let mut t4: f64 = 0.0;
    let mut t = [
        0.259597723012478e-01f64,
        0.778093339495366e-01f64,
        0.129449135396945e+00f64,
        0.180739964873425e+00f64,
        0.231543551376029e+00f64,
        0.281722937423262e+00f64,
        0.331142848268448e+00f64,
        0.379670056576798e+00f64,
        0.427173741583078e+00f64,
        0.473525841761707e+00f64,
        0.518601400058570e+00f64,
        0.562278900753945e+00f64,
        0.604440597048510e+00f64,
        0.644972828489477e+00f64,
        0.683766327381356e+00f64,
        0.720716513355730e+00f64,
        0.755723775306586e+00f64,
        0.788693739932264e+00f64,
        0.819537526162146e+00f64,
        0.848171984785930e+00f64,
        0.874519922646898e+00f64,
        0.898510310810046e+00f64,
        0.920078476177628e+00f64,
        0.939166276116423e+00f64,
        0.955722255839996e+00f64,
        0.969701788765053e+00f64,
        0.981067201752598e+00f64,
        0.989787895222222e+00f64,
        0.995840525118838e+00f64,
        0.999210123227436e+00f64,
    ];
    let mut w = [
        0.519078776312206e-01f64,
        0.517679431749102e-01f64,
        0.514884515009810e-01f64,
        0.510701560698557e-01f64,
        0.505141845325094e-01f64,
        0.498220356905502e-01f64,
        0.489955754557568e-01f64,
        0.480370318199712e-01f64,
        0.469489888489122e-01f64,
        0.457343797161145e-01f64,
        0.443964787957872e-01f64,
        0.429388928359356e-01f64,
        0.413655512355848e-01f64,
        0.396806954523808e-01f64,
        0.378888675692434e-01f64,
        0.359948980510845e-01f64,
        0.340038927249464e-01f64,
        0.319212190192963e-01f64,
        0.297524915007890e-01f64,
        0.275035567499248e-01f64,
        0.251804776215213e-01f64,
        0.227895169439978e-01f64,
        0.203371207294572e-01f64,
        0.178299010142074e-01f64,
        0.152746185967848e-01f64,
        0.126781664768159e-01f64,
        0.100475571822880e-01f64,
        0.738993116334531e-02f64,
        0.471272992695363e-02f64,
        0.202681196887362e-02f64,
    ];
    id[0] = 9;
    a1 = (a - 1.0f64);
    b1 = ((b - a) - 1.0f64);
    c = (12.0f64 / x);
    hu0 = 0.0f64;
    m = 10;
    while (m <= 100) {
        {
            hu1 = 0.0f64;
            g = ((0.5f64 * c) / ((m) as f64));
            d = g;
            j = 1;
            while (j < (m + 1)) {
                {
                    s = 0.0f64;
                    k = 1;
                    while (k <= 30) {
                        {
                            t1 = (d + (g * t[(k - 1) as usize]));
                            t2 = (d - (g * t[(k - 1) as usize]));
                            f1 = ((((-x) * t1).exp() * (t1).powf(a1)) * (1.0f64 + t1).powf(b1));
                            f2 = ((((-x) * t2).exp() * (t2).powf(a1)) * (1.0f64 + t2).powf(b1));
                            s += (w[(k - 1) as usize] * (f1 + f2));
                        }
                        {
                            k += 1;
                            k
                        };
                    }
                    hu1 += (s * g);
                    d += (2.0f64 * g);
                }
                {
                    j += 1;
                    j
                };
            }
            if ((1.0f64 - (hu0 / hu1)).abs() < 1.0e-9f64) {
                {
                    break;
                }
            }
            hu0 = hu1;
        }
        m += 5;
    }
    ga = gamma2(a);
    hu1 /= ga;
    m = 2;
    while (m <= 10) {
        {
            hu2 = 0.0f64;
            g = (0.5f64 / ((m) as f64));
            d = g;
            j = 1;
            while (j <= m) {
                {
                    s = 0.0f64;
                    k = 1;
                    while (k <= 30) {
                        {
                            t1 = (d + (g * t[(k - 1) as usize]));
                            t2 = (d - (g * t[(k - 1) as usize]));
                            t3 = (c / (1.0f64 - t1));
                            t4 = (c / (1.0f64 - t2));
                            f1 = (((((t3 * t3) / c) * ((-x) * t3).exp()) * (t3).powf(a1))
                                * (1.0f64 + t3).powf(b1));
                            f2 = (((((t4 * t4) / c) * ((-x) * t4).exp()) * (t4).powf(a1))
                                * (1.0f64 + t4).powf(b1));
                            s += (w[(k - 1) as usize] * (f1 + f2));
                        }
                        {
                            k += 1;
                            k
                        };
                    }
                    hu2 += (s * g);
                    d += (2.0f64 * g);
                }
                {
                    j += 1;
                    j
                };
            }
            if ((1.0f64 - (hu0 / hu2)).abs() < 1.0e-9f64) {
                {
                    break;
                }
            }
            hu0 = hu2;
        }
        m += 2;
    }
    ga = gamma2(a);
    hu2 /= ga;
    hu = (hu1 + hu2);
    return hu;
}

pub(super) fn lpmv(mut x: f64, mut m: i32, mut v: f64) -> f64 {
    let mut mx: i32 = 0;
    let mut neg_m: i32 = 0;
    let mut nv: i32 = 0;
    let mut j: i32 = 0;
    let mut vx: f64 = 0.0;
    let mut pmv: f64 = 0.0;
    let mut v0: f64 = 0.0;
    let mut p0: f64 = 0.0;
    let mut p1: f64 = 0.0;
    let mut g1: f64 = 0.0;
    let mut g2: f64 = 0.0;
    if ((x == (-1.0f64)) && (v != (((v) as i32) as f64))) {
        {
            if (m == 0) {
                {
                    let mut vs: f64 = ((v * M_PI).sin() / (v * M_PI));
                    pmv = if (vs > ((0) as f64)) {
                        (-1e300f64)
                    } else {
                        1e300f64
                    };
                }
            } else {
                {
                    pmv = 1e300f64;
                }
            }
            return pmv;
        }
    }
    vx = v;
    mx = m;
    if (v < ((0) as f64)) {
        {
            vx = ((-vx) - 1.0f64);
        }
    }
    neg_m = 0;
    if (m < 0) {
        {
            if ((((vx + ((m) as f64)) + ((1) as f64)) > ((0) as f64))
                || (vx != (((vx) as i32) as f64)))
            {
                {
                    neg_m = 1;
                    mx = (-m);
                }
            } else {
                {
                    return f64::NAN;
                }
            }
        }
    }
    nv = ((vx) as i32);
    v0 = (vx - ((nv) as f64));
    if ((nv > 2) && (nv > mx)) {
        {
            p0 = lpmv0((v0 + ((mx) as f64)), mx, x);
            p1 = lpmv0(((v0 + ((mx) as f64)) + ((1) as f64)), mx, x);
            pmv = p1;
            j = (mx + 2);
            while (j <= nv) {
                {
                    pmv = ((((((((2) as f64) * (v0 + ((j) as f64))) - ((1) as f64)) * x) * p1)
                        - ((((v0 + ((j) as f64)) - ((1) as f64)) + ((mx) as f64)) * p0))
                        / ((v0 + ((j) as f64)) - ((mx) as f64)));
                    p0 = p1;
                    p1 = pmv;
                }
                {
                    j += 1;
                    j
                };
            }
        }
    } else {
        {
            pmv = lpmv0(vx, mx, x);
        }
    }
    if ((neg_m != 0) && ((pmv).abs() < 1.0e300f64)) {
        {
            g1 = gamma2(((vx - ((mx) as f64)) + ((1) as f64)));
            g2 = gamma2(((vx + ((mx) as f64)) + ((1) as f64)));
            pmv = (((pmv * g1) / g2) * neg_one_pow(mx));
        }
    }
    return pmv;
}

pub(super) fn lpmv0(mut v: f64, mut m: i32, mut x: f64) -> f64 {
    let mut j: i32 = 0;
    let mut k: i32 = 0;
    let mut nv: i32 = 0;
    let mut c0: f64 = 0.0;
    let mut v0: f64 = 0.0;
    let mut vs: f64 = 0.0;
    let mut pa: f64 = 0.0;
    let mut pss: f64 = 0.0;
    let mut pv0: f64 = 0.0;
    let mut pmv: f64 = 0.0;
    let mut r: f64 = 0.0;
    let mut r0: f64 = 0.0;
    let mut r1: f64 = 0.0;
    let mut r2: f64 = 0.0;
    let mut s: f64 = 0.0;
    let mut s0: f64 = 0.0;
    let mut s1: f64 = 0.0;
    let mut s2: f64 = 0.0;
    let mut qr: f64 = 0.0;
    let mut rg: f64 = 0.0;
    let mut xq: f64 = 0.0;
    let mut pi: f64 = 3.141592653589793f64;
    let mut el: f64 = 0.5772156649015329f64;
    let mut eps: f64 = 1e-14f64;
    nv = ((v) as i32);
    v0 = (v - ((nv) as f64));
    if ((x == (-1.0f64)) && (v != ((nv) as f64))) {
        {
            if (m == 0) {
                return (-1.0e+300f64);
            }
            if (m != 0) {
                return 1.0e+300f64;
            }
        }
    }
    c0 = 1.0f64;
    if (m != 0) {
        {
            rg = (v * (v + ((m) as f64)));
            j = 1;
            while (j <= (m - 1)) {
                {
                    rg *= ((v * v) - ((j * j) as f64));
                }
                {
                    j += 1;
                    j
                };
            }
            xq = (1.0f64 - (x * x)).sqrt();
            r0 = 1.0f64;
            j = 1;
            while (j <= m) {
                {
                    r0 = (((0.5f64 * r0) * xq) / ((j) as f64));
                }
                {
                    j += 1;
                    j
                };
            }
            c0 = (r0 * rg);
        }
    }
    if (v0 == 0.0f64) {
        {
            pmv = 1.0f64;
            r = 1.0f64;
            k = 1;
            while (k <= (nv - m)) {
                {
                    r = (((((0.5f64 * r) * (((((-nv) + m) + k) as f64) - 1.0f64))
                        * (((nv + m) + k) as f64))
                        / ((k * (k + m)) as f64))
                        * (1.0f64 + x));
                    pmv += r;
                }
                {
                    k += 1;
                    k
                };
            }
            return ((neg_one_pow(nv) * c0) * pmv);
        }
    } else {
        {
            if (x >= (-0.35f64)) {
                {
                    pmv = 1.0f64;
                    r = 1.0f64;
                    k = 1;
                    while (k <= 100) {
                        {
                            r = (((((0.5f64 * r)
                                * ((((-v) + ((m) as f64)) + ((k) as f64)) - 1.0f64))
                                * ((v + ((m) as f64)) + ((k) as f64)))
                                / ((k * (m + k)) as f64))
                                * (1.0f64 - x));
                            pmv += r;
                            if ((k > 12) && ((r / pmv).abs() < eps)) {
                                {
                                    break;
                                }
                            }
                        }
                        {
                            k += 1;
                            k
                        };
                    }
                    return ((neg_one_pow(m) * c0) * pmv);
                }
            } else {
                {
                    vs = ((v * pi).sin() / pi);
                    pv0 = 0.0f64;
                    if (m != 0) {
                        {
                            qr = ((1.0f64 - x) / (1.0f64 + x)).sqrt();
                            r2 = 1.0f64;
                            j = 1;
                            while (j <= m) {
                                {
                                    r2 *= (qr * ((j) as f64));
                                }
                                {
                                    j += 1;
                                    j
                                };
                            }
                            s0 = 1.0f64;
                            r1 = 1.0f64;
                            k = 1;
                            while (k <= (m - 1)) {
                                {
                                    r1 = (((((0.5f64 * r1)
                                        * (((-v) + ((k) as f64)) - ((1) as f64)))
                                        * (v + ((k) as f64)))
                                        / ((k * (k - m)) as f64))
                                        * (1.0f64 + x));
                                    s0 += r1;
                                }
                                {
                                    k += 1;
                                    k
                                };
                            }
                            pv0 = ((((-vs) * r2) / ((m) as f64)) * s0);
                        }
                    }
                    pa = (((2.0f64 * (psi_spec(v) + el)) + (pi / (pi * v).tan())) + (1.0f64 / v));
                    s1 = 0.0f64;
                    j = 1;
                    while (j <= m) {
                        {
                            s1 += ((((j * j) as f64) + (v * v))
                                / (((j) as f64) * (((j * j) as f64) - (v * v))));
                        }
                        {
                            j += 1;
                            j
                        };
                    }
                    pmv = (((pa + s1) - (1.0f64 / (((m) as f64) - v)))
                        + (0.5f64 * (1.0f64 + x)).ln());
                    r = 1.0f64;
                    k = 1;
                    while (k <= 100) {
                        {
                            r = (((((0.5f64 * r)
                                * ((((-v) + ((m) as f64)) + ((k) as f64)) - 1.0f64))
                                * ((v + ((m) as f64)) + ((k) as f64)))
                                / ((k * (k + m)) as f64))
                                * (1.0f64 + x));
                            s = 0.0f64;
                            j = 1;
                            while (j <= m) {
                                s += (((((k + j) * (k + j)) as f64) + (v * v))
                                    / (((k + j) as f64)
                                        * ((((k + j) * (k + j)) as f64) - (v * v))));
                                {
                                    j += 1;
                                    j
                                };
                            }
                            s2 = 0.0f64;
                            j = 1;
                            while (j <= k) {
                                s2 =
                                    (s2 + (1.0f64 / (((j) as f64) * (((j * j) as f64) - (v * v)))));
                                {
                                    j += 1;
                                    j
                                };
                            }
                            pss = ((((pa + s) + (((2.0f64 * v) * v) * s2))
                                - (1.0f64 / (((m + k) as f64) - v)))
                                + (0.5f64 * (1.0f64 + x)).ln());
                            r2 = (pss * r);
                            pmv += r2;
                            if ((r2 / pmv).abs() < eps) {
                                {
                                    break;
                                }
                            }
                        }
                        {
                            k += 1;
                            k
                        };
                    }
                    return (pv0 + ((pmv * vs) * c0));
                }
            }
        }
    }
}

pub(super) fn pbdv(
    mut x: f64,
    mut v: f64,
    dv: &mut [f64],
    dp: &mut [f64],
    pdf: &mut [f64],
    pdd: &mut [f64],
) {
    let mut ja: i32 = 0;
    let mut k: i32 = 0;
    let mut l: i32 = 0;
    let mut m: i32 = 0;
    let mut nk: i32 = 0;
    let mut nv: i32 = 0;
    let mut na: i32 = 0;
    let mut xa: f64 = 0.0;
    let mut vh: f64 = 0.0;
    let mut ep: f64 = 0.0;
    let mut f: f64 = 0.0;
    let mut f0: f64 = 0.0;
    let mut f1: f64 = 0.0;
    let mut v0: f64 = 0.0;
    let mut v1: f64 = 0.0;
    let mut v2: f64 = 0.0;
    let mut pd: f64 = 0.0;
    let mut pd0: f64 = 0.0;
    let mut pd1: f64 = 0.0;
    let mut s0: f64 = 0.0;
    xa = (x).abs();
    vh = v;
    v += (1.0f64).copysign(v);
    nv = ((v) as i32);
    v0 = (v - ((nv) as f64));
    na = ((((nv) as f64).abs()) as i32);
    ep = (((-0.25f64) * x) * x).exp();
    ja = 0;
    if (na >= 1) {
        {
            ja = 1;
        }
    }
    if (v >= 0.0f64) {
        {
            if (v0 == 0.0f64) {
                {
                    pd0 = ep;
                    pd1 = (x * ep);
                }
            } else {
                {
                    l = 0;
                    while (l <= ja) {
                        {
                            v1 = (v0 + ((l) as f64));
                            if (xa <= 5.8f64) {
                                {
                                    pd1 = dvsa(x, v1);
                                }
                            } else {
                                {
                                    pd1 = dvla(x, v1);
                                }
                            }
                            if (l == 0) {
                                {
                                    pd0 = pd1;
                                }
                            }
                        }
                        {
                            l += 1;
                            l
                        };
                    }
                }
            }
            dv[(0) as usize] = pd0;
            dv[(1) as usize] = pd1;
            k = 2;
            while (k <= na) {
                {
                    pdf[0] = ((x * pd1) - (((((k) as f64) + v0) - 1.0f64) * pd0));
                    dv[(k) as usize] = pdf[0];
                    pd0 = pd1;
                    pd1 = pdf[0];
                }
                {
                    k += 1;
                    k
                };
            }
        }
    } else {
        {
            if (x <= 0.0f64) {
                {
                    if (xa <= 5.8f64) {
                        {
                            pd0 = dvsa(x, v0);
                            v1 = (v0 - 1.0f64);
                            pd1 = dvsa(x, v1);
                        }
                    } else {
                        {
                            pd0 = dvla(x, v0);
                            v1 = (v0 - 1.0f64);
                            pd1 = dvla(x, v1);
                        }
                    }
                    dv[(0) as usize] = pd0;
                    dv[(1) as usize] = pd1;
                    k = 2;
                    while (k <= na) {
                        {
                            pd = ((((-x) * pd1) + pd0) / ((((k) as f64) - 1.0f64) - v0));
                            dv[(k) as usize] = pd;
                            pd0 = pd1;
                            pd1 = pd;
                        }
                        {
                            k += 1;
                            k
                        };
                    }
                }
            } else {
                if (x <= 2.0f64) {
                    {
                        v2 = (((nv) as f64) + v0);
                        if (nv == 0) {
                            {
                                v2 -= 1.0f64;
                            }
                        }
                        nk = ((-v2) as i32);
                        f1 = dvsa(x, v2);
                        v1 = (v2 + 1.0f64);
                        f0 = dvsa(x, v1);
                        dv[(nk) as usize] = f1;
                        dv[(nk - 1) as usize] = f0;
                        k = (nk - 2);
                        while (k >= 0) {
                            {
                                f = ((x * f0) + (((((k) as f64) - v0) + 1.0f64) * f1));
                                dv[(k) as usize] = f;
                                f1 = f0;
                                f0 = f;
                            }
                            {
                                k -= 1;
                                k
                            };
                        }
                    }
                } else {
                    {
                        if (xa <= 5.8f64) {
                            {
                                pd0 = dvsa(x, v0);
                            }
                        } else {
                            {
                                pd0 = dvla(x, v0);
                            }
                        }
                        dv[(0) as usize] = pd0;
                        m = (100 + na);
                        f1 = 0.0f64;
                        f0 = 1e-30f64;
                        f = 0.0f64;
                        k = m;
                        while (k >= 0) {
                            {
                                f = ((x * f0) + (((((k) as f64) - v0) + 1.0f64) * f1));
                                if (k <= na) {
                                    {
                                        dv[(k) as usize] = f;
                                    }
                                }
                                f1 = f0;
                                f0 = f;
                            }
                            {
                                k -= 1;
                                k
                            };
                        }
                        s0 = (pd0 / f);
                        k = 0;
                        while (k <= na) {
                            {
                                dv[(k) as usize] *= s0;
                            }
                            {
                                k += 1;
                                k
                            };
                        }
                    }
                }
            }
        }
    }
    k = 0;
    while (k < na) {
        {
            v1 = ((v0).abs() + ((k) as f64));
            if (v >= 0.0f64) {
                {
                    dp[(k) as usize] = (((0.5f64 * x) * dv[(k) as usize]) - dv[(k + 1) as usize]);
                }
            } else {
                {
                    dp[(k) as usize] =
                        ((((-0.5f64) * x) * dv[(k) as usize]) - (v1 * dv[(k + 1) as usize]));
                }
            }
        }
        {
            k += 1;
            k
        };
    }
    pdf[0] = dv[(na - 1) as usize];
    pdd[0] = dp[(na - 1) as usize];
    v = vh;
    return;
}

pub(super) fn dvsa(mut x: f64, mut va: f64) -> f64 {
    let mut m: i32 = 0;
    let mut ep: f64 = 0.0;
    let mut a0: f64 = 0.0;
    let mut va0: f64 = 0.0;
    let mut pd: f64 = 0.0;
    let mut ga0: f64 = 0.0;
    let mut g1: f64 = 0.0;
    let mut g0: f64 = 0.0;
    let mut r: f64 = 0.0;
    let mut vm: f64 = 0.0;
    let mut gm: f64 = 0.0;
    let mut r1: f64 = 0.0;
    let mut vt: f64 = 0.0;
    let mut pi: f64 = 3.141592653589793f64;
    let mut eps: f64 = 1.0e-15f64;
    let mut sq2: f64 = ((2) as f64).sqrt();
    ep = (((-0.25f64) * x) * x).exp();
    va0 = (0.5f64 * (1.0f64 - va));
    if (va == 0.0f64) {
        {
            pd = ep;
        }
    } else {
        {
            if (x == 0.0f64) {
                {
                    if ((va0 <= 0.0f64) && (va0 == (((va0) as i32) as f64))) {
                        {
                            pd = 0.0f64;
                        }
                    } else {
                        {
                            ga0 = gamma2(va0);
                            pd = ((pi).sqrt() / ((2.0f64).powf(((-0.5f64) * va)) * ga0));
                        }
                    }
                }
            } else {
                {
                    g1 = gamma2((-va));
                    a0 = (((2.0f64).powf((((-0.5f64) * va) - 1.0f64)) * ep) / g1);
                    vt = ((-0.5f64) * va);
                    g0 = gamma2(vt);
                    pd = g0;
                    r = 1.0f64;
                    m = 1;
                    while (m <= 250) {
                        {
                            vm = (0.5f64 * (((m) as f64) - va));
                            gm = gamma2(vm);
                            r = ((((-r) * sq2) * x) / ((m) as f64));
                            r1 = (gm * r);
                            pd += r1;
                            if ((r1).abs() < ((pd).abs() * eps)) {
                                {
                                    break;
                                }
                            }
                        }
                        {
                            m += 1;
                            m
                        };
                    }
                    pd *= a0;
                }
            }
        }
    }
    return pd;
}

pub(super) fn dvla(mut x: f64, mut va: f64) -> f64 {
    let mut k: i32 = 0;
    let mut ep: f64 = 0.0;
    let mut a0: f64 = 0.0;
    let mut gl: f64 = 0.0;
    let mut pd: f64 = 0.0;
    let mut r: f64 = 0.0;
    let mut vl: f64 = 0.0;
    let mut x1: f64 = 0.0;
    let mut pi: f64 = 3.141592653589793f64;
    let mut eps: f64 = 1.0e-12f64;
    ep = (((-0.25f64) * x) * x).exp();
    a0 = (((x).abs()).powf(va) * ep);
    r = 1.0f64;
    pd = 1.0f64;
    k = 1;
    while (k <= 16) {
        {
            r = (((((-0.5f64) * r) * (((2.0f64 * ((k) as f64)) - va) - 1.0f64))
                * (((2.0f64 * ((k) as f64)) - va) - 2.0f64))
                / ((((k) as f64) * x) * x));
            pd += r;
            if ((r / pd).abs() < eps) {
                {
                    break;
                }
            }
        }
        {
            k += 1;
            k
        };
    }
    pd *= a0;
    if (x < 0.0f64) {
        {
            x1 = (-x);
            vl = vvla(x1, va);
            gl = gamma2((-va));
            pd = (((pi * vl) / gl) + ((pi * va).cos() * pd));
        }
    }
    return pd;
}

pub(super) fn vvla(mut x: f64, mut va: f64) -> f64 {
    let mut k: i32 = 0;
    let mut pv: f64 = 0.0;
    let mut qe: f64 = 0.0;
    let mut a0: f64 = 0.0;
    let mut r: f64 = 0.0;
    let mut x1: f64 = 0.0;
    let mut gl: f64 = 0.0;
    let mut dsl: f64 = 0.0;
    let mut pdl: f64 = 0.0;
    let mut pi: f64 = 3.141592653589793f64;
    let mut eps: f64 = 1e-12f64;
    qe = ((0.25f64 * x) * x).exp();
    a0 = ((((x).abs()).powf(((-va) - ((1) as f64))) * (2.0f64 / pi).sqrt()) * qe);
    r = 1.0f64;
    pv = 1.0f64;
    k = 1;
    while (k <= 18) {
        {
            r = ((((0.5f64 * r) * (((2.0f64 * ((k) as f64)) + va) - 1.0f64))
                * ((2.0f64 * ((k) as f64)) + va))
                / ((((k) as f64) * x) * x));
            pv += r;
            if ((r / pv).abs() < eps) {
                {
                    break;
                }
            }
        }
        {
            k += 1;
            k
        };
    }
    pv *= a0;
    if (x < 0.0f64) {
        {
            x1 = (-x);
            pdl = dvla(x1, va);
            gl = gamma2((-va));
            dsl = ((pi * va).sin() * (pi * va).sin());
            pv = ((((dsl * gl) / pi) * pdl) - ((pi * va).cos() * pv));
        }
    }
    return pv;
}

pub(super) fn pbvv(
    mut x: f64,
    mut v: f64,
    vv: &mut [f64],
    vp: &mut [f64],
    pvf: &mut [f64],
    pvd: &mut [f64],
) {
    let mut ja: i32 = 0;
    let mut k: i32 = 0;
    let mut kv: i32 = 0;
    let mut l: i32 = 0;
    let mut m: i32 = 0;
    let mut na: i32 = 0;
    let mut nv: i32 = 0;
    let mut f: f64 = 0.0;
    let mut f0: f64 = 0.0;
    let mut f1: f64 = 0.0;
    let mut pv0: f64 = 0.0;
    let mut q2p: f64 = 0.0;
    let mut qe: f64 = 0.0;
    let mut s0: f64 = 0.0;
    let mut v0: f64 = 0.0;
    let mut v1: f64 = 0.0;
    let mut v2: f64 = 0.0;
    let mut vh: f64 = 0.0;
    let mut xa: f64 = 0.0;
    let mut pi: f64 = 3.141592653589793f64;
    xa = (x).abs();
    vh = v;
    v += (1.0f64).copysign(v);
    nv = ((v) as i32);
    v0 = (v - ((nv) as f64));
    na = ((((nv) as f64).abs()) as i32);
    qe = ((0.25f64 * x) * x).exp();
    q2p = (2.0f64 / pi).sqrt();
    ja = 0;
    if (na >= 1) {
        {
            ja = 1;
        }
    }
    f = 0.0f64;
    if (v <= 0.0f64) {
        {
            if (v0 == 0.0f64) {
                {
                    if (xa <= 7.5f64) {
                        {
                            pv0 = vvsa(x, v0);
                        }
                    } else {
                        {
                            pv0 = vvla(x, v0);
                        }
                    }
                    f0 = (q2p * qe);
                    f1 = (x * f0);
                    vv[(0) as usize] = pv0;
                    vv[(1) as usize] = f0;
                    vv[(2) as usize] = f1;
                }
            } else {
                {
                    l = 0;
                    while (l <= ja) {
                        {
                            v1 = (v0 - ((l) as f64));
                            if (xa <= 7.5f64) {
                                {
                                    f1 = vvsa(x, v1);
                                }
                            } else {
                                {
                                    f1 = vvla(x, v1);
                                }
                            }
                            if (l == 0) {
                                {
                                    f0 = f1;
                                }
                            }
                        }
                        {
                            l += 1;
                            l
                        };
                    }
                    vv[(0) as usize] = f0;
                    vv[(1) as usize] = f1;
                }
            }
            kv = 2;
            if (v0 == 0.0f64) {
                {
                    kv = 3;
                }
            }
            k = kv;
            while (k <= na) {
                {
                    f = ((x * f1) + (((((k) as f64) - v0) - 2.0f64) * f0));
                    vv[(k) as usize] = f;
                    f0 = f1;
                    f1 = f;
                }
                {
                    k += 1;
                    k
                };
            }
        }
    } else {
        {
            if ((x >= 0.0f64) && (x <= 7.5f64)) {
                {
                    v2 = v;
                    if (v2 < 1.0f64) {
                        {
                            v2 = (v2 + 1.0f64);
                        }
                    }
                    f1 = vvsa(x, v2);
                    v1 = (v2 - 1.0f64);
                    kv = ((v2) as i32);
                    f0 = vvsa(x, v1);
                    vv[(kv) as usize] = f1;
                    vv[(kv - 1) as usize] = f0;
                    k = (kv - 2);
                    while (k >= 0) {
                        {
                            f = ((x * f0) - (((((k) as f64) + v0) + 2.0f64) * f1));
                            if (k <= na) {
                                {
                                    vv[(k) as usize] = f;
                                }
                            }
                            f1 = f0;
                            f0 = f;
                        }
                        {
                            k -= 1;
                            k
                        };
                    }
                }
            } else {
                if (x > 7.5f64) {
                    {
                        pv0 = vvla(x, v0);
                        m = ((((100) as f64) + ((na) as f64).abs()) as i32);
                        vv[(1) as usize] = pv0;
                        f1 = 0.0f64;
                        f0 = 1.0e-40f64;
                        k = m;
                        while (k >= 0) {
                            {
                                f = ((x * f0) - (((((k) as f64) + v0) + 2.0f64) * f1));
                                if (k <= na) {
                                    {
                                        vv[(k) as usize] = f;
                                    }
                                }
                                f1 = f0;
                                f0 = f;
                            }
                            {
                                k -= 1;
                                k
                            };
                        }
                        s0 = (pv0 / f);
                        k = 0;
                        while (k <= na) {
                            {
                                vv[(k) as usize] *= s0;
                            }
                            {
                                k += 1;
                                k
                            };
                        }
                    }
                } else {
                    {
                        if (xa <= 7.5f64) {
                            {
                                f0 = vvsa(x, v0);
                                v1 = (v0 + 1.0f64);
                                f1 = vvsa(x, v1);
                            }
                        } else {
                            {
                                f0 = vvla(x, v0);
                                v1 = (v0 + 1.0f64);
                                f1 = vvla(x, v1);
                            }
                        }
                        vv[(0) as usize] = f0;
                        vv[(1) as usize] = f1;
                        k = 2;
                        while (k <= na) {
                            {
                                f = (((x * f1) - f0) / (((k) as f64) + v0));
                                vv[(k) as usize] = f;
                                f0 = f1;
                                f1 = f;
                            }
                            {
                                k += 1;
                                k
                            };
                        }
                    }
                }
            }
        }
    }
    k = 0;
    while (k < na) {
        {
            v1 = (v0 + ((k) as f64));
            if (v >= 0.0f64) {
                {
                    vp[(k) as usize] = (((0.5f64 * x) * vv[(k) as usize])
                        - ((v1 + 1.0f64) * vv[(k + 1) as usize]));
                }
            } else {
                {
                    vp[(k) as usize] =
                        ((((-0.5f64) * x) * vv[(k) as usize]) + vv[(k + 1) as usize]);
                }
            }
        }
        {
            k += 1;
            k
        };
    }
    pvf[0] = vv[(na - 1) as usize];
    pvd[0] = vp[(na - 1) as usize];
    v = vh;
    return;
}

pub(super) fn vvsa(mut x: f64, mut va: f64) -> f64 {
    let mut a0: f64 = 0.0;
    let mut fac: f64 = 0.0;
    let mut g1: f64 = 0.0;
    let mut ga0: f64 = 0.0;
    let mut gm: f64 = 0.0;
    let mut gw: f64 = 0.0;
    let mut r: f64 = 0.0;
    let mut r1: f64 = 0.0;
    let mut sq2: f64 = 0.0;
    let mut sv: f64 = 0.0;
    let mut sv0: f64 = 0.0;
    let mut v1: f64 = 0.0;
    let mut vb0: f64 = 0.0;
    let mut vm: f64 = 0.0;
    let mut pv: f64 = 0.0;
    let mut eps: f64 = 1.0e-15f64;
    let mut pi: f64 = 3.141592653589793f64;
    let mut ep: f64 = (((-0.25f64) * x) * x).exp();
    let mut va0: f64 = (1.0f64 + (0.5f64 * va));
    if (x == 0.0f64) {
        {
            if (((va0 <= 0.0f64) && (va0 == (((va0) as i32) as f64))) || (va == 0.0f64)) {
                {
                    pv = 0.0f64;
                }
            } else {
                {
                    vb0 = ((-0.5f64) * va);
                    sv0 = (va0 * pi).sin();
                    ga0 = gamma2(va0);
                    pv = (((2.0f64).powf(vb0) * sv0) / ga0);
                }
            }
        }
    } else {
        {
            sq2 = (2.0f64).sqrt();
            a0 = (((2.0f64).powf(((-0.5f64) * va)) * ep) / (2.0f64 * pi));
            sv = ((-(va + 0.5f64)) * pi).sin();
            v1 = ((-0.5f64) * va);
            g1 = gamma2(v1);
            pv = ((sv + 1.0f64) * g1);
            r = 1.0f64;
            fac = 1.0f64;
            let mut m: i32 = 1;
            while (m <= 250) {
                {
                    vm = (0.5f64 * (((m) as f64) - va));
                    gm = gamma2(vm);
                    r = (((r * sq2) * x) / ((m) as f64));
                    fac = (-fac);
                    gw = ((fac * sv) + 1.0f64);
                    r1 = ((gw * r) * gm);
                    pv += r1;
                    if (((r1 / pv).abs() < eps) && (gw != 0.0f64)) {
                        {
                            break;
                        }
                    }
                }
                {
                    m += 1;
                    m
                };
            }
            pv *= a0;
        }
    }
    return pv;
}

pub(super) fn itairy_asymptotic(
    mut x: f64,
    apt: &mut [f64],
    bpt: &mut [f64],
    ant: &mut [f64],
    bnt: &mut [f64],
) {
    let mut k: i32 = 0;
    let mut r: f64 = 0.0;
    let mut su1: f64 = 0.0;
    let mut su2: f64 = 0.0;
    let mut su3: f64 = 0.0;
    let mut su4: f64 = 0.0;
    let mut su5: f64 = 0.0;
    let mut su6: f64 = 0.0;
    let mut xp6: f64 = 0.0;
    let mut xe: f64 = 0.0;
    let mut xr1: f64 = 0.0;
    let mut xr2: f64 = 0.0;
    let mut pi: f64 = 3.141592653589793f64;
    let mut q0: f64 = (1.0f64 / 3.0f64);
    let mut q1: f64 = (2.0f64 / 3.0f64);
    let mut q2: f64 = 1.4142135623730951f64;
    let mut a = [
        0.569444444444444f64,
        0.891300154320988f64,
        0.226624344493027e+01f64,
        0.798950124766861e+01f64,
        0.360688546785343e+02f64,
        0.198670292131169e+03f64,
        0.129223456582211e+04f64,
        0.969483869669600e+04f64,
        0.824184704952483e+05f64,
        0.783031092490225e+06f64,
        0.822210493622814e+07f64,
        0.945557399360556e+08f64,
        0.118195595640730e+10f64,
        0.159564653040121e+11f64,
        0.231369166433050e+12f64,
        0.358622522796969e+13f64,
    ];
    xe = ((x * (x).sqrt()) / 1.5f64);
    xp6 = (1.0f64 / ((6.0f64 * pi) * xe).sqrt());
    su1 = 1.0f64;
    r = 1.0f64;
    xr1 = (1.0f64 / xe);
    k = 1;
    while (k < 17) {
        {
            r = ((-r) * xr1);
            su1 += (a[(k - 1) as usize] * r);
        }
        {
            k += 1;
            k
        };
    }
    su2 = 1.0f64;
    r = 1.0f64;
    k = 1;
    while (k < 17) {
        {
            r = (r * xr1);
            su2 += (a[(k - 1) as usize] * r);
        }
        {
            k += 1;
            k
        };
    }
    apt[0] = (q0 - (((-xe).exp() * xp6) * su1));
    bpt[0] = (((2.0f64 * (xe).exp()) * xp6) * su2);
    su3 = 1.0f64;
    r = 1.0f64;
    xr2 = (1.0f64 / (xe * xe));
    k = 1;
    while (k < 9) {
        {
            r = ((-r) * xr2);
            su3 += (a[((2 * k) - 1) as usize] * r);
        }
        {
            k += 1;
            k
        };
    }
    su4 = (a[(0) as usize] * xr1);
    r = xr1;
    k = 1;
    while (k < 8) {
        {
            r = ((-r) * xr2);
            su4 += (a[(2 * k) as usize] * r);
        }
        {
            k += 1;
            k
        };
    }
    su5 = (su3 + su4);
    su6 = (su3 - su4);
    ant[0] = (q1 - ((q2 * xp6) * ((su5 * (xe).cos()) - (su6 * (xe).sin()))));
    bnt[0] = ((q2 * xp6) * ((su5 * (xe).sin()) + (su6 * (xe).cos())));
}

pub(super) fn itika(mut x: f64, ti: &mut [f64], tk: &mut [f64]) {
    let mut k: i32 = 0;
    let mut rc1: f64 = 0.0;
    let mut rc2: f64 = 0.0;
    let mut e0: f64 = 0.0;
    let mut b1: f64 = 0.0;
    let mut b2: f64 = 0.0;
    let mut rs: f64 = 0.0;
    let mut r: f64 = 0.0;
    let mut tw: f64 = 0.0;
    let mut x2: f64 = 0.0;
    let mut a = [
        0.625f64,
        1.0078125f64,
        2.5927734375f64,
        9.1868591308594f64,
        4.1567974090576e+1f64,
        2.2919635891914e+2f64,
        1.491504060477e+3f64,
        1.1192354495579e+4f64,
        9.515939374212e+4f64,
        9.0412425769041e+5f64,
    ];
    let mut pi: f64 = 3.141592653589793f64;
    let mut el: f64 = 0.5772156649015329f64;
    if (x == 0.0f64) {
        {
            ti[0] = 0.0f64;
            tk[0] = 0.0f64;
            return;
        }
    } else {
        if (x < 20.0f64) {
            {
                x2 = (x * x);
                ti[0] = 1.0f64;
                r = 1.0f64;
                k = 1;
                while (k <= 50) {
                    {
                        r = (((((0.25f64 * r) * (((2 * k) as f64) - 1.0f64))
                            / (((2 * k) as f64) + 1.0f64))
                            / ((k * k) as f64))
                            * x2);
                        ti[0] += r;
                        if ((r / ti[0]).abs() < 1.0e-12f64) {
                            {
                                break;
                            }
                        }
                    }
                    {
                        k += 1;
                        k
                    };
                }
                ti[0] *= x;
            }
        } else {
            {
                x2 = 0.0f64;
                ti[0] = 1.0f64;
                r = 1.0f64;
                k = 1;
                while (k <= 10) {
                    {
                        r = (r / x);
                        ti[0] += (a[(k - 1) as usize] * r);
                    }
                    {
                        k += 1;
                        k
                    };
                }
                rc1 = (1.0f64 / ((2.0f64 * pi) * x).sqrt());
                ti[0] = ((rc1 * (x).exp()) * ti[0]);
            }
        }
    }
    if (x < 12.0f64) {
        {
            e0 = (el + (x / 2.0f64).ln());
            b1 = (1.0f64 - e0);
            b2 = 0.0f64;
            rs = 0.0f64;
            r = 1.0f64;
            tw = 0.0f64;
            k = 1;
            while (k <= 50) {
                {
                    r = (((((0.25f64 * r) * (((2 * k) as f64) - 1.0f64))
                        / (((2 * k) as f64) + 1.0f64))
                        / ((k * k) as f64))
                        * x2);
                    b1 += (r * ((1.0f64 / (((2 * k) + 1) as f64)) - e0));
                    rs += (1.0f64 / ((k) as f64));
                    b2 += (r * rs);
                    tk[0] = (b1 + b2);
                    if (((tk[0] - tw) / tk[0]).abs() < 1.0e-12f64) {
                        {
                            break;
                        }
                    }
                    tw = tk[0];
                }
                {
                    k += 1;
                    k
                };
            }
            tk[0] *= x;
        }
    } else {
        {
            tk[0] = 1.0f64;
            r = 1.0f64;
            k = 1;
            while (k <= 10) {
                {
                    r = ((-r) / x);
                    tk[0] = (tk[0] + (a[(k - 1) as usize] * r));
                }
                {
                    k += 1;
                    k
                };
            }
            rc2 = (pi / (2.0f64 * x)).sqrt();
            tk[0] = ((pi / 2.0f64) - ((rc2 * tk[0]) * (-x).exp()));
        }
    }
    return;
}

pub(super) fn itjya(mut x: f64, tj: &mut [f64], ty: &mut [f64]) {
    let mut k: i32 = 0;
    let mut a = [0.0; (18) as usize];
    let mut a0: f64 = 0.0;
    let mut a1: f64 = 0.0;
    let mut af: f64 = 0.0;
    let mut bf: f64 = 0.0;
    let mut bg: f64 = 0.0;
    let mut r: f64 = 0.0;
    let mut r2: f64 = 0.0;
    let mut rc: f64 = 0.0;
    let mut rs: f64 = 0.0;
    let mut ty1: f64 = 0.0;
    let mut ty2: f64 = 0.0;
    let mut x2: f64 = 0.0;
    let mut xp: f64 = 0.0;
    let mut pi: f64 = 3.141592653589793f64;
    let mut el: f64 = 0.5772156649015329f64;
    let mut eps: f64 = 1.0e-12f64;
    if (x == 0.0f64) {
        {
            tj[0] = 0.0f64;
            ty[0] = 0.0f64;
        }
    } else {
        if (x <= 20.0f64) {
            {
                x2 = (x * x);
                tj[0] = x;
                r = x;
                k = 1;
                while (k < 61) {
                    {
                        r = ((((((-0.25f64) * r) * ((2.0f64 * ((k) as f64)) - 1.0f64))
                            / ((2.0f64 * ((k) as f64)) + 1.0f64))
                            / ((k * k) as f64))
                            * x2);
                        tj[0] += r;
                        if ((r).abs() < ((tj[0]).abs() * eps)) {
                            {
                                break;
                            }
                        }
                    }
                    {
                        k += 1;
                        k
                    };
                }
                ty1 = ((el + (x / 2.0f64).ln()) * tj[0]);
                rs = 0.0f64;
                ty2 = 1.0f64;
                r = 1.0f64;
                k = 1;
                while (k < 61) {
                    {
                        r = ((((((-0.25f64) * r) * ((2.0f64 * ((k) as f64)) - 1.0f64))
                            / ((2.0f64 * ((k) as f64)) + 1.0f64))
                            / ((k * k) as f64))
                            * x2);
                        rs += (1.0f64 / ((k) as f64));
                        r2 = (r * (rs + (1.0f64 / ((2.0f64 * ((k) as f64)) + 1.0f64))));
                        ty2 += r2;
                        if ((r2).abs() < ((ty2).abs() * eps)) {
                            {
                                break;
                            }
                        }
                    }
                    {
                        k += 1;
                        k
                    };
                }
                ty[0] = (((ty1 - (x * ty2)) * 2.0f64) / pi);
            }
        } else {
            {
                a0 = 1.0f64;
                a1 = (5.0f64 / 8.0f64);
                a[(0) as usize] = a1;
                let mut k: i32 = 1;
                while (k <= 16) {
                    {
                        af = (((((1.5f64 * (((k) as f64) + 0.5f64))
                            * (((k) as f64) + (5.0f64 / 6.0f64)))
                            * a1)
                            - ((((0.5f64 * (((k) as f64) + 0.5f64)) * (((k) as f64) + 0.5f64))
                                * (((k) as f64) - 0.5f64))
                                * a0))
                            / (((k) as f64) + 1.0f64));
                        a[(k) as usize] = af;
                        a0 = a1;
                        a1 = af;
                    }
                    {
                        k += 1;
                        k
                    };
                }
                bf = 1.0f64;
                r = 1.0f64;
                let mut k: i32 = 1;
                while (k <= 8) {
                    {
                        r = ((-r) / (x * x));
                        bf += (a[((2 * k) - 1) as usize] * r);
                    }
                    {
                        k += 1;
                        k
                    };
                }
                bg = (a[(0) as usize] / x);
                r = (1.0f64 / x);
                let mut k: i32 = 1;
                while (k <= 8) {
                    {
                        r = ((-r) / (x * x));
                        bg += (a[(2 * k) as usize] * r);
                    }
                    {
                        k += 1;
                        k
                    };
                }
                xp = (x + (0.25f64 * pi));
                rc = (2.0f64 / (pi * x)).sqrt();
                tj[0] = (1.0f64 - (rc * ((bf * (xp).cos()) + (bg * (xp).sin()))));
                ty[0] = (rc * ((bg * (xp).cos()) - (bf * (xp).sin())));
            }
        }
    }
    return;
}

pub(super) fn ittika(mut x: f64, tti: &mut [f64], ttk: &mut [f64]) {
    let mut k: i32 = 0;
    let mut b1: f64 = 0.0;
    let mut e0: f64 = 0.0;
    let mut r: f64 = 0.0;
    let mut r2: f64 = 0.0;
    let mut rc: f64 = 0.0;
    let mut rs: f64 = 0.0;
    let mut pi: f64 = 3.141592653589793f64;
    let mut el: f64 = 0.5772156649015329f64;
    let mut c = [
        1.625f64,
        4.1328125f64,
        1.45380859375e1f64,
        6.553353881835e1f64,
        3.6066157150269e2f64,
        2.3448727161884e3f64,
        1.7588273098916e4f64,
        1.4950639538279e5f64,
    ];
    if (x == 0.0f64) {
        {
            tti[0] = 0.0f64;
            ttk[0] = 1.0e+300f64;
            return;
        }
    }
    if (x < 700.0f64) {
        {
            tti[0] = 1.0f64;
            r = 1.0f64;
            let mut k: i32 = 2;
            while (k <= 1000) {
                {
                    r = (((((0.25f64 * r) * (((k) as f64) - 1.0f64)) / (((k * k) * k) as f64))
                        * x)
                        * x);
                    tti[0] += r;
                    if ((r / tti[0]).abs() < f64::EPSILON) {
                        {
                            break;
                        }
                    }
                }
                {
                    k += 1;
                    k
                };
            }
            tti[0] *= ((0.125f64 * x) * x);
        }
    } else {
        {
            tti[0] = 1.0f64;
            r = 1.0f64;
            let mut k: i32 = 1;
            while (k <= 8) {
                {
                    r = (r / x);
                    tti[0] += (c[(k - 1) as usize] * r);
                }
                {
                    k += 1;
                    k
                };
            }
            rc = (x * ((2.0f64 * pi) * x).sqrt());
            tti[0] = ((tti[0] * (x).exp()) / rc);
        }
    }
    if (x <= 12.0f64) {
        {
            e0 = (((((0.5f64 * (x / 2.0f64).ln()) + el) * (x / 2.0f64).ln())
                + ((pi * pi) / 24.0f64))
                + ((0.5f64 * el) * el));
            b1 = (1.5f64 - (el + (x / 2.0f64).ln()));
            rs = 1.0f64;
            r = 1.0f64;
            k = 2;
            while (k <= 50) {
                {
                    r = (((((0.25f64 * r) * (((k) as f64) - 1.0f64)) / (((k * k) * k) as f64))
                        * x)
                        * x);
                    rs += (1.0f64 / ((k) as f64));
                    r2 = (r
                        * ((rs + (1.0f64 / (2.0f64 * ((k) as f64)))) - (el + (x / 2.0f64).ln())));
                    b1 += r2;
                    if ((r2 / b1).abs() < 1.0e-12f64) {
                        {
                            break;
                        }
                    }
                }
                {
                    k += 1;
                    k
                };
            }
            ttk[0] = (e0 - (((0.125f64 * x) * x) * b1));
        }
    } else {
        {
            ttk[0] = 1.0f64;
            r = 1.0f64;
            k = 1;
            while (k <= 8) {
                {
                    r = ((-r) / x);
                    ttk[0] += (c[(k - 1) as usize] * r);
                }
                {
                    k += 1;
                    k
                };
            }
            rc = (x * ((2.0f64 / pi) * x).sqrt());
            ttk[0] = ((ttk[0] * (-x).exp()) / rc);
        }
    }
    return;
}

pub(super) fn ittjya(mut x: f64, ttj: &mut [f64], tty: &mut [f64]) {
    let mut k: i32 = 0;
    let mut l: i32 = 0;
    let mut a0: f64 = 0.0;
    let mut bj0: f64 = 0.0;
    let mut bj1: f64 = 0.0;
    let mut by0: f64 = 0.0;
    let mut by1: f64 = 0.0;
    let mut e0: f64 = 0.0;
    let mut b1: f64 = 0.0;
    let mut g0: f64 = 0.0;
    let mut g1: f64 = 0.0;
    let mut px: f64 = 0.0;
    let mut qx: f64 = 0.0;
    let mut r: f64 = 0.0;
    let mut r0: f64 = 0.0;
    let mut r1: f64 = 0.0;
    let mut r2: f64 = 0.0;
    let mut rs: f64 = 0.0;
    let mut t: f64 = 0.0;
    let mut vt: f64 = 0.0;
    let mut xk: f64 = 0.0;
    let mut pi: f64 = 3.141592653589793f64;
    let mut el: f64 = 0.5772156649015329f64;
    if (x == 0.0f64) {
        {
            ttj[0] = 0.0f64;
            tty[0] = (-1.0e+300f64);
        }
    } else {
        if (x <= 20.0f64) {
            {
                ttj[0] = 1.0f64;
                r = 1.0f64;
                k = 2;
                while (k <= 100) {
                    {
                        r = ((((((-0.25f64) * r) * (((k) as f64) - 1.0f64))
                            / (((k * k) * k) as f64))
                            * x)
                            * x);
                        ttj[0] += r;
                        if ((r).abs() < ((ttj[0]).abs() * 1.0e-12f64)) {
                            {
                                break;
                            }
                        }
                    }
                    {
                        k += 1;
                        k
                    };
                }
                ttj[0] *= ((0.125f64 * x) * x);
                e0 = ((0.5f64 * (((pi * pi) / 6.0f64) - (el * el)))
                    - (((0.5f64 * (x / 2.0f64).ln()) + el) * (x / 2.0f64).ln()));
                b1 = ((el + (x / 2.0f64).ln()) - 1.5f64);
                rs = 1.0f64;
                r = (-1.0f64);
                k = 2;
                while (k <= 100) {
                    {
                        r = ((((((-0.25f64) * r) * (((k) as f64) - 1.0f64))
                            / (((k * k) * k) as f64))
                            * x)
                            * x);
                        rs = (rs + (1.0f64 / ((k) as f64)));
                        r2 = (r
                            * ((rs + (1.0f64 / (2.0f64 * ((k) as f64))))
                                - (el + (x / 2.0f64).ln())));
                        b1 = (b1 + r2);
                        if ((r2).abs() < ((b1).abs() * 1.0e-12f64)) {
                            {
                                break;
                            }
                        }
                    }
                    {
                        k += 1;
                        k
                    };
                }
                tty[0] = ((2.0f64 / pi) * (e0 + (((0.125f64 * x) * x) * b1)));
            }
        } else {
            {
                a0 = (2.0f64 / (pi * x)).sqrt();
                bj0 = 0.0f64;
                by0 = 0.0f64;
                bj1 = 0.0f64;
                l = 0;
                while (l <= 1) {
                    {
                        vt = ((4.0f64 * ((l) as f64)) * ((l) as f64));
                        px = 1.0f64;
                        r = 1.0f64;
                        k = 1;
                        while (k <= 14) {
                            {
                                r = ((((((-0.0078125f64) * r)
                                    * (vt
                                        - ((4.0f64 * ((k) as f64)) - 3.0f64)
                                            .powf(((2) as f64))))
                                    / (x * ((k) as f64)))
                                    * (vt
                                        - ((4.0f64 * ((k) as f64)) - 1.0f64).powf(((2) as f64))))
                                    / (((2.0f64 * ((k) as f64)) - 1.0f64) * x));
                                px += r;
                                if ((r).abs() < ((px).abs() * 1.0e-12f64)) {
                                    {
                                        break;
                                    }
                                }
                            }
                            {
                                k += 1;
                                k
                            };
                        }
                        qx = 1.0f64;
                        r = 1.0f64;
                        k = 1;
                        while (k <= 14) {
                            {
                                r = ((((((-0.0078125f64) * r)
                                    * (vt
                                        - ((4.0f64 * ((k) as f64)) - 1.0f64)
                                            .powf(((2) as f64))))
                                    / (x * ((k) as f64)))
                                    * (vt
                                        - ((4.0f64 * ((k) as f64)) + 1.0f64).powf(((2) as f64))))
                                    / (((2.0f64 * ((k) as f64)) + 1.0f64) * x));
                                qx += r;
                                if ((r).abs() < ((qx).abs() * 1.0e-12f64)) {
                                    {
                                        break;
                                    }
                                }
                            }
                            {
                                k += 1;
                                k
                            };
                        }
                        qx = (((0.125f64 * (vt - 1.0f64)) / x) * qx);
                        xk = (x - ((0.25f64 + (0.5f64 * ((l) as f64))) * pi));
                        bj1 = (a0 * ((px * (xk).cos()) - (qx * (xk).sin())));
                        by1 = (a0 * ((px * (xk).sin()) + (qx * (xk).cos())));
                        if (l == 0) {
                            {
                                bj0 = bj1;
                                by0 = by1;
                            }
                        }
                    }
                    {
                        l += 1;
                        l
                    };
                }
                t = (2.0f64 / x);
                g0 = 1.0f64;
                r0 = 1.0f64;
                k = 1;
                while (k <= 10) {
                    {
                        r0 = (((((-r0) * ((k) as f64)) * ((k) as f64)) * t) * t);
                        g0 += r0;
                    }
                    {
                        k += 1;
                        k
                    };
                }
                g1 = 1.0f64;
                r1 = 1.0f64;
                k = 1;
                while (k <= 10) {
                    {
                        r1 = (((((-r1) * ((k) as f64)) * (((k) as f64) + 1.0f64)) * t) * t);
                        g1 += r1;
                    }
                    {
                        k += 1;
                        k
                    };
                }
                ttj[0] = ((((((2.0f64 * g1) * bj0) / (x * x)) - ((g0 * bj1) / x)) + el)
                    + (x / 2.0f64).ln());
                tty[0] = ((((2.0f64 * g1) * by0) / (x * x)) - ((g0 * by1) / x));
            }
        }
    }
    return;
}

pub(super) fn itth0(mut x: f64) -> f64 {
    let mut k: i32 = 0;
    let mut f0: f64 = 0.0;
    let mut g0: f64 = 0.0;
    let mut r: f64 = 0.0;
    let mut s: f64 = 0.0;
    let mut t: f64 = 0.0;
    let mut tth: f64 = 0.0;
    let mut tty: f64 = 0.0;
    let mut xt: f64 = 0.0;
    let mut pi: f64 = 3.141592653589793f64;
    s = 1.0f64;
    r = 1.0f64;
    if (x < 24.5f64) {
        {
            k = 1;
            while (k < 61) {
                {
                    r = (((((-r) * x) * x) * ((2.0f64 * ((k) as f64)) - 1.0f64))
                        / ((2.0f64 * ((k) as f64)) + 1.0f64).powf(((3) as f64)));
                    s += r;
                    if ((r).abs() < ((s).abs() * 1.0e-12f64)) {
                        {
                            break;
                        }
                    }
                }
                {
                    k += 1;
                    k
                };
            }
            tth = ((pi / 2.0f64) - (((2.0f64 / pi) * x) * s));
        }
    } else {
        {
            k = 1;
            while (k < 11) {
                {
                    r = (((-r) * ((2.0f64 * ((k) as f64)) - 1.0f64).powf(((3) as f64)))
                        / ((((2.0f64 * ((k) as f64)) + 1.0f64) * x) * x));
                    s += r;
                    if ((r).abs() < ((s).abs() * 1.0e-12f64)) {
                        {
                            break;
                        }
                    }
                }
                {
                    k += 1;
                    k
                };
            }
            tth = ((2.0f64 / (pi * x)) * s);
            t = (8.0f64 / x);
            xt = (x + (0.25f64 * pi));
            f0 = ((((((((((((0.18118e-2f64 * t) - 0.91909e-2f64) * t) + 0.017033f64) * t)
                - 0.9394e-3f64)
                * t)
                - 0.051445f64)
                * t)
                - 0.11e-5f64)
                * t)
                + 0.7978846f64);
            g0 = ((((((((((((-0.23731e-2f64) * t) + 0.59842e-2f64) * t) + 0.24437e-2f64) * t)
                - 0.0233178f64)
                * t)
                + 0.595e-4f64)
                * t)
                + 0.1620695f64)
                * t);
            tty = (((f0 * (xt).sin()) - (g0 * (xt).cos())) / ((x).sqrt() * x));
            tth = (tth + tty);
        }
    }
    return tth;
}

pub(super) fn itsh0(mut x: f64) -> f64 {
    let mut k: i32 = 0;
    let mut a = [0.0; (25) as usize];
    let mut a0: f64 = 0.0;
    let mut a1: f64 = 0.0;
    let mut af: f64 = 0.0;
    let mut bf: f64 = 0.0;
    let mut bg: f64 = 0.0;
    let mut r: f64 = 0.0;
    let mut rd: f64 = 0.0;
    let mut s: f64 = 0.0;
    let mut s0: f64 = 0.0;
    let mut th0: f64 = 0.0;
    let mut ty: f64 = 0.0;
    let mut xp: f64 = 0.0;
    let mut pi: f64 = 3.141592653589793f64;
    let mut el: f64 = 0.57721566490153f64;
    r = 1.0f64;
    if (x <= 30.0f64) {
        {
            s = 0.5f64;
            k = 1;
            while (k < 101) {
                {
                    rd = 1.0f64;
                    if (k == 1) {
                        {
                            rd = 0.5f64;
                        }
                    }
                    r = (((((-r) * rd) * ((k) as f64)) / (((k) as f64) + 1.0f64))
                        * (x / ((2.0f64 * ((k) as f64)) + 1.0f64)).powf(((2) as f64)));
                    s += r;
                    if ((r).abs() < ((s).abs() * 1.0e-12f64)) {
                        {
                            break;
                        }
                    }
                }
                {
                    k += 1;
                    k
                };
            }
            th0 = ((((2.0f64 / pi) * x) * x) * s);
        }
    } else {
        {
            s = 1.0f64;
            k = 1;
            while (k < 13) {
                {
                    r = ((((-r) * ((k) as f64)) / (((k) as f64) + 1.0f64))
                        * (((2.0f64 * ((k) as f64)) + 1.0f64) / x).powf(((2) as f64)));
                    s += r;
                    if ((r).abs() < ((s).abs() * 1.0e-12f64)) {
                        {
                            break;
                        }
                    }
                }
                {
                    k += 1;
                    k
                };
            }
            s0 = ((s / ((pi * x) * x)) + ((2.0f64 / pi) * ((2.0f64 * x).ln() + el)));
            a0 = 1.0f64;
            a1 = (5.0f64 / 8.0f64);
            a[(0) as usize] = a1;
            k = 1;
            while (k < 21) {
                {
                    af = (((((1.5f64 * (((k) as f64) + 0.5f64))
                        * (((k) as f64) + (5.0f64 / 6.0f64)))
                        * a1)
                        - ((((0.5f64 * (((k) as f64) + 0.5f64)) * (((k) as f64) + 0.5f64))
                            * (((k) as f64) - 0.5f64))
                            * a0))
                        / (((k) as f64) + 1.0f64));
                    a[(k) as usize] = af;
                    a0 = a1;
                    a1 = af;
                }
                {
                    k += 1;
                    k
                };
            }
            bf = 1.0f64;
            r = 1.0f64;
            k = 1;
            while (k < 11) {
                {
                    r = ((-r) / (x * x));
                    bf += (a[((2 * k) - 1) as usize] * r);
                }
                {
                    k += 1;
                    k
                };
            }
            bg = (a[(0) as usize] / x);
            r = (1.0f64 / x);
            k = 1;
            while (k <= 10) {
                {
                    r = ((-r) / (x * x));
                    bg += (a[(2 * k) as usize] * r);
                }
                {
                    k += 1;
                    k
                };
            }
            xp = (x + (0.25f64 * pi));
            ty = ((2.0f64 / (pi * x)).sqrt() * ((bg * (xp).cos()) - (bf * (xp).sin())));
            th0 = (ty + s0);
        }
    }
    return th0;
}

pub(super) fn itsl0(mut x: f64) -> f64 {
    let mut k: i32 = 0;
    let mut a = [0.0; (18) as usize];
    let mut a0: f64 = 0.0;
    let mut a1: f64 = 0.0;
    let mut af: f64 = 0.0;
    let mut r: f64 = 0.0;
    let mut rd: f64 = 0.0;
    let mut s: f64 = 0.0;
    let mut s0: f64 = 0.0;
    let mut ti: f64 = 0.0;
    let mut tl0: f64 = 0.0;
    let mut pi: f64 = 3.141592653589793f64;
    let mut el: f64 = 0.57721566490153f64;
    r = 1.0f64;
    if (x <= 20.0f64) {
        {
            s = 0.5f64;
            k = 1;
            while (k < 101) {
                {
                    rd = 1.0f64;
                    if (k == 1) {
                        {
                            rd = 0.5f64;
                        }
                    }
                    r = ((((r * rd) * ((k) as f64)) / (((k) as f64) + 1.0f64))
                        * (x / ((2.0f64 * ((k) as f64)) + 1.0f64)).powf(((2) as f64)));
                    s += r;
                    if ((r / s).abs() < 1.0e-12f64) {
                        {
                            break;
                        }
                    }
                }
                {
                    k += 1;
                    k
                };
            }
            tl0 = ((((2.0f64 / pi) * x) * x) * s);
        }
    } else {
        {
            s = 1.0f64;
            k = 1;
            while (k < 11) {
                {
                    r = (((r * ((k) as f64)) / (((k) as f64) + 1.0f64))
                        * (((2.0f64 * ((k) as f64)) + 1.0f64) / x).powf(((2) as f64)));
                    s += r;
                    if ((r / s).abs() < 1.0e-12f64) {
                        {
                            break;
                        }
                    }
                }
                {
                    k += 1;
                    k
                };
            }
            s0 = (((-s) / ((pi * x) * x)) + ((2.0f64 / pi) * ((2.0f64 * x).ln() + el)));
            a0 = 1.0f64;
            a1 = (5.0f64 / 8.0f64);
            a[(0) as usize] = a1;
            k = 1;
            while (k < 11) {
                {
                    af = (((((1.5f64 * (((k) as f64) + 0.50f64))
                        * (((k) as f64) + (5.0f64 / 6.0f64)))
                        * a1)
                        - (((0.5f64 * (((k) as f64) + 0.5f64).powf(((2) as f64)))
                            * (((k) as f64) - 0.5f64))
                            * a0))
                        / (((k) as f64) + 1.0f64));
                    a[(k) as usize] = af;
                    a0 = a1;
                    a1 = af;
                }
                {
                    k += 1;
                    k
                };
            }
            ti = 1.0f64;
            r = 1.0f64;
            k = 1;
            while (k < 11) {
                {
                    r = (r / x);
                    ti += (a[(k - 1) as usize] * r);
                }
                {
                    k += 1;
                    k
                };
            }
            tl0 = (((ti / ((((2) as f64) * pi) * x).sqrt()) * (x).exp()) + s0);
        }
    }
    return tl0;
}
