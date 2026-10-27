//! Real Cephes I_v, adapted from XSF's cephes/scipy_iv.h.
//! Copyright Stephen L. Moshier (1984–2000), Xiaogang Zhang (2006),
//! Pauli Virtanen (2009), and SciPy developers (2024).
//! See LICENSES/cephes-iv.txt and LICENSES/xsf-bundled.txt.
//!
//! K is carried as exp(x) K throughout Temme's method. The Wronskian
//! therefore yields exp(-x) I directly without overflowing exponentials.
use super::{MACHEP, gamma, lgam, sinpi};
use core::f64::consts::{FRAC_2_PI, PI};

const MAX_ITER: usize = 1000;

#[allow(clippy::excessive_precision)]
const U: [&[f64]; 11] = [
    &[1.0],
    &[-0.20833333333333334, 0.125],
    &[0.3342013888888889, -0.4010416666666667, 0.0703125],
    &[
        -1.0258125964506173,
        1.8464626736111112,
        -0.8912109375,
        0.0732421875,
    ],
    &[
        4.669584423426247,
        -11.207002616222995,
        8.78912353515625,
        -2.3640869140625,
        0.112152099609375,
    ],
    &[
        -28.212072558200244,
        84.63621767460074,
        -91.81824154324003,
        42.53499874538846,
        -7.368794359479631,
        0.22710800170898438,
    ],
    &[
        212.5701300392171,
        -765.2524681411816,
        1059.9904525279999,
        -699.5796273761327,
        218.1905117442116,
        -26.491430486951554,
        0.5725014209747314,
    ],
    &[
        -1919.4576623184068,
        8061.722181737308,
        -13586.550006434136,
        11655.393336864536,
        -5305.646978613405,
        1200.9029132163525,
        -108.09091978839464,
        1.7277275025844574,
    ],
    &[
        20204.29133096615,
        -96980.5983886375,
        192547.0012325315,
        -203400.17728041555,
        122200.46498301747,
        -41192.65496889756,
        7109.514302489364,
        -493.915304773088,
        6.074042001273483,
    ],
    &[
        -242919.18790055133,
        1311763.614662977,
        -2998015.918538106,
        3763271.297656404,
        -2813563.226586534,
        1268365.2733216248,
        -331645.1724845636,
        45218.76898136274,
        -2499.830481811209,
        24.380529699556064,
    ],
    &[
        3284469.8530720375,
        -19706819.11843222,
        50952602.49266463,
        -74105148.21153264,
        66344512.27472903,
        -37567176.66076335,
        13288767.16642182,
        -2785618.128086455,
        308186.40461266245,
        -13886.089753717039,
        110.01714026924674,
    ],
];

fn asymptotic(v: f64, x: f64) -> f64 {
    let mu = 4.0 * v * v;
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..=100 {
        let odd = (2 * k - 1) as f64;
        term *= -((mu - odd * odd) / x) / (8.0 * k as f64);
        sum += term;
        if term.abs() <= MACHEP * sum.abs() {
            return sum / (2.0 * PI).sqrt() / x.sqrt();
        }
    }
    f64::NAN
}

fn uniform(v: f64, x: f64, scaled: bool) -> f64 {
    let a = v.abs();
    let z = x / a;
    let r = z.hypot(1.0);
    let t = r.recip();
    let t2 = t * t;
    let mut i_sum = 1.0;
    let mut k_sum = 1.0;
    let mut power = 1.0;
    for (n, coefficients) in U.iter().enumerate().skip(1) {
        let polynomial = coefficients.iter().fold(0.0, |p, c| p * t2 + c);
        power *= t / a;
        let term = polynomial * power;
        i_sum += term;
        k_sum += if n % 2 == 0 { term } else { -term };
        if term.abs() < MACHEP {
            break;
        }
    }
    // eta = sqrt(1+z²) - asinh(1/z). Rationalize r-z for the
    // scaled exponent, and use logarithms when z underflows.
    let log_ratio = if z < 1e-150 {
        x.ln() - a.ln() - core::f64::consts::LN_2
    } else {
        -z.recip().asinh()
    };
    let exponent = if scaled {
        a * (r.recip() / (1.0 + z / r) + log_ratio)
    } else {
        a * (r + log_ratio)
    };
    let log_prefactor = -0.5 * ((2.0 * PI).ln() + a.ln() + r.ln());
    let value = (exponent + log_prefactor + i_sum.ln()).exp();
    if v < 0.0 && v != v.floor() {
        let k_exponent = if scaled {
            -a * (r + log_ratio) - x
        } else {
            -exponent
        };
        let reflection = FRAC_2_PI * sinpi(a);
        value
            + reflection.signum()
                * (k_exponent + log_prefactor + PI.ln() + k_sum.ln() + reflection.abs().ln()).exp()
    } else {
        value
    }
}

// Temme's series for |u| <= 1/2, x <= 2.
fn temme_series(u: f64, x: f64) -> Option<(f64, f64)> {
    let gp = gamma(1.0 + u) - 1.0;
    let gm = gamma(1.0 - u) - 1.0;
    let a = (0.5 * x).ln();
    let b = (u * a).exp();
    let sigma = -a * u;
    let c = if u.abs() < MACHEP {
        1.0
    } else {
        sinpi(u) / (u * PI)
    };
    let d = if sigma.abs() < MACHEP {
        1.0
    } else {
        sigma.sinh() / sigma
    };
    let gamma1 = if u.abs() < MACHEP {
        -0.5772156649015329
    } else {
        0.5 * (gp - gm) * c / u
    };
    let gamma2 = 0.5 * (2.0 + gp + gm) * c;
    let mut p = (1.0 + gp) / (2.0 * b);
    let mut q = (1.0 + gm) * b / 2.0;
    let mut f = (sigma.cosh() * gamma1 - d * a * gamma2) / c;
    let mut coefficient = 1.0;
    let mut sum = f;
    let mut sum1 = p;
    for k in 1..MAX_ITER {
        let k = k as f64;
        f = (k * f + p + q) / (k * k - u * u);
        p /= k - u;
        q /= k + u;
        let h = p - k * f;
        coefficient *= x * x / (4.0 * k);
        sum += coefficient * f;
        sum1 += coefficient * h;
        if (coefficient * f).abs() < sum.abs() * MACHEP {
            return Some((sum * x.exp(), 2.0 * sum1 / x * x.exp()));
        }
    }
    None
}

// Steed's CF2 for exp(x) K_u and exp(x) K_(u+1), x > 2.
fn cf2(u: f64, x: f64) -> Option<(f64, f64)> {
    let mut a = u * u - 0.25;
    let mut b = 2.0 * (x + 1.0);
    let mut d = b.recip();
    let mut f = d;
    let mut delta = d;
    let mut previous = 0.0;
    let mut current = 1.0;
    let mut c = -a;
    let mut q_sum = c;
    let mut sum = 1.0 + q_sum * delta;
    let mut k = 1.0;
    for _ in 2..MAX_ITER {
        a -= 2.0 * k;
        k += 1.0;
        b += 2.0;
        d = (b + a * d).recip();
        delta *= b * d - 1.0;
        f += delta;
        let q = (previous - (b - 2.0) * current) / a;
        previous = current;
        current = q;
        c *= -a / k;
        q_sum += c * q;
        sum += q_sum * delta;
        if (q_sum * delta).abs() < sum.abs() * MACHEP {
            let k = (PI / 2.0).sqrt() / x.sqrt() / sum;
            return Some((k, k * (0.5 + u + x + (u * u - 0.25) * f) / x));
        }
    }
    None
}

// Modified Lentz CF1, I_(v+1)/I_v.
fn cf1(v: f64, x: f64) -> Option<f64> {
    let tiny = f64::MAX.sqrt().recip();
    let mut c = tiny;
    let mut f = tiny;
    let mut d = 0.0;
    let two_over_x = 2.0 / x;
    let mut vk = v;
    for _ in 1..MAX_ITER {
        vk += 1.0;
        let b = vk * two_over_x;
        c = b + c.recip();
        d += b;
        if c == 0.0 {
            c = tiny;
        }
        if d == 0.0 {
            d = tiny;
        }
        d = d.recip();
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() <= 2.0 * MACHEP {
            return Some(f);
        }
    }
    None
}

pub(crate) fn iv(v: f64, x: f64, scaled: bool) -> f64 {
    // Domain, zero, infinity, parity, and half orders are handled by the caller.
    let v = if v < 0.0 && v == v.floor() { -v } else { v };
    if v.abs() > 50.0 {
        return uniform(v, x, scaled);
    }
    if x * x < (v + 1.0).abs() * MACHEP {
        let g = gamma(v + 1.0);
        return g.signum()
            * (v * (x.ln() - core::f64::consts::LN_2)
                - lgam(v + 1.0)
                - if scaled { x } else { 0.0 })
            .exp();
    }
    let a = v.abs();
    let lim = (4.0 * a * a + 10.0) / x / 8.0;
    // CF1 needs O(x) terms and exhausts MAX_ITER for large x. Beyond x = 1000
    // the large-x expansion has peak terms of order one for |v| <= 50.
    if x > 1000.0 || (x > 100.0 && lim.powi(4) / 24.0 < 10.0 * MACHEP) {
        return unscale(asymptotic(a, x), x, scaled);
    }
    let n = a.round() as usize;
    let u = a - n as f64;
    let Some((mut previous, mut current)) = (if x <= 2.0 {
        temme_series(u, x)
    } else {
        cf2(u, x)
    }) else {
        return f64::NAN;
    };
    let mut log_scale = 0.0;
    for k in 1..=n {
        if current.abs() > 1e200 {
            previous *= 1e-200;
            current *= 1e-200;
            log_scale += 200.0 * core::f64::consts::LN_10;
        }
        let next = 2.0 * (u + k as f64) * current / x + previous;
        previous = current;
        current = next;
    }
    let Some(ratio) = cf1(a, x) else {
        return f64::NAN;
    };
    let denominator = previous * ratio + current;
    let mut value = if log_scale == 0.0 {
        denominator.recip() / x
    } else {
        (-denominator.ln() - x.ln() - log_scale).exp()
    };
    if v < 0.0 {
        let reflection = FRAC_2_PI * sinpi(a);
        value += if log_scale == 0.0 {
            reflection * previous * (-2.0 * x).exp()
        } else {
            reflection.signum()
                * (reflection.abs().ln() + previous.ln() + log_scale - 2.0 * x).exp()
        };
    }
    unscale(value, x, scaled)
}

fn unscale(value: f64, x: f64, scaled: bool) -> f64 {
    if scaled {
        value
    } else {
        let half = (0.5 * x).exp();
        (value * half) * half
    }
}
