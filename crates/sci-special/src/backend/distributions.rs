//! Additional discrete distribution functions, using existing beta/gamma kernels.
use crate::backend::{StrErr, cephes64, gamma_extra::inverse_beta, utils::finish};
#[allow(unused_imports)]
use sci_core::Float;

fn binomial(k: f64, n: i32, p: f64, complement: bool) -> f64 {
    let k = k.floor();
    if !(0.0..=1.0).contains(&p) || n < 0 || k > n as f64 {
        return f64::NAN;
    }
    if k < 0.0 {
        return if complement { 1.0 } else { f64::NAN };
    }
    if k == n as f64 {
        return if complement { 0.0 } else { 1.0 };
    }
    if k == 0.0 {
        let exponent = n as f64 * (-p).ln_1p();
        return if complement {
            -exponent.exp_m1()
        } else {
            exponent.exp()
        };
    }
    if complement {
        cephes64::incbet(k + 1.0, n as f64 - k, p)
    } else {
        cephes64::incbet(n as f64 - k, k + 1.0, 1.0 - p)
    }
}

fn inverse_binomial(k: f64, n: i32, y: f64) -> f64 {
    let k = k.floor();
    if !(0.0..=1.0).contains(&y) || !k.is_finite() || k < 0.0 || k >= n as f64 {
        return f64::NAN;
    }
    if k == 0.0 {
        return -(y.ln() / n as f64).exp_m1();
    }
    if y > 0.5 {
        inverse_beta(k + 1.0, n as f64 - k, 1.0 - y)
    } else {
        1.0 - inverse_beta(n as f64 - k, k + 1.0, y)
    }
}

macro_rules! binomial_api {
    ($(#[$meta:meta])* $name:ident($k:ident, $n:ident, $p:ident) $body:block) => {
        $(#[$meta])*
        pub fn $name<T: Float>($k: T, $n: i32, $p: T) -> Result<T, StrErr> {
            if $k.is_nan() || $p.is_nan() { return Err(concat!(stringify!($name), ": Arguments cannot be NAN.")); }
            let $k = $k.to_f64().ok_or("Binomial: Invalid argument.")?;
            let $p = $p.to_f64().ok_or("Binomial: Invalid probability.")?;
            finish($body, concat!(stringify!($name), ": Invalid domain."))
        }
    };
}

binomial_api! {
    /// Binomial CDF through floor(k), for 0 <= floor(k) <= n and 0 <= p <= 1.
    bdtr(k, n, p) { binomial(k, n, p, false) }
}
binomial_api! {
    /// Binomial survival probability above floor(k), for n >= 0 and 0 <= p <= 1.
    /// Returns one for k < 0; floor(k) > n is outside the domain.
    bdtrc(k, n, p) { binomial(k, n, p, true) }
}
binomial_api! {
    /// Success probability p such that bdtr(k,n,p) = y.
    /// Requires 0 <= floor(k) < n and 0 <= y <= 1.
    bdtri(k, n, y) { inverse_binomial(k, n, y) }
}

macro_rules! negative_binomial_api {
    ($(#[$meta:meta])* $name:ident($k:ident, $n:ident, $p:ident) $body:block) => {
        $(#[$meta])*
        pub fn $name<T: Float>($k: i32, $n: i32, $p: T) -> Result<T, StrErr> {
            let $p = $p.to_f64().ok_or("Negative binomial: Invalid probability.")?;
            if $k < 0 || $n <= 0 || !(0.0..=1.0).contains(&$p) {
                return Err(concat!(stringify!($name), ": Requires k >= 0, n > 0 and probability in [0,1]."));
            }
            // Convert before adding, including on 32-bit targets at i32::MAX.
            let $k = $k as f64;
            let $n = $n as f64;
            finish($body, concat!(stringify!($name), ": Undefined result."))
        }
    };
}
negative_binomial_api! {
    /// Probability of at most k failures before the nth success, with success probability p.
    /// Requires k >= 0, n > 0 and 0 <= p <= 1.
    nbdtr(k, n, p) { cephes64::incbet(n, k+1.0, p) }
}
negative_binomial_api! {
    /// Probability of more than k failures before the nth success.
    /// Requires k >= 0, n > 0 and 0 <= p <= 1.
    nbdtrc(k, n, p) {
        if k == 0.0 { -(n*p.ln()).exp_m1() }
        else { cephes64::incbet(k+1.0, n, 1.0-p) }
    }
}
negative_binomial_api! {
    /// Success probability p such that nbdtr(k,n,p) = y.
    /// Requires k >= 0, n > 0 and 0 <= y <= 1.
    nbdtri(k, n, y) { inverse_beta(n, k+1.0, y) }
}

/// Poisson mean m such that pdtr(k,m) = y, for k >= 0 and 0 <= y < 1.
pub fn pdtri<T: Float>(k: i32, y: T) -> Result<T, StrErr> {
    let y = y.to_f64().ok_or("pdtri: Invalid probability.")?;
    finish(
        cephes64::pdtri(k as isize, y),
        "pdtri: Requires k >= 0 and 0 <= y < 1.",
    )
}
