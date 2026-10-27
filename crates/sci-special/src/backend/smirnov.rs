//! Exact finite-sample, one-sided Kolmogorov–Smirnov distribution.
use crate::backend::{StrErr, double_double::Dd, utils::finish};
use sci_core::Float;

const ERROR: StrErr = "Smirnov: Requires 1 <= n <= 1000000 and an argument in [0,1].";

fn validate(n: u32, x: f64) -> Result<(), StrErr> {
    if n == 0 || n > 1_000_000 || !(0.0..=1.0).contains(&x) {
        Err(ERROR)
    } else {
        Ok(())
    }
}

// Returns log CDF, log survival, and the survival derivative.
fn tails(n: u32, x: f64) -> (f64, f64, f64) {
    let nf = n as f64;
    if x == 0.0 {
        return (f64::NEG_INFINITY, 0.0, -1.0);
    }
    if x == 1.0 {
        return (0.0, f64::NEG_INFINITY, if n == 1 { -1.0 } else { 0.0 });
    }
    if nf * x <= 1.0 {
        let logcdf = x.ln() + (nf - 1.0) * x.ln_1p();
        let derivative = -(logcdf.exp() / x) * (1.0 + (nf - 1.0) * x / (1.0 + x));
        return (logcdf, (-logcdf.exp()).ln_1p(), derivative);
    }
    // Smirnov's finite positive sum. Extended precision in logarithmic
    // binomial accumulation limits drift for large sample sizes.
    let end = (nf * (1.0 - x)).floor() as u32;
    let mut binomial = Dd::from(0.0);
    let mut scale = f64::NEG_INFINITY;
    let mut sum = 0.0;
    let mut correction = 0.0;
    let mut derivative = 0.0;
    for j in 0..=end {
        let jf = j as f64;
        let a = x + jf / nf;
        // Subtract in this order to retain the last, very small interval.
        let b = (nf - jf) / nf - x;
        let exponent = n - j;
        if j > 0 {
            binomial = binomial + Dd::from(((n - j + 1) as f64 / jf).ln());
        }
        let (logterm, slope) = if j == 0 {
            (nf * (-x).ln_1p(), -nf / (1.0 - x))
        } else if b > 0.0 {
            (
                x.ln() + binomial.hi + binomial.lo + (jf - 1.0) * a.ln() + exponent as f64 * b.ln(),
                1.0 / x + (jf - 1.0) / a - exponent as f64 / b,
            )
        } else {
            // At the moving endpoint only exponent=1 has a nonzero derivative.
            if exponent == 1 {
                let d = -(x.ln() + binomial.hi + binomial.lo + (jf - 1.0) * a.ln()).exp();
                derivative += if scale.is_finite() {
                    d * (-scale).exp()
                } else {
                    0.0
                };
            }
            continue;
        };
        if logterm > scale {
            let rescale = (scale - logterm).exp();
            sum *= rescale;
            correction *= rescale;
            derivative *= rescale;
            scale = logterm;
        }
        let weight = (logterm - scale).exp();
        let term = weight - correction;
        let next = sum + term;
        correction = (next - sum) - term;
        sum = next;
        derivative += weight * slope;
    }
    let logsf = (scale + sum.ln()).min(0.0);
    ((-logsf.exp_m1()).ln(), logsf, derivative * scale.exp())
}

macro_rules! distribution {
    ($($name:ident,$component:tt,$doc:literal;)+)=>{$(
        #[doc=$doc]
        pub fn $name<T:Float>(n:u32,x:T)->Result<T,StrErr> {
            let x=x.to_f64().ok_or(ERROR)?;
            validate(n,x)?;
            let values=tails(n,x);
            let value=distribution!(@value values,$component);
            finish(value,ERROR)
        }
    )+};
    (@value $v:ident,lower)=>{if $v.0 < -core::f64::consts::LN_2 {$v.0.exp()} else {-$v.1.exp_m1()}};
    (@value $v:ident,upper)=>{if $v.1 < -core::f64::consts::LN_2 {$v.1.exp()} else {-$v.0.exp_m1()}};
    (@value $v:ident,slope)=>{$v.2};
}
distribution! {
    smirnov,upper,"One-sided finite-sample Kolmogorov–Smirnov survival probability; n is in 1..=1000000.";
    smirnovc,lower,"Complement of smirnov, evaluated directly for small arguments.";
    smirnovp,slope,"Derivative of smirnov with respect to its argument (nonpositive).";
}

fn inverse(n: u32, p: f64, mut lower: bool) -> Result<f64, StrErr> {
    validate(n, p)?;
    if p == 0.0 {
        return Ok(if lower { 0.0 } else { 1.0 });
    }
    if p == 1.0 {
        return Ok(if lower { 1.0 } else { 0.0 });
    }
    let p = if p > 0.5 {
        lower = !lower;
        1.0 - p
    } else {
        p
    };
    let target = p.ln();
    // Bisection in log x resolves both subnormal lower probabilities and
    // very small distances from one without an absolute x tolerance.
    let mut left = -746.0;
    let mut right = 0.0;
    for _ in 0..80 {
        let logx = 0.5 * (left + right);
        let values = tails(n, logx.exp());
        let value = if lower { values.0 } else { values.1 };
        if (value < target) == lower {
            left = logx;
        } else {
            right = logx;
        }
    }
    Ok((0.5 * (left + right)).exp())
}
macro_rules! inverse_api {
    ($($name:ident,$lower:literal,$doc:literal;)+)=>{$(
        #[doc=$doc]
        pub fn $name<T:Float>(n:u32,p:T)->Result<T,StrErr> {
            finish(inverse(n,p.to_f64().ok_or(ERROR)?,$lower)?,ERROR)
        }
    )+};
}
inverse_api! {
    smirnovi,false,"Inverse finite-sample Smirnov survival function, including endpoint probabilities.";
    smirnovci,true,"Inverse finite-sample Smirnov CDF, preserving subnormal probabilities.";
}
