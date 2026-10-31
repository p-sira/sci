//! Kelvin functions: ber/bei from direct series and asymptotic kernels, ker/kei
//! from a power series and a trapezoidal integral at rotated arguments.
use crate::backend::{
    StrErr,
    complex::finish,
    double_double::{Dd, DdFactor, split},
};
use core::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_4, PI, SQRT_2};
use num_complex::{Complex, Complex64};
#[allow(unused_imports)]
use sci_core::Float;

/// Kelvin values (ber + i bei, ker + i kei, ber′ + i bei′, ker′ + i kei′).
pub type Kelvin<T> = (Complex<T>, Complex<T>, Complex<T>, Complex<T>);

// Evaluate both the ordinary and harmonic-weighted I_0 series. Keeping the
// derivative series separate preserves its leading term when x*x underflows.
fn small(x: f64) -> Kelvin<f64> {
    if x < 1e-150 {
        let logarithm = x.ln() - core::f64::consts::LN_2 + 0.5772156649015328606;
        return (
            Complex64::new(1.0, (x * 0.5) * (x * 0.5)),
            Complex64::new(-logarithm, -FRAC_PI_4),
            Complex64::new(-((x * 0.25) * x) * (x * 0.25), x * 0.5),
            Complex64::new(-1.0 / x, x * (0.25 - 0.5 * logarithm)),
        );
    }
    let q = Complex64::new(0.0, (x * 0.5) * (x * 0.5));
    let mut term = Complex64::new(1.0, 0.0);
    let mut derivative = Complex64::new(0.0, x * 0.5);
    let mut be = term;
    let mut bep = derivative;
    let mut harmonic = 0.0;
    let mut weighted = Complex64::new(0.0, 0.0);
    let mut weighted_derivative = derivative;
    for k in 1..=32 {
        let k = k as f64;
        harmonic += 1.0 / k;
        term *= q / (k * k);
        be += term;
        weighted += term * harmonic;
        derivative *= q / (k * (k + 1.0));
        bep += derivative;
        weighted_derivative += derivative * (harmonic + 1.0 / (k + 1.0));
        if term.norm() <= f64::EPSILON * be.norm() && derivative.norm() <= f64::EPSILON * bep.norm()
        {
            break;
        }
    }
    let logarithm = Complex64::new(
        x.ln() - core::f64::consts::LN_2 + 0.5772156649015328606,
        FRAC_PI_4,
    );
    let ke = weighted - logarithm * be;
    let kep = weighted_derivative - logarithm * bep - Complex64::new(be.re / x, be.im / x);
    (be, ke, bep, kep)
}

fn argument<T: Float>(x: T) -> Result<f64, StrErr> {
    let x = x.to_f64().ok_or("kelvin: Invalid argument.")?;
    if !x.is_finite() {
        return Err("kelvin: Argument must be finite.");
    }
    Ok(x)
}

/// Which of the value and derivative a caller needs from [`pair`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum Need {
    Both,
    Value,
    Derivative,
}

/// Below this argument, ber + i bei comes from the power series; above it,
/// from the large-argument expansion. Both reach binary64 accuracy here: the
/// asymptotic terms fall below 1e-17 before they start to grow.
const ASYMPTOTIC: f64 = 20.0;
/// 1/√2 - FRAC_1_SQRT_2 and π/8 - PI/8, the binary64 rounding errors.
const FRAC_1_SQRT_2_LO: f64 = -4.833646656726457e-17;
const PI_8_LO: f64 = 1.5308084989341915e-17;

/// Below this argument the power series loses less than about one ulp to
/// cancellation and is summed in binary64; above it, in double-double.
const DOUBLE_DOUBLE: f64 = 4.0;

/// ber + i bei = I₀(x e^{iπ/4}) and its x-derivative for 0.5 <= x < 20 from
/// the power series Σ i^k (x/2)^{2k} / (k!)².
///
/// The absolute terms sum to I₀(x) while the result has modulus about
/// e^{x/√2}/√(2πx), so the series loses up to e^{(1-1/√2)x} in relative
/// accuracy. Beyond x = 4, terms and partial sums are kept in double-double.
/// Index k mod 4 selects the +re, +im, -re or -im partial sum; the
/// derivative series Σ i^k k (x/2)^{2k} / (k!)² is divided by x/2 at the end.
fn first_kind_series(x: f64, need: Need) -> (Complex64, Complex64) {
    let half = 0.5 * x;
    let value = need != Need::Derivative;
    let derivative = need != Need::Value;
    let nan = Complex64::new(f64::NAN, f64::NAN);
    let combine = |s: [f64; 4]| Complex64::new(s[0] - s[2], s[1] - s[3]);
    if x < DOUBLE_DOUBLE {
        let t = half * half;
        let mut sums = [1.0, 0.0, 0.0, 0.0];
        let mut derivative_sums = [0.0; 4];
        let mut term = 1.0;
        let mut k = 1.0_f64;
        let mut slot = 1;
        loop {
            term *= t / (k * k);
            sums[slot] += term;
            derivative_sums[slot] += term * k;
            // Terms peak by k = 2 here, and the value and derivative have
            // moduli above 1 and 0.2, so an absolute tolerance suffices.
            if term * k <= 1e-18 {
                break;
            }
            k += 1.0;
            slot = (slot + 1) & 3;
        }
        return (
            combine(sums),
            Complex64::new(
                (derivative_sums[0] - derivative_sums[2]) / half,
                (derivative_sums[1] - derivative_sums[3]) / half,
            ),
        );
    }
    let (value_sum, derivative_sum) = match need {
        Need::Both => series_dd::<true, true>(half),
        Need::Value => series_dd::<true, false>(half),
        Need::Derivative => series_dd::<false, true>(half),
    };
    (
        if value { value_sum } else { nan },
        if derivative {
            derivative_sum / half
        } else {
            nan
        },
    )
}

/// Double-double sums of Σ i^k t^k / (k!)² and Σ i^k k t^k / (k!)², with
/// t = (x/2)², for 4 <= x < 20. Four terms are taken per iteration, one into
/// each of the +re, +im, -re and -im partial sums, so that every partial sum
/// adds terms of one sign.
#[inline(always)]
fn series_dd<const VALUE: bool, const DERIVATIVE: bool>(half: f64) -> (Complex64, Complex64) {
    let (half_hi, half_lo) = split(half);
    let t_hi = half * half;
    let t = Dd {
        hi: t_hi,
        lo: (half_hi * half_hi - t_hi) + 2.0 * half_hi * half_lo + half_lo * half_lo,
    };
    let factor = DdFactor::new(t);
    let zero = Dd::from(0.0);
    let (mut re_plus, mut im_plus, mut re_minus, mut im_minus) = (Dd::from(1.0), zero, zero, zero);
    let (mut d_re_plus, mut d_im_plus, mut d_re_minus, mut d_im_minus) = (zero, zero, zero, zero);
    let mut term = Dd::from(1.0);
    let mut k = 0.0_f64;
    let mut step = |sum: &mut Dd, derivative_sum: &mut Dd, k: f64| -> (f64, f64) {
        term = factor.mul(term).div_small(k * k);
        if VALUE {
            *sum = sum.add_same_sign(term);
        }
        let mut weighted = 0.0;
        if DERIVATIVE {
            let product = term.mul_small(k);
            *derivative_sum = derivative_sum.add_same_sign(product);
            weighted = product.hi;
        }
        (term.hi, weighted)
    };
    // At most 64 groups (k <= 256) are needed for x < 20.
    for _ in 0..64 {
        step(&mut im_plus, &mut d_im_plus, k + 1.0);
        step(&mut re_minus, &mut d_re_minus, k + 2.0);
        step(&mut im_minus, &mut d_im_minus, k + 3.0);
        let (last, last_weighted) = step(&mut re_plus, &mut d_re_plus, k + 4.0);
        k += 4.0;
        // Terms grow until k ≈ x/2; the last terms of the group are compared
        // with the moduli of the series they entered.
        if k > half {
            let small = |last: f64, rp: Dd, ip: Dd, rm: Dd, im: Dd| {
                last <= 1e-18 * ((rp.hi - rm.hi).abs() + (ip.hi - im.hi).abs())
            };
            if (!VALUE || small(last, re_plus, im_plus, re_minus, im_minus))
                && (!DERIVATIVE
                    || small(last_weighted, d_re_plus, d_im_plus, d_re_minus, d_im_minus))
            {
                break;
            }
        }
    }
    let combine = |re_plus: Dd, im_plus: Dd, re_minus: Dd, im_minus: Dd| {
        let re = re_plus - re_minus;
        let im = im_plus - im_minus;
        Complex64::new(re.hi + re.lo, im.hi + im.lo)
    };
    (
        combine(re_plus, im_plus, re_minus, im_minus),
        combine(d_re_plus, d_im_plus, d_re_minus, d_im_minus),
    )
}

/// ber + i bei and its x-derivative for x >= 20, multiplied by e^{-x/√2} when
/// `scaled`, from DLMF 10.40.5
/// at z = x e^{iπ/4}:
///
/// I_ν(z) ~ e^z/√(2πz) Σ (-1)^k a_k(ν)/z^k + e^{(ν+½)πi} e^{-z}/√(2πz) Σ a_k(ν)/z^k.
///
/// The recessive term is kept because its relative size e^{-√2 x} still
/// exceeds rounding at x = 20. The derivative is e^{iπ/4} I₁(z).
fn first_kind_asymptotic(x: f64, scaled: bool, need: Need) -> (Complex64, Complex64) {
    let value = need != Need::Derivative;
    let derivative = need != Need::Value;
    // z^{-k} = e^{-ikπ/4}/x^k; the powers of x are folded into a_k.
    let unit = Complex64::new(FRAC_1_SQRT_2, -FRAC_1_SQRT_2);
    let mut power = Complex64::new(1.0, 0.0);
    let (mut a0, mut a1) = (1.0_f64, 1.0_f64);
    let one = Complex64::new(1.0, 0.0);
    let (mut dominant0, mut recessive0) = (one, one);
    let (mut dominant1, mut recessive1) = (one, one);
    let mut k = 1.0_f64;
    while k < 64.0 {
        let odd = 2.0 * k - 1.0;
        let odd2 = odd * odd;
        let denominator = 8.0 * k * x;
        a0 *= -odd2 / denominator;
        a1 *= (4.0 - odd2) / denominator;
        power *= unit;
        let sign = if k as u32 % 2 == 1 { -1.0 } else { 1.0 };
        if value {
            let term = power * a0;
            dominant0 += term * sign;
            recessive0 += term;
        }
        if derivative {
            let term = power * a1;
            dominant1 += term * sign;
            recessive1 += term;
        }
        if a0.abs().max(a1.abs()) <= 1e-18 {
            break;
        }
        k += 1.0;
    }
    // e^{i x/√2} and e^{-√2 x} e^{-i x/√2}, both divided by √(2πz).
    // The phase x/√2 - π/8 is formed in double-double: rounding it in
    // binary64 would cost about x ulps in the oscillating result.
    let phase = Dd::from(x).mul_f64(FRAC_1_SQRT_2) + Dd::from(x * FRAC_1_SQRT_2_LO);
    let shifted = phase
        - Dd {
            hi: PI / 8.0,
            lo: PI_8_LO,
        };
    let (sin, cos) = shifted.hi.sin_cos();
    let (sin, cos) = (sin + shifted.lo * cos, cos - shifted.lo * sin);
    let mut modulus = 1.0 / (2.0 * PI * x).sqrt();
    // The unscaled growth e^{x/√2} is applied as two half factors so that it
    // may overflow only together with the result, and uses the same
    // double-double exponent (its rounding would also cost about x ulps).
    let half_scale = if scaled {
        1.0
    } else {
        modulus *= 1.0 + phase.lo;
        (0.5 * phase.hi).exp()
    };
    let restore = |w: Complex64| {
        Complex64::new(
            (w.re * half_scale) * half_scale,
            (w.im * half_scale) * half_scale,
        )
    };
    let growing = Complex64::new(cos, sin) * modulus;
    let phase = phase.hi;
    let (sin, cos) = (-phase - PI / 8.0).sin_cos();
    let decaying = Complex64::new(cos, sin) * (modulus * (-SQRT_2 * x).exp());
    let i = Complex64::new(0.0, 1.0);
    let nan = Complex64::new(f64::NAN, f64::NAN);
    let rotation = Complex64::new(FRAC_1_SQRT_2, FRAC_1_SQRT_2);
    (
        if value {
            restore(growing * dominant0 + i * decaying * recessive0)
        } else {
            nan
        },
        if derivative {
            restore(rotation * (growing * dominant1 - i * decaying * recessive1))
        } else {
            nan
        },
    )
}

/// Below this argument, ker + i kei comes from the binary64 power series in
/// [`small`], which loses less than about 4 ulps to cancellation there; above
/// it, from [`second_kind_integral`].
const SECOND_KIND_SERIES: f64 = 2.0;

/// e^z K₀(z) and e^z K₁(z) at z = x e^{iπ/4} for x >= 2.
///
/// Substituting cosh t = 1 + s² in K_ν(z) = ∫₀^∞ e^{-z cosh t} cosh(νt) dt
/// (DLMF 10.32.9) gives
///
/// e^z K₀(z) = ∫₀^∞ 2 e^{-z s²} / √(s² + 2) ds,
/// e^z K₁(z) = ∫₀^∞ 2 e^{-z s²} (1 + s²) / √(s² + 2) ds.
///
/// The integrands are even and analytic for |Im s| < √2, so the trapezoidal
/// rule with step h converges geometrically; on the line Im s = σ they grow by
/// at most e^{√2 x σ²}, which bounds the error by about
/// e^{√2 x σ² - 2πσ/h}. The step min(0.17, 0.38/√x) keeps this below 1e-17
/// with a margin of 10% in h; the sum then stops once e^{-z s²} is below 1e-18
/// (at most about 32 nodes). The nodes e^{-z h² k²} come from the recurrence
/// g_{k+1} = g_k r_k, r_{k+1} = r_k e^{-2zh²}, so the loop needs only complex
/// products and one square root per node.
fn second_kind_integral(x: f64) -> (Complex64, Complex64) {
    let h = (0.38 / x.sqrt()).min(0.17);
    let zh = Complex64::new(x * FRAC_1_SQRT_2, x * FRAC_1_SQRT_2) * (h * h);
    let mut ratio = (-zh).exp();
    let step = ratio * ratio;
    let mut node = Complex64::new(1.0, 0.0);
    // Half of the s = 0 node, 2/√2 for both integrands.
    let mut k0 = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let mut k1 = k0;
    for k in 1..128 {
        node *= ratio;
        ratio *= step;
        let s2 = (k as f64 * h) * (k as f64 * h);
        let term = node * (2.0 / (s2 + 2.0).sqrt());
        k0 += term;
        k1 += term * (1.0 + s2);
        if node.norm_sqr() < 1e-36 {
            break;
        }
    }
    (k0 * h, k1 * h)
}

/// e^{-z} at z = x e^{iπ/4}, or its unit phase e^{-i x/√2} when `scaled`.
/// The angle and exponent x/√2 are formed in double-double: rounding them in
/// binary64 would cost about x ulps in the oscillating result.
fn decay(x: f64, scaled: bool) -> Complex64 {
    let phase = Dd::from(x).mul_f64(FRAC_1_SQRT_2) + Dd::from(x * FRAC_1_SQRT_2_LO);
    let (sin, cos) = phase.hi.sin_cos();
    let (sin, cos) = (sin + phase.lo * cos, cos - phase.lo * sin);
    let modulus = if scaled {
        1.0
    } else {
        (-phase.hi).exp() * (1.0 - phase.lo)
    };
    Complex64::new(cos * modulus, -sin * modulus)
}

fn pair(x: f64, second: bool, scaled: bool) -> Result<(Complex64, Complex64), StrErr> {
    pair_needed(x, second, scaled, Need::Both)
}

fn pair_needed(
    x: f64,
    second: bool,
    scaled: bool,
    need: Need,
) -> Result<(Complex64, Complex64), StrErr> {
    if second && x < 0.0 {
        return Err("kelvin: ker and kei require x >= 0.");
    }
    let a = x.abs();
    let (value, mut derivative) = if a == 0.0 {
        if second {
            (
                Complex64::new(f64::INFINITY, -FRAC_PI_4),
                Complex64::new(f64::NEG_INFINITY, 0.0),
            )
        } else {
            (Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0))
        }
    } else if a < 0.5 {
        let (be, ke, bep, kep) = small(a);
        if second { (ke, kep) } else { (be, bep) }
    } else {
        let rotation = Complex64::new(FRAC_1_SQRT_2, FRAC_1_SQRT_2);
        if second {
            if a < SECOND_KIND_SERIES {
                let (_, ke, _, kep) = small(a);
                if scaled {
                    let scale = (a * FRAC_1_SQRT_2).exp();
                    (ke * scale, kep * scale)
                } else {
                    (ke, kep)
                }
            } else {
                let (k0, k1) = second_kind_integral(a);
                let scale = decay(a, scaled);
                (k0 * scale, -rotation * k1 * scale)
            }
        } else {
            let (value, derivative) = if a < ASYMPTOTIC {
                let (value, derivative) = first_kind_series(a, need);
                if scaled {
                    let scale = (-a * FRAC_1_SQRT_2).exp();
                    (value * scale, derivative * scale)
                } else {
                    (value, derivative)
                }
            } else {
                first_kind_asymptotic(a, scaled, need)
            };
            (value, derivative)
        }
    };
    if !second && x.is_sign_negative() {
        derivative = -derivative;
    }
    Ok((value, derivative))
}

/// All four complex Kelvin pairs. Requires finite x >= 0.
///
/// At zero, ker and ker′ return their infinite limits. Backend precision
/// failures are reported as errors, as for the complex Bessel functions.
pub fn kelvin<T: Float>(x: T) -> Result<Kelvin<T>, StrErr> {
    let x = argument(x)?;
    let (be, bep) = pair(x, false, false)?;
    let (ke, kep) = pair(x, true, false)?;
    Ok((finish(be)?, finish(ke)?, finish(bep)?, finish(kep)?))
}

macro_rules! kelvin_component {
    ($($name:ident, $second:literal, $index:tt, $component:ident, $doc:literal;)+) => {$(
        #[doc = $doc]
        pub fn $name<T: Float>(x: T) -> Result<T, StrErr> {
            let need = if $index == 0 { Need::Value } else { Need::Derivative };
            let value = pair_needed(argument(x)?, $second, false, need)?.$index.$component;
            crate::backend::utils::finish(value, concat!(stringify!($name), ": Undefined result."))
        }
    )+};
}
kelvin_component! {
    ber, false, 0, re, "Kelvin ber(x), even in x; requires a finite argument.";
    bei, false, 0, im, "Kelvin bei(x), even in x; requires a finite argument.";
    berp, false, 1, re, "Derivative of Kelvin ber(x), odd in x; requires a finite argument.";
    beip, false, 1, im, "Derivative of Kelvin bei(x), odd in x; requires a finite argument.";
    ker, true, 0, re, "Kelvin ker(x), for finite x >= 0; ker(0) = infinity.";
    kei, true, 0, im, "Kelvin kei(x), for finite x >= 0; kei(0) = -pi/4.";
    kerp, true, 1, re, "Derivative of Kelvin ker(x), for finite x >= 0; ker′(0) = -infinity.";
    keip, true, 1, im, "Derivative of Kelvin kei(x), for finite x >= 0; kei′(0) = 0.";
}

/// Select a Kelvin function or its first derivative for zero finding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KelvinFunction {
    /// ber(x).
    Ber,
    /// bei(x).
    Bei,
    /// ker(x).
    Ker,
    /// kei(x).
    Kei,
    /// ber′(x).
    BerPrime,
    /// bei′(x).
    BeiPrime,
    /// ker′(x).
    KerPrime,
    /// kei′(x).
    KeiPrime,
}

impl KelvinFunction {
    /// Scaled value alone, for bracketing; one kernel evaluation.
    fn value(self, x: f64) -> Result<f64, StrErr> {
        use KelvinFunction::*;
        let second = matches!(self, Ker | Kei | KerPrime | KeiPrime);
        let prime = matches!(self, BerPrime | BeiPrime | KerPrime | KeiPrime);
        let imaginary = matches!(self, Bei | Kei | BeiPrime | KeiPrime);
        let need = if prime { Need::Derivative } else { Need::Value };
        let (value, derivative) = pair_needed(x, second, true, need)?;
        let value = if prime { derivative } else { value };
        Ok(if imaginary { value.im } else { value.re })
    }

    /// Scaled value and its first two derivatives at `x > 0`.
    ///
    /// Both w = ber + i bei and w = ker + i kei satisfy
    /// w'' = i w - w'/x (DLMF 10.61), so the second derivative needs no extra
    /// Bessel evaluation. Value and derivatives share one scale factor, which
    /// leaves the Newton and Halley ratios unchanged.
    fn derivatives(self, x: f64) -> Result<(f64, f64, f64), StrErr> {
        use KelvinFunction::*;
        let second = matches!(self, Ker | Kei | KerPrime | KeiPrime);
        let prime = matches!(self, BerPrime | BeiPrime | KerPrime | KeiPrime);
        let imaginary = matches!(self, Bei | Kei | BeiPrime | KeiPrime);
        let (value, derivative) = pair_needed(x, second, true, Need::Both)?;
        let i = Complex64::new(0.0, 1.0);
        let curvature = i * value - derivative / x;
        let (f, slope, curvature) = if prime {
            let third = i * derivative - curvature / x + derivative / (x * x);
            (derivative, curvature, third)
        } else {
            (value, derivative, curvature)
        };
        Ok(if imaginary {
            (f.im, slope.im, curvature.im)
        } else {
            (f.re, slope.re, curvature.re)
        })
    }
}

/// Halley iteration started at `x` and safeguarded by the sign-changing
/// bracket `[left, right]`, where `left_negative` is the sign of the function
/// at `left`. Steps leaving the bracket fall back to bisection.
fn bracketed_halley(
    function: KelvinFunction,
    mut left: f64,
    left_negative: bool,
    mut right: f64,
    mut x: f64,
) -> Result<f64, StrErr> {
    for _ in 0..64 {
        let (value, slope, curvature) = function.derivatives(x)?;
        if !value.is_finite() {
            return Err("Zero finder encountered a nonfinite value.");
        }
        if value == 0.0 || right - left <= 4.0 * f64::EPSILON * x.abs().max(1.0) {
            return Ok(x);
        }
        if value.is_sign_negative() == left_negative {
            left = x;
        } else {
            right = x;
        }
        let denominator = 2.0 * slope * slope - value * curvature;
        let step = if denominator != 0.0 && denominator.is_finite() {
            2.0 * value * slope / denominator
        } else {
            value / slope
        };
        let next = x - step;
        // At rounding level, a further evaluation cannot improve x. Accept it
        // even when rounding noise points the step just outside the bracket,
        // which would otherwise restart from a bisection.
        if step.abs() <= 2.0 * f64::EPSILON * x.abs() {
            return Ok(if next >= left && next <= right {
                next
            } else {
                x
            });
        }
        x = if next > left && next < right {
            next
        } else {
            0.5 * (left + right)
        };
    }
    Err("Zero finder did not converge.")
}

/// Fill a caller-owned slice with successive positive Kelvin zeros.
///
/// The origin is excluded. Uses a bracketed Halley iteration on scaled
/// kernels, starting from the spacing of the previous zeros. Evaluations do
/// not allocate. On failure, earlier entries remain populated.
/// An empty slice succeeds without evaluating any functions.
pub fn kelvin_zeros<T: Float>(function: KelvinFunction, zeros: &mut [T]) -> Result<(), StrErr> {
    let mut left = 0.5;
    // Previous zero and spacing, used to predict the next zero.
    let mut previous: Option<(f64, f64)> = None;
    for zero in zeros {
        let mut fl = function.value(left)?;
        // (right end, its value, Halley starting point)
        let mut bracket = None;
        if let Some((last, spacing)) = previous {
            // Spacings vary slowly (towards pi*sqrt(2)), so the previous one
            // predicts the next zero closely. The trial bracket
            // [last + 3.5, guess + 0.5] is shorter than the minimum spacing,
            // so a sign change there isolates exactly the next zero.
            let guess = last + spacing;
            let right = (guess + 0.5).min(left + 4.0);
            let fr = function.value(right)?;
            if fl.is_sign_negative() != fr.is_sign_negative() || fr == 0.0 {
                bracket = Some((right, fr, guess.clamp(left, right)));
            }
        }
        if bracket.is_none() {
            let mut right = left + 1.0;
            for _ in 0..32 {
                let fr = function.value(right)?;
                if fl.is_sign_negative() != fr.is_sign_negative() || fr == 0.0 {
                    bracket = Some((right, fr, 0.5 * (left + right)));
                    break;
                }
                left = right;
                fl = fr;
                right += 1.0;
            }
        }
        let (right, fr, start) = bracket.ok_or("kelvin_zeros: Could not bracket a root.")?;
        let x = if fl == 0.0 {
            left
        } else if fr == 0.0 {
            right
        } else {
            bracketed_halley(function, left, fl.is_sign_negative(), right, start)?
        };
        *zero = T::from(x).ok_or("kelvin_zeros: Cannot represent root.")?;
        let spacing = match previous {
            Some((last, _)) => x - last,
            None => core::f64::consts::PI * core::f64::consts::SQRT_2,
        };
        previous = Some((x, spacing));
        // Consecutive positive zeros of every Kelvin function and derivative
        // are at least 4.3 apart (asymptotically pi*sqrt(2)), so the next
        // scan can start 3.5 beyond this zero without skipping one.
        left = x + 3.5;
    }
    Ok(())
}
