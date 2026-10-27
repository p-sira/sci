//! Adapters and shared algorithms for the remaining real special functions.
use super::{cephes64 as c, specfun as s};
use num_complex::Complex64;
use std::f64::consts::{FRAC_PI_2, PI, SQRT_2};
use std::sync::LazyLock;

pub(super) fn gamma_ratio(a: f64, b: f64, extra: f64) -> f64 {
    (c::lgam(a) - c::lgam(b) + extra).exp()
        * super::gammasgn(a).unwrap_or(f64::NAN)
        * super::gammasgn(b).unwrap_or(f64::NAN)
}
pub(super) fn lgam_sgn(x: f64, sign: &mut [i32]) -> f64 {
    sign[0] = super::gammasgn(x).unwrap_or(f64::NAN) as i32;
    c::lgam(x)
}

// Fourier recurrence matrices from DLMF 28.4. The even-even first coefficient
// has weight two, so the symmetric basis uses sqrt(2)*A_0.
fn mathieu_spectrum(kd: i32, m: i32, q: f64, vectors: bool) -> (f64, Vec<f64>) {
    // Coefficient vectors keep their established length. Characteristic
    // values alone use a shorter truncation: Fourier coefficients beyond
    // m/2 + 4*sqrt(|q|) decay faster than (|q|/(4j^2))^j, so twenty extra
    // rows leave the eigenvalue unchanged at binary64 precision.
    let margin = if vectors { 48 } else { 20 };
    let size = (m as usize / 2 + margin + (4.0 * q.abs().sqrt()) as usize).min(251);
    let rank = match kd {
        1 | 2 => m / 2,
        3 => (m - 1) / 2,
        _ => m / 2 - 1,
    } as usize;
    let mut diagonal: Vec<_> = (0..size)
        .map(|j| {
            let frequency = match kd {
                1 => 2 * j,
                4 => 2 * j + 2,
                _ => 2 * j + 1,
            } as f64;
            frequency * frequency
        })
        .collect();
    if kd == 2 {
        diagonal[0] += q;
    }
    if kd == 3 {
        diagonal[0] -= q;
    }
    if q == 0.0 {
        let mut coefficients = vec![0.0; size];
        coefficients[rank] = if kd == 1 && rank == 0 {
            1.0 / SQRT_2
        } else {
            1.0
        };
        return (diagonal[rank], coefficients);
    }
    let mut off = vec![q; size - 1];
    if kd == 1 {
        off[0] *= SQRT_2;
    }
    // Sturm counts isolate only the requested eigenvalue of this symmetric
    // tridiagonal Fourier recurrence. Each pass also accumulates
    // d/dλ log det(T − λI), so once the bracket isolates the eigenvalue,
    // safeguarded Newton steps replace most bisection steps.
    let sturm = |shift: f64| {
        // A (nearly) zero pivot is perturbed to a tiny negative value before
        // it is counted, so the count and recurrence see the same sign.
        let mut pivot = diagonal[0] - shift;
        let mut slope = -1.0;
        let mut exact = pivot == 0.0;
        if pivot.abs() < 1e-300 {
            pivot = -1e-300;
        }
        let mut count = usize::from(pivot < 0.0);
        let mut log_derivative = slope / pivot;
        for j in 1..size {
            let ratio = off[j - 1] * off[j - 1] / pivot;
            slope = -1.0 + ratio * slope / pivot;
            pivot = diagonal[j] - shift - ratio;
            exact |= pivot == 0.0;
            if pivot.abs() < 1e-300 {
                pivot = -1e-300;
            }
            count += usize::from(pivot < 0.0);
            log_derivative += slope / pivot;
        }
        (count, log_derivative, exact)
    };
    // The off-diagonal part has 2-norm at most `radius`, so by Weyl's
    // inequality the requested eigenvalue lies within `radius` of the
    // diagonal entry of the same rank. Fall back to the Gershgorin interval
    // if rounding defeats that bracket.
    let radius = 2.0 * SQRT_2 * q.abs();
    let mut sorted = diagonal.clone();
    sorted.sort_unstable_by(f64::total_cmp);
    let (mut lower, mut upper) = (sorted[rank] - radius, sorted[rank] + radius);
    let (mut lower_count, mut lower_slope, _) = sturm(lower);
    let (mut upper_count, mut upper_slope, _) = sturm(upper);
    if lower_count > rank || upper_count <= rank {
        lower = sorted[0] - radius;
        upper = sorted[size - 1] + radius;
        (lower_count, lower_slope, _) = sturm(lower);
        (upper_count, upper_slope, _) = sturm(upper);
    }
    let newton = |at: f64, log_derivative: f64| at - 1.0 / log_derivative;
    let mut guess = if lower_count == rank && upper_count == rank + 1 {
        Some(newton(lower, lower_slope))
            .filter(|x| *x > lower && *x < upper)
            .or(Some(newton(upper, upper_slope)))
    } else {
        None
    };
    let mut converged = None;
    for _ in 0..128 {
        let midpoint = match guess.take() {
            Some(x) if x > lower && x < upper => x,
            _ => lower + 0.5 * (upper - lower),
        };
        if midpoint <= lower || midpoint >= upper {
            break;
        }
        let (count, log_derivative, exact) = sturm(midpoint);
        if count <= rank {
            lower = midpoint;
            lower_count = count;
        } else {
            upper = midpoint;
            upper_count = count;
        }
        if lower_count == rank && upper_count == rank + 1 && !exact && log_derivative.is_finite() {
            let next = newton(midpoint, log_derivative);
            if (next - midpoint).abs() <= 2.0 * f64::EPSILON * midpoint.abs().max(1.0)
                && next >= lower
                && next <= upper
            {
                // A tiny step can also come from cancellation near an
                // eigenvalue of a leading minor; confirm with Sturm counts.
                let delta = 4.0 * f64::EPSILON * next.abs().max(1.0);
                let (below, above) = (next - delta, next + delta);
                if sturm(below).0 <= rank && sturm(above).0 > rank {
                    converged = Some(next);
                    break;
                }
                // Not a verified eigenvalue: continue by bisection.
                continue;
            }
            guess = Some(next);
        }
    }
    let eigenvalue = converged.unwrap_or(lower + 0.5 * (upper - lower));
    if !vectors {
        return (eigenvalue, Vec::new());
    }
    // Inverse iteration computes its eigenvector with O(size) storage/work
    // per solve, avoiding a dense decomposition of all unused eigenpairs.
    let shift = eigenvalue + 8.0 * f64::EPSILON * eigenvalue.abs().max(1.0);
    let mut coefficients = vec![1.0; size];
    let mut pivots = vec![0.0; size];
    for _ in 0..4 {
        pivots[0] = diagonal[0] - shift;
        if pivots[0].abs() < 1e-300 {
            pivots[0] = 1e-300;
        }
        for j in 1..size {
            let factor = off[j - 1] / pivots[j - 1];
            pivots[j] = diagonal[j] - shift - factor * off[j - 1];
            if pivots[j].abs() < 1e-300 {
                pivots[j] = 1e-300;
            }
            coefficients[j] -= factor * coefficients[j - 1];
        }
        coefficients[size - 1] /= pivots[size - 1];
        for j in (0..size - 1).rev() {
            coefficients[j] = (coefficients[j] - off[j] * coefficients[j + 1]) / pivots[j];
        }
        let norm = coefficients.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !norm.is_finite() || norm == 0.0 {
            return (f64::NAN, vec![f64::NAN; size]);
        }
        coefficients.iter_mut().for_each(|x| *x /= norm);
    }
    if kd == 1 {
        coefficients[0] /= SQRT_2;
    }
    let phase = if coefficients.iter().sum::<f64>() < 0.0 {
        -1.0
    } else {
        1.0
    };
    coefficients.iter_mut().for_each(|x| *x *= phase);
    (eigenvalue, coefficients)
}
pub(super) fn cva2(kd: i32, m: i32, q: f64) -> f64 {
    mathieu_spectrum(kd, m, q, false).0
}
pub(super) fn fcoef(kd: i32, m: i32, q: f64, _a: f64, out: &mut [f64]) {
    let (_, coefficients) = mathieu_spectrum(kd, m, q, true);
    out.fill(0.0);
    for (a, b) in out.iter_mut().zip(coefficients) {
        *a = b;
    }
}
pub(crate) fn mathieu_a(m: u32, q: f64) -> f64 {
    cva2(if m.is_multiple_of(2) { 1 } else { 2 }, m as i32, q)
}
pub(crate) fn mathieu_b(m: u32, q: f64) -> f64 {
    if m == 0 {
        f64::NAN
    } else {
        cva2(if m.is_multiple_of(2) { 4 } else { 3 }, m as i32, q)
    }
}
fn angular(m: u32, q: f64, x: f64, odd: bool) -> (f64, f64) {
    if odd && m == 0 {
        return (0.0, 0.0);
    }
    let kd = if odd {
        if m.is_multiple_of(2) { 4 } else { 3 }
    } else if m.is_multiple_of(2) {
        1
    } else {
        2
    };
    let (_, coefficients) = mathieu_spectrum(kd, m as i32, q, true);
    let x = x.to_radians();
    // The coefficient vector has unit 2-norm (up to the A_0 weight), so terms
    // with |A_k|*max(k, 1) below 1e-20 cannot affect either sum at binary64
    // precision; skipping them avoids their trigonometric evaluations.
    coefficients
        .iter()
        .enumerate()
        .fold((0.0, 0.0), |(v, d), (j, &a)| {
            let k = match kd {
                1 => 2 * j,
                4 => 2 * j + 2,
                _ => 2 * j + 1,
            } as f64;
            if a.abs() * k.max(1.0) < 1e-20 {
                return (v, d);
            }
            let (sin, cos) = (k * x).sin_cos();
            if odd {
                (v + a * sin, d + a * k * cos)
            } else {
                (v + a * cos, d - a * k * sin)
            }
        })
}
pub(crate) fn mathieu_cem(m: u32, q: f64, x: f64) -> (f64, f64) {
    angular(m, q, x, false)
}
pub(crate) fn mathieu_sem(m: u32, q: f64, x: f64) -> (f64, f64) {
    angular(m, q, x, true)
}
fn modified(m: u32, q: f64, x: f64, odd: bool, kind: i32) -> (f64, f64) {
    if q <= 0.0 || !q.is_finite() || !x.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    let (mut f, mut d, mut g, mut e) = ([f64::NAN], [f64::NAN], [f64::NAN], [f64::NAN]);
    let status = s::mtu12(
        if odd { 2 } else { 1 },
        kind,
        m as i32,
        q,
        x,
        &mut f,
        &mut d,
        &mut g,
        &mut e,
    );
    if status != 0 {
        return (f64::NAN, f64::NAN);
    }
    if kind == 1 {
        (f[0], d[0])
    } else {
        (g[0], e[0])
    }
}
fn coefficient_count(m: u32, q: f64) -> usize {
    let qm = if q <= 1.0 {
        7.5 + 56.1 * q.sqrt() - 134.7 * q + 90.7 * q.sqrt() * q
    } else {
        17.0 + 3.1 * q.sqrt() - 0.126 * q + 0.0037 * q.sqrt() * q
    };
    (qm + 0.5 * m as f64) as usize
}
pub(crate) fn mathieu_even_coef(m: u32, q: f64) -> Result<Vec<f64>, &'static str> {
    let (_, mut out) = mathieu_spectrum(if m.is_multiple_of(2) { 1 } else { 2 }, m as i32, q, true);
    out.resize(coefficient_count(m, q), 0.0);
    Ok(out)
}
pub(crate) fn mathieu_odd_coef(m: u32, q: f64) -> Result<Vec<f64>, &'static str> {
    if m == 0 {
        return Err("odd Mathieu order must be positive");
    }
    let (_, mut out) = mathieu_spectrum(if m.is_multiple_of(2) { 4 } else { 3 }, m as i32, q, true);
    out.resize(coefficient_count(m, q), 0.0);
    Ok(out)
}

fn characteristic(m: u64, n: u64, c: f64, kd: i32) -> f64 {
    if !c.is_finite() {
        return f64::NAN;
    }
    let mut cv = [f64::NAN];
    let mut eg = vec![0.0; (n - m + 1) as usize];
    s::segv(m as i32, n as i32, c.abs(), kd, &mut cv, &mut eg);
    cv[0]
}
fn spheroidal(m: u64, n: u64, c: f64, cv: f64, x: f64, kd: i32, radial: i32) -> (f64, f64) {
    if !c.is_finite() || !cv.is_finite() || !x.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    let (mut f, mut d, mut g, mut e) = ([f64::NAN], [f64::NAN], [f64::NAN], [f64::NAN]);
    let c = c.abs();
    let status = if radial == 0 {
        if x.abs() > 1.0 {
            return (f64::NAN, f64::NAN);
        }
        s::aswfa(x, m as i32, n as i32, c, kd, cv, &mut f, &mut d)
    } else if kd == 1 {
        if x <= 1.0 {
            return (f64::NAN, f64::NAN);
        }
        s::rswfp(
            m as i32, n as i32, c, x, cv, radial, &mut f, &mut d, &mut g, &mut e,
        )
    } else {
        if x < 0.0 {
            return (f64::NAN, f64::NAN);
        }
        s::rswfo(
            m as i32, n as i32, c, x, cv, radial, &mut f, &mut d, &mut g, &mut e,
        )
    };
    if status != 0 {
        return (f64::NAN, f64::NAN);
    }
    if radial == 2 {
        (g[0], e[0])
    } else {
        (f[0], d[0])
    }
}

pub(crate) fn polygamma(n: u32, x: f64) -> f64 {
    if n == 0 {
        return super::digamma(x).unwrap_or(f64::NAN);
    }
    if x.is_nan() || x == f64::NEG_INFINITY {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return if n.is_multiple_of(2) { -0.0 } else { 0.0 };
    }
    if x <= 0.0 && x == x.floor() {
        return if n % 2 == 1 { f64::INFINITY } else { f64::NAN };
    }
    let factorial = c::gamma(n as f64 + 1.0);
    let mut a = x;
    let mut correction = 0.0;
    for _ in 0..100_000 {
        if a > 0.0 {
            break;
        }
        correction += (-1.0_f64).powi(n as i32 + 1) * factorial / a.powi(n as i32 + 1);
        a += 1.0;
    }
    if a <= 0.0 {
        return f64::NAN;
    }
    (-1.0_f64).powi(n as i32 + 1) * factorial * c::zeta(n as f64 + 1.0, a) + correction
}
pub(crate) fn multigammaln(a: f64, d: u32) -> f64 {
    d as f64 * (d as f64 - 1.0) * 0.25 * PI.ln()
        + (0..d).map(|j| c::lgam(a - j as f64 * 0.5)).sum::<f64>()
}
pub(crate) fn gdtrib(a: f64, p: f64, x: f64) -> f64 {
    if a <= 0.0 || x <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::INFINITY;
    }
    if p == 1.0 {
        return 0.0;
    }
    let (mut lo, mut hi) = (0.0, 1.0);
    // P(b, y) → 1 as b → 0+, so the residual at the origin is 1 − p.
    let (mut f_lo, mut f_hi) = (1.0 - p, f64::NAN);
    for _ in 0..1024 {
        f_hi = c::igam(hi, a * x) - p;
        if f_hi <= 0.0 {
            break;
        }
        lo = hi;
        f_lo = f_hi;
        hi *= 2.0;
    }
    // P(b, y) decreases in b; Brent's method converges superlinearly on the
    // bracket. Fall back to bisection if it meets a non-finite residual.
    if f_hi <= 0.0
        && let Some(root) = super::utils::brent_root(lo, hi, f_lo, f_hi, |b| c::igam(b, a * x) - p)
    {
        return root;
    }
    for _ in 0..128 {
        let mid = lo + (hi - lo) * 0.5;
        if c::igam(mid, a * x) > p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo + (hi - lo) * 0.5
}
pub(crate) fn hyp1f1(a: f64, b: f64, x: f64) -> f64 {
    if !a.is_finite() || !b.is_finite() || !x.is_finite() || a.abs() > 100_000.0 {
        return f64::NAN;
    }
    if a == 0.0 || x == 0.0 {
        return 1.0;
    }
    if a == b {
        return x.exp();
    }
    if b <= 0.0 && b == b.floor() && !(a <= 0.0 && a == a.floor() && a > b) {
        return f64::INFINITY;
    }
    // Nonpositive integer a terminates the defining hypergeometric series.
    // Evaluating that polynomial directly avoids the gamma poles introduced
    // by the general asymptotic formulas for negative x.
    if a < 0.0 && a == a.floor() {
        let (mut term, mut sum) = (1.0, 1.0);
        for k in 0..(-a) as usize {
            let k = k as f64;
            term *= (a + k) * x / ((b + k) * (k + 1.0));
            sum += term;
        }
        return sum;
    }
    // The Kummer-transformed series still has catastrophic cancellation when
    // b-a is large and negative. Continue the analytic solution outwards instead
    // of summing exponentially larger alternating terms at the target.
    if a > 0.0 && b < a && (-1000.0..0.0).contains(&x) && (-8.0..=64.0).contains(&b) {
        return super::kummer::negative_argument(a, b, x);
    }
    // In the moderate-argument region the defining series is more accurate
    // than the truncated large-x expansion. Kummer's transformation makes
    // the argument positive and avoids alternating exponential cancellation.
    if x.abs() <= 100.0 && a.abs().max(b.abs()) <= 64.0 {
        let (alpha, argument, scale) = if x < 0.0 {
            (b - a, -x, x.exp())
        } else {
            (a, x, 1.0)
        };
        let (mut term, mut sum, mut correction) = (1.0, 1.0, 0.0);
        for k in 0..1000 {
            let k = k as f64;
            term *= (alpha + k) * argument / ((b + k) * (k + 1.0));
            let adjusted = term - correction;
            let next = sum + adjusted;
            correction = (next - sum) - adjusted;
            sum = next;
            if term.abs() <= f64::EPSILON * sum.abs() {
                return scale * sum;
            }
        }
    }
    s::chgm(x, a, b)
}
pub(crate) fn hyp2f1(a: f64, b: f64, c: f64, x: f64) -> f64 {
    if [a, b, c, x].iter().any(|v| !v.is_finite()) || a.abs().max(b.abs()).max(c.abs()) > 100_000.0
    {
        return f64::NAN;
    }
    s::hyp2f1(a, b, c, x)
}
pub(crate) fn hypu(a: f64, b: f64, x: f64) -> f64 {
    if !a.is_finite()
        || !b.is_finite()
        || !x.is_finite()
        || x < 0.0
        || a.abs().max(b.abs()) > 100_000.0
    {
        return f64::NAN;
    }
    if x == 0.0 {
        return if b > 1.0 {
            f64::INFINITY
        } else {
            super::poch(1.0 - b + a, -a).unwrap_or(f64::NAN)
        };
    }
    if a == 0.0 {
        return 1.0;
    }
    // DLMF 13.3.7 avoids the ill-conditioned b=1 series near a=0.
    // Shift a into the stable range before calling the shared kernel.
    if b == 1.0 && x < 1.0 && a > -0.25 && a < 0.3 {
        return (x + 1.0 + 2.0 * a) * hypu(a + 1.0, 1.0, x)
            - (a + 1.0).powi(2) * hypu(a + 2.0, 1.0, x);
    }
    let mut digits = [0];
    let mut error = [0];
    s::chgu(x, a, b, &mut digits, &mut error)
}
pub(crate) fn pmv(m: i32, v: f64, x: f64) -> f64 {
    if !v.is_finite()
        || !x.is_finite()
        || m.unsigned_abs() > 4096
        || v.abs() > 100_000.0
        || x.abs() > 1.0
    {
        return f64::NAN;
    }
    let value = s::lpmv(x, m, v);
    // Zhang–Jin uses a finite sentinel at the singular endpoint.
    if value.abs() == 1e300 {
        f64::INFINITY.copysign(value)
    } else {
        value
    }
}
pub(crate) fn pbdv(v: f64, x: f64) -> (f64, f64) {
    cylinder(v, x, false)
}
pub(crate) fn pbvv(v: f64, x: f64) -> (f64, f64) {
    cylinder(v, x, true)
}
fn cylinder(v: f64, x: f64, second: bool) -> (f64, f64) {
    if !v.is_finite() || !x.is_finite() || v.abs() > 4096.0 {
        return (f64::NAN, f64::NAN);
    }
    let size = v.abs() as usize + 4;
    let mut values = vec![0.0; size];
    let mut slopes = vec![0.0; size];
    let (mut f, mut d) = ([f64::NAN], [f64::NAN]);
    if second {
        s::pbvv(x, v, &mut values, &mut slopes, &mut f, &mut d);
    } else {
        s::pbdv(x, v, &mut values, &mut slopes, &mut f, &mut d);
    }
    (f[0], d[0])
}
pub(crate) fn pbwa(a: f64, x: f64) -> (f64, f64) {
    if a.abs() > 5.0 || x.abs() > 5.0 || !a.is_finite() || !x.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    let g1 = super::complex::gamma(Complex64::new(0.25, 0.5 * a))
        .map(|v| v.norm())
        .unwrap_or(f64::NAN);
    let g2 = super::complex::gamma(Complex64::new(0.75, 0.5 * a))
        .map(|v| v.norm())
        .unwrap_or(f64::NAN);
    let f1 = (g1 / g2).sqrt();
    let f2 = (2.0 * g2 / g1).sqrt();
    // Entire power series of y''+(x²/4-a)y=0, with even/odd initial data.
    let mut coefficients = [0.0; 202];
    coefficients[0] = 1.0;
    coefficients[1] = 0.0;
    let mut odd = [0.0; 202];
    odd[1] = 1.0;
    for k in 0..200 {
        let denom = ((k + 1) * (k + 2)) as f64;
        coefficients[k + 2] = (a * coefficients[k]
            - if k >= 2 {
                0.25 * coefficients[k - 2]
            } else {
                0.0
            })
            / denom;
        odd[k + 2] = (a * odd[k] - if k >= 2 { 0.25 * odd[k - 2] } else { 0.0 }) / denom;
    }
    let eval = |c: &[f64]| {
        c.iter()
            .rev()
            .fold((0.0, 0.0), |(y, d), &t| (y * x + t, d * x + y))
    };
    let (ev, ed) = eval(&coefficients);
    let (ov, od) = eval(&odd);
    let scale = 2.0_f64.powf(-0.75);
    (scale * (f1 * ev - f2 * ov), scale * (f1 * ed - f2 * od))
}

pub(crate) fn prolate_segv(m: u64, n: u64, c: f64) -> f64 {
    characteristic(m, n, c, 1)
}
pub(crate) fn prolate_aswfa(m: u64, n: u64, c: f64, cv: f64, x: f64) -> (f64, f64) {
    spheroidal(m, n, c, cv, x, 1, 0)
}
pub(crate) fn prolate_aswfa_nocv(m: u64, n: u64, c: f64, x: f64) -> (f64, f64) {
    prolate_aswfa(m, n, c, characteristic(m, n, c, 1), x)
}
pub(crate) fn prolate_radial1(m: u64, n: u64, c: f64, cv: f64, x: f64) -> (f64, f64) {
    spheroidal(m, n, c, cv, x, 1, 1)
}
pub(crate) fn prolate_radial1_nocv(m: u64, n: u64, c: f64, x: f64) -> (f64, f64) {
    prolate_radial1(m, n, c, characteristic(m, n, c, 1), x)
}
pub(crate) fn prolate_radial2(m: u64, n: u64, c: f64, cv: f64, x: f64) -> (f64, f64) {
    spheroidal(m, n, c, cv, x, 1, 2)
}
pub(crate) fn prolate_radial2_nocv(m: u64, n: u64, c: f64, x: f64) -> (f64, f64) {
    prolate_radial2(m, n, c, characteristic(m, n, c, 1), x)
}

pub(crate) fn oblate_segv(m: u64, n: u64, c: f64) -> f64 {
    characteristic(m, n, c, -1)
}
pub(crate) fn oblate_aswfa(m: u64, n: u64, c: f64, cv: f64, x: f64) -> (f64, f64) {
    spheroidal(m, n, c, cv, x, -1, 0)
}
pub(crate) fn oblate_aswfa_nocv(m: u64, n: u64, c: f64, x: f64) -> (f64, f64) {
    oblate_aswfa(m, n, c, characteristic(m, n, c, -1), x)
}
pub(crate) fn oblate_radial1(m: u64, n: u64, c: f64, cv: f64, x: f64) -> (f64, f64) {
    spheroidal(m, n, c, cv, x, -1, 1)
}
pub(crate) fn oblate_radial1_nocv(m: u64, n: u64, c: f64, x: f64) -> (f64, f64) {
    oblate_radial1(m, n, c, characteristic(m, n, c, -1), x)
}
pub(crate) fn oblate_radial2(m: u64, n: u64, c: f64, cv: f64, x: f64) -> (f64, f64) {
    spheroidal(m, n, c, cv, x, -1, 2)
}
pub(crate) fn oblate_radial2_nocv(m: u64, n: u64, c: f64, x: f64) -> (f64, f64) {
    oblate_radial2(m, n, c, characteristic(m, n, c, -1), x)
}
pub(crate) fn mathieu_modcem1(m: u32, q: f64, x: f64) -> (f64, f64) {
    modified(m, q, x, false, 1)
}
pub(crate) fn mathieu_modcem2(m: u32, q: f64, x: f64) -> (f64, f64) {
    modified(m, q, x, false, 2)
}
pub(crate) fn mathieu_modsem1(m: u32, q: f64, x: f64) -> (f64, f64) {
    modified(m, q, x, true, 1)
}
pub(crate) fn mathieu_modsem2(m: u32, q: f64, x: f64) -> (f64, f64) {
    modified(m, q, x, true, 2)
}

pub(crate) fn airy(x: f64) -> [f64; 4] {
    let (a, b, c, d) = super::airy(x).unwrap_or((f64::NAN, f64::NAN, f64::NAN, f64::NAN));
    [a, b, c, d]
}
pub(crate) fn airy_scaled(x: f64) -> [f64; 4] {
    let (a, b, c, d) = super::airye(x).unwrap_or((f64::NAN, f64::NAN, f64::NAN, f64::NAN));
    [a, b, c, d]
}
pub(crate) fn airy_integrals(x: f64) -> [f64; 4] {
    if !x.is_finite() {
        return [f64::NAN; 4];
    }
    if x.abs() <= 10.0 {
        // Integrate y''=x*y's two power-series solutions. Double-double
        // summation retains oscillatory cancellation without quadrature.
        // Constants are Ai(0), -Ai'(0), and sqrt(3) (DLMF 9.2.3–6).
        use super::double_double::Dd;
        let c0 = Dd {
            hi: 0.3550280538878172,
            lo: 2.05233632436212e-17,
        };
        let c1 = Dd {
            hi: 0.2588194037928068,
            lo: -2.5222431116108322e-17,
        };
        let root3 = Dd {
            hi: 1.7320508075688772,
            lo: 1.0035084221806903e-16,
        };
        let integral = |x: f64| {
            let x = Dd::from(x);
            let cube = x * x * x;
            let (mut f, mut g) = (x, x * x / Dd::from(2.0));
            let (mut tf, mut tg) = (f, g);
            for k in 1..128 {
                let k = k as f64;
                // Integer denominators below 2^53 are exact in binary64.
                tf = tf * cube * Dd::from(3.0 * k - 2.0)
                    / Dd::from((3.0 * k + 1.0) * (3.0 * k) * (3.0 * k - 1.0));
                tg = tg * cube * Dd::from(3.0 * k - 1.0)
                    / Dd::from((3.0 * k + 2.0) * (3.0 * k) * (3.0 * k + 1.0));
                f = f + tf;
                g = g + tg;
                if tf.hi.abs().max(tg.hi.abs()) <= 1e-31 * f.hi.abs().max(g.hi.abs()) {
                    break;
                }
            }
            ((c0 * f - c1 * g).hi, (root3 * (c0 * f + c1 * g)).hi)
        };
        let positive = integral(x);
        let negative = integral(-x);
        return [positive.0, positive.1, -negative.0, -negative.1];
    }
    if x.abs() <= 16.0 {
        return [
            quadrature(0.0, x, |t| airy(t)[0]),
            quadrature(0.0, x, |t| airy(t)[2]),
            quadrature(0.0, x, |t| airy(-t)[0]),
            quadrature(0.0, x, |t| airy(-t)[2]),
        ];
    }
    let (mut a, mut b, mut c, mut d) = ([0.0], [0.0], [0.0], [0.0]);
    s::itairy_asymptotic(x.abs(), &mut a, &mut b, &mut c, &mut d);
    if x < 0.0 {
        [-c[0], -d[0], -a[0], -b[0]]
    } else {
        [a[0], b[0], c[0], d[0]]
    }
}
pub(crate) fn ellipj(u: f64, m: f64) -> (f64, f64, f64, f64) {
    super::ellipj(u, m).unwrap_or((f64::NAN, f64::NAN, f64::NAN, f64::NAN))
}
pub(crate) fn sici(x: f64) -> (f64, f64) {
    super::sici(x).unwrap_or((f64::NAN, f64::NAN))
}
pub(crate) fn shichi(x: f64) -> (f64, f64) {
    super::shichi(x).unwrap_or((f64::NAN, f64::NAN))
}
pub(crate) fn fresnel(x: f64) -> (f64, f64) {
    super::fresnel(x).unwrap_or((f64::NAN, f64::NAN))
}
pub(crate) fn smirnov(n: i32, x: f64) -> f64 {
    super::smirnov(n as u32, x).unwrap_or(f64::NAN)
}
pub(crate) fn smirnovi(n: i32, p: f64) -> f64 {
    super::smirnovi(n as u32, p).unwrap_or(f64::NAN)
}
pub(crate) fn assoc_legendre_p(n: i32, m: i32, x: f64) -> f64 {
    if n < 0 {
        f64::NAN
    } else {
        super::assoc_legendre_p(n as u32, m, x).unwrap_or(f64::NAN)
    }
}
pub(crate) fn sph_legendre_p(n: i32, m: i32, x: f64) -> f64 {
    if n < 0 {
        f64::NAN
    } else {
        super::sph_legendre_p(n as u32, m, x).unwrap_or(f64::NAN)
    }
}
pub(crate) fn legendre_p_all(n: usize, x: f64) -> Vec<f64> {
    super::orthogonal::legendre_all(n, x)
}
pub(crate) fn assoc_legendre_p_all(n: usize, m: usize, x: f64) -> Vec<Vec<f64>> {
    super::legendre::assoc_legendre_table(n, m, x)
}
pub(crate) fn sph_legendre_p_all(n: usize, m: usize, x: f64) -> Vec<Vec<f64>> {
    super::legendre::sph_legendre_table(n, m, x)
}
pub(crate) fn sph_harm_y_all(n: usize, m: usize, theta: f64, phi: f64) -> Vec<Vec<Complex64>> {
    super::legendre::sph_harm_table(n, m, theta, phi)
}
pub(crate) fn legendre_q_all(n: usize, x: f64) -> (Vec<f64>, Vec<f64>) {
    let mut values = vec![0.0; n + 1];
    let mut slopes = vec![0.0; n + 1];
    if x.is_finite() {
        s::lqmns(
            0,
            n as i32,
            if x < -1.0 { -x } else { x },
            &mut values,
            &mut slopes,
        );
        if x < -1.0 {
            for degree in 0..=n {
                let phase = if degree % 2 == 0 { -1.0 } else { 1.0 };
                values[degree] *= phase;
                slopes[degree] *= -phase;
            }
        }
    } else {
        values.fill(f64::NAN);
        slopes.fill(f64::NAN);
    }
    (values, slopes)
}
pub(crate) fn assoc_legendre_q_all(n: usize, m: usize, x: f64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut values = vec![vec![0.0; m + 1]; n + 1];
    let mut slopes = values.clone();
    for order in 0..=m {
        let mut v = vec![0.0; n + 1];
        let mut d = v.clone();
        if x.is_finite() {
            s::lqmns(
                order as i32,
                n as i32,
                if x < -1.0 { -x } else { x },
                &mut v,
                &mut d,
            );
        } else {
            v.fill(f64::NAN);
            d.fill(f64::NAN);
        }
        for degree in 0..=n {
            let phase = if x.abs() < 1.0 && order % 2 == 1 {
                -1.0
            } else {
                1.0
            };
            let reflection = if x < -1.0 {
                if (degree + order) % 2 == 0 { -1.0 } else { 1.0 }
            } else {
                1.0
            };
            values[degree][order] = phase * reflection * v[degree];
            slopes[degree][order] =
                phase * if x < -1.0 { -reflection } else { reflection } * d[degree];
        }
    }
    (values, slopes)
}

pub(crate) fn it1j0y0(x: f64) -> (f64, f64) {
    bessel_integrals(x, s::itjya, true)
}
pub(crate) fn it2j0y0(x: f64) -> (f64, f64) {
    bessel_integrals(x, s::ittjya, false)
}
pub(crate) fn it1i0k0(x: f64) -> (f64, f64) {
    bessel_integrals(x, s::itika, true)
}
pub(crate) fn it2i0k0(x: f64) -> (f64, f64) {
    bessel_integrals(x, s::ittika, false)
}
fn bessel_integrals(x: f64, kernel: fn(f64, &mut [f64], &mut [f64]), odd: bool) -> (f64, f64) {
    if !x.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    let (mut a, mut b) = ([0.0], [0.0]);
    kernel(x.abs(), &mut a, &mut b);
    if x < 0.0 {
        (if odd { -a[0] } else { a[0] }, f64::NAN)
    } else {
        (a[0], b[0])
    }
}
pub(crate) fn itstruve0(x: f64) -> f64 {
    if !x.is_finite() {
        f64::NAN
    } else if x.abs() <= 24.0 {
        // Integrate H_0's defining power series term by term. Double-double
        // arithmetic preserves the cancellation near the transition without
        // repeatedly evaluating H_0 at quadrature nodes.
        use super::double_double::Dd;
        let square = Dd::from(x) * Dd::from(x);
        let (mut term, mut sum) = (Dd::from(1.0), Dd::from(1.0));
        for k in 1..128 {
            let k = k as f64;
            // Integer denominators below 2^53 are exact in binary64.
            term = -term * square * Dd::from(k)
                / Dd::from((k + 1.0) * (2.0 * k + 1.0) * (2.0 * k + 1.0));
            sum = sum + term;
            if term.hi.abs() <= 1e-31 * sum.hi.abs() {
                break;
            }
        }
        (square * sum / Dd::from(PI)).hi
    } else if x.abs() <= 128.0 {
        quadrature(0.0, x.abs(), |t| {
            super::struve_h(0.0, t).unwrap_or(f64::NAN)
        })
    } else {
        s::itsh0(x.abs())
    }
}
pub(crate) fn itmodstruve0(x: f64) -> f64 {
    if !x.is_finite() {
        f64::NAN
    } else {
        s::itsl0(x.abs())
    }
}
pub(crate) fn it2struve0(x: f64) -> f64 {
    if !x.is_finite() {
        f64::NAN
    } else if x < 0.0 {
        PI - s::itth0(-x)
    } else {
        s::itth0(x)
    }
}
pub(crate) fn besselpoly(a: f64, lambda: f64, mut v: f64) -> f64 {
    if !a.is_finite() || !lambda.is_finite() || !v.is_finite() {
        return f64::NAN;
    }
    if a == 0.0 {
        return if v == 0.0 { 1.0 / (lambda + 1.0) } else { 0.0 };
    }
    let mut sign = 1.0;
    if v < 0.0 && v == v.floor() {
        v = -v;
        if v % 2.0 != 0.0 {
            sign = -1.0;
        }
    }
    let mut term = (v * a.ln()).exp() / (c::gamma(v + 1.0) * (lambda + v + 1.0));
    let mut sum = 0.0;
    for m in 0..1000 {
        sum += term;
        let m = m as f64;
        term *= -a * a * (lambda + v + 1.0 + 2.0 * m)
            / ((v + m + 1.0) * (m + 1.0) * (lambda + v + 3.0 + 2.0 * m));
        if term == 0.0 {
            break;
        }
    }
    sign * sum
}
pub(crate) fn jahnke_emden_lambda(v: f64, x: f64) -> (Vec<f64>, Vec<f64>) {
    let size = v.floor() as usize;
    let base = v.fract();
    (0..=size)
        .map(|i| {
            let degree = base + i as f64;
            let value = super::hyp0f1(degree + 1.0, -x * x * 0.25).unwrap_or(f64::NAN);
            let slope = -x / (2.0 * (degree + 1.0))
                * super::hyp0f1(degree + 2.0, -x * x * 0.25).unwrap_or(f64::NAN);
            (value, slope)
        })
        .unzip()
}

pub(crate) fn wofz(z: Complex64) -> Complex64 {
    super::wofz(z).unwrap_or(Complex64::new(f64::NAN, f64::NAN))
}
pub(crate) fn kelvin(x: f64) -> [Complex64; 4] {
    let (a, b, c, d) = super::kelvin(x).unwrap_or((
        Complex64::new(f64::NAN, f64::NAN),
        Complex64::new(f64::NAN, f64::NAN),
        Complex64::new(f64::NAN, f64::NAN),
        Complex64::new(f64::NAN, f64::NAN),
    ));
    [a, b, c, d]
}
fn modified_fresnel(x: f64, minus: bool) -> (Complex64, Complex64) {
    let z = Complex64::new(1.0, -1.0) * (x * std::f64::consts::FRAC_1_SQRT_2);
    let erfc = super::complex::erfc(z).unwrap_or(Complex64::new(f64::NAN, f64::NAN));
    let f = Complex64::new(1.0, 1.0) * (PI.sqrt() / (2.0 * SQRT_2)) * erfc;
    let k = Complex64::from_polar(1.0 / PI.sqrt(), -(x * x + PI / 4.0)) * f;
    if minus { (f.conj(), k.conj()) } else { (f, k) }
}
pub(crate) fn modified_fresnel_plus(x: f64) -> (Complex64, Complex64) {
    modified_fresnel(x, false)
}
pub(crate) fn modified_fresnel_minus(x: f64) -> (Complex64, Complex64) {
    modified_fresnel(x, true)
}
pub(crate) fn fresnel_zeros(nt: usize) -> Result<(Vec<Complex64>, Vec<Complex64>), &'static str> {
    let zeros = |sine: bool| {
        let mut out = Vec::with_capacity(nt);
        for i in 1..=nt {
            let p = if sine {
                2.0 * (i as f64).sqrt()
            } else {
                (4.0 * i as f64 - 1.0).sqrt()
            };
            let mut z = Complex64::new(
                p - (PI * p).ln() / (PI * PI * p.powi(3)),
                (PI * p).ln() / (PI * p),
            );
            let mut converged = false;
            for _ in 0..64 {
                let Ok((s, c)) = super::complex::fresnel(z) else {
                    break;
                };
                let phase = FRAC_PI_2 * z * z;
                let slope = if sine { phase.sin() } else { phase.cos() };
                let value = if sine { s } else { c };
                let step = value / slope;
                z -= step;
                if step.norm() <= 4.0 * f64::EPSILON * z.norm() {
                    converged = true;
                    break;
                }
            }
            if !converged || !z.re.is_finite() || !z.im.is_finite() {
                return Err("Fresnel zero refinement did not converge");
            }
            out.push(z);
        }
        Ok(out)
    };
    Ok((zeros(true)?, zeros(false)?))
}

// Log-positive defining series, with an explicit upper bound on total work.
pub(crate) fn log_wright_bessel(a: f64, b: f64, x: f64) -> f64 {
    if a < 0.0 || b < 0.0 || x < 0.0 || [a, b, x].iter().any(|v| !v.is_finite()) {
        return f64::NAN;
    }
    if a == 0.0 {
        return x - c::lgam(b);
    }
    if x == 0.0 {
        return -c::lgam(b);
    }
    let lx = x.ln();
    let mut scale = f64::NEG_INFINITY;
    let mut sum = 0.0;
    let mut previous = f64::NEG_INFINITY;
    for k in 0..100_000 {
        let k = k as f64;
        let parameter = a * k + b;
        let term = k * lx - c::lgam(k + 1.0) - c::lgam(parameter);
        if term == f64::NEG_INFINITY {
            continue;
        }
        if term > scale {
            sum *= (scale - term).exp();
            scale = term;
        }
        let relative = (term - scale).exp();
        sum += relative;
        if term < previous && relative <= f64::EPSILON * sum {
            return scale + sum.ln();
        }
        previous = term;
    }
    f64::NAN
}
pub(crate) fn wright_bessel(a: f64, b: f64, x: f64) -> f64 {
    log_wright_bessel(a, b, x).exp()
}

pub(crate) fn owens_t(h: f64, a: f64) -> f64 {
    if h.is_nan() || a.is_nan() {
        return f64::NAN;
    }
    if h.is_infinite() || a == 0.0 {
        return 0.0_f64.copysign(a);
    }
    let sign = a.signum();
    let a = a.abs();
    let h = h.abs();
    if h == 0.0 {
        return sign * a.atan() / (2.0 * PI);
    }
    if a > 1.0 {
        let ph = 0.5 * super::erfc(h / std::f64::consts::SQRT_2).unwrap_or(f64::NAN);
        let pa = 0.5 * super::erfc(a * h / std::f64::consts::SQRT_2).unwrap_or(f64::NAN);
        return sign * (0.5 * (ph + pa) - ph * pa - owens_t(a * h, 1.0 / a));
    }
    // Gauss–Legendre on atan(t): the denominator cancels under t=tan(theta).
    let upper = a.atan();
    let mut sum = 0.0;
    // The integrand's width in theta is about 1/h; the 32-point rule matches
    // the 64-point rule to rounding level for h <= 12 (checked against
    // independent Owen's T values), halving the exponential evaluations.
    let rule: &[(f64, f64)] = if h <= 12.0 { &*GAUSS_32 } else { &*GAUSS_64 };
    for &(z, weight) in rule {
        for theta in [0.5 * upper * (1.0 - z), 0.5 * upper * (1.0 + z)] {
            sum += weight * (-0.5 * (h / theta.cos()).powi(2)).exp();
        }
    }
    sign * upper * sum / (4.0 * PI)
}
pub(crate) fn hankel_1(v: f64, x: f64) -> Complex64 {
    super::cyl_hankel_1(v, Complex64::new(x, 0.0)).unwrap_or(Complex64::new(f64::NAN, f64::NAN))
}
pub(crate) fn hankel_1e(v: f64, x: f64) -> Complex64 {
    super::cyl_hankel_1e(v, Complex64::new(x, 0.0)).unwrap_or(Complex64::new(f64::NAN, f64::NAN))
}
pub(crate) fn hankel_2(v: f64, x: f64) -> Complex64 {
    super::cyl_hankel_2(v, Complex64::new(x, 0.0)).unwrap_or(Complex64::new(f64::NAN, f64::NAN))
}
pub(crate) fn hankel_2e(v: f64, x: f64) -> Complex64 {
    super::cyl_hankel_2e(v, Complex64::new(x, 0.0)).unwrap_or(Complex64::new(f64::NAN, f64::NAN))
}
pub(crate) fn ber_zeros(nt: usize) -> Result<Vec<f64>, &'static str> {
    let mut out = vec![0.0; nt];
    super::kelvin_zeros(super::KelvinFunction::Ber, &mut out)?;
    Ok(out)
}
pub(crate) fn bei_zeros(nt: usize) -> Result<Vec<f64>, &'static str> {
    let mut out = vec![0.0; nt];
    super::kelvin_zeros(super::KelvinFunction::Bei, &mut out)?;
    Ok(out)
}
pub(crate) fn ker_zeros(nt: usize) -> Result<Vec<f64>, &'static str> {
    let mut out = vec![0.0; nt];
    super::kelvin_zeros(super::KelvinFunction::Ker, &mut out)?;
    Ok(out)
}
pub(crate) fn kei_zeros(nt: usize) -> Result<Vec<f64>, &'static str> {
    let mut out = vec![0.0; nt];
    super::kelvin_zeros(super::KelvinFunction::Kei, &mut out)?;
    Ok(out)
}
pub(crate) fn berp_zeros(nt: usize) -> Result<Vec<f64>, &'static str> {
    let mut out = vec![0.0; nt];
    super::kelvin_zeros(super::KelvinFunction::BerPrime, &mut out)?;
    Ok(out)
}
pub(crate) fn beip_zeros(nt: usize) -> Result<Vec<f64>, &'static str> {
    let mut out = vec![0.0; nt];
    super::kelvin_zeros(super::KelvinFunction::BeiPrime, &mut out)?;
    Ok(out)
}
pub(crate) fn kerp_zeros(nt: usize) -> Result<Vec<f64>, &'static str> {
    let mut out = vec![0.0; nt];
    super::kelvin_zeros(super::KelvinFunction::KerPrime, &mut out)?;
    Ok(out)
}

// Nodes and weights depend only on the rule, so compute them once rather than
// repeating the Legendre root solves for every integral and subinterval.
static GAUSS_32: LazyLock<[(f64, f64); 16]> = LazyLock::new(gauss_rule::<16>);
static GAUSS_64: LazyLock<[(f64, f64); 32]> = LazyLock::new(gauss_rule::<32>);

fn gauss_rule<const HALF: usize>() -> [(f64, f64); HALF] {
    let order = 2 * HALF;
    std::array::from_fn(|i| {
        let mut z = (PI * (i as f64 + 0.75) / (order as f64 + 0.5)).cos();
        let mut derivative = 0.0;
        for _ in 0..10 {
            let (mut p0, mut p1) = (1.0, z);
            for n in 2..=order {
                let p = ((2 * n - 1) as f64 * z * p1 - (n - 1) as f64 * p0) / n as f64;
                p0 = p1;
                p1 = p;
            }
            derivative = order as f64 * (z * p1 - p0) / (z * z - 1.0);
            let step = p1 / derivative;
            z -= step;
            if step.abs() < f64::EPSILON {
                break;
            }
        }
        (z, 2.0 / ((1.0 - z * z) * derivative * derivative))
    })
}

// Composite 32-point Gaussian quadrature. Intervals are at most four units;
// callers bound their integration domains before entering this kernel.
fn quadrature(left: f64, right: f64, evaluate: impl Fn(f64) -> f64) -> f64 {
    let pieces = ((right - left).abs() / 4.0).ceil().max(1.0) as usize;
    let width = (right - left) / pieces as f64;
    let mut total = 0.0;
    for piece in 0..pieces {
        let midpoint = left + (piece as f64 + 0.5) * width;
        for &(z, weight) in &*GAUSS_32 {
            total += weight
                * width
                * 0.5
                * (evaluate(midpoint - z * width * 0.5) + evaluate(midpoint + z * width * 0.5));
        }
    }
    total
}

pub(crate) fn keip_zeros(nt: usize) -> Result<Vec<f64>, &'static str> {
    let mut out = vec![0.0; nt];
    super::kelvin_zeros(super::KelvinFunction::KeiPrime, &mut out)?;
    Ok(out)
}
