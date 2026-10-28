//! Adapters and shared algorithms for the remaining real special functions.
use super::{cephes64 as c, specfun as s};
use num_complex::Complex64;
use std::f64::consts::{FRAC_PI_2, PI, SQRT_2};

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
fn mathieu_spectrum(kd: i32, m: i32, q: f64, vector_margin: Option<usize>) -> (f64, Vec<f64>) {
    // Characteristic values use a short truncation. Beyond row
    // j = m/2 + 4*sqrt(|q|), (2j)^2 exceeds the eigenvalue by more than 60|q|,
    // so successive Fourier coefficients shrink by more than a factor of 60
    // and ten extra rows leave the eigenvalue unchanged at binary64 precision.
    // Coefficient vectors use `vector_margin` extra rows instead; their
    // inverse iteration reuses the short-truncation eigenvalue.
    let extent = m as usize / 2 + (4.0 * q.abs().sqrt()) as usize;
    let rank = match kd {
        1 | 2 => m / 2,
        3 => (m - 1) / 2,
        _ => m / 2 - 1,
    } as usize;
    // Row j has diagonal (first + 2j)^2, except that the first diagonal
    // entry of the odd-order recurrences is shifted by ±q. The matrix is
    // never stored by the eigenvalue search.
    let first = match kd {
        1 => 0.0,
        4 => 2.0,
        _ => 1.0,
    };
    let diagonal_zero = match kd {
        2 => first * first + q,
        3 => first * first - q,
        _ => first * first,
    };
    let diagonal_at = |j: usize| {
        if j == 0 {
            diagonal_zero
        } else {
            let frequency = first + 2.0 * j as f64;
            frequency * frequency
        }
    };
    // By Weyl's inequality the eigenvalue is at most `upper_bound`. Where
    // d_j − λ > 2|q|, the minimal solution of the three-term recurrence
    // shrinks by at least ρ_j = |q|/(d_j − λ − |q|) per row, so truncating
    // once q·(Π ρ_j)² < 1e-20 moves the eigenvalue by less than 1e-20; three
    // further rows are kept as a margin. Small |q| needs far fewer rows than
    // the fixed truncation, which remains an upper limit.
    let radius = 2.0 * SQRT_2 * q.abs();
    let size = {
        let fixed = (extent + 10).min(251);
        let upper_bound = diagonal_zero.max(diagonal_at(rank)) + radius;
        let mut decay = 1.0;
        let mut size = fixed;
        for j in rank + 1..fixed {
            let excess = diagonal_at(j) - upper_bound;
            if excess > 2.0 * q.abs() {
                decay *= q.abs() / (excess - q.abs());
                if decay * decay * q.abs() < 1e-20 {
                    size = (j + 4).min(fixed);
                    break;
                }
            }
        }
        size
    };
    let vector = |eigenvalue: f64| match vector_margin {
        Some(margin) => mathieu_vector(
            kd,
            q,
            rank,
            eigenvalue,
            (extent + margin).min(251),
            diagonal_at,
        ),
        None => (eigenvalue, Vec::new()),
    };
    if q == 0.0 {
        return vector(diagonal_at(rank));
    }
    let b2 = q * q;
    let b2_first = if kd == 1 {
        (q * SQRT_2) * (q * SQRT_2)
    } else {
        b2
    };
    // Sturm counts isolate only the requested eigenvalue of this symmetric
    // tridiagonal Fourier recurrence. Each pass also accumulates the first
    // two derivatives of log|det(T − λI)| for the Laguerre steps below. Each
    // pivot is inverted once and the reciprocal reused by the next row's
    // elimination, derivative recurrences and logarithmic-derivative terms.
    let sturm = |shift: f64| {
        // A (nearly) zero pivot is perturbed to a tiny negative value before
        // it is counted, so the count and recurrence see the same sign.
        let mut pivot = diagonal_zero - shift;
        let mut slope = -1.0;
        let mut curvature = 0.0;
        let mut exact = pivot == 0.0;
        if pivot.abs() < 1e-300 {
            pivot = -1e-300;
        }
        let mut count = usize::from(pivot < 0.0);
        let mut inverse = 1.0 / pivot;
        let mut gradient = slope * inverse;
        let mut hessian = gradient * gradient;
        let mut frequency = first;
        let mut off_squared = b2_first;
        for _ in 1..size {
            frequency += 2.0;
            let ratio = off_squared * inverse;
            off_squared = b2;
            let scale = ratio * inverse;
            curvature = scale * (curvature - 2.0 * slope * slope * inverse);
            slope = -1.0 + scale * slope;
            pivot = frequency * frequency - shift - ratio;
            exact |= pivot == 0.0;
            if pivot.abs() < 1e-300 {
                pivot = -1e-300;
            }
            count += usize::from(pivot < 0.0);
            inverse = 1.0 / pivot;
            let term = slope * inverse;
            gradient += term;
            hessian += term * term - curvature * inverse;
        }
        (count, gradient, hessian, exact)
    };
    // The eigenvalue count alone, for confirming a converged Laguerre step;
    // it repeats the pivot recurrence of `sturm` without the derivatives.
    let count_below = |shift: f64| {
        let mut pivot = diagonal_zero - shift;
        if pivot.abs() < 1e-300 {
            pivot = -1e-300;
        }
        let mut count = usize::from(pivot < 0.0);
        let mut frequency = first;
        let mut off_squared = b2_first;
        for _ in 1..size {
            frequency += 2.0;
            // Same rounding as `sturm`: multiply by the pivot reciprocal.
            pivot = frequency * frequency - shift - off_squared * (1.0 / pivot);
            off_squared = b2;
            if pivot.abs() < 1e-300 {
                pivot = -1e-300;
            }
            count += usize::from(pivot < 0.0);
        }
        count
    };
    // Rows j >= 1 are already in increasing order, so the k-th smallest
    // diagonal entry follows from the position of the first entry.
    let below_first = (1..size)
        .take_while(|&j| diagonal_at(j) < diagonal_zero)
        .count();
    let sorted = |k: usize| match k.cmp(&below_first) {
        std::cmp::Ordering::Less => diagonal_at(k + 1),
        std::cmp::Ordering::Equal => diagonal_zero,
        std::cmp::Ordering::Greater => diagonal_at(k),
    };
    // Laguerre's method for the real-rooted characteristic polynomial: from a
    // point with exactly `rank` eigenvalues below it, the upward step lands
    // between that point and the next eigenvalue (and symmetrically
    // downward), converging cubically and monotonically to the requested
    // eigenvalue. Here g = Σ 1/(λ-λᵢ) and h = Σ 1/(λ-λᵢ)².
    let degree = size as f64;
    let laguerre = |at: f64, g: f64, h: f64, upward: bool| {
        let root = ((degree - 1.0) * (degree * h - g * g)).max(0.0).sqrt();
        let denominator = if upward { g - root } else { g + root };
        at - degree / denominator
    };
    // Bisection safeguards the Laguerre steps inside a bracket whose lower
    // end has at most `rank` eigenvalues below it and whose upper end has
    // more; any step that leaves the bracket is replaced by its midpoint.
    let search = |mut lower: f64, mut upper: f64, start: Option<f64>| {
        let (mut lower_count, mut upper_count) = (0, size);
        let mut guess = start;
        for _ in 0..128 {
            let midpoint = match guess.take() {
                Some(x) if x > lower && x < upper => x,
                _ => lower + 0.5 * (upper - lower),
            };
            if midpoint <= lower || midpoint >= upper {
                break;
            }
            let (count, g, h, exact) = sturm(midpoint);
            if count <= rank {
                lower = midpoint;
                lower_count = count;
            } else {
                upper = midpoint;
                upper_count = count;
            }
            let upward = count == rank;
            if (upward || count == rank + 1) && !exact && g.is_finite() && h.is_finite() {
                let next = laguerre(midpoint, g, h, upward);
                // Laguerre converges cubically, and eigenvalues of the same
                // recurrence are at least about four apart, so a step δ leaves
                // an error of order δ³/16. Once δ³ is below a quarter ulp of
                // the eigenvalue, `next` needs no further Laguerre pass.
                let step = (next - midpoint).abs();
                let scale = next.abs().max(1.0);
                if (step <= 2.0 * f64::EPSILON * midpoint.abs().max(1.0)
                    || step * step * step <= 0.25 * f64::EPSILON * scale)
                    && next >= lower
                    && next <= upper
                {
                    // A tiny step can also come from cancellation near an
                    // eigenvalue of a leading minor; confirm with Sturm
                    // counts. A side already bracketed by this or an earlier
                    // pass needs no extra count, because counts are monotone.
                    // Rounding perturbs the pivots by about ε·max(|λ|, |q|),
                    // so the counts cannot confirm a narrower interval.
                    let delta = 4.0 * f64::EPSILON * next.abs().max(q.abs()).max(1.0);
                    let (below, above) = (next - delta, next + delta);
                    if ((below <= lower && lower_count <= rank) || count_below(below) <= rank)
                        && ((above >= upper && upper_count > rank) || count_below(above) > rank)
                    {
                        return Ok(next);
                    }
                    // Not a verified eigenvalue: continue by bisection.
                    continue;
                }
                guess = Some(next);
            }
        }
        Err(lower + 0.5 * (upper - lower))
    };
    // The off-diagonal part has 2-norm at most `radius`, so by Weyl's
    // inequality the requested eigenvalue lies within `radius` of the
    // diagonal entry of the same rank. Fall back to the Gershgorin interval
    // if rounding defeats that bracket. The first pass starts from the DLMF
    // small- or large-q expansion instead of the bracket midpoint; a guess
    // outside the bracket is ignored.
    let start = mathieu_guess(kd, m, q);
    let eigenvalue = search(sorted(rank) - radius, sorted(rank) + radius, start)
        .or_else(|_| search(sorted(0) - radius, sorted(size - 1) + radius, None))
        .unwrap_or_else(|midpoint| midpoint);
    vector(eigenvalue)
}
// Starting estimate of the characteristic value for the recurrence `kd`
// (1: a_{2k}, 2: a_{2k+1}, 3: b_{2k+1}, 4: b_{2k+2}). Small |q| uses the power
// series of DLMF 28.6.1-28.6.14 and large |q| the asymptotic expansion of
// DLMF 28.8.1; a_{2k}(-q) = a_{2k}(q), b_{2k}(-q) = b_{2k}(q) and
// a_{2k+1}(-q) = b_{2k+1}(q) reduce negative q. The estimate only seeds the
// safeguarded search, so it need not be accurate in the transition region.
fn mathieu_guess(kd: i32, m: i32, q: f64) -> Option<f64> {
    let (kd, q) = if q < 0.0 {
        (
            match kd {
                2 => 3,
                3 => 2,
                kd => kd,
            },
            -q,
        )
    } else {
        (kd, q)
    };
    if !q.is_finite() || m < 0 {
        return None;
    }
    let mf = m as f64;
    let q2 = q * q;
    // Approximate |q| where the large-q expansion becomes the closer
    // estimate, located on a grid of exact eigenvalues; about 0.9·m² for
    // larger orders.
    let small_q_limit = match (kd, m) {
        (1, 0) => 1.6,
        (2, 1) => 4.6,
        (3, 1) => 3.0,
        (1, 2) => 2.0,
        (4, 2) => 6.2,
        (2, 3) => 8.5,
        (3, 3) => 5.2,
        (_, 4) => 9.1,
        (2, 5) => 25.0,
        (3, 5) => 16.0,
        _ => 0.9 * mf * mf,
    };
    if q <= small_q_limit {
        let q4 = q2 * q2;
        let guess = match (kd, m) {
            (1, 0) => {
                q2 * (-0.5 + q2 * (7.0 / 128.0 + q2 * (-29.0 / 2304.0 + q2 * 68687.0 / 18874368.0)))
            }
            (2 | 3, 1) => {
                let sign = if kd == 2 { 1.0 } else { -1.0 };
                let p = sign * q;
                1.0 + p - q2 / 8.0 - p * q2 / 64.0 - q4 / 1536.0
                    + 11.0 * p * q4 / 36864.0
                    + 49.0 * q4 * q2 / 589824.0
                    - 55.0 * p * q4 * q2 / 9437184.0
            }
            (1, 2) => {
                4.0 + q2 * (5.0 / 12.0 + q2 * (-763.0 / 13824.0 + q2 * 1002401.0 / 79626240.0))
            }
            (4, 2) => 4.0 + q2 * (-1.0 / 12.0 + q2 * (5.0 / 13824.0 - q2 * 289.0 / 79626240.0)),
            (2 | 3, 3) => {
                let sign = if kd == 2 { 1.0 } else { -1.0 };
                let p = sign * q;
                9.0 + q2 / 16.0 + p * q2 / 64.0 + 13.0 * q4 / 20480.0 - 5.0 * p * q4 / 16384.0
            }
            (1, 4) => 16.0 + q2 / 30.0 + 433.0 * q4 / 864000.0,
            (4, 4) => 16.0 + q2 / 30.0 - 317.0 * q4 / 864000.0,
            _ => {
                let m2 = mf * mf;
                let d1 = m2 - 1.0;
                let d4 = m2 - 4.0;
                let mut guess =
                    m2 + q2 / (2.0 * d1) + (5.0 * m2 + 7.0) * q4 / (32.0 * d1 * d1 * d1 * d4);
                if m >= 7 {
                    guess += (9.0 * m2 * m2 + 58.0 * m2 + 29.0) * q4 * q2
                        / (64.0 * d1.powi(5) * d4 * (m2 - 9.0));
                }
                guess
            }
        };
        return Some(guess);
    }
    let s = if kd <= 2 {
        2.0 * mf + 1.0
    } else {
        2.0 * mf - 1.0
    };
    let h = q.sqrt();
    let s2 = s * s;
    Some(
        -2.0 * q + 2.0 * s * h
            - (s2 + 1.0) / 8.0
            - s * (s2 + 3.0) / (128.0 * h)
            - (5.0 * s2 * s2 + 34.0 * s2 + 9.0) / (4096.0 * q)
            - s * (33.0 * s2 * s2 + 410.0 * s2 + 405.0) / (131072.0 * q * h)
            - (63.0 * s2 * s2 * s2 + 1260.0 * s2 * s2 + 2943.0 * s2 + 486.0) / (8388608.0 * q2)
            - s * (527.0 * s2 * s2 * s2 + 15617.0 * s2 * s2 + 69001.0 * s2 + 41607.0)
                / (536870912.0 * q2 * h),
    )
}
// Inverse iteration computes the eigenvector of the Fourier recurrence with
// O(size) storage/work per solve, avoiding a dense decomposition of all unused
// eigenpairs. The shifted factorization does not change between solves.
fn mathieu_vector(
    kd: i32,
    q: f64,
    rank: usize,
    eigenvalue: f64,
    size: usize,
    diagonal_at: impl Fn(usize) -> f64,
) -> (f64, Vec<f64>) {
    let off_at = |j: usize| if kd == 1 && j == 0 { q * SQRT_2 } else { q };
    if q == 0.0 {
        let mut coefficients = vec![0.0; size];
        coefficients[rank] = if kd == 1 && rank == 0 {
            1.0 / SQRT_2
        } else {
            1.0
        };
        return (eigenvalue, coefficients);
    }
    let shift = eigenvalue + 8.0 * f64::EPSILON * eigenvalue.abs().max(1.0);
    let mut pivots = vec![0.0; size];
    let mut factors = vec![0.0; size];
    pivots[0] = diagonal_at(0) - shift;
    if pivots[0].abs() < 1e-300 {
        pivots[0] = 1e-300;
    }
    for j in 1..size {
        let off = off_at(j - 1);
        factors[j] = off / pivots[j - 1];
        pivots[j] = diagonal_at(j) - shift - factors[j] * off;
        if pivots[j].abs() < 1e-300 {
            pivots[j] = 1e-300;
        }
    }
    let mut coefficients = vec![1.0; size];
    for _ in 0..4 {
        for j in 1..size {
            coefficients[j] -= factors[j] * coefficients[j - 1];
        }
        coefficients[size - 1] /= pivots[size - 1];
        for j in (0..size - 1).rev() {
            coefficients[j] = (coefficients[j] - off_at(j) * coefficients[j + 1]) / pivots[j];
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
/// Extra rows of the established coefficient-vector truncation.
const COEFFICIENT_MARGIN: usize = 48;
pub(super) fn cva2(kd: i32, m: i32, q: f64) -> f64 {
    mathieu_spectrum(kd, m, q, None).0
}
pub(super) fn fcoef(kd: i32, m: i32, q: f64, _a: f64, out: &mut [f64]) {
    let (_, coefficients) = mathieu_spectrum(kd, m, q, Some(COEFFICIENT_MARGIN));
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
    // Coefficients more than 16 rows beyond the significant ones are below
    // 1e-28 (see `mathieu_spectrum`) and cannot affect either sum.
    let (_, coefficients) = mathieu_spectrum(kd, m as i32, q, Some(16));
    let x = x.to_radians();
    // cos(kx) and sin(kx) for k = first + 2j follow from one rotation by 2x
    // per term instead of a trigonometric evaluation per term. The
    // coefficient vector has unit 2-norm (up to the A_0 weight), so terms
    // with |A_k|*max(k, 1) below 1e-20 cannot affect either sum.
    let first = match kd {
        1 => 0.0,
        4 => 2.0,
        _ => 1.0,
    };
    let (step_sin, step_cos) = (2.0 * x).sin_cos();
    let (mut sin, mut cos) = (first * x).sin_cos();
    let mut k = first;
    let (mut value, mut derivative) = (0.0, 0.0);
    for &a in &coefficients {
        if a.abs() * k.max(1.0) >= 1e-20 {
            if odd {
                value += a * sin;
                derivative += a * k * cos;
            } else {
                value += a * cos;
                derivative -= a * k * sin;
            }
        }
        (sin, cos) = (
            sin * step_cos + cos * step_sin,
            cos * step_cos - sin * step_sin,
        );
        k += 2.0;
    }
    (value, derivative)
}
pub(crate) fn mathieu_cem(m: u32, q: f64, x: f64) -> (f64, f64) {
    angular(m, q, x, false)
}
pub(crate) fn mathieu_sem(m: u32, q: f64, x: f64) -> (f64, f64) {
    angular(m, q, x, true)
}
fn modified(m: u32, q: f64, x: f64, odd: bool, kind: i32) -> (f64, f64) {
    // The odd functions start at order one, as for `mathieu_b`.
    if q <= 0.0 || !q.is_finite() || !x.is_finite() || (odd && m == 0) {
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
    let (_, mut out) = mathieu_spectrum(
        if m.is_multiple_of(2) { 1 } else { 2 },
        m as i32,
        q,
        Some(COEFFICIENT_MARGIN),
    );
    out.resize(coefficient_count(m, q), 0.0);
    Ok(out)
}
pub(crate) fn mathieu_odd_coef(m: u32, q: f64) -> Result<Vec<f64>, &'static str> {
    if m == 0 {
        return Err("odd Mathieu order must be positive");
    }
    let (_, mut out) = mathieu_spectrum(
        if m.is_multiple_of(2) { 4 } else { 3 },
        m as i32,
        q,
        Some(COEFFICIENT_MARGIN),
    );
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
    let y = a * x;
    if let Some(root) = gdtrib_from_guess(y, p) {
        return root;
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
/// Solves P(b, y) = p for b by bracketing around an approximate start and
/// refining with Brent's method.
///
/// Gamma(b) has mean and variance b, so b + z√b = y with z = Φ⁻¹(p) gives a
/// first estimate, refined with the Wilson–Hilferty approximation. The
/// bracket grows geometrically from there. P(b, y) decreases
/// in b, so the root is unique. Returns `None` if no finite bracket is found
/// within the stepping budget, leaving the caller's search from the origin.
fn gdtrib_from_guess(y: f64, p: f64) -> Option<f64> {
    let z = c::ndtri(p);
    let root_b = 0.5 * (-z + (z * z + 4.0 * y).sqrt());
    let mut start = root_b * root_b;
    if !start.is_finite() || start <= 1e-3 {
        return None;
    }
    // Refine with the Wilson–Hilferty approximation
    // 3√b ((y/b)^{1/3} − 1 + 1/(9b)) = z, which is more accurate in the tails.
    let wilson_hilferty = |b: f64| 3.0 * b.sqrt() * ((y / b).cbrt() - 1.0 + 1.0 / (9.0 * b)) - z;
    for _ in 0..4 {
        let h = 1e-6 * start;
        let derivative = (wilson_hilferty(start + h) - wilson_hilferty(start - h)) / (2.0 * h);
        let next = start - wilson_hilferty(start) / derivative;
        if !next.is_finite() || next <= 0.5 * start || next >= 2.0 * start {
            break;
        }
        start = next;
    }
    let residual = |b: f64| c::igam(b, y) - p;
    let mut a = start;
    let mut ra = residual(a);
    if ra == 0.0 {
        return Some(a);
    }
    if !ra.is_finite() {
        return None;
    }
    // Residual above zero: the root lies at larger b.
    let up = ra > 0.0;
    let mut factor = 1.0 + 0.05 / (start + 1.0).sqrt();
    for _ in 0..64 {
        let b = if up { a * factor } else { a / factor };
        if !(1e-3..=1e300).contains(&b) {
            return None;
        }
        let rb = residual(b);
        if !rb.is_finite() {
            return None;
        }
        if rb == 0.0 {
            return Some(b);
        }
        if (rb > 0.0) != (ra > 0.0) {
            let (lo, hi, rl, rh) = if up { (a, b, ra, rb) } else { (b, a, rb, ra) };
            return super::utils::brent_root(lo, hi, rl, rh, residual);
        }
        a = b;
        ra = rb;
        factor *= factor;
    }
    None
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
    // At b=0 the legacy dispatcher can leave its initialized zero for
    // negative a, or accept a cancellation-damaged approximation near a=0.
    // Kummer's U(a,0,x)=x U(a+1,2,x) supplies a regular evaluation. Retain
    // the original path if intermediate rescaling exceeds binary64 range.
    if b == 0.0 {
        let transformed = x * hypu(a + 1.0, 2.0, x);
        if transformed.is_finite() {
            return transformed;
        }
    }
    // DLMF 13.3.7 avoids the ill-conditioned integer-b series near a=0
    // for b=1 and near a=1 for b=2.
    // Shift a into the stable range before calling the shared kernel.
    if (b == 1.0 && x < 1.0 && a > -0.25 && a < 0.3) || (b == 2.0 && x < 1.0 && a > 1.0 && a < 2.0)
    {
        return (x + 2.0 + 2.0 * a - b) * hypu(a + 1.0, b, x)
            - (a + 1.0) * (a + 2.0 - b) * hypu(a + 2.0, b, x);
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
        // The terms at -x are those at x with signs (-1)^(k+1) for f and
        // (-1)^k for g, so sum even and odd k separately and evaluate both
        // arguments from one pass over the terms.
        let x = Dd::from(x);
        let cube = x * x * x;
        let (mut tf, mut tg) = (x, (x * x).mul_f64(0.5));
        let (mut f_even, mut f_odd) = (tf, Dd::from(0.0));
        let (mut g_even, mut g_odd) = (tg, Dd::from(0.0));
        for n in 1..128u32 {
            let k = f64::from(n);
            // Integer factors below 2^53 are exact in binary64.
            tf = (tf * cube)
                .mul_f64(3.0 * k - 2.0)
                .div_f64((3.0 * k + 1.0) * (3.0 * k) * (3.0 * k - 1.0));
            tg = (tg * cube)
                .mul_f64(3.0 * k - 1.0)
                .div_f64((3.0 * k + 2.0) * (3.0 * k) * (3.0 * k + 1.0));
            // For fixed parity every term shares the sign of x^(3n), so
            // the cheaper same-sign addition is exact to double-double precision.
            if n % 2 == 0 {
                f_even = f_even.add_same_sign(tf);
                g_even = g_even.add_same_sign(tg);
            } else {
                f_odd = f_odd.add_same_sign(tf);
                g_odd = g_odd.add_same_sign(tg);
            }
            let scale = |f: f64, g: f64| f.abs().max(g.abs());
            let positive = scale(f_even.hi + f_odd.hi, g_even.hi + g_odd.hi);
            let negative = scale(f_odd.hi - f_even.hi, g_even.hi - g_odd.hi);
            // Once the next term ratios, at most |x|³/((3k+3)(3k+2)), are
            // below one half, each tail is at most twice its current term,
            // and terms below 1e-20 of the sums are beyond binary64 resolution.
            let tolerance = if 2.0 * (cube.hi.abs()) < (3.0 * k + 3.0) * (3.0 * k + 2.0) {
                1e-20
            } else {
                1e-31
            };
            if tf.hi.abs().max(tg.hi.abs()) <= tolerance * positive.min(negative) {
                break;
            }
        }
        let integral = |f: Dd, g: Dd| ((c0 * f - c1 * g).hi, (root3 * (c0 * f + c1 * g)).hi);
        let positive = integral(f_even + f_odd, g_even + g_odd);
        let negative = integral(f_odd - f_even, g_even - g_odd);
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
    let (first, mut second) = bessel_integrals(x, s::ittika, false);
    if x >= 4.0 && x.is_finite() {
        // Shift the defining K tail: exp(-x) ∫ exp(-u) K0e(x+u)/(x+u) du.
        // The old eight-term asymptotic loses ~2e-6 relatively at x=19.9;
        // the small-x series also cancels before its x=12 transition.
        // A single 32-point Gaussian rule resolves the smooth shifted
        // integrand on [0,48]; the omitted tail is below 1e-20 relatively.
        // The nodes and their exp(-u) weights do not depend on x, so they
        // are tabulated as (u, 24 w exp(-u)), sorted by u. The six nodes beyond
        // u = 44 have weights below 1e-19 and K0e(x+u)/(x+u) only decreases,
        // so together they change the sum by under 1e-19 relatively and are
        // dropped; a unit test re-derives the table.
        const NODES: [(f64, f64); 26] = [
            (0.06566731561244232, 0.15774056799892153),
            (0.34532372291355884, 0.2765310482031931),
            (0.8457058658998466, 0.2615916500409001),
            (1.562254177494248, 0.17246299683299984),
            (2.48829226161475, 0.08538223504570523),
            (3.6151772704183207, 0.03293922306990191),
            (4.932388896769383, 0.010153654031119359),
            (6.427629150233047, 0.002553288646578059),
            (8.086937593674834, 0.0005339642208046494),
            (9.894821826221705, 9.464933659751057e-5),
            (11.834402185626494, 1.449782460042314e-5),
            (13.887569372864753, 1.9573964074942813e-6),
            (16.035153545228937, 2.3774039031616317e-7),
            (18.25710330594871, 2.6525304499069184e-8),
            (20.532672922012885, 2.777272208722698e-9),
            (22.840616023494277, 2.7884665049272796e-10),
            (25.159383976505723, 2.7437043088658582e-11),
            (27.467327077987115, 2.7035625036831377e-12),
            (29.74289669405129, 2.7254843297461865e-13),
            (31.964846454771067, 2.870286460342419e-14),
            (34.11243062713525, 3.2220564647021825e-15),
            (36.1655978143735, 3.930056930103593e-16),
            (38.1051781737783, 5.30292494105746e-17),
            (39.91306240632517, 8.046402826858624e-18),
            (41.572370849766955, 1.3929403727552428e-18),
            (43.06761110323062, 2.784234374611805e-19),
        ];
        second = 0.0;
        for &(u, weight) in &NODES {
            second += weight * c::k0e(x + u) / (x + u);
        }
        second *= (-x).exp();
    }
    (first, second)
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
        // Terms alternate in sign; summing even and odd k separately keeps
        // each partial sum single-signed, so the cheaper same-sign addition
        // is exact to double-double precision. One full addition combines them.
        let mut term = Dd::from(1.0);
        let (mut even, mut odd) = (term, Dd::from(0.0));
        for n in 1..128u32 {
            let k = f64::from(n);
            // Integer denominators below 2^53 are exact in binary64.
            term = (term * square)
                .mul_f64(-k)
                .div_f64((k + 1.0) * (2.0 * k + 1.0) * (2.0 * k + 1.0));
            if n % 2 == 0 {
                even = even.add_same_sign(term);
            } else {
                odd = odd.add_same_sign(term);
            }
            // Once the next ratio x²(k+1)/((k+2)(2k+3)²) is below one half,
            // the tail is at most twice the current term, and terms below
            // 1e-20 of the sum are beyond binary64 resolution.
            let tolerance = if x * x < (k + 2.0) * (2.0 * k + 3.0) {
                1e-20
            } else {
                1e-31
            };
            if term.hi.abs() <= tolerance * (even.hi + odd.hi).abs() {
                break;
            }
        }
        (square * (even + odd) / Dd::from(PI)).hi
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
    // Λ_ν(x) = 0F1(; ν + 1; -x²/4) and Λ_ν′(x) = -x Λ_{ν+1}(x) / (2(ν + 1)),
    // so each derivative reuses the value at the next order.
    let z = -x * x * 0.25;
    let lambda = |i: usize| super::hyp0f1(base + i as f64 + 1.0, z).unwrap_or(f64::NAN);
    let mut values = vec![0.0; size + 2];
    values[size + 1] = lambda(size + 1);
    values[size] = lambda(size);
    // Lower orders follow from J's three-term recurrence (DLMF 10.6.1),
    // Λ_(ν-1) = Λ_ν + z Λ_(ν+1) / (ν(ν+1)), run downwards. J is the minimal
    // solution for orders beyond the turning point ν ≈ x, where the backward
    // direction is strongly stable. Closer to or below x the recurrence is
    // only neutrally stable and loses relative accuracy near zeros, so those
    // orders are evaluated directly. Above x = 50 the anchors come from the
    // Bessel path and the recurrence gains little, so it is not used there.
    // Double-double steps keep the rounding of up to 256 steps from
    // accumulating, so recurred orders are as accurate as the two anchors.
    use super::double_double::Dd;
    let recur_from = if x <= 50.0 {
        x + 2.0 * x.cbrt()
    } else {
        f64::INFINITY
    };
    let (mut upper, mut current) = (Dd::from(values[size + 1]), Dd::from(values[size]));
    for i in (0..size).rev() {
        let order = base + (i + 1) as f64;
        if order - 1.0 >= recur_from {
            let next =
                current + (upper * Dd::from(z)).div_dd(Dd::from(order) * Dd::from(order + 1.0));
            upper = current;
            current = next;
            values[i] = next.hi;
        } else {
            values[i] = lambda(i);
            upper = current;
            current = Dd::from(values[i]);
        }
    }
    (0..=size)
        .map(|i| {
            let degree = base + i as f64;
            (values[i], -x / (2.0 * (degree + 1.0)) * values[i + 1])
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
    // T(h, a) = exp(-h^2/2)/(2π) ∫_0^a exp(-h^2 t^2/2)/(1 + t^2) dt by
    // Gauss–Legendre in t. For a <= 1 the poles at ±i are far enough from
    // [0, a] that the rational factor converges beyond binary64 precision,
    // and no trigonometric evaluations are needed at the nodes.
    // The Gaussian's width is about 1/h; the 32-point rule matches the
    // 64-point rule to rounding level for h <= 12 (checked against
    // independent Owen's T values), halving the exponential evaluations.
    // With t = a(1+z)/2 the Gaussian factor is exp(-(ha)²(1+z)²/8); for
    // ha <= 4 it and the poles at ±i (Bernstein ellipse radius above 4.6
    // for a <= 1) are resolved by the 16-point rule beyond binary64
    // precision, again halving the exponentials.
    let rule: &[(f64, f64)] = if h * a <= 4.0 {
        &GAUSS_16
    } else if h <= 12.0 {
        &GAUSS_32
    } else {
        &GAUSS_64
    };
    let half = 0.5 * a;
    let scale = -0.5 * h * h;
    if scale == f64::NEG_INFINITY {
        return 0.0_f64.copysign(sign);
    }
    let mut sum = 0.0;
    for &(z, weight) in rule {
        for t in [half * (1.0 - z), half * (1.0 + z)] {
            let square = t * t;
            sum += weight * (scale * square).exp() / (1.0 + square);
        }
    }
    sign * half * scale.exp() * sum / (2.0 * PI)
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

// Positive Gauss–Legendre nodes and weights for the 16-, 32- and 64-point rules,
// as (node, weight) pairs. They are tabulated rather than solved lazily so
// no evaluation pays for the Legendre root solves; `gauss_rule` reproduces
// them bit for bit and a unit test checks the tables against it.
const GAUSS_16: [(f64, f64); 8] = [
    (0.9894009349916499, 0.027152459411754058),
    (0.9445750230732326, 0.062253523938647776),
    (0.8656312023878318, 0.0951585116824929),
    (0.755404408355003, 0.12462897125553395),
    (0.6178762444026438, 0.14959598881657682),
    (0.45801677765722737, 0.16915651939500256),
    (0.2816035507792589, 0.1826034150449236),
    (0.09501250983763744, 0.18945061045506847),
];
const GAUSS_32: [(f64, f64); 16] = [
    (0.9972638618494816, 0.007018610009469521),
    (0.9856115115452684, 0.01627439473090571),
    (0.9647622555875064, 0.02539206530926214),
    (0.9349060759377397, 0.03427386291302141),
    (0.8963211557660521, 0.042835898022226704),
    (0.84936761373257, 0.050998059262376154),
    (0.7944837959679424, 0.05868409347853558),
    (0.7321821187402897, 0.06582222277636181),
    (0.6630442669302152, 0.07234579410884862),
    (0.5877157572407623, 0.07819389578707044),
    (0.5068999089322294, 0.08331192422694672),
    (0.42135127613063533, 0.08765209300440374),
    (0.33186860228212767, 0.0911738786957639),
    (0.23928736225213706, 0.09384439908080441),
    (0.1444719615827965, 0.09563872007927485),
    (0.04830766568773831, 0.09654008851472785),
];
const GAUSS_64: [(f64, f64); 32] = [
    (0.9993050417357722, 0.0017832807216962678),
    (0.9963401167719552, 0.004147033260562521),
    (0.9910133714767443, 0.006504457968978368),
    (0.983336253884626, 0.008846759826363935),
    (0.973326827789911, 0.011168139460131076),
    (0.9610087996520538, 0.013463047896718736),
    (0.9464113748584028, 0.015726030476024788),
    (0.9295691721319396, 0.017951715775697225),
    (0.9105221370785028, 0.02013482315353018),
    (0.8893154459951141, 0.022270173808383212),
    (0.8659993981540929, 0.024352702568710933),
    (0.8406292962525804, 0.026377469715054645),
    (0.8132653151227975, 0.02833967261425946),
    (0.7839723589433414, 0.030234657072402426),
    (0.7528199072605319, 0.032057928354851606),
    (0.7198818501716109, 0.03380516183714161),
    (0.6852363130542333, 0.03547221325688236),
    (0.6489654712546573, 0.03705512854024007),
    (0.6111553551723933, 0.0385501531786156),
    (0.571895646202634, 0.03995374113272036),
    (0.5312794640198946, 0.04126256324262355),
    (0.48940314570705296, 0.042473515123653625),
    (0.4463660172534641, 0.04358372452932347),
    (0.4022701579639916, 0.04459055816375658),
    (0.3572201583376681, 0.04549162792741812),
    (0.31132287199021097, 0.046284796581314396),
    (0.2646871622087674, 0.04696818281620999),
    (0.21742364374000708, 0.047540165714830315),
    (0.1696444204239928, 0.04799938859645833),
    (0.12146281929612056, 0.0483447622348029),
    (0.07299312178779904, 0.048575467441503394),
    (0.02435029266342443, 0.04869095700913973),
];

#[cfg(test)]
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
        for &(z, weight) in &GAUSS_32 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_tables_match_legendre_root_solves() {
        assert_eq!(GAUSS_16, gauss_rule::<8>());
        assert_eq!(GAUSS_32, gauss_rule::<16>());
        assert_eq!(GAUSS_64, gauss_rule::<32>());
    }

    #[test]
    fn it2i0k0_tabulated_nodes_match_gauss_rule() {
        let mut nodes = [(0.0, 0.0); 32];
        for (pair, &(z, weight)) in nodes
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(&gauss_rule::<16>())
        {
            for (node, u) in pair.iter_mut().zip([24.0 * (1.0 - z), 24.0 * (1.0 + z)]) {
                *node = (u, 24.0 * weight * (-u).exp());
            }
        }
        nodes.sort_by(|a, b| a.0.total_cmp(&b.0));
        let kept: Vec<_> = nodes.iter().copied().filter(|&(_, w)| w >= 1e-19).collect();
        assert_eq!(kept.len(), 26);
        for x in [4.0, 6.0, 12.5, 40.0] {
            let sum = |nodes: &[(f64, f64)]| -> f64 {
                nodes
                    .iter()
                    .map(|&(u, weight)| weight * c::k0e(x + u) / (x + u))
                    .sum()
            };
            assert_eq!(it2i0k0(x).1, sum(&kept) * (-x).exp());
            let full = sum(&nodes) * (-x).exp();
            assert!((it2i0k0(x).1 - full).abs() <= 1e-18 * full);
        }
    }
}
