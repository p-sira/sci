//! Outward Taylor continuation of the real Kummer solution.
//!
//! DLMF 13.2.1 and 13.29(ii): propagate M along the negative real ray,
//! starting from its analytic defining series rather than an asymptotic mixture.
use super::double_double::Dd;

pub(super) fn negative_argument(a: f64, b: f64, x: f64) -> f64 {
    let a = Dd::from(a);
    let b = Dd::from(b);
    let target = Dd::from(x);
    let degree = a - b;
    if degree.lo == 0.0 && degree.hi == degree.hi.floor() {
        // M(b+n,b,x)=exp(x)*M(-n,b,-x). The latter is a normalized
        // Laguerre polynomial. It is a recessive ODE solution at negative x;
        // propagating it outwards would amplify roundoff into the other mode.
        let (mut previous, mut current) = (Dd::from(1.0), Dd::from(1.0) + target / b);
        for n in 1..degree.hi as usize {
            let n = Dd::from(n as f64);
            let next = ((Dd::from(2.0) * n + b + target) * current - n * previous) / (n + b);
            previous = current;
            current = next;
        }
        let scale = (0.5 * x).exp();
        return ((current.hi + current.lo) * scale) * scale;
    }
    // Binary scaling keeps the starting coordinate exact while limiting
    // cancellation in the origin series, including large positive a.
    let mut start = -0.25_f64;
    while a.hi * -start > 16.0 {
        start *= 0.5;
    }
    let mut center = Dd::from(start.max(x));
    // a + n and b + n are exact in double-double and the integer factors
    // need no promotion. The derivative sum Σ (n+1) term_(n+1) / center is
    // divided by the centre once, after the loop.
    let (mut term, mut y, mut weighted) = (Dd::from(1.0), Dd::from(1.0), Dd::from(0.0));
    let mut converged = false;
    for n in 0..256 {
        let n = n as f64;
        term = (term * Dd::sum(a.hi, n) * center).div_dd(Dd::sum(b.hi, n).mul_f64(n + 1.0));
        y = y + term;
        weighted = weighted + term.mul_f64(n + 1.0);
        if term.hi.abs() <= 1e-30 * y.hi.abs() && n > 16.0_f64.max(-b.hi + 1.0) {
            converged = true;
            break;
        }
    }
    if !converged {
        return f64::NAN;
    }
    let mut slope = weighted.div_dd(center);
    for _ in 0..8192 {
        let remaining = target - center;
        if remaining.hi >= 0.0 {
            return y.hi + y.lo;
        }
        // Stay within half the distance to the regular singular point, and
        // bound the oscillatory phase per step. DD coordinates avoid rounding
        // a target or accumulating coordinate drift over many steps.
        let step = (-center.hi * 0.5).min(0.5).min((-center.hi / a.hi).sqrt());
        let h = if remaining.hi.abs() <= step {
            remaining
        } else {
            Dd::from(-step)
        };
        // a + n is exact in double-double; integer factors need no
        // promotion; h^2 and z - b are formed once per step. The derivative
        // sum Σ (n+2) w_n / h is divided by h once, after the loop.
        let squared = h * h;
        let shifted = center - b;
        let (mut u, mut v) = (y, slope * h);
        let initial = v;
        let mut next_y = u + v;
        let mut weighted = Dd::from(0.0);
        let mut small_terms = 0;
        converged = false;
        for n in 0..128 {
            // If w=sum c_n*t^n at center z, Kummer's equation gives
            // z(n+2)(n+1)c_(n+2) = (a+n)c_n + (n+1)(z-b-n)c_(n+1).
            // Store c_n*h^n so powers cannot overflow independently.
            let n = n as f64;
            let (n1, n2) = (n + 1.0, n + 2.0);
            let numerator =
                Dd::sum(a.hi, n) * u * squared + (shifted.add_f64(-n) * v).mul_f64(n1) * h;
            let w = numerator.div_dd(center.mul_f64(n2 * n1));
            let derivative_term = w.mul_f64(n2);
            next_y = next_y + w;
            weighted = weighted + derivative_term;
            u = v;
            v = w;
            // Two consecutive small terms avoid stopping on an isolated zero
            // of this second-order coefficient recurrence. Check the derivative
            // as well, since it seeds the next step.
            if n > 8.0
                && derivative_term.hi.abs()
                    <= 1e-30 * next_y.hi.abs().min((initial + weighted).hi.abs())
            {
                small_terms += 1;
                if small_terms == 2 {
                    converged = true;
                    break;
                }
            } else {
                small_terms = 0;
            }
        }
        let next_slope = slope + weighted.div_dd(h);
        if !converged || !next_y.hi.is_finite() || !next_slope.hi.is_finite() {
            return f64::NAN;
        }
        y = next_y;
        slope = next_slope;
        center = center + h;
    }
    if (target - center).hi >= 0.0 {
        y.hi + y.lo
    } else {
        f64::NAN
    }
}
