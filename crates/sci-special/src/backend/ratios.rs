//! Bounded Perron continued fraction for adjacent modified Bessel functions.
use super::double_double::Dd;

pub(crate) fn iv_ratio_pair(v: f64, x: f64) -> (f64, f64) {
    if v.is_nan() || x.is_nan() || v < 0.5 || x < 0.0 || (v.is_infinite() && x.is_infinite()) {
        return (f64::NAN, f64::NAN);
    }
    if x == 0.0 {
        return (x, 1.0);
    }
    if v.is_infinite() {
        return (0.0, 1.0);
    }
    if x.is_infinite() {
        return (1.0, 0.0);
    }
    if v == 0.5 {
        let e = (-x).exp();
        let e2 = e * e;
        return (x.tanh(), 2.0 * e2 / (1.0 + e2));
    }
    // Scale the coefficients before products; DD preserves the small complement
    // when the continued-fraction tail nearly cancels its leading 2v term.
    let c = Dd::from(1.0 / v.max(x));
    let vc = Dd::from(v) * c;
    let xc = Dd::from(x) * c;
    let two = Dd::from(2.0);
    let one = Dd::from(1.0);
    let coefficients = |n: usize| {
        (
            -(two * vc + Dd::from((2 * n - 1) as f64) * c) * xc,
            two * (vc + xc) + Dd::from(n as f64) * c,
        )
    };
    let (a, mut previous_b) = coefficients(1);
    let mut term = a / previous_b;
    let mut ratio = Dd::from(0.0);
    let mut sum = two * vc;
    for n in 1..=1000 {
        sum = sum + term;
        if term.hi.abs() <= f64::EPSILON * sum.hi.abs() {
            let denominator = xc + sum;
            return ((xc / denominator).hi, (sum / denominator).hi);
        }
        let (a, b) = coefficients(n + 1);
        let p = a * (one + ratio);
        ratio = -p / (p + b * previous_b);
        term = term * ratio;
        previous_b = b;
    }
    (f64::NAN, f64::NAN)
}
