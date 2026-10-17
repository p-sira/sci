//! Uncensored AS R94 adaptation of SciPy 1.17.0's
//! `scipy/stats/_ansari_swilk_statistics.pyx` (BSD-3-Clause).
//! Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
//! <https://github.com/scipy/scipy/blob/v1.17.0/scipy/stats/_ansari_swilk_statistics.pyx>
//! See ../SCIPY_LICENSE.txt for the retained license.

use super::*;

fn polynomial(coefficients: &[f64], x: f64) -> f64 {
    coefficients.iter().rev().fold(0., |p, c| p * x + c)
}

// Validated, finite, nonconstant, shifted/scaled observations; n in 3..=5000.
pub(super) fn evaluate(values: &[f64]) -> Result<TestResult> {
    let mut x = values.to_vec();
    x.sort_unstable_by(f64::total_cmp);
    let n = x.len();
    let an = n as f64;
    let normal = normal(0., 1.)?;
    let mut weights = vec![std::f64::consts::FRAC_1_SQRT_2; n / 2];
    if n > 3 {
        for (i, a) in weights.iter_mut().enumerate() {
            *a = normal.inverse_cdf((i as f64 + 0.625) / (an + 0.25));
        }
        let sum = 2. * weights.iter().map(|a| a * a).sum::<f64>();
        let norm = sum.sqrt();
        let r = 1. / an.sqrt();
        let first = polynomial(&[0., 0.221157, -0.147981, -2.07119, 4.434685, -2.706056], r)
            - weights[0] / norm;
        let (start, scale) = if n > 5 {
            let second = -weights[1] / norm
                + polynomial(
                    &[0., 0.042981, -0.293762, -1.752461, 5.682633, -3.582633],
                    r,
                );
            let scale = ((sum - 2. * (weights[0].powi(2) + weights[1].powi(2)))
                / (1. - 2. * (first.powi(2) + second.powi(2))))
            .sqrt();
            weights[1] = second;
            (2, scale)
        } else {
            (
                1,
                ((sum - 2. * weights[0].powi(2)) / (1. - 2. * first.powi(2))).sqrt(),
            )
        };
        weights[0] = first;
        for a in &mut weights[start..] {
            *a *= -1. / scale;
        }
    }
    let coefficient = |i: usize| {
        let j = n - 1 - i;
        match i.cmp(&j) {
            std::cmp::Ordering::Less => -weights[i],
            std::cmp::Ordering::Equal => 0.,
            std::cmp::Ordering::Greater => weights[j],
        }
    };
    let range = x[n - 1] - x[0];
    let center = x.iter().map(|v| v / range).sum::<f64>() / an;
    let weight_center = (0..n).map(coefficient).sum::<f64>() / an;
    let (mut xx, mut aa, mut ax) = (0., 0., 0.);
    for (i, value) in x.iter().enumerate() {
        let a = coefficient(i) - weight_center;
        let v = value / range - center;
        xx += v * v;
        aa += a * a;
        ax += a * v;
    }
    let norm = (aa * xx).sqrt();
    let complement = (norm - ax) * (norm + ax) / (aa * xx);
    let mut statistic = finite(1. - complement)?;
    let pvalue = if n == 3 {
        statistic = statistic.clamp(0.75, 1.);
        1. - 6. / std::f64::consts::PI * statistic.sqrt().acos()
    } else {
        let mut y = complement.ln();
        let (m, s) = if n <= 11 {
            let gamma = polynomial(&[-2.273, 0.459], an);
            if y >= gamma {
                return Ok(TestResult {
                    statistic,
                    pvalue: 1e-19,
                });
            }
            y = -(gamma - y).ln();
            (
                polynomial(&[0.544, -0.39978, 0.025054, -0.0006714], an),
                polynomial(&[1.3822, -0.77857, 0.062767, -0.0020322], an).exp(),
            )
        } else {
            (
                polynomial(&[-1.5861, -0.31082, -0.083751, 0.0038915], an.ln()),
                polynomial(&[-0.4803, -0.082676, 0.0030302], an.ln()).exp(),
            )
        };
        normal.sf((y - m) / s)
    };
    Ok(TestResult {
        statistic,
        pvalue: finite(pvalue)?.clamp(0., 1.),
    })
}
