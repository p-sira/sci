//! Small private solvers for finite quadrature rules.

use crate::{Error, Result};

/// Solve a small dense system with scaled partial pivoting.
pub(crate) fn solve_pivoted(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>> {
    let size = b.len();
    let mut scales: Vec<f64> = a
        .iter()
        .map(|row| row.iter().fold(0.0_f64, |v, x| v.max(x.abs())))
        .collect();
    for col in 0..size {
        let pivot = (col..size)
            .max_by(|&i, &j| {
                (a[i][col].abs() / scales[i]).total_cmp(&(a[j][col].abs() / scales[j]))
            })
            .unwrap();
        if !a[pivot][col].is_finite() || a[pivot][col].abs() <= f64::MIN_POSITIVE {
            return Err(Error::NumericalFailure("singular Newton-Cotes system"));
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        scales.swap(col, pivot);
        for row in col + 1..size {
            let factor = a[row][col] / a[col][col];
            a[row][col] = 0.0;
            let (preceding, current) = a.split_at_mut(row);
            for (value, &pivot_value) in current[0].iter_mut().zip(&preceding[col]).skip(col + 1) {
                *value -= factor * pivot_value;
            }
            b[row] -= factor * b[col];
        }
    }
    for col in (0..size).rev() {
        let mut rhs = b[col];
        for k in col + 1..size {
            rhs -= a[col][k] * b[k];
        }
        b[col] = rhs / a[col][col];
        if !b[col].is_finite() {
            return Err(Error::NonFiniteEvaluation);
        }
    }
    Ok(b)
}

/// Positive roots via Newton iteration, mirrored to guarantee symmetric weights.
pub(crate) fn legendre_nodes_weights(n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    for i in 0..n.div_ceil(2) {
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut converged = false;
        for _ in 0..32 {
            let (p, previous) = legendre_pair(n, x);
            let derivative = n as f64 * (x * p - previous) / (x * x - 1.0);
            let delta = p / derivative;
            x -= delta;
            if delta.abs() <= 4.0 * f64::EPSILON * x.abs().max(1.0) {
                converged = true;
                break;
            }
        }
        if !converged || !x.is_finite() {
            return Err(Error::NumericalFailure("Legendre root iteration failed"));
        }
        let (p, previous) = legendre_pair(n, x);
        let derivative = n as f64 * (x * p - previous) / (x * x - 1.0);
        let weight = 2.0 / ((1.0 - x * x) * derivative * derivative);
        if !weight.is_finite() || weight <= 0.0 {
            return Err(Error::NumericalFailure("invalid Legendre weight"));
        }
        nodes[i] = -x;
        nodes[n - 1 - i] = x;
        weights[i] = weight;
        weights[n - 1 - i] = weight;
    }
    // The exact Legendre mass is two. Correct small accumulated weight drift
    // before fixed_quad rescales integrals near f64::MAX; otherwise a constant
    // finite integral can spuriously overflow (notably the two-node rule).
    let mass: f64 = weights.iter().sum();
    let normalization = 2.0 / mass;
    for weight in &mut weights {
        *weight *= normalization;
    }
    Ok((nodes, weights))
}

fn legendre_pair(n: usize, x: f64) -> (f64, f64) {
    let (mut previous, mut current) = (1.0, x);
    if n == 1 {
        return (current, previous);
    }
    for k in 2..=n {
        let next = ((2 * k - 1) as f64 * x * current - (k - 1) as f64 * previous) / k as f64;
        previous = current;
        current = next;
    }
    (current, previous)
}
