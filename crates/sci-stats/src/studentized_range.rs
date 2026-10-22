//! Studentized-range mixture integral from SciPy 1.17.0's
//! `scipy/stats/_stats.pyx` (BSD-3-Clause), adapted to direct survival
//! probabilities and bounded composite Gaussian quadrature.
//! Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
//! <https://github.com/scipy/scipy/blob/v1.17.0/scipy/stats/_stats.pyx>
//! See ../SCIPY_LICENSE.txt for the retained license.
//!
//! Conditional on S=sqrt(chi²_ν/ν), integrate over the minimum Z of k
//! independent standard normals:
//! P(range > q*S) = E_S[k ∫ φ(z) (sf(z)^(k-1)
//!                              - (sf(z)-sf(z+q*S))^(k-1)) dz].
//! Use y=sqrt(ν/2)*log(chi²_ν/ν) for a resolved scale density at large ν.

use super::*;
use std::sync::OnceLock;

struct Rule {
    normal: Vec<(f64, f64)>,
    scale: Vec<(f64, f64)>,
}

fn tail(z: f64) -> f64 {
    0.5 * statrs::function::erf::erfc(z * std::f64::consts::FRAC_1_SQRT_2)
}

fn rule(order: usize) -> Result<&'static Rule> {
    static COARSE: OnceLock<Result<Rule>> = OnceLock::new();
    static FINE: OnceLock<Result<Rule>> = OnceLock::new();
    let cache = if order == 32 { &COARSE } else { &FINE };
    cache
        .get_or_init(|| {
            let (nodes, weights) = crate::special::roots_legendre(order)?;
            let composite = |edges: &[f64]| {
                edges
                    .windows(2)
                    .flat_map(|edge| {
                        let half = (edge[1] - edge[0]) / 2.;
                        let mid = (edge[0] + edge[1]) / 2.;
                        nodes
                            .iter()
                            .zip(&weights)
                            .map(move |(z, w)| (mid + half * z, half * w))
                    })
                    .collect()
            };
            Ok(Rule {
                normal: composite(&[-10., -8., -6., -4., -2., 0., 2., 4., 6., 8., 10.]),
                scale: composite(&[-32., -16., -8., -4., 0., 4., 8., 12.]),
            })
        })
        .as_ref()
        .map_err(Clone::clone)
}

// Log normalization of the standardized log-gamma density, avoiding
// cancellation of a*log(a)-a-logGamma(a) for large a.
fn log_normalizer(a: f64) -> f64 {
    if a < 8. {
        return a * a.ln() - a - crate::special::gammaln(a) - 0.5 * a.ln();
    }
    let r = 1. / a;
    let r2 = r * r;
    let correction = r
        * (1. / 12.
            + r2 * (-1. / 360.
                + r2 * (1. / 1260.
                    + r2 * (-1. / 1680. + r2 * (1. / 1188. - r2 * 691. / 360360.)))));
    -0.5 * (2. * std::f64::consts::PI).ln() - correction
}

pub(super) struct StudentizedRange {
    groups: usize,
    df: f64,
}

impl StudentizedRange {
    pub(super) fn new(groups: usize, df: f64) -> Self {
        Self { groups, df }
    }

    fn integrate(&self, q: f64, order: usize) -> Result<f64> {
        let rule = rule(order)?;
        let a = self.df / 2.;
        let root = a.sqrt();
        let log_norm = log_normalizer(a);
        let power = (self.groups - 1) as f64;
        let mut total = 0.;
        for &(y, weight) in &rule.scale {
            let v = y / root;
            let scale = (v / 2.).exp();
            let density = (log_norm + a * (v - v.exp_m1())).exp();
            let mut conditional = 0.;
            for &(z, w) in &rule.normal {
                let survival = tail(z);
                let ratio = (tail(z + q * scale) / survival).min(1.);
                // Compute the difference of powers without tail cancellation.
                let difference = -(power * (-ratio).ln_1p()).exp_m1();
                let phi = (-z * z / 2.).exp() / (2. * std::f64::consts::PI).sqrt();
                conditional += w * phi * survival.powi(self.groups as i32 - 1) * difference;
            }
            total += weight * density * self.groups as f64 * conditional;
        }
        finite(total)
    }

    // Called with 3..=16 groups, integer residual df in 3..=65533 and q>=0.
    // At most 358400 integrand evaluations per pair (orders 32 and 64).
    pub(super) fn sf(&self, q: f64) -> Result<f64> {
        if q == 0. {
            return Ok(1.);
        }
        if q == f64::INFINITY {
            return Ok(0.);
        }
        let coarse = self.integrate(q, 32)?;
        let fine = self.integrate(q, 64)?;
        converged_probability(coarse, fine)
    }
}

// Validate the quadrature estimate separately from producing it: ordinary
// supported inputs should converge, but failures must remain checked.
fn converged_probability(coarse: f64, fine: f64) -> Result<f64> {
    if !coarse.is_finite() || (coarse - fine).abs() > 2e-10 || !(0. ..=1. + 2e-10).contains(&fine) {
        return Err(Error::NumericalFailure(
            "studentized-range quadrature did not converge",
        ));
    }
    Ok(fine.clamp(0., 1.))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadrature_rejects_disagreement_and_invalid_probabilities() {
        for (coarse, fine) in [
            (0.5, 0.5 + 3e-10),
            (-1e-12, -1e-12),
            (1. + 3e-10, 1. + 3e-10),
            (f64::NAN, 0.5),
            (0.5, f64::NAN),
            (f64::INFINITY, f64::INFINITY),
        ] {
            assert_eq!(
                converged_probability(coarse, fine),
                Err(Error::NumericalFailure(
                    "studentized-range quadrature did not converge"
                ))
            );
        }
        assert_eq!(converged_probability(0.5, 0.5), Ok(0.5));
        assert_eq!(converged_probability(1., 1. + 1e-10), Ok(1.));
    }
}
