//! # Sci-Filter
//!
//! Functional Bayesian state estimation on shared row-major [`Matrix`] values.
//!
//! Part of [Sci](https://github.com/p-sira/sci), a unified scientific computing library for Rust.
//!
//! Linear and extended Kalman steps and the scaled unscented filter delegate to
//! private Smelly backends. Particle propagation and moments use kalman_filters;
//! log-weight normalization and reproducible systematic resampling are native.
//! Sample filtering and FIR/IIR design remain in `sci-signal`.
//!
//! Every call validates finite values, shapes, covariance matrices, and work
//! limits. Functions return owned results; failures leave their inputs unchanged.
//! Models use additive, independent process and observation noise. Coordinates
//! are Euclidean: angle wrapping and manifold residuals are not supported.
//! Covariances are symmetric positive-semidefinite on input; sigma-point
//! filtering additionally requires positive definiteness. Singular innovation
//! matrices and nonfinite evaluations are reported through [`Error`] rather
//! than converted to a plausible estimate.
//!
//! ```
//! use sci_filter::{Gaussian, kalman_predict, kalman_update};
//! let prior = Gaussian { mean: vec![0.], covariance: vec![vec![1.]] };
//! let predicted = kalman_predict(&prior, &[vec![1.]], &[vec![0.1]])?;
//! let updated = kalman_update(&predicted, &[2.], &[vec![1.]], &[vec![0.5]])?;
//! assert!(updated.estimate.mean[0] > 1.);
//! # Ok::<(), sci_filter::Error>(())
//! ```
//!
//! ## Acknowledgments
//!
//! We thank `smelly`, `kalman_filters`, and
//! `nalgebra` for their state-estimation algorithms.

#![forbid(unsafe_code)]
#![deny(missing_docs, rustdoc::broken_intra_doc_links)]

use nalgebra::{DMatrix, DVector};
pub use sci_core::{Error, Matrix, Result};

mod kalman;
mod particle;
pub use kalman::*;
pub use particle::*;

/// A Gaussian state estimate, with covariance in squared coordinate units.
///
/// `mean` has length n and `covariance` has n rows of n entries. Fields are
/// validated at every numerical call; constructing this value performs no work.
#[derive(Clone, Debug, PartialEq)]
pub struct Gaussian {
    /// Mean coordinates in caller-defined state order and units.
    pub mean: Vec<f64>,
    /// Symmetric positive-semidefinite covariance in row-major order.
    pub covariance: Matrix,
}

/// A posterior Gaussian estimate and the observation innovation.
#[derive(Clone, Debug, PartialEq)]
pub struct Update {
    /// Posterior mean and covariance.
    pub estimate: Gaussian,
    /// Observed minus predicted observation, in observation order and units.
    pub innovation: Vec<f64>,
}

/// Van der Merwe sigma-point scaling, with λ = α²(n + κ) − n.
///
/// Finite α > 0, β ≥ 0, and n + κ > 0 are required. The effective scale
/// α²(n + κ) must be finite and at least 1e-6 n to bound weight cancellation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SigmaPoints {
    /// Spread of points about the mean.
    pub alpha: f64,
    /// Distribution parameter; 2 is conventional for Gaussian states.
    pub beta: f64,
    /// Secondary scaling parameter; 0 is conventional.
    pub kappa: f64,
}

impl Default for SigmaPoints {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            beta: 2.,
            kappa: 0.,
        }
    }
}

const MAX_DIM: usize = 32;
const MAX_WORK: usize = 16_777_216;

fn dimension(n: usize) -> Result<()> {
    if (1..=MAX_DIM).contains(&n) {
        Ok(())
    } else {
        Err(Error::InvalidInput("filter dimensions must be in 1..=32"))
    }
}

fn vector(x: &[f64], n: usize) -> Result<()> {
    if x.len() != n {
        return Err(Error::DimensionMismatch);
    }
    if x.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(Error::InvalidInput("filter inputs must be finite"))
    }
}

fn matrix(a: &[Vec<f64>], rows: usize, cols: usize) -> Result<DMatrix<f64>> {
    if a.len() != rows || a.iter().any(|row| row.len() != cols) {
        return Err(Error::DimensionMismatch);
    }
    for row in a {
        vector(row, cols)?;
    }
    Ok(DMatrix::from_fn(rows, cols, |i, j| a[i][j]))
}

// Scale the eigensystem so that covariance validation itself cannot overflow.
// Tiny negative eigenvalues from roundoff are tolerated, never projected away.
fn covariance(a: &[Vec<f64>], n: usize) -> Result<DMatrix<f64>> {
    let mut a = matrix(a, n, n)?;
    let scale = a.iter().map(|v| v.abs()).fold(0., f64::max);
    if scale == 0. {
        return Ok(a);
    }
    let tolerance = 128. * f64::EPSILON * n as f64;
    for i in 0..n {
        for j in 0..i {
            if (a[(i, j)] / scale - a[(j, i)] / scale).abs() > tolerance {
                return Err(Error::InvalidInput("covariance must be symmetric"));
            }
            let average = 0.5 * a[(i, j)] + 0.5 * a[(j, i)];
            a[(i, j)] = average;
            a[(j, i)] = average;
        }
    }
    let eigen = nalgebra::linalg::SymmetricEigen::try_new(&a / scale, f64::EPSILON, 10_000).ok_or(
        Error::NumericalFailure("covariance validation did not converge"),
    )?;
    if eigen
        .eigenvalues
        .iter()
        .any(|v| !v.is_finite() || *v < -tolerance)
    {
        return Err(Error::InvalidInput(
            "covariance must be positive semidefinite",
        ));
    }
    Ok(a)
}

fn gaussian(x: &Gaussian) -> Result<(DVector<f64>, DMatrix<f64>)> {
    dimension(x.mean.len())?;
    vector(&x.mean, x.mean.len())?;
    Ok((
        DVector::from_column_slice(&x.mean),
        covariance(&x.covariance, x.mean.len())?,
    ))
}

fn rows(a: &DMatrix<f64>) -> Matrix {
    (0..a.nrows())
        .map(|i| (0..a.ncols()).map(|j| a[(i, j)]).collect())
        .collect()
}

fn output(mean: &DVector<f64>, p: &DMatrix<f64>) -> Result<Gaussian> {
    if mean.iter().chain(p.iter()).any(|v| !v.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    let p = covariance(&rows(p), mean.len()).map_err(|e| match e {
        Error::InvalidInput(_) => Error::NumericalFailure("filter produced an invalid covariance"),
        e => e,
    })?;
    Ok(Gaussian {
        mean: mean.as_slice().to_vec(),
        covariance: rows(&p),
    })
}

fn backend(e: smelly::FilterError) -> Error {
    use smelly::FilterError::*;
    match e {
        Dimension { .. } => Error::DimensionMismatch,
        SingularInnovation => Error::NumericalFailure("innovation covariance is singular"),
        NonPositiveDefiniteCovariance => {
            Error::NumericalFailure("sigma points require positive definite covariance")
        }
        InvalidSigmaPointScaling => Error::InvalidInput("invalid sigma-point scaling"),
        PredictRequired => Error::NumericalFailure("unscented prediction is missing"),
    }
}
