//! # Shared numerical types and errors
//!
//! Common errors, matrix storage, convolution modes, and spline representations
//! used across the Sci crate family.

#![forbid(unsafe_code)]
#![deny(missing_docs, rustdoc::broken_intra_doc_links)]

use std::fmt;

/// A validation or numerical evaluation failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// A finite-difference step must be finite and positive.
    InvalidStep,
    /// A parameter is outside its documented domain.
    InvalidInput(&'static str),
    /// Input shapes do not agree.
    DimensionMismatch,
    /// A callback returned vectors with inconsistent lengths.
    InconsistentOutputDimension {
        /// Length of the newly evaluated output.
        positive: usize,
        /// Length of the previous output.
        negative: usize,
    },
    /// An evaluation returned a nonfinite value.
    NonFiniteEvaluation,
    /// An underlying numerical algorithm failed.
    NumericalFailure(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStep => f.write_str("step must be finite and greater than zero"),
            Self::InvalidInput(message) | Self::NumericalFailure(message) => f.write_str(message),
            Self::DimensionMismatch => f.write_str("input dimensions do not agree"),
            Self::InconsistentOutputDimension { positive, negative } => {
                write!(f, "output dimension changed from {negative} to {positive}")
            }
            Self::NonFiniteEvaluation => f.write_str("evaluation returned a nonfinite value"),
        }
    }
}

impl std::error::Error for Error {}

/// The shared result type used by fallible numerical operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Checks a finite numerical result.
#[doc(hidden)]
pub fn finite(value: f64) -> Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(Error::NonFiniteEvaluation)
    }
}

/// Checks that coordinates are finite.
#[doc(hidden)]
pub fn finite_slice(values: &[f64]) -> Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(Error::InvalidInput("coordinates must be finite"))
    }
}

/// An owned real matrix with one vector per row.
pub type Matrix = Vec<Vec<f64>>;

/// Scalar B-spline `S(x) = Σ coefficients[i] B[i,degree](x)`.
///
/// Knots are nondecreasing; `coefficients.len() = knots.len() - degree - 1`.
/// The base interval is `[knots[degree], knots[coefficients.len()]]`.
/// Numerical APIs validate public fields before use. Coefficients have sample units.
#[derive(Clone, Debug, PartialEq)]
pub struct Spline {
    /// Complete knot vector, including endpoint multiplicities.
    pub knots: Vec<f64>,
    /// B-spline coefficients in increasing basis-index order.
    pub coefficients: Vec<f64>,
    /// Polynomial degree, rather than order (order is degree plus one).
    pub degree: usize,
}

/// Parametric B-spline curve with one scalar spline per coordinate.
#[derive(Clone, Debug, PartialEq)]
pub struct ParametricSpline {
    /// Coordinate splines, all using the same parameter and knot vector.
    pub components: Vec<Spline>,
    /// Increasing sample parameters used for the fit.
    pub parameters: Vec<f64>,
}

/// Tensor-product spline `S(x,y) = Σ c[i,j] B[i,kx](x) B[j,ky](y)`.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceSpline {
    /// Complete nondecreasing x knot vector.
    pub x_knots: Vec<f64>,
    /// Complete nondecreasing y knot vector.
    pub y_knots: Vec<f64>,
    /// Flattened coefficients: x basis index first, y basis index fastest.
    pub coefficients: Vec<f64>,
    /// Degree in the first coordinate.
    pub x_degree: usize,
    /// Degree in the second coordinate.
    pub y_degree: usize,
}

/// Output selection for convolution and correlation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// All `n + m - 1` samples, assuming zero extension.
    Full,
    /// `n` samples centered on the full convolution, where `n` is the first length.
    Same,
    /// `|n - m| + 1` samples with complete overlap.
    Valid,
}
