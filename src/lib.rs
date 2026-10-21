//! # Sci
//!
//! A unified interface to scientific computing in Rust: free functions, `f64`
//! scalars, shared data types, and consistent errors across the Sci crate family.
//!
//! Sci favors established Rust crates maintained by domain experts for complex
//! algorithms. It adds validation and interoperability around private backends;
//! simple formulas use standard-library operations or direct implementations.
//! Numerical behavior is defined by each module's Rust API and documentation:
//! shared `f64` values, explicit shapes and units, bounded work, and shared
//! error handling. External libraries are implementation details, not runtime
//! compatibility requirements.
//!
//! All numerical modules are enabled by default. Select individual Cargo features
//! with `default-features = false`, or use the `sci-*` crates directly. A module's
//! dependencies compile without exposing additional modules through this facade.
//!
//! ```
//! let error = sci::Error::DimensionMismatch;
//! assert_eq!(error.to_string(), "input dimensions do not agree");
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs, rustdoc::broken_intra_doc_links)]

pub use sci_core::{
    Error, Matrix, Mode, ParametricSpline, Result, SparseArray, SparseMatrix, Spline, SurfaceSpline,
};

/// Clustering and hierarchy; requires the `cluster` feature.
#[cfg(feature = "cluster")]
pub use sci_cluster as cluster;

/// Complex arithmetic and elementary functions; requires the `complex` feature.
#[cfg(feature = "complex")]
pub use sci_complex as complex;

/// Physical and mathematical constants; requires the `constants` feature.
#[cfg(feature = "constants")]
pub use sci_constants as constants;

/// Numerical differentiation; requires the `differentiate` feature.
#[cfg(feature = "differentiate")]
pub use sci_differentiate as differentiate;

/// Bayesian state estimation; requires the `filter` feature.
#[cfg(feature = "filter")]
pub use sci_filter as filter;

/// Fourier, cosine, sine, and Hankel transforms; requires the `fft` feature.
#[cfg(feature = "fft")]
pub use sci_fft as fft;

/// Numerical integration; requires the `integrate` feature.
#[cfg(feature = "integrate")]
pub use sci_integrate as integrate;

/// Polynomial and piecewise interpolation; requires the `interpolate` feature.
#[cfg(feature = "interpolate")]
pub use sci_interpolate as interpolate;

/// Linear algebra; requires the `linalg` feature.
#[cfg(feature = "linalg")]
pub use sci_linalg as linalg;

/// Numerical optimization; requires the `optimize` feature.
#[cfg(feature = "optimize")]
pub use sci_optimize as optimize;

/// Signal processing; requires the `signal` feature.
#[cfg(feature = "signal")]
pub use sci_signal as signal;

/// Spatial algorithms and distances; requires the `spatial` feature.
#[cfg(feature = "spatial")]
pub use sci_spatial as spatial;

/// Special mathematical functions; requires the `special` feature.
#[cfg(feature = "special")]
pub use sci_special as special;

/// Statistics and probability distributions; requires the `stats` feature.
#[cfg(feature = "stats")]
pub use sci_stats as stats;

/// Sparse arrays, graphs, linear algebra and NPZ interchange; requires the `sparse` feature.
#[cfg(feature = "sparse")]
pub use sci_sparse as sparse;

/// Scientific file interchange; requires the `io` feature.
#[cfg(feature = "io")]
pub use sci_io as io;
