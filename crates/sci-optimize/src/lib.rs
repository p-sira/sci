//! # Sci-Optimize
//!
//! Numerical optimization and minimization.
//!
//! Part of [Sci](https://github.com/p-sira/sci), a unified scientific computing library for Rust.
//!
//! Unconstrained minimization using Basin internally. Objective and gradient
//! callbacks take the same `&[f64]` used by differentiation and linear algebra.
//! Callers never implement backend traits or construct solver state.
//!
//! ## Algorithms
//!
//! [`minimize::minimize`] uses Nelder–Mead, a derivative-free simplex method.
//! [`minimize::minimize_with_gradient`] uses fixed-step gradient descent. Its learning
//! rate must suit the objective; convergence is not guaranteed for arbitrary
//! functions. Limits return a result with a status, while invalid evaluations
//! return the shared [`Error`].
//! Objective values and callback outputs must be finite. A successful status
//! means the method met its configured stopping rule, not that a global
//! minimum was proved; inspect the returned point, value, and status together.
//!
//! ## Examples
//!
//! ```
//! let result = sci_optimize::minimize::minimize(
//!     |x| (x[0] - 2.0).powi(2),
//!     &[0.0], sci_optimize::Options::default(),
//! )?;
//! assert!((result.point[0] - 2.0).abs() < 1e-6);
//! # Ok::<(), sci_core::Error>(())
//! ```
//!
//! ## References
//!
//! - Nelder and Mead (1965), A simplex method for function minimization,
//!   [doi:10.1093/comjnl/7.4.308](https://doi.org/10.1093/comjnl/7.4.308).
//!
//! ## Acknowledgments
//!
//! We thank `basin` for its private optimization backends.

#![forbid(unsafe_code)]
#![deny(missing_docs, rustdoc::broken_intra_doc_links)]

use sci_core as error;
use sci_core as linalg;
pub use sci_core::{Error, Result};
#[path = "mod.rs"]
mod implementation;
pub use implementation::{Minimum, Options, Status};

/// Objective minimization and its solver configuration and results.
pub mod minimize {
    pub use crate::implementation::{
        Minimum, Options, Status, minimize, minimize_with_gradient, try_minimize,
        try_minimize_with_gradient,
    };
}

/// Numerical derivatives and Rosenbrock benchmark functions.
pub mod derivatives {
    pub use crate::implementation::{
        approx_fprime, check_grad, rosen, rosen_der, rosen_hess, rosen_hess_prod,
    };
}
