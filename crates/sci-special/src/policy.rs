//! Explicit, immutable error policies for functional composition.

use crate::Result;

/// Error handling for a computed scalar, with no process-global state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ErrorPolicy {
    /// Preserve IEEE NaNs and infinities, like SciPy's default numerical policy.
    #[default]
    Ieee,
    /// Reject a nonfinite result using the shared error type.
    Finite,
}

/// Returns the default, immutable special-function error policy.
///
/// ## SciPy adaptation
///
/// The default is [`ErrorPolicy::Ieee`]. Unlike SciPy, Sci has no mutable
/// thread-local or global error configuration. Pass a policy explicitly to
/// [`seterr`] when composing evaluations. There is no warning-channel policy.
///
/// # Examples
///
/// ```
/// use sci_special::{geterr,ErrorPolicy};
/// assert_eq!(geterr(),ErrorPolicy::Ieee);
/// ```
pub const fn geterr() -> ErrorPolicy {
    ErrorPolicy::Ieee
}

/// Applies an explicit error policy to a computed `f64` value.
///
/// ## SciPy adaptation
///
/// This is a pure counterpart to SciPy's stateful `seterr`: it validates this
/// value only and never changes later calls, backend state, or other threads.
/// IEEE mode returns NaNs and infinities unchanged; finite mode returns
/// [`crate::Error::NonFiniteEvaluation`] for either. Backend domain warnings are
/// not intercepted. Use functions returning [`Result`] for explicit domain errors.
///
/// # Examples
///
/// ```
/// use sci_special::{gamma,seterr,ErrorPolicy};
/// assert_eq!(seterr(gamma(5.0),ErrorPolicy::Finite)?,24.0);
/// # Ok::<(),sci_core::Error>(())
/// ```
pub fn seterr(value: f64, policy: ErrorPolicy) -> Result<f64> {
    match policy {
        ErrorPolicy::Ieee => Ok(value),
        ErrorPolicy::Finite => crate::error::finite(value),
    }
}
