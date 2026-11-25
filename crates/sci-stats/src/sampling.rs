//! Random sampling from discrete weights and continuous probability densities.
//!
//! All engines are implemented in Rust, accept explicit seeded random streams,
//! and enforce numerical domain and allocation limits. Their APIs are adapted
//! from the mathematical methods in SciPy's `stats.sampling` namespace.

mod discrete;
pub use discrete::*;

mod inversion;
pub use inversion::*;

mod rejection;
pub use rejection::*;
