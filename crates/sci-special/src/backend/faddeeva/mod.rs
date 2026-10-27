//! Rust translation of Steven G. Johnson's Faddeeva implementation.
mod auxilliary;
mod complexerrorfunctions;
mod realerrorfunctions;
pub(crate) use complexerrorfunctions::{ComplexErrorFunctions, w_with_relerror};
