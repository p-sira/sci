//! Scientific file interchange with bounded input and explicit data types.
//!
//! Default features are `matrix-market`, `wav`, and `matlab`; each can be
//! selected independently. Readers accept Rust byte streams and writers accept
//! byte destinations. Paths are opened by callers; extensions are never added.
//! Backend types remain private. I/O errors retain their original source.
//! Through the Sci facade, select `io-matrix-market`, `io-wav`, or `io-matlab`.
//! With facade defaults disabled, `io` alone exposes shared I/O types without
//! file codecs. Matrix Market uses shared `sci-core` CSR helpers without
//! depending on `sci-sparse` or enabling sparse NPZ interchange.
//!
//! File data may contain NaN, infinities, and signed zeros. Conversion into a
//! numerical Sci module remains subject to that module's domain validation.
//! Inputs are limited to 64 MiB, arrays/samples to 1,000,000 scalar values,
//! axes to 1,000,000, rank to 32, and MATLAB files to 1,024 variables.
//! Format modules document narrower supported subsets and conversion rules.
//!
//! ```
//! let values = sci_io::NumericData::I64(vec![9_007_199_254_740_993]);
//! // The original integer remains exact; f64 conversion is explicit.
//! assert_eq!(values.to_f64(), vec![9_007_199_254_740_992.0]);
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs, rustdoc::broken_intra_doc_links)]

#[cfg(any(feature = "matrix-market", feature = "wav", feature = "matlab"))]
use std::io::Read;

pub use sci_core::{Matrix, SparseMatrix};

/// Scientific file decoding, validation, or stream failure.
#[derive(Debug)]
pub enum Error {
    /// An underlying stream operation failed.
    Io(std::io::Error),
    /// File contents or public record fields violate format invariants or limits.
    InvalidData(&'static str),
    /// A recognized format feature is outside the supported subset.
    Unsupported(&'static str),
    /// A private codec rejected a file; its diagnostic is retained as text.
    Codec(String),
    /// A shared Sci numerical validation or conversion failed.
    Numerical(sci_core::Error),
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => e.fmt(f),
            Self::InvalidData(s) | Self::Unsupported(s) => f.write_str(s),
            Self::Codec(s) => f.write_str(s),
            Self::Numerical(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Numerical(e) => Some(e),
            _ => None,
        }
    }
}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<sci_core::Error> for Error {
    fn from(e: sci_core::Error) -> Self {
        Self::Numerical(e)
    }
}
/// A fallible scientific I/O operation.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(any(feature = "matrix-market", feature = "wav", feature = "matlab"))]
const BYTES: usize = 64 * 1024 * 1024;
#[cfg(any(feature = "matrix-market", feature = "wav", feature = "matlab"))]
const VALUES: usize = 1_000_000;

#[cfg(any(feature = "matrix-market", feature = "wav", feature = "matlab"))]
fn bounded_read(reader: impl Read) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader.take(BYTES as u64 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > BYTES {
        return Err(Error::InvalidData("input exceeds 64 MiB"));
    }
    Ok(bytes)
}

#[cfg(feature = "matlab")]
fn volume(shape: &[usize]) -> Result<usize> {
    if shape.is_empty() || shape.len() > 32 || shape.iter().any(|&n| n > VALUES) {
        return Err(Error::InvalidData("invalid rank or axis exceeds 1,000,000"));
    }
    shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d).filter(|&n| n <= VALUES))
        .ok_or(Error::InvalidData("array exceeds 1,000,000 values"))
}

macro_rules! numeric_data {
    ($($variant:ident($ty:ty)),+ $(,)?) => {
        /// Typed scalar storage, without implicit normalization or conversion.
        #[derive(Clone, Debug, PartialEq)]
        pub enum NumericData {
            $(#[doc = concat!("Values stored as `", stringify!($ty), "`.")]
            $variant(Vec<$ty>)),+
        }
        impl NumericData {
            /// Number of stored scalar values.
            pub fn len(&self) -> usize { match self { $(Self::$variant(v) => v.len()),+ } }
            /// Whether scalar storage is empty.
            pub fn is_empty(&self) -> bool { self.len() == 0 }
            /// Convert each scalar to `f64`, preserving ordering and units.
            ///
            /// # Parameters and conventions
            /// `self` contains typed scalars in the caller's storage order.
            /// # Domain and special values
            /// All values are accepted, including NaN, infinities and signed zero.
            /// # Returns
            /// A newly allocated vector with `output[i] = self[i] as f64`.
            /// # Numerical behavior
            /// IEEE casts preserve f32 values exactly and floating special values.
            /// Integers beyond ±2^53 may round to the nearest representable value.
            /// This conversion does not normalize audio or reject nonfinite values.
            /// # Examples
            /// ```
            /// assert_eq!(sci_io::NumericData::I16(vec![-2, 3]).to_f64(), [-2.0, 3.0]);
            /// ```
            /// # References
            /// [Rust numeric casts](https://doc.rust-lang.org/reference/expressions/operator-expr.html#numeric-cast).
            pub fn to_f64(&self) -> Vec<f64> {
                match self { $(Self::$variant(v) => v.iter().map(|&x| x as f64).collect()),+ }
            }
        }
    }
}
numeric_data!(
    I8(i8),
    U8(u8),
    I16(i16),
    U16(u16),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    F32(f32),
    F64(f64)
);

/// Matrix Market interchange in coordinate and array storage formats.
#[cfg(feature = "matrix-market")]
#[path = "market.rs"]
pub mod matrix_market;

/// Typed WAV audio and explicit conversion to signal samples.
#[cfg(feature = "wav")]
pub mod wavfile;

/// Dense numerical MATLAB Level-5 files, including bounded zlib compression.
#[cfg(feature = "matlab")]
pub mod matlab;
