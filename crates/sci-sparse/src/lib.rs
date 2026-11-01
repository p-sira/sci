//! Sparse matrices with real `f64` values and shared [`SparseMatrix`] storage.
//!
//! Construction and arithmetic use canonical CSR: sorted, unique columns per
//! row. Duplicate coordinates are summed; explicit zeros are retained unless
//! [`operations::eliminate_zeros`] is requested. Numerical inputs reject NaN and infinity, invalid
//! storage, axes above 1,000,000, and storage above 1,000,000 entries. Arithmetic
//! rejects nonfinite results. Dense conversion is limited to 1,000,000 cells;
//! sparse multiplication to 20,000,000 scalar products and block assembly to
//! 20,000,000 total input rows. Empty axes are supported.
//!
//! Shared [`SparseArray`] COO storage supports rank 1..32 and axis operations.
//! Optional graph algorithms, iterative solves, spectral decompositions, matrix
//! functions, random sampling, and NPZ interchange cover the modern SciPy function
//! inventory through documented Rust adaptations. Spectral and iterative axes are bounded
//! at 256; spectral routines use dense fallbacks. Python mutable storage classes,
//! deprecated compatibility namespaces and legacy solver controls are excluded.
//! Private faer, petgraph, lapjv, RNG and ZIP backends do not appear in the API.
//! See the repository's sparse coverage report for signature and work limits.
//! Defaults enable `full` (the `solver`, `graph`, and `random` backends) and `npz`.
//! Disable defaults for lightweight CSR/COO construction and arithmetic with only
//! `sci-core`; opt into each backend independently. The Sci facade's `sparse`
//! feature retains the full API, and `sparse-npz` additionally enables NPZ I/O.
//!
//! ```
//! use sci_sparse::{construct::from_coo, operations::matvec};
//! let a = from_coo(2, 2, &[(0, 0, 2.0), (1, 1, 4.0)])?;
//! assert_eq!(matvec(&a, &[3.0, 2.0])?, vec![6.0, 8.0]);
//! # Ok::<(), sci_sparse::Error>(())
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs, rustdoc::broken_intra_doc_links)]

pub use sci_core::{Error, Matrix, Result, SparseArray, SparseMatrix};

/// Sparse multidimensional arrays and CSR index conversions.
pub mod array;
/// Sparse matrix constructors and format conversion.
pub mod construct;
#[cfg(feature = "graph")]
/// Algorithms for weighted graphs represented by CSR matrices.
pub mod csgraph;
#[cfg(feature = "npz")]
/// Sparse NPZ serialization and deserialization.
pub mod io;
#[cfg(feature = "solver")]
/// Sparse linear algebra using finite real matrices.
pub mod linalg;
/// CSR validation, arithmetic, structural transforms, and multiplication.
pub mod operations;

// Private names used by the existing in-crate graph, array, and solver backends.
use construct::from_coo;
#[cfg(feature = "npz")]
use construct::from_csr;
#[cfg(feature = "solver")]
use construct::{from_dense, identity, to_dense};
#[cfg(feature = "solver")]
use operations::{matmul, matvec};
use operations::{transpose, validate};

const LIMIT: usize = 1_000_000;
const WORK: usize = 20_000_000;

fn shape(rows: usize, cols: usize) -> Result<()> {
    if rows > LIMIT || cols > LIMIT {
        return Err(Error::InvalidInput("sparse axis exceeds 1,000,000"));
    }
    Ok(())
}

fn entries(a: &SparseMatrix) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
    (0..a.rows).flat_map(move |r| {
        (a.indptr[r]..a.indptr[r + 1]).map(move |p| (r, a.indices[p], a.data[p]))
    })
}
