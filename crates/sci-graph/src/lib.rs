//! Graph algorithms on canonical real CSR adjacency matrices.
//!
//! Shared [`SparseMatrix`] values represent weighted directed or undirected
//! graphs. An absent entry means no edge; stored zero weights are edges except
//! where an algorithm documents different behavior. Graph algorithms can be
//! used independently of the `sci-sparse` crate, through `sci-graph` or
//! `sci::graph`. Numerical backends remain private.
//!
//! Graphs are currently bounded at 256 vertices (up to 256 per side for
//! bipartite algorithms), with additional work limits on some operations.
//!
//! ```
//! let graph = sci_graph::csr::from_coo(2, 2, &[(0, 1, 2.0)])?;
//! let paths = sci_graph::dijkstra(&graph, &Default::default())?;
//! assert_eq!(paths.distances[0][1], 2.0);
//! # Ok::<(), sci_graph::Error>(())
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs, rustdoc::broken_intra_doc_links)]

pub use sci_core::{Error, Matrix, Result, SparseMatrix, csr};

mod algorithms;
pub use algorithms::*;

use sci_core::csr::{from_coo, validate};

const WORK: usize = 20_000_000;

fn entries(a: &SparseMatrix) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
    (0..a.rows).flat_map(move |r| {
        (a.indptr[r]..a.indptr[r + 1]).map(move |p| (r, a.indices[p], a.data[p]))
    })
}
