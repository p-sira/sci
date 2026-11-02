# Sci-Graph

`sci-graph` provides graph algorithms on shared, finite `sci-core::SparseMatrix`
CSR adjacency matrices. It can be used directly without `sci-sparse`, or through
the `sci::graph` facade with the `graph` feature.

```toml
sci-graph = "0.1"
```

```rust
let a = sci_graph::csr::from_coo(2, 2, &[(0, 1, 2.0)])?;
let paths = sci_graph::dijkstra(&a, &Default::default())?;
assert_eq!(paths.distances[0][1], 2.0);
# Ok::<(), sci_graph::Error>(())
```

Algorithms include shortest paths, traversal, connected components, spanning
trees, matching, flow, graph Laplacians, reorderings, and graph conversions.
Their results and domain restrictions remain the same as the previous
`sci-sparse::csgraph` implementation. A stored zero is an edge except where
an algorithm explicitly treats it as absent. Algorithms typically support up to
256 vertices and document additional work limits.

`sci-sparse` retains an optional `graph` feature that re-exports this crate as
`sci_sparse::csgraph` for source compatibility.
