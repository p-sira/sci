# Changelog

## 0.1.0

- Initial standalone release of graph algorithms previously provided by `sci-sparse::csgraph`. Use `sci_graph` directly or `sci::graph` with the facade's `graph` feature.
- Provide shortest paths, traversals, connected components, minimum spanning trees, Laplacians, reordering, bipartite matching, structural rank, maximum flow, and Yen's shortest paths.
- Accept shared `sci-core 0.3` CSR `SparseMatrix` adjacency matrices, with dense and masked graph conversions. Graphs are limited to 256 vertices, or 256 per side for bipartite algorithms, with additional per-operation work limits.
- Reduce temporary storage in path reconstruction and Yen's path enumeration.

See the [migration guide](../../MIGRATING.md) for the removed sparse graph paths and feature changes.
