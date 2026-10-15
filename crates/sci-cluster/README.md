# Sci-Cluster

Clustering of finite row-major observations.

Part of [Sci](https://github.com/p-sira/sci), a unified scientific computing library for Rust.

`vq` provides vector quantization; `hierarchy` provides agglomerative
clustering and linkage-tree utilities with explicit numerical work limits.
Rows are observations and columns are coordinates. Distances and linkage
updates use the ordering documented by each function; ties are resolved
deterministically. Inputs are not mutated, and invalid shapes or
nonfinite values return `Error`.

## Acknowledgments

We thank `kodama` for its clustering algorithms.
