# Sci

<p align="center">
    <a href="https://opensource.org/license/BSD-3-clause" style="text-decoration:none">
        <img src="https://img.shields.io/badge/License-BSD--3--Clause-brightgreen.svg" alt="License">
    </a>
    <a href="https://crates.io/crates/sci" style="text-decoration:none">
        <img src="https://img.shields.io/crates/v/sci" alt="Crate">
    </a>
    <a href="https://crates.io/crates/sci" style="text-decoration: none">
        <img src="https://img.shields.io/crates/d/sci" alt="Total Downloads">
    </a>
    <a href="https://docs.rs/sci" style="text-decoration:none">
        <img src="https://img.shields.io/badge/Docs-docs.rs-blue" alt="Documentation">
    </a>
    <a href="https://codecov.io/gh/p-sira/sci" style="text-decoration:none"> 
        <img src="https://codecov.io/gh/p-sira/sci/graph/badge.svg?token=JJVn2k9hM4"/> 
    </a>
    <a href="https://app.codspeed.io/p-sira/sci?utm_source=badge" style="text-decoration:none">
        <img src="https://img.shields.io/endpoint?url=https://codspeed.io/badge.json" alt="CodSpeed"/>
    </a>
</p>

**Sci** is a family of scientific computing crates with a unified Rust interface. The philosophy of Sci is to build on Rust's scientific ecosystem maintained by domain experts, while Sci supplies a consistent interface, input validation, and interoperability. Shared types let results flow between modules, from differentiation into optimization or Fourier analysis into statistics.

## Installation

```shell
cargo add sci
```

All 15 modules are enabled by default. Select only the modules you need with:

```shell
cargo add sci --no-default-features --features fft,complex
```

Rust 1.89 or later is required. Builds enabling `fft`, `special`, `cluster`,
`integrate`, `linalg`, `signal`, or `stats` require a C++17 compiler and libclang
for the private `xsf` backend, including through transitive dependencies.

## Modules

| Area | Modules and Cargo features |
| --- | --- |
| Numerical methods | [`differentiate`](https://docs.rs/sci-differentiate), [`integrate`](https://docs.rs/sci-integrate), [`interpolate`](https://docs.rs/sci-interpolate), [`optimize`](https://docs.rs/sci-optimize) |
| Data analysis | [`filter`](https://docs.rs/sci-filter), [`cluster`](https://docs.rs/sci-cluster), [`linalg`](https://docs.rs/sci-linalg), [`sparse`](https://docs.rs/sci-sparse), [`spatial`](https://docs.rs/sci-spatial), [`stats`](https://docs.rs/sci-stats) |
| Transforms and signals | [`fft`](https://docs.rs/sci-fft), [`signal`](https://docs.rs/sci-signal) |
| Mathematical functions | [`complex`](https://docs.rs/sci-complex), [`constants`](https://docs.rs/sci-constants), [`special`](https://docs.rs/sci-special) |

Linear algebra accepts shared rows and faer, ndarray, or nalgebra storage.
Sparse operations use shared CSR/COO storage, with graph algorithms, direct and iterative solves, spectral routines, and NPZ interchange.

## Documentation

- For individual module documentations, please follow the link in [Modules](#modules).
- [Data types and numerical conventions](docs/conventions.md).
- [Development, reference fixtures, and validation](docs/development.md).
