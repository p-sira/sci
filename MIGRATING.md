# Migrating to Sci 0.3

All workspace packages advance one minor version: `0.2.x` becomes `0.3.0`,
while `sci-io` and `sci-sparse` advance from `0.1.x` to `0.2.0`.
Use matching workspace releases so shared error and numerical types agree.

Sci uses SciPy for independent numerical references. Its public API follows
Rust types and ownership; SciPy inventory names do not require extra exports.

## Direct numerical calls

`linalg::find_best_blas_type`, `get_blas_funcs`, `get_lapack_funcs`, `BlasType`,
`BlasFunction`, and `LapackFunction` are removed. Call `dot`, `matvec`, `matmul`,
`solve`, `lu_factor`, `lu_solve`, `cholesky`, `cho_solve`, `eig`, `eigh`, `svd`,
or `qz` directly. Scalar precision is already `f64`. Direct `dot`, `matvec`, and `matmul` retain
their documented IEEE propagation; the removed dispatcher wrappers performed
additional finite-value checks. Use `sci_core::finite_slice` and check results
when that stricter behavior is required. For a vector Euclidean
norm with overflow-resistant accumulation, use `x.iter().fold(0.0_f64,
|norm, &value| norm.hypot(value))`; validate inputs when required.

`stats::abs(x)`, `exp(x)`, and `log(x)` become `x.abs()`, `x.exp()`, and `x.ln()`.
`special::cbrt`, `exp2`, `expm1`, `log1p`, and `round` become `f64::cbrt`,
`exp2`, `exp_m1`, `ln_1p`, and **`round_ties_even`**, respectively.
`special::exp10(x)` becomes `10.0_f64.powf(x)`.
`special::geterr`, `seterr`, and `ErrorPolicy` are removed. Use
`sci_core::finite(value)` to reject NaN and infinity, or retain IEEE results.
Stable special functions such as `exprel`, `xlogy`, and `log_expit` remain.

## Sparse types and ownership

`issparse` is removed; use concrete types or traits. `get_index_dtype`,
`IndexDtype`, and `safely_cast_index_arrays` are replaced by concrete checked
conversions:

```rust
let a = sci::sparse::identity(2)?;
let indices = sci::sparse::csr_indices_i32(&a)?;
assert_eq!(indices.indptr, vec![0_i32, 1, 2]);
assert_eq!(indices.indices, vec![0_i32, 1]);
# Ok::<(), sci::Error>(())
```

Use `csr_indices_i64` for signed 64-bit storage. Both return named fields,
validate CSR and dimensions, preserve ordering, and reject unrepresentable
values. Matrix storage continues to use `usize` internally.

`aslinearoperator(&a)` becomes `LinearOperator::try_from(&a)`. This owned
conversion copies A and its transpose. Solvers already accept `&a` through
`LinearMap`; pass it directly when an owned callback operator is unnecessary.

## FFT lengths

`next_fast_len(target, real)` and `prev_fast_len(target, real)` become
`next_fast_len(target)` and `prev_fast_len(target)`. Both select lengths whose
prime factors are at most 11, following RustFFT guidance. This is a heuristic,
not CPU-specific performance tuning or exact SciPy real-transform parity.
Zero remains zero and the search remains bounded at 1,048,576 samples.

## Convolution algorithm selection

`signal::choose_conv_method` and `ConvolutionMethod` are removed. Their timing
constants were calibrated for SciPy and did not predict Sci's Rust backends.
Call `convolve`, `fftconvolve`, or `oaconvolve` explicitly and benchmark the
relevant input sizes when performance matters. These operations retain their
existing output conventions and work limits.

## Canonical names

The following aliases are removed. Their numerical operations remain available
under the canonical names. Existing independent SciPy fixture labels may retain
the old spelling while Rust tests call the canonical API.

| Removed name | Replacement |
| `stats::kstest(x, y, alternative)` | `stats::ks_2samp(x, y, alternative)`; use `ks_1samp` for a continuous null |
| `special::psi(x)` | `special::digamma(x)` |
| `special::assoc_laguerre(x, n, k)` | `special::eval_genlaguerre(n, k, x)` (arguments reordered) |
| `special::ellipf(phi, m)` | `special::ellipkinc(phi, m)` |
| `signal::closest_STFT_dual_window` | `signal::closest_stft_dual_window` |
| --- | --- |
| `c_roots` | `roots_chebyc` |
| `cg_roots` | `roots_gegenbauer` |
| `h_roots` | `roots_hermite` |
| `he_roots` | `roots_hermitenorm` |
| `j_roots` | `roots_jacobi` |
| `js_roots` | `roots_sh_jacobi` |
| `l_roots` | `roots_laguerre` |
| `la_roots` | `roots_genlaguerre` |
| `p_roots` | `roots_legendre` |
| `ps_roots` | `roots_sh_legendre` |
| `s_roots` | `roots_chebys` |
| `t_roots` | `roots_chebyt` |
| `ts_roots` | `roots_sh_chebyt` |
| `u_roots` | `roots_chebyu` |
| `us_roots` | `roots_sh_chebyu` |
| `solve_lyapunov` | `solve_continuous_lyapunov` |
| `maxRstat` | `max_rstat` |
| `py_vq` | `vq` |
| `sosfreqz` | `freqz_sos` |
| `check_COLA` | `check_cola` |
| `check_NOLA` | `check_nola` |
| `freqresp` | `freqs` |
| `dfreqresp` | `freqz` |
| `splrep` | `make_splrep` |
| `splprep` | `make_splprep` |
| `factorized` | `splu` |
| `bmat` | `block_array` |
| `diags_array` | `diags` |
| `eye_array` | `eye` |
| `matrix_transpose` | `transpose` |

`ishermitian` remains a mathematical predicate for real input, equivalent to
`issymmetric` in the supported scalar domain. The similarly named `spl*`,
`bispl*`, factorization, and transform functions that perform distinct
mathematical operations remain available.
