//! Typed dispatch for the supported real functional linear algebra surface.
use super::{
    CholeskyFactor, EigenResult, GeneralEigenResult, LuFactor, Matrix, QzResult, SvdResult,
};
use crate::{Error, Result};
/// Fixed scalar/storage metadata for the shared f64 matrix representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlasType {
    /// BLAS scalar prefix; always 'd' (double precision real).
    pub prefix: char,
    /// Scalar type name; always "f64".
    pub dtype: &'static str,
    /// Whether input prefers Fortran order; always false for row-major matrices.
    pub prefer_fortran: bool,
}
/// Typed callable BLAS operation with owned/vector return conventions.
/// These functional signatures replace mutable buffers, leading dimensions,
/// scalar alpha/beta, and increments. Kernels are private native Rust backends.
// Typed variants deliberately retain the callable operation signatures.
#[allow(clippy::type_complexity)]
#[derive(Clone, Copy, Debug)]
pub enum BlasFunction {
    /// `dot(x,y)` returns their inner product.
    Dot(fn(&[f64], &[f64]) -> Result<f64>),
    /// `nrm2(x)` returns the Euclidean vector norm.
    Nrm2(fn(&[f64]) -> Result<f64>),
    /// `gemv(a,x)` returns A x.
    Gemv(fn(&[Vec<f64>], &[f64]) -> Result<Vec<f64>>),
    /// `gemm(a,b)` returns A B.
    Gemm(fn(&[Vec<f64>], &[Vec<f64>]) -> Result<Matrix>),
}
/// Typed callable mathematical operation selected by LAPACK name.
/// Uses Sci shapes/errors and factor/result types. Backend implementations
/// use faer kernels rather than an actual LAPACK library.
// Typed variants deliberately retain the callable operation signatures.
#[allow(clippy::type_complexity)]
#[derive(Clone, Copy, Debug)]
pub enum LapackFunction {
    /// `gesv(a,b)` returns the solution vector.
    Gesv(fn(&[Vec<f64>], &[f64]) -> Result<Vec<f64>>),
    /// `getrf(a)` returns an owned square LU factor.
    Getrf(fn(&[Vec<f64>]) -> Result<LuFactor>),
    /// `getrs(factor,b,transpose)` solves using that factor.
    Getrs(fn(&LuFactor, &[f64], bool) -> Result<Vec<f64>>),
    /// `potrf(a,lower)` returns a Cholesky factor matrix.
    Potrf(fn(&[Vec<f64>], bool) -> Result<Matrix>),
    /// `potrs(factor,b)` solves using an owned Cholesky factor.
    Potrs(fn(&CholeskyFactor, &[f64]) -> Result<Vec<f64>>),
    /// `geev(a)` returns right complex eigenpairs.
    Geev(fn(&[Vec<f64>]) -> Result<GeneralEigenResult>),
    /// `syev(a,lower)` returns a symmetric eigensystem.
    Syev(fn(&[Vec<f64>], bool) -> Result<EigenResult>),
    /// `gesvd(a)` returns economy singular factors.
    Gesvd(fn(&[Vec<f64>]) -> Result<SvdResult>),
    /// `gges(a,b)` returns complex generalized Schur factors.
    Gges(fn(&[Vec<f64>], &[Vec<f64>]) -> Result<QzResult>),
}
fn dot(x: &[f64], y: &[f64]) -> Result<f64> {
    crate::error::finite_slice(x)?;
    crate::error::finite_slice(y)?;
    crate::error::finite(super::dot(x, y)?)
}
fn nrm2(x: &[f64]) -> Result<f64> {
    crate::error::finite_slice(x)?;
    crate::error::finite(x.iter().fold(0.0_f64, |s, &v| s.hypot(v)))
}
fn gemv(a: &[Vec<f64>], x: &[f64]) -> Result<Vec<f64>> {
    super::backend::dense(a)?;
    crate::error::finite_slice(x)?;
    let y = super::matvec(a, x)?;
    if y.iter().any(|v| !v.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok(y)
}
fn gemm(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Matrix> {
    super::backend::dense(a)?;
    super::backend::dense(b)?;
    super::backend::output(super::matmul(a, b)?)
}
/// Selects the BLAS precision prefix and storage preference.
///
/// ## Parameters
///
/// `arrays` accept [`super::MatrixInput`] storage. An empty list is accepted.
/// No dtype argument is needed because all scalar inputs are f64.
///
/// ## Domain
///
/// Each matrix must be rectangular; zero-size shapes and IEEE scalar values
/// are accepted because only shapes/type/storage are inspected.
///
/// ## Returns
///
/// BlasType { prefix: 'd', dtype: "f64", prefer_fortran: false }.
///
/// ## Errors
///
/// Ragged matrix or shared constructor allocation bound exceeded.
///
/// ## Numerical behavior
///
/// Native shape inspection only. Reports the row-major shared representation
/// used by dispatch handles, independent of input storage; complex, single
/// precision and promotion are absent. At most 1,000,000 matrices are accepted.
///
/// # Examples
///
/// ```
/// let t = sci_linalg::find_best_blas_type(&[] as &[sci_linalg::Matrix])?;
/// assert_eq!(t.prefix,'d');
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `find_best_blas_type`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.find_best_blas_type.html).
pub fn find_best_blas_type(arrays: &[impl super::MatrixInput]) -> Result<BlasType> {
    if arrays.len() > super::backend::MAX_ENTRIES {
        return Err(Error::InvalidInput("too many matrices"));
    }
    for a in arrays {
        super::input::checked_shape(a)?;
    }
    Ok(BlasType {
        prefix: 'd',
        dtype: "f64",
        prefer_fortran: false,
    })
}
/// Resolves BLAS names to typed real mathematical operations.
///
/// ## Parameters
///
/// `names` supports dot, nrm2, gemv, gemm, optionally prefixed with d.
/// Result order matches name order. Match the returned enum to call its
/// function pointer; signatures are owned functional adaptations.
///
/// ## Domain
///
/// An empty name list returns an empty vector. Unknown names and other
/// precision prefixes are rejected. Kernel inputs follow the selected operation.
///
/// ## Returns
///
/// Owned list of typed BlasFunction handles using shared f64 types.
///
/// ## Errors
///
/// Unknown operation. Kernel calls reject shape/nonfinite inputs or outputs.
///
/// ## Numerical behavior
///
/// Dispatch is O(number of names); dot/gemv/gemm delegate to faer, and
/// nrm2 uses native hypot accumulation. Matrix axes are bounded at 256. Vector norm uses stable hypot.
/// Raw BLAS overloads, increments, mutation and alpha/beta are unsupported.
///
/// # Examples
///
/// ```
/// use sci_linalg::{get_blas_funcs,BlasFunction};
/// if let BlasFunction::Dot(f) = get_blas_funcs(&["dot"])?[0] {
///     assert_eq!(f(&[1.,2.],&[3.,4.])?,11.);
/// }
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `get_blas_funcs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.get_blas_funcs.html).
pub fn get_blas_funcs(names: &[&str]) -> Result<Vec<BlasFunction>> {
    names
        .iter()
        .map(|name| {
            Ok(
                match if *name == "dot" {
                    name
                } else {
                    name.strip_prefix('d').unwrap_or(name)
                } {
                    "dot" => BlasFunction::Dot(dot),
                    "nrm2" => BlasFunction::Nrm2(nrm2),
                    "gemv" => BlasFunction::Gemv(gemv),
                    "gemm" => BlasFunction::Gemm(gemm),
                    _ => return Err(Error::InvalidInput("unsupported real BLAS operation")),
                },
            )
        })
        .collect()
}
/// Resolves LAPACK names to typed shared numerical operations.
///
/// ## Parameters
///
/// `names` supports gesv, getrf, getrs, potrf, potrs, geev, syev,
/// syevd, gesvd, gesdd, and gges, optionally with a d prefix.
/// Driver aliases reuse the same implementation. Handles accept owned
/// Sci factors and return shared results rather than LAPACK tuples.
///
/// ## Domain
///
/// An empty name list is accepted. Unknown names and non-double-real
/// prefixes are rejected. Kernel domains follow the resolved public API.
///
/// ## Returns
///
/// Owned list of LapackFunction handles, in requested order.
///
/// ## Errors
///
/// Unknown operation; numerical calls propagate their public API errors.
///
/// ## Numerical behavior
///
/// O(number of names) dispatch. Uses private established Rust decompositions.
/// This is a typed functional registry, with the documented supported names;
/// raw work arrays, info codes, driver selection, mutation, and the hundreds
/// of low-level LAPACK bindings are unsupported.
///
/// # Examples
///
/// ```
/// use sci_linalg::{get_lapack_funcs,LapackFunction};
/// if let LapackFunction::Gesv(f) = get_lapack_funcs(&["gesv"])?[0] {
///     assert_eq!(f(&[vec![2.]],&[4.])?,vec![2.]);
/// }
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `get_lapack_funcs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.get_lapack_funcs.html).
pub fn get_lapack_funcs(names: &[&str]) -> Result<Vec<LapackFunction>> {
    names
        .iter()
        .map(|name| {
            Ok(match name.strip_prefix('d').unwrap_or(name) {
                "gesv" => LapackFunction::Gesv(super::solve),
                "getrf" => LapackFunction::Getrf(super::lu_factor),
                "getrs" => LapackFunction::Getrs(super::lu_solve),
                "potrf" => LapackFunction::Potrf(super::cholesky),
                "potrs" => LapackFunction::Potrs(super::cho_solve),
                "geev" => LapackFunction::Geev(super::eig),
                "syev" | "syevd" => LapackFunction::Syev(super::eigh),
                "gesvd" | "gesdd" => LapackFunction::Gesvd(super::svd),
                "gges" => LapackFunction::Gges(super::qz),
                _ => return Err(Error::InvalidInput("unsupported real LAPACK operation")),
            })
        })
        .collect()
}
