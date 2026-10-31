//! Structured matrices with finite real entries.
// SciPy-derived layouts/identities retain their BSD notice in LICENSES/scipy-linalg.txt.
use super::backend::{output, size};
use crate::{Error, Result, complex::Complex};
use std::f64::consts::TAU;

/// Orientation of a Pascal matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PascalKind {
    /// Symmetric entries `binomial(i+j, i)`.
    Symmetric,
    /// Lower triangular entries `binomial(i,j)`.
    Lower,
    /// Upper triangular entries `binomial(j,i)`.
    Upper,
}
/// Scaling of the negative-exponent DFT matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DftScale {
    /// No normalization.
    None,
    /// Divide by the square root of the order.
    SqrtN,
    /// Divide by the order.
    N,
}
fn binomial(n: usize, k: usize) -> f64 {
    let k = k.min(n - k);
    (1..=k).fold(1.0, |v, i| v * (n - k + i) as f64 / i as f64)
}
/// Constructs a matrix constant along diagonals: `T[i,j] = c[i-j]` for `i>=j`, otherwise `r[j-i]`.
///
/// ## Parameters
///
/// `c` is the first column; optional `r` is the first row. `r[0]` is ignored; `None` uses `c`. Real, unbatched adaptation of SciPy.
///
/// ## Domain
///
/// Inputs must be finite. Output has at most 1,000,000 entries and each axis at most 1,000,000. Zero-sized outputs are allowed unless stated otherwise.
///
/// ## Returns
///
/// A row-major `len(c)` by `len(r)` matrix.
///
/// ## Errors
///
/// Returns a shared error for nonfinite inputs, invalid parameters, or the work bound; nonfinite output is an evaluation error.
///
/// ## Numerical behavior
///
/// O(rows × columns) copying; no numerical algorithm.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::toeplitz(&[1.,2.], Some(&[9.,3.]))?, vec![vec![1.,3.],vec![2.,1.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `toeplitz`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.toeplitz.html).
pub fn toeplitz(
    c: &(impl super::VectorInput + ?Sized),
    r: Option<&[f64]>,
) -> Result<Vec<Vec<f64>>> {
    let c_input = super::input::vector_slice(c)?;
    let c = c_input.as_ref();
    let r = r.unwrap_or(c);
    size(c.len(), r.len())?;
    crate::error::is_finite_slice(c)?;
    crate::error::is_finite_slice(r)?;
    Ok((0..c.len())
        .map(|i| {
            (0..r.len())
                .map(|j| if i >= j { c[i - j] } else { r[j - i] })
                .collect()
        })
        .collect())
}
/// Constructs a matrix constant along anti-diagonals: `H[i,j]=c[i+j]` if `i+j<len(c)`, otherwise `r[i+j-len(c)+1]`.
///
/// ## Parameters
///
/// `c` is the first column, optional `r` the last row. `r[0]` is ignored. `None` supplies `len(c)` zeros. No batching.
///
/// ## Domain
///
/// Inputs must be finite. Output has at most 1,000,000 entries and each axis at most 1,000,000. Zero-sized outputs are allowed unless stated otherwise.
///
/// ## Returns
///
/// A row-major `len(c)` by `len(r)` matrix.
///
/// ## Errors
///
/// Returns a shared error for nonfinite inputs, invalid parameters, or the work bound; nonfinite output is an evaluation error.
///
/// ## Numerical behavior
///
/// O(rows × columns) copying.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::hankel(&[1.,2.],None)?,vec![vec![1.,2.],vec![2.,0.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `hankel`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.hankel.html).
pub fn hankel(c: &(impl super::VectorInput + ?Sized), r: Option<&[f64]>) -> Result<Vec<Vec<f64>>> {
    let c_input = super::input::vector_slice(c)?;
    let c = c_input.as_ref();
    let n = r.map_or(c.len(), <[f64]>::len);
    size(c.len(), n)?;
    crate::error::is_finite_slice(c)?;
    if let Some(r) = r {
        crate::error::is_finite_slice(r)?;
    }
    Ok((0..c.len())
        .map(|i| {
            (0..n)
                .map(|j| {
                    if i + j < c.len() {
                        c[i + j]
                    } else {
                        r.map_or(0.0, |r| r[i + j - c.len() + 1])
                    }
                })
                .collect()
        })
        .collect())
}
/// Constructs `C[i,j]=c[(i-j) mod n]`.
///
/// ## Parameters
///
/// `c` is the first column; subsequent columns are cyclic downward shifts. No batching.
///
/// ## Domain
///
/// Inputs must be finite. Output has at most 1,000,000 entries and each axis at most 1,000,000. Zero-sized outputs are allowed unless stated otherwise.
///
/// ## Returns
///
/// An `n` by `n` real matrix.
///
/// ## Errors
///
/// Returns a shared error for nonfinite inputs, invalid parameters, or the work bound; nonfinite output is an evaluation error.
///
/// ## Numerical behavior
///
/// O(n²) copying.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::circulant(&[1.,2.])?,vec![vec![1.,2.],vec![2.,1.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `circulant`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.circulant.html).
pub fn circulant(c: &(impl super::VectorInput + ?Sized)) -> Result<Vec<Vec<f64>>> {
    let c_input = super::input::vector_slice(c)?;
    let c = c_input.as_ref();
    let n = c.len();
    size(n, n)?;
    crate::error::is_finite_slice(c)?;
    Ok((0..n)
        .map(|i| {
            (0..n)
                .map(|j| c[if i >= j { i - j } else { n + i - j }])
                .collect()
        })
        .collect())
}
/// Constructs the Hilbert matrix `H[i,j]=1/(i+j+1)`, using zero-based indices.
///
/// ## Parameters
///
/// `n` is the square matrix order.
///
/// ## Domain
///
/// Inputs must be finite. Output has at most 1,000,000 entries and each axis at most 1,000,000. Zero-sized outputs are allowed unless stated otherwise.
///
/// ## Returns
///
/// An `n` by `n` matrix.
///
/// ## Errors
///
/// Returns a shared error for nonfinite inputs, invalid parameters, or the work bound; nonfinite output is an evaluation error.
///
/// ## Numerical behavior
///
/// O(n²); Hilbert matrices rapidly become ill-conditioned.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::hilbert(1)?,vec![vec![1.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `hilbert`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.hilbert.html).
pub fn hilbert(n: usize) -> Result<Vec<Vec<f64>>> {
    size(n, n)?;
    Ok((0..n)
        .map(|i| (0..n).map(|j| 1.0 / (i + j + 1) as f64).collect())
        .collect())
}
/// Constructs `H⁻¹[i,j]=(-1)^(i+j) (i+j+1) C(n+i,n-j-1) C(n+j,n-i-1) C(i+j,i)²`.
///
/// ## Parameters
///
/// `n` is the Hilbert order. SciPy arbitrary-precision integer mode is omitted.
///
/// ## Domain
///
/// Inputs must be finite. Output has at most 1,000,000 entries and each axis at most 1,000,000. Zero-sized outputs are allowed unless stated otherwise. Order must be at most 128.
///
/// ## Returns
///
/// The inverse Hilbert matrix as `f64`.
///
/// ## Errors
///
/// Returns a shared error for nonfinite inputs, invalid parameters, or the work bound; nonfinite output is an evaluation error.
///
/// ## Numerical behavior
///
/// Direct binomial formula, O(n³) bounded at n=128. Integers above 2^53 are rounded; multiplying by H can lose nearly all precision.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::invhilbert(2)?,vec![vec![4.,-6.],vec![-6.,12.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `invhilbert`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.invhilbert.html).
pub fn invhilbert(n: usize) -> Result<Vec<Vec<f64>>> {
    if n > 128 {
        return Err(Error::InvalidInput("inverse Hilbert order exceeds 128"));
    }
    output(
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
                        sign * (i + j + 1) as f64
                            * binomial(n + i, n - j - 1)
                            * binomial(n + j, n - i - 1)
                            * binomial(i + j, i).powi(2)
                    })
                    .collect()
            })
            .collect(),
    )
}
/// Constructs Pascal entries `P[i,j]=C(i+j,i)` (symmetric), `C(i,j)` for `j<=i` (lower), or its transpose (upper).
///
/// ## Parameters
///
/// `n` is the order; `kind` selects orientation. SciPy exact integer mode is omitted.
///
/// ## Domain
///
/// Inputs must be finite. Output has at most 1,000,000 entries and each axis at most 1,000,000. Zero-sized outputs are allowed unless stated otherwise. Order must be at most 256.
///
/// ## Returns
///
/// An `n` by `n` matrix.
///
/// ## Errors
///
/// Returns a shared error for nonfinite inputs, invalid parameters, or the work bound; nonfinite output is an evaluation error.
///
/// ## Numerical behavior
///
/// Binomial recurrence with f64 rounding above 2^53; O(n³), n<=256.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::pascal(2,sci_linalg::PascalKind::Lower)?,vec![vec![1.,0.],vec![1.,1.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `pascal`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.pascal.html).
pub fn pascal(n: usize, kind: PascalKind) -> Result<Vec<Vec<f64>>> {
    if n > 256 {
        return Err(Error::InvalidInput("Pascal order exceeds 256"));
    }
    output(
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| match kind {
                        PascalKind::Symmetric => binomial(i + j, i),
                        PascalKind::Lower => {
                            if i >= j {
                                binomial(i, j)
                            } else {
                                0.0
                            }
                        }
                        PascalKind::Upper => {
                            if j >= i {
                                binomial(j, i)
                            } else {
                                0.0
                            }
                        }
                    })
                    .collect()
            })
            .collect(),
    )
}
/// Constructs the inverse Pascal matrix: triangular entries are `(-1)^(i+j) P[i,j]`; the symmetric inverse is `L⁻ᵀ L⁻¹`.
///
/// ## Parameters
///
/// `n` and `kind` match [`pascal`]. Integer exact mode is omitted.
///
/// ## Domain
///
/// Inputs must be finite. Output has at most 1,000,000 entries and each axis at most 1,000,000. Zero-sized outputs are allowed unless stated otherwise. Order must be at most 256.
///
/// ## Returns
///
/// The inverse as an `n` by `n` matrix.
///
/// ## Errors
///
/// Returns a shared error for nonfinite inputs, invalid parameters, or the work bound; nonfinite output is an evaluation error.
///
/// ## Numerical behavior
///
/// O(n³); large binomial values round in f64 and can amplify cancellation.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::invpascal(2,sci_linalg::PascalKind::Symmetric)?,vec![vec![2.,-1.],vec![-1.,1.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `invpascal`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.invpascal.html).
pub fn invpascal(n: usize, kind: PascalKind) -> Result<Vec<Vec<f64>>> {
    let mut p = pascal(
        n,
        if kind == PascalKind::Symmetric {
            PascalKind::Lower
        } else {
            kind
        },
    )?;
    for (i, row) in p.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            if (i + j) % 2 == 1 {
                *v = -*v;
            }
        }
    }
    if kind == PascalKind::Symmetric {
        return output(
            (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| (i.max(j)..n).map(|k| p[k][i] * p[k][j]).sum())
                        .collect()
                })
                .collect(),
        );
    }
    Ok(p)
}
/// Constructs Sylvester Hadamard matrices from `H₁=[1]` and `H₂n=[[Hn,Hn],[Hn,-Hn]]`, so `H Hᵀ=n I`.
///
/// ## Parameters
///
/// `n` is a positive power of two. Values use f64, without SciPy dtype selection.
///
/// ## Domain
///
/// Inputs must be finite. Output has at most 1,000,000 entries and each axis at most 1,000,000. Zero-sized outputs are allowed unless stated otherwise. Zero and orders that are not powers of two are invalid.
///
/// ## Returns
///
/// An `n` by `n` matrix of ±1.
///
/// ## Errors
///
/// Returns a shared error for nonfinite inputs, invalid parameters, or the work bound; nonfinite output is an evaluation error.
///
/// ## Numerical behavior
///
/// O(n²); entries are exact.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::hadamard(2)?,vec![vec![1.,1.],vec![1.,-1.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `hadamard`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.hadamard.html).
pub fn hadamard(n: usize) -> Result<Vec<Vec<f64>>> {
    size(n, n)?;
    if !n.is_power_of_two() {
        return Err(Error::InvalidInput(
            "Hadamard order must be a positive power of two",
        ));
    }
    Ok((0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if (i & j).count_ones() % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                })
                .collect()
        })
        .collect())
}
/// Constructs orthonormal contrasts: row k (1<=k<n) has `1/sqrt(k(k+1))` in columns 0..k, `-k/sqrt(k(k+1))` in column k, and zero elsewhere.
///
/// ## Parameters
///
/// `n` is positive; `full` includes the first row `1/sqrt(n)`. No batching.
///
/// ## Domain
///
/// Inputs must be finite. Output has at most 1,000,000 entries and each axis at most 1,000,000. Zero-sized outputs are allowed unless stated otherwise. n=0 is invalid.
///
/// ## Returns
///
/// An `n` by `n` matrix if full, otherwise `(n-1)` by `n`. For n=1 and full=false the empty representation loses its column count.
///
/// ## Errors
///
/// Returns a shared error for nonfinite inputs, invalid parameters, or the work bound; nonfinite output is an evaluation error.
///
/// ## Numerical behavior
///
/// O(n²); rows are orthonormal to rounding precision.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::helmert(1,true)?,vec![vec![1.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `helmert`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.helmert.html).
pub fn helmert(n: usize, full: bool) -> Result<Vec<Vec<f64>>> {
    size(n, n)?;
    if n == 0 {
        return Err(Error::InvalidInput("Helmert order must be positive"));
    }
    Ok((if full { 0 } else { 1 }..n)
        .map(|k| {
            (0..n)
                .map(|j| {
                    if k == 0 {
                        1.0 / (n as f64).sqrt()
                    } else {
                        let d = ((k * (k + 1)) as f64).sqrt();
                        if j < k {
                            1.0 / d
                        } else if j == k {
                            -(k as f64) / d
                        } else {
                            0.0
                        }
                    }
                })
                .collect()
        })
        .collect())
}
/// Constructs the symmetric distance matrix `F[i,j]=|a[i]-a[j]|`.
///
/// ## Parameters
///
/// `a` is a vector of real values. No batching.
///
/// ## Domain
///
/// Inputs must be finite. Output has at most 1,000,000 entries and each axis at most 1,000,000. Zero-sized outputs are allowed unless stated otherwise.
///
/// ## Returns
///
/// A `len(a)` square matrix.
///
/// ## Errors
///
/// Returns a shared error for nonfinite inputs, invalid parameters, or the work bound; nonfinite output is an evaluation error.
///
/// ## Numerical behavior
///
/// O(n²); subtraction overflow is reported.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::fiedler(&[2.,5.])?,vec![vec![0.,3.],vec![3.,0.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `fiedler`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.fiedler.html).
pub fn fiedler(a: &(impl super::VectorInput + ?Sized)) -> Result<Vec<Vec<f64>>> {
    let a_input = super::input::vector_slice(a)?;
    let a = a_input.as_ref();
    size(a.len(), a.len())?;
    crate::error::is_finite_slice(a)?;
    output(
        a.iter()
            .map(|x| a.iter().map(|y| (x - y).abs()).collect())
            .collect(),
    )
}
/// Constructs a polynomial companion matrix with first row `-a[1:]/a[0]` and ones on the first subdiagonal. Its eigenvalues are roots of `Σ a[k] x^(n-k)`.
///
/// ## Parameters
///
/// `a` contains descending polynomial coefficients. No batching.
///
/// ## Domain
///
/// Inputs must be finite. Output has at most 1,000,000 entries and each axis at most 1,000,000. Zero-sized outputs are allowed unless stated otherwise. At least two coefficients and a nonzero leading coefficient are required.
///
/// ## Returns
///
/// An `(len(a)-1)` square matrix.
///
/// ## Errors
///
/// Returns a shared error for nonfinite inputs, invalid parameters, or the work bound; nonfinite output is an evaluation error.
///
/// ## Numerical behavior
///
/// O(n²); coefficient ratios can overflow. Polynomial roots can be ill-conditioned.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::companion(&[1.,-3.,2.])?,vec![vec![3.,-2.],vec![1.,0.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `companion`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.companion.html).
pub fn companion(a: &(impl super::VectorInput + ?Sized)) -> Result<Vec<Vec<f64>>> {
    let a_input = super::input::vector_slice(a)?;
    let a = a_input.as_ref();
    if a.len() < 2 || a[0] == 0.0 {
        return Err(Error::InvalidInput(
            "companion requires degree >=1 and nonzero leading coefficient",
        ));
    }
    let n = a.len() - 1;
    size(n, n)?;
    crate::error::is_finite_slice(a)?;
    let mut c = vec![vec![0.0; n]; n];
    for j in 0..n {
        c[0][j] = -a[j + 1] / a[0];
    }
    for (i, row) in c.iter_mut().enumerate().skip(1) {
        row[i - 1] = 1.0;
    }
    output(c)
}
/// Constructs the pentadiagonal Fiedler companion of `Σ a[k] x^(n-k)`; its eigenvalues are the polynomial roots.
///
/// ## Parameters
///
/// `a` contains descending coefficients, with nonzero leading coefficient when nonempty. Uses SciPy's alternating pentadiagonal ordering; no batching.
///
/// ## Domain
///
/// Inputs must be finite. Output has at most 1,000,000 entries and each axis at most 1,000,000. Zero-sized outputs are allowed unless stated otherwise. Zero or one coefficient returns an empty matrix.
///
/// ## Returns
///
/// An `(len(a)-1)` square matrix for degree >=1.
///
/// ## Errors
///
/// Returns a shared error for nonfinite inputs, invalid parameters, or the work bound; nonfinite output is an evaluation error.
///
/// ## Numerical behavior
///
/// O(n²); ratios can overflow and roots can be ill-conditioned. See Fiedler (2003), DOI 10.1016/S0024-3795(03)00548-2.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::fiedler_companion(&[1.,-3.,2.])?,vec![vec![3.,-2.],vec![1.,0.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `fiedler_companion`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.fiedler_companion.html).
pub fn fiedler_companion(a: &(impl super::VectorInput + ?Sized)) -> Result<Vec<Vec<f64>>> {
    let a_input = super::input::vector_slice(a)?;
    let a = a_input.as_ref();
    crate::error::is_finite_slice(a)?;
    if a.is_empty() {
        return Ok(vec![]);
    }
    if a[0] == 0.0 {
        return Err(Error::InvalidInput("leading coefficient must be nonzero"));
    }
    if a.len() == 1 {
        return Ok(vec![]);
    }
    if a.len() == 2 {
        return companion(a);
    }
    let n = a.len() - 1;
    size(n, n)?;
    let mut c = vec![vec![0.0; n]; n];
    c[0][0] = -a[1] / a[0];
    c[1][0] = 1.0;
    for i in (2..n).step_by(2) {
        c[i][i - 1] = -a[i + 1] / a[0];
    }
    for i in (3..n).step_by(2) {
        c[i][i - 2] = 1.0;
    }
    for i in (0..n - 1).step_by(2) {
        c[i][i + 1] = -a[i + 2] / a[0];
    }
    for i in (0..n - 2).step_by(2) {
        c[i][i + 2] = 1.0;
    }
    output(c)
}
/// Constructs a population transition matrix with fecundities `L[0,j]=f[j]`, survival rates `L[i+1,i]=s[i]`, and zero elsewhere.
///
/// ## Parameters
///
/// `f` has n fecundities; `s` has n-1 survival rates. Values are not constrained to physical probabilities, matching SciPy.
///
/// ## Domain
///
/// Inputs must be finite. Output has at most 1,000,000 entries and each axis at most 1,000,000. Zero-sized outputs are allowed unless stated otherwise. n>=2 and matching lengths are required.
///
/// ## Returns
///
/// An `n` square matrix.
///
/// ## Errors
///
/// Returns a shared error for nonfinite inputs, invalid parameters, or the work bound; nonfinite output is an evaluation error.
///
/// ## Numerical behavior
///
/// O(n²) copying.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::leslie(&[2.,3.],&[0.5])?,vec![vec![2.,3.],vec![0.5,0.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `leslie`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.leslie.html).
pub fn leslie(
    f: &(impl super::VectorInput + ?Sized),
    s: &(impl super::VectorInput + ?Sized),
) -> Result<Vec<Vec<f64>>> {
    let f_input = super::input::vector_slice(f)?;
    let f = f_input.as_ref();
    let s_input = super::input::vector_slice(s)?;
    let s = s_input.as_ref();
    let n = f.len();
    if n < 2 || s.len() != n - 1 {
        return Err(Error::DimensionMismatch);
    }
    size(n, n)?;
    crate::error::is_finite_slice(f)?;
    crate::error::is_finite_slice(s)?;
    let mut a = vec![vec![0.0; n]; n];
    a[0].copy_from_slice(f);
    for (i, &v) in s.iter().enumerate() {
        a[i + 1][i] = v;
    }
    Ok(a)
}
/// Constructs `diag(A₀,A₁,...)`, placing input matrices on disjoint diagonal blocks and zeros elsewhere.
///
/// ## Parameters
///
/// `blocks` contains rectangular [`super::MatrixInput`] values. Libraries may
/// be mixed using `&dyn MatrixInput` entries. Only matrices are accepted;
/// SciPy scalar/vector promotion is absent. Empty library axes are retained
/// when summing dimensions; an empty shared block represents 0×0.
///
/// ## Domain
///
/// Inputs must be finite. Output has at most 1,000,000 entries and each axis at most 1,000,000. Zero-sized outputs are allowed unless stated otherwise.
///
/// ## Returns
///
/// A matrix whose row and column counts are the corresponding sums. An empty list returns an empty matrix.
///
/// ## Errors
///
/// Returns a shared error for nonfinite inputs, invalid parameters, or the work bound; nonfinite output is an evaluation error.
///
/// ## Numerical behavior
///
/// O(output entries); no arithmetic on input entries.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::block_diag(&[vec![vec![2.]],vec![vec![3.]]])?,vec![vec![2.,0.],vec![0.,3.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `block_diag`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.block_diag.html).
pub fn block_diag(blocks: &[impl super::MatrixInput]) -> Result<Vec<Vec<f64>>> {
    if blocks.len() > 1_000_000 {
        return Err(Error::InvalidInput("too many blocks"));
    }
    let mut dims = Vec::with_capacity(blocks.len());
    let (mut m, mut n) = (0usize, 0usize);
    for block in blocks {
        let (r, c) = super::input::checked_shape(block)?;
        for i in 0..r {
            for j in 0..c {
                match block.matrix_entry(i, j) {
                    value if value.is_finite() => Ok(value),
                    _ => Err(Error::NonFiniteEvaluation),
                }?;
            }
        }
        m = m
            .checked_add(r)
            .ok_or(Error::InvalidInput("block dimensions overflow"))?;
        n = n
            .checked_add(c)
            .ok_or(Error::InvalidInput("block dimensions overflow"))?;
        size(m, n)?;
        dims.push((r, c));
    }
    let mut result = vec![vec![0.0; n]; m];
    let (mut r0, mut c0) = (0, 0);
    for (block, (r, c)) in blocks.iter().zip(dims) {
        for i in 0..r {
            for j in 0..c {
                result[r0 + i][c0 + j] = block.matrix_entry(i, j);
            }
        }
        r0 += r;
        c0 += c;
    }
    Ok(result)
}
/// Constructs `C[i,j]=a[i+offset-j]` when the kernel index is valid, otherwise zero, so `C x` equals the selected linear convolution.
///
/// ## Parameters
///
/// `a` is a nonempty kernel; `n` is the positive input length; `mode` reuses [`sci_core::Mode`]. Same mode returns max(kernel length,n) rows, matching SciPy.
///
/// ## Domain
///
/// Inputs must be finite. Output has at most 1,000,000 entries and each axis at most 1,000,000. Zero-sized outputs are allowed unless stated otherwise. Both lengths must be positive.
///
/// ## Returns
///
/// Full: len(a)+n-1 rows; same: max(len(a),n); valid: |len(a)-n|+1. All outputs have n columns.
///
/// ## Errors
///
/// Returns a shared error for nonfinite inputs, invalid parameters, or the work bound; nonfinite output is an evaluation error.
///
/// ## Numerical behavior
///
/// O(rows × n); coefficients are copied exactly.
///
/// # Examples
///
/// ```
/// let c=sci_linalg::convolution_matrix(&[1.,2.],2,sci_signal::Mode::Full)?;
/// assert_eq!(c,vec![vec![1.,0.],vec![2.,1.],vec![0.,2.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `convolution_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.convolution_matrix.html).
pub fn convolution_matrix(
    a: &(impl super::VectorInput + ?Sized),
    n: usize,
    mode: sci_core::Mode,
) -> Result<Vec<Vec<f64>>> {
    let a_input = super::input::vector_slice(a)?;
    let a = a_input.as_ref();
    if a.is_empty() || n == 0 {
        return Err(Error::InvalidInput("convolution lengths must be positive"));
    }
    crate::error::is_finite_slice(a)?;
    let total = a
        .len()
        .checked_add(n - 1)
        .ok_or(Error::InvalidInput("convolution size overflow"))?;
    let (m, offset) = match mode {
        sci_core::Mode::Full => (total, 0),
        sci_core::Mode::Same => (a.len().max(n), (a.len().min(n) - 1) / 2),
        sci_core::Mode::Valid => (a.len().abs_diff(n) + 1, a.len().min(n) - 1),
    };
    size(m, n)?;
    Ok((0..m)
        .map(|i| {
            (0..n)
                .map(|j| {
                    (i + offset)
                        .checked_sub(j)
                        .filter(|&k| k < a.len())
                        .map_or(0.0, |k| a[k])
                })
                .collect()
        })
        .collect())
}
/// Constructs `F[j,k]=exp(-2π i j k/n)/s`, the negative-exponent discrete Fourier transform matrix.
///
/// ## Parameters
///
/// `n` is positive. `scale` sets s=1, sqrt(n), or n. Complex entries use [`Complex`].
///
/// ## Domain
///
/// Inputs must be finite. Output has at most 1,000,000 entries and each axis at most 1,000,000. Zero-sized outputs are allowed unless stated otherwise. n=0 is invalid.
///
/// ## Returns
///
/// An `n` square row-major complex matrix.
///
/// ## Errors
///
/// Returns a shared error for nonfinite inputs, invalid parameters, or the work bound; nonfinite output is an evaluation error.
///
/// ## Numerical behavior
///
/// O(n²); angles are reduced modulo n before evaluating sin/cos. No batching.
///
/// # Examples
///
/// ```
/// let f=sci_linalg::dft(2,sci_linalg::DftScale::None)?;
/// assert!((f[1][1].re+1.).abs()<1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `dft`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.dft.html).
pub fn dft(n: usize, scale: DftScale) -> Result<Vec<Vec<Complex>>> {
    size(n, n)?;
    if n == 0 {
        return Err(Error::InvalidInput("DFT order must be positive"));
    }
    let s = match scale {
        DftScale::None => 1.0,
        DftScale::SqrtN => (n as f64).sqrt(),
        DftScale::N => n as f64,
    };
    Ok((0..n)
        .map(|j| {
            (0..n)
                .map(|k| {
                    let (im, re) = (-TAU * ((j * k) % n) as f64 / n as f64).sin_cos();
                    Complex {
                        re: re / s,
                        im: im / s,
                    }
                })
                .collect()
        })
        .collect())
}
/// Constructs the rectangular diagonal SVD matrix `Σ[i,i]=s[i]`, with all other entries zero.
///
/// ## Parameters
///
/// `s` has min(m,n) entries, in supplied order; `m` and `n` are output dimensions. No requirement for positive or sorted values, matching SciPy.
///
/// ## Domain
///
/// Finite values are required. Zero dimensions are permitted with an empty s. At most 1,000,000 entries and each axis at most 1,000,000.
///
/// ## Returns
///
/// An m×n row-major matrix. Zero rows lose their column count.
///
/// ## Errors
///
/// Returns a shared error for incorrect length, nonfinite inputs, or work-bound violations.
///
/// ## Numerical behavior
///
/// O(mn) allocation; input entries are copied exactly.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::diagsvd(&[2.,1.],2,3)?,vec![vec![2.,0.,0.],vec![0.,1.,0.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `diagsvd`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.diagsvd.html).
pub fn diagsvd(
    s: &(impl super::VectorInput + ?Sized),
    m: usize,
    n: usize,
) -> Result<Vec<Vec<f64>>> {
    let s_input = super::input::vector_slice(s)?;
    let s = s_input.as_ref();
    size(m, n)?;
    if s.len() != m.min(n) {
        return Err(Error::DimensionMismatch);
    }
    crate::error::is_finite_slice(s)?;
    let mut a = vec![vec![0.0; n]; m];
    for (i, &v) in s.iter().enumerate() {
        a[i][i] = v;
    }
    Ok(a)
}
