//! Principal complex matrix functions using private Schur and solve backends.
use super::{ComplexMatrix, backend::*};
use crate::{
    Error, Result,
    complex::{Complex, native, owned},
};
use faer::{Mat, Scale};
use num_complex::Complex64 as C;
pub(super) fn complex_rows(a: &Mat<C>) -> Result<ComplexMatrix> {
    if a.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok((0..a.nrows())
        .map(|i| (0..a.ncols()).map(|j| owned(a[(i, j)])).collect())
        .collect())
}
pub(super) fn complex_dense(a: &[Vec<Complex>]) -> Result<Mat<C>> {
    let n = a.first().map_or(0, Vec::len);
    if a.is_empty() || n == 0 || a.len() > MAX_DIM || n > MAX_DIM || a.iter().any(|r| r.len() != n)
    {
        return Err(Error::DimensionMismatch);
    }
    if a.iter()
        .flatten()
        .any(|z| !z.re.is_finite() || !z.im.is_finite())
    {
        return Err(Error::InvalidInput("complex entries must be finite"));
    }
    Ok(Mat::from_fn(a.len(), n, |i, j| native(a[i][j])))
}
fn schur(a: Mat<C>) -> Result<(Mat<C>, Mat<C>)> {
    let scale = a
        .iter()
        .fold(0.0_f64, |s, z| s.max(z.re.abs()).max(z.im.abs()));
    let scale = if scale == 0. { 1. } else { scale };
    let normalized = a.map(|z| C::new(z.re / scale, z.im / scale));
    let (q, t) = complex_schur(normalized)?;
    let t = t * Scale(C::new(scale, 0.));
    complex_rows(&t)?;
    Ok((q, t))
}
fn triangular_root(t: &Mat<C>) -> Result<Mat<C>> {
    let n = t.nrows();
    let mut r = Mat::<C>::zeros(n, n);
    for i in 0..n {
        r[(i, i)] = t[(i, i)].sqrt();
    }
    for width in 1..n {
        for i in 0..n - width {
            let j = i + width;
            let mut value = t[(i, j)];
            for k in i + 1..j {
                value -= r[(i, k)] * r[(k, j)];
            }
            let denominator = r[(i, i)] + r[(j, j)];
            if denominator == C::new(0., 0.) {
                if value != C::new(0., 0.) {
                    return Err(Error::NumericalFailure(
                        "matrix has no supported principal square root",
                    ));
                }
            } else {
                r[(i, j)] = value / denominator;
            }
        }
    }
    complex_rows(&r)?;
    Ok(r)
}
fn root(a: Mat<C>) -> Result<Mat<C>> {
    let (q, t) = schur(a)?;
    Ok(&q * triangular_root(&t)? * q.adjoint())
}
fn logarithm(a: Mat<C>) -> Result<Mat<C>> {
    let (q, mut t) = schur(a)?;
    let n = t.nrows();
    if (0..n).any(|i| t[(i, i)] == C::new(0., 0.)) {
        return Err(Error::InvalidInput(
            "matrix logarithm requires nonsingular input",
        ));
    }
    let identity = Mat::<C>::identity(n, n);
    let mut count = 0;
    while (&t - &identity).norm_l2() > 0.25 {
        if count == 64 {
            return Err(Error::NumericalFailure(
                "logarithm inverse scaling limit reached",
            ));
        }
        t = triangular_root(&t)?;
        count += 1;
    }
    // log(I+E) = integral_0^1 (I+xE)^-1 E dx. Reuse the shared
    // Gauss-Legendre rule already used by the integration module.
    let (nodes, weights) = crate::special::roots_legendre(16)?;
    let e = t - &identity;
    let mut value = Mat::<C>::zeros(n, n);
    for (node, weight) in nodes.into_iter().zip(weights) {
        let x = (node + 1.) / 2.;
        let w = weight / 2.;
        let integrand = (&identity + &e * Scale(C::new(x, 0.)))
            .solve_checked(&e)
            .ok_or(Error::NumericalFailure(
                "singular logarithm quadrature system",
            ))?;
        value += integrand * Scale(C::new(w, 0.));
    }
    value *= Scale(C::new(2.0_f64.powi(count), 0.));
    Ok(&q * value * q.adjoint())
}
fn input(a: &[Vec<f64>]) -> Result<(Mat<C>, f64)> {
    let (a, scale) = scaled(square(a)?);
    Ok((a.map(|x| C::new(*x, 0.)), scale))
}
/// Computes a principal matrix square root `R R = A`.
///
/// ## Parameters
///
/// `a` is row-major; complex output uses shared Complex scalars.
///
/// ## Domain
///
/// Finite nonempty real square matrices, order at most 256.
/// Outputs can be complex, including on the negative real axis. NaN/infinity
/// are rejected. Branches use principal scalar arguments in (-π,π].
/// Singular matrices are accepted when the triangular root recurrence exists.
///
/// ## Returns
///
/// Complex square root of shape n × n.
///
/// ## Errors
///
/// Invalid shape/domain, nonfinite input/output, failed Schur/solve, or
/// iteration/work limit reached.
///
/// ## Numerical behavior
///
/// Private faer complex Schur and linear solves; O(n³) work.
/// Schur iteration uses faer’s bounded QZ sweeps on (A,I). Nonnormal or ill-conditioned
/// inputs can lose accuracy; overflow returns an error.
/// A native scalar triangular recurrence completes the backend Schur reduction.
/// A zero diagonal sum with nonzero numerator reports an unsupported root.
///
/// # Examples
///
/// ```
/// let r = sci_linalg::sqrtm(&[vec![-4.]])?;
/// assert!((r[0][0].im-2.).abs()<1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `sqrtm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sqrtm.html).
pub fn sqrtm(a: &(impl super::MatrixInput + ?Sized)) -> Result<ComplexMatrix> {
    let a_input = super::input::matrix_rows(a)?;
    let a = a_input.as_ref();
    let (a, scale) = input(a)?;
    complex_rows(&(root(a)? * Scale(C::new(scale.sqrt(), 0.))))
}
/// Computes the principal logarithm `exp(L) = A`.
///
/// ## Parameters
///
/// `a` is row-major. Negative-real eigenvalues use the upper-bank scalar branch.
///
/// ## Domain
///
/// Finite nonempty real square matrices, order at most 256.
/// Outputs can be complex, including on the negative real axis. NaN/infinity
/// are rejected. Branches use principal scalar arguments in (-π,π].
/// Input must be nonsingular; singular logarithms return errors.
///
/// ## Returns
///
/// Complex logarithm of shape n × n.
///
/// ## Errors
///
/// Invalid shape/domain, nonfinite input/output, failed Schur/solve, or
/// iteration/work limit reached.
///
/// ## Numerical behavior
///
/// Private faer complex Schur and linear solves; O(n³) work.
/// Schur iteration uses faer’s bounded QZ sweeps on (A,I). Nonnormal or ill-conditioned
/// inputs can lose accuracy; overflow returns an error.
/// Uses at most 64 triangular square roots followed by 16-point Gauss–Legendre
/// quadrature of the resolvent, evaluated with backend LU. Root scaling
/// reduces ||T-I||F to at most 0.25. Negative-axis results are branch-sensitive.
///
/// # Examples
///
/// ```
/// let l = sci_linalg::logm(&[vec![1.]])?;
/// assert!(l[0][0].re.abs()<1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `logm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.logm.html).
pub fn logm(a: &(impl super::MatrixInput + ?Sized)) -> Result<ComplexMatrix> {
    let a_input = super::input::matrix_rows(a)?;
    let a = a_input.as_ref();
    let (a, scale) = input(a)?;
    let mut value = logarithm(a)?;
    for i in 0..value.nrows() {
        value[(i, i)] += C::new(scale.ln(), 0.);
    }
    complex_rows(&value)
}
/// Computes the principal real power `A^t = exp(t log(A))`.
///
/// ## Parameters
///
/// `t` is a finite real exponent with |t| ≤ 2³². Integer powers use
/// backend exponentiation, including singular matrices for nonnegative powers.
///
/// ## Domain
///
/// Finite nonempty real square matrices, order at most 256.
/// Outputs can be complex, including on the negative real axis. NaN/infinity
/// are rejected. Branches use principal scalar arguments in (-π,π].
/// Noninteger or negative powers require a nonsingular input.
///
/// ## Returns
///
/// Complex power of shape n × n.
///
/// ## Errors
///
/// Invalid shape/domain, nonfinite input/output, failed Schur/solve, or
/// iteration/work limit reached.
///
/// ## Numerical behavior
///
/// Private faer complex Schur and linear solves; O(n³) work.
/// Schur iteration uses faer’s bounded QZ sweeps on (A,I). Nonnormal or ill-conditioned
/// inputs can lose accuracy; overflow returns an error.
/// Noninteger powers reuse logm and a Padé exponential evaluated through faer with pre-scaling.
/// Integer powers cover the entire closed exponent range, including ±2³²,
/// with at most 33 exponentiation iterations. The logarithm one-norm times |t|
/// must be at most 2³² for noninteger powers; squaring is bounded.
///
/// # Examples
///
/// ```
/// let p = sci_linalg::fractional_matrix_power(&[vec![4.]],0.5)?;
/// assert!((p[0][0].re-2.).abs()<1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `fractional_matrix_power`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.fractional_matrix_power.html).
pub fn fractional_matrix_power(
    a: &(impl super::MatrixInput + ?Sized),
    t: f64,
) -> Result<ComplexMatrix> {
    let a_input = super::input::matrix_rows(a)?;
    let a = a_input.as_ref();
    if !t.is_finite() || t.abs() > 4_294_967_296. {
        return Err(Error::InvalidInput(
            "exponent must be finite with magnitude at most 2^32",
        ));
    }
    let a = square(a)?.map(|x| C::new(*x, 0.));
    if t == t.trunc() {
        let base = if t < 0. {
            a.inverse_checked()
                .ok_or(Error::NumericalFailure("negative power of singular matrix"))?
        } else {
            a
        };
        return complex_rows(&matrix_power(base, t.abs() as u64));
    }
    let (a, scale) = scaled(a.map(|z| z.re));
    let mut value = logarithm(a.map(|x| C::new(*x, 0.)))?;
    for i in 0..value.nrows() {
        value[(i, i)] += C::new(scale.ln(), 0.);
    }
    value *= Scale(C::new(t, 0.));
    let norm = (0..value.ncols())
        .map(|j| value.col(j).iter().map(|z| z.norm()).sum::<f64>())
        .fold(0., f64::max);
    if !norm.is_finite() || norm > 4_294_967_296. {
        return Err(Error::InvalidInput(
            "matrix power exponential exceeds work bound",
        ));
    }
    let steps = if norm <= 0.5 {
        0
    } else {
        (norm.log2().ceil() + 1.) as u32
    };
    value *= Scale(C::new(2.0_f64.powi(-(steps as i32)), 0.));
    value = pade_exponential(&value)?;
    for _ in 0..steps {
        value = &value * &value;
    }
    complex_rows(&value)
}
/// Computes a matrix function `f(A)` by the Schur–Parlett recurrence.
///
/// ## Parameters
///
/// `function` maps a shared Complex eigenvalue to f(λ). It is evaluated once
/// per diagonal entry; outputs must be finite. The caller selects branches.
///
/// ## Domain
///
/// Finite nonempty real square matrices, order at most 256.
/// Outputs can be complex, including on the negative real axis. NaN/infinity
/// are rejected. Branches use principal scalar arguments in (-π,π].
/// For coupled repeated eigenvalues, scalar values do not determine derivative
/// terms; these cases return an error instead of silently omitting derivatives.
///
/// ## Returns
///
/// Complex f(A) of shape n × n.
///
/// ## Errors
///
/// Invalid shape/domain, nonfinite input/output, failed Schur/solve, or
/// iteration/work limit reached.
///
/// ## Numerical behavior
///
/// Private faer complex Schur and linear solves; O(n³) work.
/// Schur iteration uses faer’s bounded QZ sweeps on (A,I). Nonnormal or ill-conditioned
/// inputs can lose accuracy; overflow returns an error.
/// A native scalar divided-difference recurrence follows backend Schur.
/// Closely spaced eigenvalues can cause cancellation; repeated coupled
/// eigenvalues need derivatives and are unsupported. Callback panics propagate.
///
/// # Examples
///
/// ```
/// let f = sci_linalg::funm(&[vec![1.,0.],vec![0.,2.]],
///     |z| sci_complex::exp(z))?;
/// assert!((f[0][0].re-std::f64::consts::E).abs()<1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `funm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.funm.html).
pub fn funm<F: FnMut(Complex) -> Complex>(
    a: &(impl super::MatrixInput + ?Sized),
    mut function: F,
) -> Result<ComplexMatrix> {
    let a_input = super::input::matrix_rows(a)?;
    let a = a_input.as_ref();
    let a = square(a)?.map(|x| C::new(*x, 0.));
    let (q, t) = schur(a)?;
    let n = t.nrows();
    let mut f = Mat::<C>::zeros(n, n);
    for i in 0..n {
        f[(i, i)] = native(function(owned(t[(i, i)])));
    }
    complex_rows(&f)?;
    for width in 1..n {
        for i in 0..n - width {
            let j = i + width;
            let mut numerator = t[(i, j)] * (f[(j, j)] - f[(i, i)]);
            for k in i + 1..j {
                numerator += t[(i, k)] * f[(k, j)] - f[(i, k)] * t[(k, j)];
            }
            let denominator = t[(j, j)] - t[(i, i)];
            if denominator.norm() <= f64::EPSILON * (1. + t[(i, i)].norm() + t[(j, j)].norm()) {
                if t[(i, j)] != C::new(0., 0.) || numerator != C::new(0., 0.) {
                    return Err(Error::NumericalFailure(
                        "funm requires derivatives for coupled repeated eigenvalues",
                    ));
                }
            } else {
                f[(i, j)] = numerator / denominator;
            }
        }
    }
    complex_rows(&(&q * f * q.adjoint()))
}
/// Computes the matrix sign, `sign(A) = A (A²)^(-1/2)`.
///
/// ## Parameters
///
/// `a` is row-major. Scalar eigenvalues with positive/negative real part
/// map to +1/-1. Eigenvalues on the imaginary axis are excluded.
///
/// ## Domain
///
/// Finite nonempty real square matrices, order at most 256.
/// Outputs can be complex, including on the negative real axis. NaN/infinity
/// are rejected. Branches use principal scalar arguments in (-π,π].
/// Requires no eigenvalue on or within relative machine epsilon of
/// the imaginary axis.
///
/// ## Returns
///
/// Complex sign matrix of shape n × n.
///
/// ## Errors
///
/// Invalid shape/domain, nonfinite input/output, failed Schur/solve, or
/// iteration/work limit reached.
///
/// ## Numerical behavior
///
/// Private faer complex Schur and linear solves; O(n³) work.
/// Schur iteration uses faer’s bounded QZ sweeps on (A,I). Nonnormal or ill-conditioned
/// inputs can lose accuracy; overflow returns an error.
/// Uses backend Schur to validate the spectrum, then principal sqrt(A²)
/// and a backend LU solve. Squaring can overflow for extreme inputs.
///
/// # Examples
///
/// ```
/// let s = sci_linalg::signm(&[vec![-2.,0.],vec![0.,3.]])?;
/// assert!((s[0][0].re+1.).abs()<1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `signm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.signm.html).
pub fn signm(a: &(impl super::MatrixInput + ?Sized)) -> Result<ComplexMatrix> {
    let a_input = super::input::matrix_rows(a)?;
    let a = a_input.as_ref();
    let (a, _) = input(a)?;
    let (_, t) = schur(a.clone())?;
    if (0..t.nrows()).any(|i| t[(i, i)].re.abs() <= f64::EPSILON * t[(i, i)].norm()) {
        return Err(Error::InvalidInput(
            "matrix sign excludes imaginary-axis eigenvalues",
        ));
    }
    let root = root(&a * &a)?;
    let value = root
        .solve_checked(&a)
        .ok_or(Error::NumericalFailure("singular matrix-sign denominator"))?;
    complex_rows(&value)
}
/// Converts `A = Z T Zᵀ` into a complex triangular Schur form.
///
/// ## Parameters
///
/// `t` and `z` are equal-order real square matrices. z must be orthogonal
/// and t quasi-upper-triangular (only disjoint two-by-two subdiagonal blocks).
///
/// ## Domain
///
/// Finite real matrices of order at most 256. NaN/infinity are rejected.
///
/// ## Returns
///
/// Tuple (complex T, complex Z), preserving A. Backend ordering and phases
/// can differ from SciPy.
///
/// ## Errors
///
/// Invalid shape/domain, nonfinite input/output, failed Schur/solve, or
/// iteration/work limit reached.
///
/// ## Numerical behavior
///
/// Private faer complex Schur and linear solves; O(n³) work.
/// Schur iteration uses faer’s bounded QZ sweeps on (A,I). Nonnormal or ill-conditioned
/// inputs can lose accuracy; overflow returns an error.
/// Reuses complex Schur on the supplied real Schur form. Orthogonality and
/// structure are checked with 1e-10 absolute tolerance.
///
/// # Examples
///
/// ```
/// let (t,z) = sci_linalg::rsf2csf(&[vec![0.,-1.],vec![1.,0.]],
///     &[vec![1.,0.],vec![0.,1.]])?;
/// assert_eq!(t.len(),z.len());
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `rsf2csf`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.rsf2csf.html).
pub fn rsf2csf(
    t: &(impl super::MatrixInput + ?Sized),
    z: &(impl super::MatrixInput + ?Sized),
) -> Result<(ComplexMatrix, ComplexMatrix)> {
    let t_input = super::input::matrix_rows(t)?;
    let t = t_input.as_ref();
    let z_input = super::input::matrix_rows(z)?;
    let z = z_input.as_ref();
    let t = square(t)?;
    let z = square(z)?;
    let n = t.nrows();
    if z.nrows() != n {
        return Err(Error::DimensionMismatch);
    }
    if (z.transpose() * &z - Mat::<f64>::identity(n, n)).norm_max() > 1e-10 {
        return Err(Error::InvalidInput("Schur vectors must be orthogonal"));
    }
    for i in 0..n {
        for j in 0..i.saturating_sub(1) {
            if t[(i, j)].abs() > 1e-10 {
                return Err(Error::InvalidInput("T must be quasi triangular"));
            }
        }
        if i > 1 && t[(i, i - 1)] != 0. && t[(i - 1, i - 2)] != 0. {
            return Err(Error::InvalidInput("overlapping Schur blocks"));
        }
    }
    let (q, t) = schur(t.map(|x| C::new(*x, 0.)))?;
    Ok((
        complex_rows(&t)?,
        complex_rows(&(z.map(|x| C::new(*x, 0.)) * q))?,
    ))
}

fn matrix_power(mut a: Mat<C>, mut exponent: u64) -> Mat<C> {
    let mut result = Mat::<C>::identity(a.nrows(), a.ncols());
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = &result * &a;
        }
        exponent >>= 1;
        if exponent != 0 {
            a = &a * &a;
        }
    }
    result
}
