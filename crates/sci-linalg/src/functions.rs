//! Matrix functions using bounded backend scaling and squaring.
use super::backend::*;
use super::dense::{Matrix, svdvals};
use crate::{Error, Result};
use faer::{Mat, Scale};
use num_complex::Complex64;
const MAX_ONE_NORM: f64 = 4_294_967_296.0;
fn one_norm(a: &Mat<f64>) -> f64 {
    (0..a.ncols())
        .map(|j| a.col(j).iter().map(|x| x.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}
fn scaling(a: &Mat<f64>) -> Result<u32> {
    let norm = one_norm(a);
    if !norm.is_finite() || norm > MAX_ONE_NORM {
        return Err(Error::InvalidInput("matrix function one-norm exceeds 2^32"));
    }
    // Bound Padé intermediates and the number of subsequent squarings.
    Ok(if norm <= 0.5 {
        0
    } else {
        (norm.log2().ceil() + 1.0) as u32
    })
}
fn exponential(a: &Mat<f64>) -> Result<Mat<f64>> {
    let steps = scaling(a)?;
    let mut result =
        pade_exponential(&a.map(|x| Complex64::new(x * 2.0_f64.powi(-(steps as i32)), 0.0)))?
            .map(|z| z.re);
    for _ in 0..steps {
        result = &result * &result;
        if result.iter().any(|x| !x.is_finite()) {
            return Err(Error::NonFiniteEvaluation);
        }
    }
    if result.iter().any(|x| !x.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok(result)
}
fn circular(a: &(impl super::MatrixInput + ?Sized), sine: bool) -> Result<Matrix> {
    let a = square(a)?;
    let steps = scaling(&a)?;
    let imaginary = a.map(|x| Complex64::new(0.0, x / 2.0_f64.powi(steps as i32)));
    let mut e = pade_exponential(&imaginary)?;
    for _ in 0..steps {
        e = &e * &e;
        if e.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
            return Err(Error::NonFiniteEvaluation);
        }
    }
    rows(&e.map(|z| if sine { z.im } else { z.re }))
}
// Evaluate the even and odd series independently after pre-scaling. This
// avoids subtracting nearly identical exponentials for a small sinh(A).
fn hyperbolic_base(a: &Mat<f64>, steps: u32) -> (Mat<f64>, Mat<f64>) {
    let a = a * Scale(2.0_f64.powi(-(steps as i32)));
    let a2 = &a * &a;
    let mut sine = a.clone();
    let mut cosine = Mat::<f64>::identity(a.nrows(), a.ncols());
    let mut odd = a;
    let mut even = cosine.clone();
    // ||A||_1 <= 1/2: the first omitted scalar term is below 2e-34.
    for k in 1..=12 {
        odd = (&odd * &a2) * Scale(1.0 / ((2 * k) * (2 * k + 1)) as f64);
        even = (&even * &a2) * Scale(1.0 / ((2 * k - 1) * (2 * k)) as f64);
        sine += &odd;
        cosine += &even;
    }
    (sine, cosine)
}
fn hyperbolic(a: &Mat<f64>) -> Result<(Mat<f64>, Mat<f64>)> {
    let steps = scaling(a)?;
    let (mut sine, mut cosine) = hyperbolic_base(a, steps);
    let identity = Mat::<f64>::identity(a.nrows(), a.ncols());
    for _ in 0..steps {
        let next_cosine = &identity + (&sine * &sine) * Scale(2.0);
        sine = (&sine * &cosine) * Scale(2.0);
        cosine = next_cosine;
        if sine.iter().chain(cosine.iter()).any(|x| !x.is_finite()) {
            return Err(Error::NonFiniteEvaluation);
        }
    }
    Ok((sine, cosine))
}
fn quotient(numerator: Matrix, denominator: Matrix) -> Result<Matrix> {
    let n = dense(&numerator)?;
    let d = square(&denominator)?;
    rows(&d.solve_checked(&n).ok_or(Error::NumericalFailure(
        "singular matrix-function denominator",
    ))?)
}
/// Computes the matrix exponential `exp(A)=Σk>=0 A^k/k!`.
///
/// ## Parameters
///
/// `a` is a real square matrix.
///
/// ## Domain
///
/// Real finite square matrices of order 1..=256; no NaN or infinity. The input matrix (or augmented matrix) one-norm must be at most 2^32.
///
/// ## Returns
///
/// A real square matrix exp(A).
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, nonfinite inputs, work-bound violations, singular denominators where applicable, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private [13/13] Padé evaluation through faer products and LU solves with explicit pre-scaling and at most 34 squarings. Computation is O(n³) per product. No SciPy batching or overwrite options. Accuracy degrades for large norms and nonnormal matrices. Pre-scaling bounds Padé intermediates and squaring work. True exponential overflow is an error; underflow may return zero.
///
/// # Examples
///
/// ```
/// let e=sci_linalg::expm(&[vec![0.,1.],vec![0.,0.]])?;
/// let expected=[[1.,1.],[0.,1.]];
/// assert!(e.iter().flatten().zip(expected.iter().flatten()).all(|(a,b)| (a-b).abs()<1e-14));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [Higham (2005), scaling and squaring](https://doi.org/10.1137/04061101X).
///
/// [SciPy `expm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html).
pub fn expm(a: &(impl super::MatrixInput + ?Sized)) -> Result<Matrix> {
    rows(&exponential(&square(a)?)?)
}
/// Computes the matrix cosine `cos(A)=Σk>=0 (-1)^k A^(2k)/(2k)!`.
///
/// ## Parameters
///
/// `a` is a real square matrix; angles are radians in the scalar case.
///
/// ## Domain
///
/// Real finite square matrices of order 1..=256; no NaN or infinity. The input matrix (or augmented matrix) one-norm must be at most 2^32.
///
/// ## Returns
///
/// The real matrix cosine.
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, nonfinite inputs, work-bound violations, singular denominators where applicable, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private [13/13] Padé evaluation through faer products and LU solves with explicit pre-scaling and at most 34 squarings. Computation is O(n³) per product. No SciPy batching or overwrite options. Accuracy degrades for large norms and nonnormal matrices. Uses the imaginary or real part of exp(i A), computed with the private complex backend. A real input always returns real entries.
///
/// # Examples
///
/// ```
/// assert!((sci_linalg::cosm(&[vec![0.]])?[0][0]-1.).abs()<1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [Higham (2005), scaling and squaring](https://doi.org/10.1137/04061101X).
///
/// [SciPy `cosm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cosm.html).
pub fn cosm(a: &(impl super::MatrixInput + ?Sized)) -> Result<Matrix> {
    circular(a, false)
}
/// Computes the matrix sine `sin(A)=Σk>=0 (-1)^k A^(2k+1)/(2k+1)!`.
///
/// ## Parameters
///
/// `a` is a real square matrix; angles are radians in the scalar case.
///
/// ## Domain
///
/// Real finite square matrices of order 1..=256; no NaN or infinity. The input matrix (or augmented matrix) one-norm must be at most 2^32.
///
/// ## Returns
///
/// The real matrix sine.
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, nonfinite inputs, work-bound violations, singular denominators where applicable, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private [13/13] Padé evaluation through faer products and LU solves with explicit pre-scaling and at most 34 squarings. Computation is O(n³) per product. No SciPy batching or overwrite options. Accuracy degrades for large norms and nonnormal matrices. Uses the imaginary or real part of exp(i A), computed with the private complex backend. A real input always returns real entries.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::sinm(&[vec![0.]])?,vec![vec![0.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [Higham (2005), scaling and squaring](https://doi.org/10.1137/04061101X).
///
/// [SciPy `sinm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sinm.html).
pub fn sinm(a: &(impl super::MatrixInput + ?Sized)) -> Result<Matrix> {
    circular(a, true)
}
/// Computes `cosh(A)=(exp(A)+exp(-A))/2`.
///
/// ## Parameters
///
/// `a` is a real square matrix.
///
/// ## Domain
///
/// Real finite square matrices of order 1..=256; no NaN or infinity. The input matrix (or augmented matrix) one-norm must be at most 2^32.
///
/// ## Returns
///
/// The real matrix hyperbolic cosine.
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, nonfinite inputs, work-bound violations, singular denominators where applicable, or nonfinite output.
///
/// ## Numerical behavior
///
/// Independent even/odd Taylor series through private faer products, pre-scaled to one-norm <= 1/2. Twelve updates bound the omitted term below 2e-34, followed by at most 34 double-angle steps. This preserves small sine values without exponential subtraction. Computation is O(n³) per product; large or nonnormal arguments remain sensitive to roundoff. Overflow returns an error.
///
/// # Examples
///
/// ```
/// assert!((sci_linalg::coshm(&[vec![0.]])?[0][0]-1.).abs()<1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [Higham (2005), scaling and squaring](https://doi.org/10.1137/04061101X).
///
/// [SciPy `coshm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.coshm.html).
pub fn coshm(a: &(impl super::MatrixInput + ?Sized)) -> Result<Matrix> {
    let a = square(a)?;
    rows(&hyperbolic(&a)?.1)
}
/// Computes `sinh(A)=(exp(A)-exp(-A))/2`.
///
/// ## Parameters
///
/// `a` is a real square matrix.
///
/// ## Domain
///
/// Real finite square matrices of order 1..=256; no NaN or infinity. The input matrix (or augmented matrix) one-norm must be at most 2^32.
///
/// ## Returns
///
/// The real matrix hyperbolic sine.
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, nonfinite inputs, work-bound violations, singular denominators where applicable, or nonfinite output.
///
/// ## Numerical behavior
///
/// Independent even/odd Taylor series through private faer products, pre-scaled to one-norm <= 1/2. Twelve updates bound the omitted term below 2e-34, followed by at most 34 double-angle steps. This preserves small sine values without exponential subtraction. Computation is O(n³) per product; large or nonnormal arguments remain sensitive to roundoff. Overflow returns an error.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::sinhm(&[vec![0.]])?,vec![vec![0.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [Higham (2005), scaling and squaring](https://doi.org/10.1137/04061101X).
///
/// [SciPy `sinhm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sinhm.html).
pub fn sinhm(a: &(impl super::MatrixInput + ?Sized)) -> Result<Matrix> {
    let a = square(a)?;
    rows(&hyperbolic(&a)?.0)
}
/// Computes `tan(A)=cos(A)⁻¹ sin(A)` wherever the denominator is invertible.
///
/// ## Parameters
///
/// `a` is a real square matrix.
///
/// ## Domain
///
/// Real finite square matrices of order 1..=256; no NaN or infinity. The input matrix (or augmented matrix) one-norm must be at most 2^32. cos(A) must be invertible.
///
/// ## Returns
///
/// The matrix tangent.
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, nonfinite inputs, work-bound violations, singular denominators where applicable, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private [13/13] Padé evaluation through faer products and LU solves with explicit pre-scaling and at most 34 squarings. Computation is O(n³) per product. No SciPy batching or overwrite options. Accuracy degrades for large norms and nonnormal matrices. Uses pivoted LU after sin/cos evaluation. Near poles, finite results can have large relative error; no conditioning warning is emitted.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::tanm(&[vec![0.]])?,vec![vec![0.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [Higham (2005), scaling and squaring](https://doi.org/10.1137/04061101X).
///
/// [SciPy `tanm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.tanm.html).
pub fn tanm(a: &(impl super::MatrixInput + ?Sized)) -> Result<Matrix> {
    quotient(sinm(a)?, cosm(a)?)
}
/// Computes `tanh(A)=cosh(A)⁻¹ sinh(A)` wherever the denominator is invertible.
///
/// ## Parameters
///
/// `a` is a real square matrix.
///
/// ## Domain
///
/// Real finite square matrices of order 1..=256; no NaN or infinity. The input matrix (or augmented matrix) one-norm must be at most 2^32. cosh(A) must be invertible.
///
/// ## Returns
///
/// The matrix hyperbolic tangent.
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, nonfinite inputs, work-bound violations, singular denominators where applicable, or nonfinite output.
///
/// ## Numerical behavior
///
/// Even/odd Taylor series at one-norm <= 1/2, followed by the rational
/// double-angle identity `T(2A)=2T(A)(I+T(A)^2)⁻¹`, at most 34 times.
/// Large real eigenvalues can saturate without forming overflowing exponentials.
/// Private faer products and LU solves take O(n³) per step. Denominators must be
/// invertible; nonnormality and proximity to poles can reduce accuracy.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_linalg::tanhm(&[vec![0.]])?,vec![vec![0.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [Higham (2005), scaling and squaring](https://doi.org/10.1137/04061101X).
///
/// [SciPy `tanhm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.tanhm.html).
pub fn tanhm(a: &(impl super::MatrixInput + ?Sized)) -> Result<Matrix> {
    let a = square(a)?;
    let steps = scaling(&a)?;
    let (sine, cosine) = hyperbolic_base(&a, steps);
    let mut tangent = cosine.solve_checked(&sine).ok_or(Error::NumericalFailure(
        "singular matrix-function denominator",
    ))?;
    let identity = Mat::<f64>::identity(a.nrows(), a.ncols());
    for _ in 0..steps {
        let denominator = &identity + &tangent * &tangent;
        tangent =
            denominator
                .solve_checked(&(tangent * Scale(2.0)))
                .ok_or(Error::NumericalFailure(
                    "singular matrix-function denominator",
                ))?;
        if tangent.iter().any(|x| !x.is_finite()) {
            return Err(Error::NonFiniteEvaluation);
        }
    }
    rows(&tangent)
}
/// Computes the Fréchet derivative `Lexp(A,E)=d/dt exp(A+tE)|t=0`, together with exp(A).
///
/// ## Parameters
///
/// `a` and `e` are equal-size real square matrices. The augmented block `[[A,E],[0,A]]` must have order <=256, so input order is <=128. SciPy method and compute_expm switches are omitted.
///
/// ## Domain
///
/// Real finite square matrices of order 1..=256; no NaN or infinity. The input matrix (or augmented matrix) one-norm must be at most 2^32.
///
/// ## Returns
///
/// (exp(A),Lexp(A,E)), matching SciPy's default output order.
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, nonfinite inputs, work-bound violations, singular denominators where applicable, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private [13/13] Padé evaluation through faer products and LU solves with explicit pre-scaling and at most 34 squarings. Computation is O(n³) per product. No SciPy batching or overwrite options. Accuracy degrades for large norms and nonnormal matrices. Uses the exact block-exponential identity, replacing SciPy's default differentiated scaling-and-squaring algorithm.
///
/// # Examples
///
/// ```
/// let (exp,d)=sci_linalg::expm_frechet(&[vec![0.]],&[vec![2.]])?;
/// assert!((exp[0][0]-1.).abs()<1e-14 && (d[0][0]-2.).abs()<1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [Higham (2005), scaling and squaring](https://doi.org/10.1137/04061101X).
///
/// [SciPy `expm_frechet`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm_frechet.html).
pub fn expm_frechet(
    a: &(impl super::MatrixInput + ?Sized),
    e: &(impl super::MatrixInput + ?Sized),
) -> Result<(Matrix, Matrix)> {
    let a = square(a)?;
    let e = square(e)?;
    if a.shape() != e.shape() {
        return Err(Error::DimensionMismatch);
    }
    let n = a.nrows();
    if n > MAX_DIM / 2 {
        return Err(Error::InvalidInput("Frechet derivative order exceeds 128"));
    }
    let block = Mat::from_fn(2 * n, 2 * n, |i, j| {
        if i < n && j < n {
            a[(i, j)]
        } else if i < n {
            e[(i, j - n)]
        } else if j < n {
            0.0
        } else {
            a[(i - n, j - n)]
        }
    });
    let exp = exponential(&block)?;
    Ok((
        rows(&exp.as_ref().submatrix(0, 0, n, n))?,
        rows(&exp.as_ref().submatrix(0, n, n, n))?,
    ))
}
/// Computes the relative Frobenius condition number `κ=||Lexp(A)||F→F ||A||F / ||exp(A)||F`.
///
/// ## Parameters
///
/// `a` is real square of order <=8. The derivative operator is represented by all n² coordinate directions; no estimate mode.
///
/// ## Domain
///
/// Real finite square matrices of order 1..=256; no NaN or infinity. The input matrix (or augmented matrix) one-norm must be at most 2^32. The exponential norm must not underflow to zero.
///
/// ## Returns
///
/// A nonnegative scalar; A=0 returns zero.
///
/// ## Errors
///
/// Returns a shared error for invalid shapes, nonfinite inputs, work-bound violations, singular denominators where applicable, or nonfinite output.
///
/// ## Numerical behavior
///
/// Private [13/13] Padé evaluation through faer products and LU solves with explicit pre-scaling and at most 34 squarings. Computation is O(n³) per product. No SciPy batching or overwrite options. Accuracy degrades for large norms and nonnormal matrices. n² block exponentials and an n²×n² SVD; fixed order bound 8 controls O(n⁶) work and storage. Relative norm calculations may overflow.
///
/// # Examples
///
/// ```
/// assert!((sci_linalg::expm_cond(&[vec![2.]])?-2.).abs()<1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [Higham (2005), scaling and squaring](https://doi.org/10.1137/04061101X).
///
/// [SciPy `expm_cond`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm_cond.html).
pub fn expm_cond(a: &(impl super::MatrixInput + ?Sized)) -> Result<f64> {
    let a = square(a)?;
    let n = a.nrows();
    if n > 8 {
        return Err(Error::InvalidInput(
            "exponential condition number order exceeds 8",
        ));
    }
    let a_rows = rows(&a)?;
    let exp = exponential(&a)?;
    let norm = a.iter().fold(0.0_f64, |v, &x| v.hypot(x));
    if norm == 0.0 {
        return Ok(0.0);
    }
    let exp_norm = exp.iter().fold(0.0_f64, |v, &x| v.hypot(x));
    if exp_norm == 0.0 {
        return Err(Error::NumericalFailure(
            "exponential norm underflowed to zero",
        ));
    }
    let d = n * n;
    let mut operator = vec![vec![0.0; d]; d];
    for column in 0..d {
        let mut direction = vec![vec![0.0; n]; n];
        direction[column / n][column % n] = 1.0;
        let (_, derivative) = expm_frechet(&a_rows, &direction)?;
        for (i, row) in derivative.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                operator[i * n + j][column] = v;
            }
        }
    }
    match (svdvals(&operator)?[0] / exp_norm) * norm {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}
