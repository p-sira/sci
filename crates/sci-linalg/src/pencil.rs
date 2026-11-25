//! Generalized complex Schur reductions with faer privately.
use super::{
    ComplexMatrix,
    backend::*,
    principal::{complex_dense, complex_rows},
};
use crate::{
    Error, Result,
    complex::{Complex, owned},
};
use dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::{Mat, Scale};
use num_complex::Complex64 as C;
/// Generalized Schur factorization `A=Q S Zᴴ`, `B=Q T Zᴴ`.
#[derive(Clone, Debug)]
pub struct QzResult {
    /// Complex upper triangular S.
    pub s: ComplexMatrix,
    /// Complex upper triangular T.
    pub t: ComplexMatrix,
    /// Left unitary Schur vectors.
    pub q: ComplexMatrix,
    /// Right unitary Schur vectors.
    pub z: ComplexMatrix,
    /// Homogeneous eigenvalue numerators, diagonal of S.
    pub alpha: Vec<Complex>,
    /// Homogeneous eigenvalue denominators, diagonal of T.
    pub beta: Vec<Complex>,
    /// Number of eigenvalues moved to the leading block; zero for unsorted QZ.
    pub selected: usize,
}
fn factor(s: Mat<C>, t: Mat<C>, q: Mat<C>, z: Mat<C>, selected: usize) -> Result<QzResult> {
    let n = s.nrows();
    Ok(QzResult {
        alpha: (0..n).map(|i| owned(s[(i, i)])).collect(),
        beta: (0..n).map(|i| owned(t[(i, i)])).collect(),
        s: complex_rows(&s)?,
        t: complex_rows(&t)?,
        q: complex_rows(&q)?,
        z: complex_rows(&z)?,
        selected,
    })
}
/// Computes generalized Schur forms `A=Q S Zᴴ`, `B=Q T Zᴴ`.
///
/// ## Parameters
///
/// `a` and `b` are equal-order real square matrices. Always returns complex
/// triangular output, replacing SciPy output="real"/"complex" selection.
///
/// ## Domain
///
/// Finite nonempty matrices with order at most 256. Singular B is supported.
/// Zero beta denotes an infinite eigenvalue; alpha=beta=0 is indeterminate.
/// NaN/infinity inputs are rejected.
///
/// ## Returns
///
/// Owned QzResult with factors, homogeneous alpha/beta, and selected=0.
/// Eigenvalue order and phases are unspecified.
///
/// ## Errors
///
/// Invalid dimensions, nonfinite factors, or a failed triangular reduction.
///
/// ## Numerical behavior
///
/// Uses private faer QR and real generalized Hessenberg/QZ to separate
/// real blocks, then faer complex QZ to obtain triangular output.
/// Backend sweep limits scale with order. Upper-triangular output is verified;
/// ill-conditioned matrix pencils can lose accuracy. O(n³) work, O(n²) storage.
///
/// # Examples
///
/// ```
/// let f = sci_linalg::eigen::qz(&[vec![2.]], &[vec![1.]])?;
/// assert_eq!(f.alpha.len(),1);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `qz`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qz.html).
pub fn qz(
    a: &(impl super::MatrixInput + ?Sized),
    b: &(impl super::MatrixInput + ?Sized),
) -> Result<QzResult> {
    let a = square(a)?;
    let b = square(b)?;
    if a.shape() != b.shape() {
        return Err(Error::DimensionMismatch);
    }
    let n = a.nrows();
    let (a, sa) = scaled(a);
    let (b, sb) = scaled(b);
    // First separate real Schur blocks. This avoids the complex QZ zero-beta
    // chase losing accumulated rotations on exactly singular real pencils.
    let (q, r) = b.qr_factors();
    let mut s = q.transpose() * &a;
    let mut t = r;
    let mut q = q;
    let mut z = Mat::<f64>::identity(n, n);
    let hp = faer::auto!(f64);
    let qp = faer::auto!(f64);
    let request = StackReq::any_of(&[
        faer::linalg::gevd::gen_hessenberg::generalized_hessenberg_scratch::<f64>(n, hp),
        faer::linalg::gevd::qz_real::hessenberg_to_qz_scratch::<f64>(n, faer::Par::Seq, qp),
    ]);
    let mut buffer = MemBuffer::new(request);
    faer::linalg::gevd::gen_hessenberg::generalized_hessenberg(
        s.as_mut(),
        t.as_mut(),
        Some(q.as_mut()),
        Some(z.as_mut()),
        faer::Par::Seq,
        MemStack::new(&mut buffer),
        hp,
    );
    let mut ar = faer::Col::zeros(n);
    let mut ai = faer::Col::zeros(n);
    let mut beta = faer::Col::zeros(n);
    faer::linalg::gevd::qz_real::hessenberg_to_qz(
        s.as_mut(),
        t.as_mut(),
        Some(q.as_mut()),
        Some(z.as_mut()),
        ar.as_mut(),
        ai.as_mut(),
        beta.as_mut(),
        faer::linalg::gevd::ComputeEigenvectors::Yes,
        faer::Par::Seq,
        qp,
        MemStack::new(&mut buffer),
    );
    let mut s = s.map(|x| C::new(*x, 0.));
    let mut t = t.map(|x| C::new(*x, 0.));
    let mut q = q.map(|x| C::new(*x, 0.));
    let mut z = z.map(|x| C::new(*x, 0.));
    let mut alpha = faer::Col::<C>::zeros(n);
    let mut beta = faer::Col::<C>::zeros(n);
    let hparams = faer::auto!(C);
    let qparams = faer::auto!(C);
    let request = StackReq::any_of(&[
        faer::linalg::gevd::gen_hessenberg::generalized_hessenberg_scratch::<C>(n, hparams),
        faer::linalg::gevd::qz_cplx::hessenberg_to_qz_scratch::<C>(n, faer::Par::Seq, qparams),
    ]);
    let mut buffer = MemBuffer::new(request);
    faer::linalg::gevd::gen_hessenberg::generalized_hessenberg(
        s.as_mut(),
        t.as_mut(),
        Some(q.as_mut()),
        Some(z.as_mut()),
        faer::Par::Seq,
        MemStack::new(&mut buffer),
        hparams,
    );
    faer::linalg::gevd::qz_cplx::hessenberg_to_qz(
        s.as_mut(),
        t.as_mut(),
        Some(q.as_mut()),
        Some(z.as_mut()),
        alpha.as_mut(),
        beta.as_mut(),
        faer::linalg::gevd::ComputeEigenvectors::Yes,
        faer::Par::Seq,
        qparams,
        MemStack::new(&mut buffer),
    );
    let q = Mat::from_fn(n, n, |i, j| q[(i, j)]);
    let z = Mat::from_fn(n, n, |i, j| z[(i, j)]);
    // Recover transformed forms from the original scaled pencil. This also
    // avoids relying on backend in-place sign conventions at zero pivots.
    let mut s = q.adjoint() * a.map(|x| C::new(*x, 0.)) * &z;
    let mut t = q.adjoint() * b.map(|x| C::new(*x, 0.)) * &z;
    for i in 0..n {
        for j in 0..i {
            if s[(i, j)].norm() > 1e-10 || t[(i, j)].norm() > 1e-10 {
                return Err(Error::NumericalFailure("QZ did not reach triangular form"));
            }
            s[(i, j)] = C::new(0., 0.);
            t[(i, j)] = C::new(0., 0.);
        }
    }
    factor(
        s * Scale(C::new(sa, 0.)),
        t * Scale(C::new(sb, 0.)),
        q,
        z,
        0,
    )
}
fn swap_pair(
    s: &mut Mat<C>,
    t: &mut Mat<C>,
    q: &mut Mat<C>,
    z: &mut Mat<C>,
    i: usize,
) -> Result<()> {
    let alpha = s[(i + 1, i + 1)];
    let beta = t[(i + 1, i + 1)];
    let v0 = alpha * t[(i, i + 1)] - beta * s[(i, i + 1)];
    let v1 = beta * s[(i, i)] - alpha * t[(i, i)];
    let norm = v0.norm().hypot(v1.norm());
    if norm == 0. || !norm.is_finite() {
        return Err(Error::NumericalFailure(
            "cannot separate selected QZ eigenvalues",
        ));
    }
    let v0 = v0 / norm;
    let v1 = v1 / norm;
    let right = faer::mat![[v0, -v1.conj()], [v1, v0.conj()]];
    let ss = s.as_ref().submatrix(i, i, 2, 2) * &right;
    let tt = t.as_ref().submatrix(i, i, 2, 2) * &right;
    let column = if ss.col(0).norm_l2() >= tt.col(0).norm_l2() {
        ss.col(0).to_owned()
    } else {
        tt.col(0).to_owned()
    };
    let norm = column.norm_l2();
    if norm == 0. {
        return Err(Error::NumericalFailure(
            "indeterminate QZ eigenvalue cannot be reordered",
        ));
    }
    let u0 = column[0] / norm;
    let u1 = column[1] / norm;
    let left = faer::mat![[u0, -u1.conj()], [u1, u0.conj()]];
    for a in [&mut *s, &mut *t, &mut *z] {
        let cols = a.subcols(i, 2) * &right;
        a.as_mut().subcols_mut(i, 2).copy_from(&cols);
    }
    for a in [&mut *s, &mut *t] {
        let rows = left.adjoint() * a.subrows(i, 2);
        a.as_mut().subrows_mut(i, 2).copy_from(&rows);
    }
    let cols = q.subcols(i, 2) * &left;
    q.as_mut().subcols_mut(i, 2).copy_from(&cols);
    s[(i + 1, i)] = C::new(0., 0.);
    t[(i + 1, i)] = C::new(0., 0.);
    Ok(())
}
/// Reorders a generalized Schur form so selected eigenvalues come first.
///
/// ## Parameters
///
/// `select(alpha,beta)` receives homogeneous complex eigenvalues once per
/// backend eigenvalue. Return true to move it into the leading block.
/// Callers can implement left-half-plane or inside-unit-circle selection
/// without division (and handle infinite/indeterminate eigenvalues explicitly).
///
/// ## Domain
///
/// Same finite equal-order real matrices as qz, order at most 256.
/// NaN/infinity are rejected. Cannot reorder an inseparable selected pencil.
///
/// ## Returns
///
/// QzResult with selected count, leading selected eigenvalues and complex
/// unitary factors satisfying both reconstruction equations.
///
/// ## Errors
///
/// qz errors or ill-conditioned/indeterminate adjacent reordering.
///
/// ## Numerical behavior
///
/// Uses faer QZ followed by local unitary two-by-two swaps; at most n²
/// swaps, O(n³) work. Adjacent swaps solve a homogeneous scalar eigenvector
/// relation. Selection callback panics propagate. Nearly coincident eigenvalues
/// can make ordering sensitive to roundoff.
///
/// # Examples
///
/// ```
/// let f = sci_linalg::eigen::ordqz(&[vec![-2.,0.],vec![0.,3.]],
///     &[vec![1.,0.],vec![0.,1.]], |a,b| (a.re*b.re+a.im*b.im)<0.)?;
/// assert_eq!(f.selected,1);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `ordqz`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.ordqz.html).
pub fn ordqz<F: FnMut(Complex, Complex) -> bool>(
    a: &(impl super::MatrixInput + ?Sized),
    b: &(impl super::MatrixInput + ?Sized),
    mut select: F,
) -> Result<QzResult> {
    let f = qz(a, b)?;
    let n = f.alpha.len();
    let mut flags: Vec<_> = f
        .alpha
        .iter()
        .zip(&f.beta)
        .map(|(&a, &b)| select(a, b))
        .collect();
    let mut s = complex_dense(&f.s)?;
    let mut t = complex_dense(&f.t)?;
    let mut q = complex_dense(&f.q)?;
    let mut z = complex_dense(&f.z)?;
    let mut count = 0;
    for j in 0..n {
        if flags[j] {
            for i in (count..j).rev() {
                swap_pair(&mut s, &mut t, &mut q, &mut z, i)?;
                flags.swap(i, i + 1);
            }
            count += 1;
        }
    }
    factor(s, t, q, z, count)
}
