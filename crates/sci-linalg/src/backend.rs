//! Private conversions and work bounds for dense linear algebra.
use crate::{Error, Result};
use dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::{Mat, Scale, mat::AsMatRef, prelude::Solve, traits::ComplexField};
use num_complex::Complex64 as C;
pub(super) const MAX_DIM: usize = 256;
pub(super) const MAX_ENTRIES: usize = 1_000_000;
pub(super) fn scaled(mut a: Mat<f64>) -> (Mat<f64>, f64) {
    let scale = a.iter().fold(0.0_f64, |s, &x| s.max(x.abs()));
    let scale = if scale == 0.0 { 1.0 } else { scale };
    for col in a.col_iter_mut() {
        for x in col.iter_mut() {
            *x /= scale;
        }
    }
    (a, scale)
}
pub(super) fn shape(a: &[Vec<f64>]) -> Result<(usize, usize)> {
    let n = a.first().map_or(0, Vec::len);
    if a.iter().any(|row| row.len() != n) {
        return Err(Error::DimensionMismatch);
    }
    size(a.len(), n)?;
    Ok((a.len(), n))
}
pub(super) fn size(m: usize, n: usize) -> Result<()> {
    if m > MAX_ENTRIES || n > MAX_ENTRIES || m.checked_mul(n).is_none_or(|s| s > MAX_ENTRIES) {
        return Err(Error::InvalidInput(
            "matrix exceeds 1,000,000 entries or an axis exceeds 1,000,000",
        ));
    }
    Ok(())
}
pub(super) fn dense(a: &[Vec<f64>]) -> Result<Mat<f64>> {
    let (m, n) = shape(a)?;
    if m == 0 || n == 0 || m > MAX_DIM || n > MAX_DIM {
        return Err(Error::InvalidInput(
            "dense algorithms require axes in 1..=256",
        ));
    }
    for row in a {
        crate::error::finite_slice(row)?;
    }
    Ok(Mat::from_fn(m, n, |i, j| a[i][j]))
}
pub(super) fn square(a: &[Vec<f64>]) -> Result<Mat<f64>> {
    let a = dense(a)?;
    if !a.is_square() {
        return Err(Error::DimensionMismatch);
    }
    Ok(a)
}
pub(super) fn rows(
    a: &(impl AsMatRef<T = f64, Rows = usize, Cols = usize> + ?Sized),
) -> Result<Vec<Vec<f64>>> {
    let a = a.as_mat_ref();
    if a.col_iter().flat_map(|c| c.iter()).any(|v| !v.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok((0..a.nrows())
        .map(|i| (0..a.ncols()).map(|j| a[(i, j)]).collect())
        .collect())
}
pub(super) fn output(a: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>> {
    if a.iter().flatten().any(|v| !v.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok(a)
}
pub(super) fn tolerance(v: f64) -> Result<f64> {
    if !v.is_finite() || v < 0.0 {
        return Err(Error::InvalidInput(
            "tolerances must be finite and nonnegative",
        ));
    }
    Ok(v)
}
pub(super) fn relative(v: Option<f64>, m: usize, n: usize) -> Result<f64> {
    tolerance(v.unwrap_or(f64::EPSILON * m.max(n) as f64))
}
pub(super) fn symmetric(a: &[Vec<f64>], lower: bool) -> Result<Mat<f64>> {
    let mut a = square(a)?;
    for i in 0..a.nrows() {
        for j in 0..i {
            let v = if lower { a[(i, j)] } else { a[(j, i)] };
            a[(i, j)] = v;
            a[(j, i)] = v;
        }
    }
    Ok(a)
}

/// Small helpers for entrywise operations on the private faer representation.
pub(super) trait Entries<T: Copy> {
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a;
    fn is_square(&self) -> bool;
    fn solve_checked(&self, rhs: &Mat<T>) -> Option<Mat<T>>;
    fn inverse_checked(&self) -> Option<Mat<T>>;
    fn qr_factors(&self) -> (Mat<T>, Mat<T>);
    fn swap_rows(&mut self, i: usize, j: usize);
    fn swap_columns(&mut self, i: usize, j: usize);
}
impl<T: ComplexField + Copy + PartialEq> Entries<T> for Mat<T> {
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a,
    {
        self.col_iter().flat_map(|c| c.iter())
    }
    fn is_square(&self) -> bool {
        self.nrows() == self.ncols()
    }
    fn solve_checked(&self, rhs: &Mat<T>) -> Option<Mat<T>> {
        let lu = self.partial_piv_lu();
        let zero = faer::traits::math_utils::zero::<T>();
        if (0..self.nrows()).any(|i| {
            lu.U()[(i, i)] == zero || !faer::traits::math_utils::is_finite(&lu.U()[(i, i)])
        }) {
            return None;
        }
        Some(lu.solve(rhs))
    }
    fn inverse_checked(&self) -> Option<Mat<T>> {
        self.solve_checked(&Mat::identity(self.nrows(), self.ncols()))
    }
    fn qr_factors(&self) -> (Mat<T>, Mat<T>) {
        let d = self.qr();
        (d.compute_thin_Q(), d.thin_R().to_owned())
    }
    fn swap_rows(&mut self, i: usize, j: usize) {
        for k in 0..self.ncols() {
            let x = self[(i, k)];
            self[(i, k)] = self[(j, k)];
            self[(j, k)] = x;
        }
    }
    fn swap_columns(&mut self, i: usize, j: usize) {
        for k in 0..self.nrows() {
            let x = self[(k, i)];
            self[(k, i)] = self[(k, j)];
            self[(k, j)] = x;
        }
    }
}

/// Generalized Hessenberg reduction of (A,I) supplies an ordinary reduction.
pub(super) fn real_schur(a: Mat<f64>, finish: bool) -> Result<(Mat<f64>, Mat<f64>)> {
    let n = a.nrows();
    let mut h = a.clone();
    let mut b = Mat::identity(n, n);
    let mut q = Mat::identity(n, n);
    let mut z = Mat::identity(n, n);
    let hp = faer::auto!(f64);
    let qp = faer::auto!(f64);
    let req = StackReq::any_of(&[
        faer::linalg::gevd::gen_hessenberg::generalized_hessenberg_scratch::<f64>(n, hp),
        faer::linalg::gevd::qz_real::hessenberg_to_qz_scratch::<f64>(n, faer::Par::Seq, qp),
    ]);
    let mut mem = MemBuffer::new(req);
    faer::linalg::gevd::gen_hessenberg::generalized_hessenberg(
        h.as_mut(),
        b.as_mut(),
        Some(q.as_mut()),
        Some(z.as_mut()),
        faer::Par::Seq,
        MemStack::new(&mut mem),
        hp,
    );
    if finish {
        let mut ar = faer::Col::zeros(n);
        let mut ai = faer::Col::zeros(n);
        let mut beta = faer::Col::zeros(n);
        faer::linalg::gevd::qz_real::hessenberg_to_qz(
            h.as_mut(),
            b.as_mut(),
            Some(q.as_mut()),
            Some(z.as_mut()),
            ar.as_mut(),
            ai.as_mut(),
            beta.as_mut(),
            faer::linalg::evd::ComputeEigenvectors::Yes,
            faer::Par::Seq,
            qp,
            MemStack::new(&mut mem),
        );
    }
    // B=I implies Q and Z differ only by the diagonal signs in triangular B.
    let mut t = z.transpose() * &a * &z;
    for i in 0..n {
        for j in 0..i.saturating_sub(1) {
            if t[(i, j)].abs() > 1e-10 {
                return Err(Error::NumericalFailure(
                    "Schur/Hessenberg reduction did not converge",
                ));
            }
            t[(i, j)] = 0.0;
        }
    }
    if finish {
        for i in 1..n {
            if h[(i, i - 1)] == 0.0 {
                if t[(i, i - 1)].abs() > 1e-10 {
                    return Err(Error::NumericalFailure(
                        "Schur reduction lost its block structure",
                    ));
                }
                t[(i, i - 1)] = 0.0;
            } else if i > 1 && h[(i - 1, i - 2)] != 0.0 {
                return Err(Error::NumericalFailure("Schur reduction did not converge"));
            }
        }
    }
    rows(&t)?;
    rows(&z)?;
    Ok((t, z))
}

pub(super) fn complex_schur(a: Mat<C>) -> Result<(Mat<C>, Mat<C>)> {
    let n = a.nrows();
    let mut h = a.clone();
    let mut b = Mat::identity(n, n);
    let mut q = Mat::identity(n, n);
    let mut z = Mat::identity(n, n);
    let mut alpha = faer::Col::zeros(n);
    let mut beta = faer::Col::zeros(n);
    let hp = faer::auto!(C);
    let qp = faer::auto!(C);
    let req = StackReq::any_of(&[
        faer::linalg::gevd::gen_hessenberg::generalized_hessenberg_scratch::<C>(n, hp),
        faer::linalg::gevd::qz_cplx::hessenberg_to_qz_scratch::<C>(n, faer::Par::Seq, qp),
    ]);
    let mut mem = MemBuffer::new(req);
    faer::linalg::gevd::gen_hessenberg::generalized_hessenberg(
        h.as_mut(),
        b.as_mut(),
        Some(q.as_mut()),
        Some(z.as_mut()),
        faer::Par::Seq,
        MemStack::new(&mut mem),
        hp,
    );
    faer::linalg::gevd::qz_cplx::hessenberg_to_qz(
        h.as_mut(),
        b.as_mut(),
        Some(q.as_mut()),
        Some(z.as_mut()),
        alpha.as_mut(),
        beta.as_mut(),
        faer::linalg::gevd::ComputeEigenvectors::Yes,
        faer::Par::Seq,
        qp,
        MemStack::new(&mut mem),
    );
    let mut t = z.adjoint() * &a * &z;
    for i in 0..n {
        for j in 0..i {
            if t[(i, j)].norm() > 1e-10 {
                return Err(Error::NumericalFailure(
                    "complex Schur reduction did not converge",
                ));
            }
            t[(i, j)] = C::new(0.0, 0.0);
        }
    }
    super::principal::complex_rows(&t)?;
    super::principal::complex_rows(&z)?;
    Ok((z, t))
}

/// [13/13] Padé approximant on inputs pre-scaled to one-norm <= 1/2.
/// Higham (2005), SIAM J. Matrix Anal. Appl. 26, 1179–1193.
/// All matrix products and the rational solve are delegated to faer.
pub(super) fn pade_exponential(a: &Mat<C>) -> Result<Mat<C>> {
    const B: [f64; 14] = [
        64764752532480000.,
        32382376266240000.,
        7771770303897600.,
        1187353796428800.,
        129060195264000.,
        10559470521600.,
        670442572800.,
        33522128640.,
        1323241920.,
        40840800.,
        960960.,
        16380.,
        182.,
        1.,
    ];
    let s = |i| Scale(C::new(B[i], 0.));
    let eye = Mat::<C>::identity(a.nrows(), a.ncols());
    let a2 = a * a;
    let a4 = &a2 * &a2;
    let a6 = &a4 * &a2;
    let u = a
        * (&a6 * (&a6 * s(13) + &a4 * s(11) + &a2 * s(9))
            + &a6 * s(7)
            + &a4 * s(5)
            + &a2 * s(3)
            + &eye * s(1));
    let v = &a6 * (&a6 * s(12) + &a4 * s(10) + &a2 * s(8))
        + &a6 * s(6)
        + &a4 * s(4)
        + &a2 * s(2)
        + &eye * s(0);
    let result = (&v - &u)
        .solve_checked(&(&v + &u))
        .ok_or(Error::NumericalFailure(
            "singular matrix exponential Pade denominator",
        ))?;
    super::principal::complex_rows(&result)?;
    Ok(result)
}

pub(super) struct LuFactors {
    pub l: Mat<f64>,
    pub u: Mat<f64>,
    pub permutation: Vec<usize>,
}
/// faer 0.24's unblocked LU divides by exact zero pivots. Retry such inputs
/// one column at a time, skipping zero columns; faer supplies nonzero pivot
/// factorizations and the trailing matrix updates. Singular factors stay finite.
pub(super) fn lu_factors(a: Mat<f64>) -> Result<LuFactors> {
    let d = a.partial_piv_lu();
    if d.L()
        .col_iter()
        .flat_map(|c| c.iter())
        .all(|x| x.is_finite())
        && d.U()
            .col_iter()
            .flat_map(|c| c.iter())
            .all(|x| x.is_finite())
    {
        return Ok(LuFactors {
            l: d.L().to_owned(),
            u: d.U().to_owned(),
            permutation: d.P().arrays().0.to_vec(),
        });
    }
    let (m, n) = a.shape();
    let k = m.min(n);
    let mut work = a;
    let mut l = Mat::<f64>::identity(m, k);
    let mut permutation = (0..m).collect::<Vec<_>>();
    for j in 0..k {
        let column = work.submatrix(j, j, m - j, 1).to_owned();
        if column.iter().all(|&x| x == 0.0) {
            continue;
        }
        let (column, _) = scaled(column);
        let f = column.partial_piv_lu();
        let pivot = j + f.P().arrays().0[0];
        work.swap_rows(j, pivot);
        permutation.swap(j, pivot);
        for c in 0..j {
            let x = l[(j, c)];
            l[(j, c)] = l[(pivot, c)];
            l[(pivot, c)] = x;
        }
        for i in j + 1..m {
            l[(i, j)] = f.L()[(i - j, 0)];
            work[(i, j)] = 0.0;
        }
        if j + 1 < m && j + 1 < n {
            let update =
                l.submatrix(j + 1, j, m - j - 1, 1) * work.submatrix(j, j + 1, 1, n - j - 1);
            let trailing = work.submatrix(j + 1, j + 1, m - j - 1, n - j - 1) - update;
            work.as_mut()
                .submatrix_mut(j + 1, j + 1, m - j - 1, n - j - 1)
                .copy_from(&trailing);
        }
    }
    rows(&l)?;
    rows(&work)?;
    Ok(LuFactors {
        l,
        u: work.subrows(0, k).to_owned(),
        permutation,
    })
}
