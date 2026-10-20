//! Bounded real iterative methods. QMR/TFQMR recurrences are adapted from
//! SciPy 1.18.1; see SCIPY_LICENSE.txt for its BSD redistribution notice.
use crate::*;
use std::sync::Arc;

/// Real linear map with forward and transpose actions.
///
/// Implementations must return finite vectors of the declared output length.
/// Callbacks are responsible for bounding their own work; solvers bound calls.
pub trait LinearMap {
    /// Number of output components.
    fn rows(&self) -> usize;
    /// Number of input components.
    fn cols(&self) -> usize;
    /// Compute `y[i] = Σ_j A[i,j] x[j]`.
    ///
    /// # Parameters and conventions
    /// x has [`LinearMap::cols`] components; output uses logical row order.
    /// # Domain and special values
    /// Finite vectors only; empty axes are allowed by the map implementation.
    /// # Returns
    /// A finite vector of length [`LinearMap::rows`].
    /// # Errors
    /// Incompatible lengths, invalid matrix storage, callback failure or overflow.
    /// # Numerical behavior
    /// CSR products scan stored entries; callback maps define their own accuracy
    /// and work limits. No implicit transpose is performed.
    /// # Examples
    /// ```
    /// use sci_sparse::linalg::LinearMap;
    /// assert_eq!(sci_sparse::identity(1)?.apply(&[2.])?,[2.]);
    /// # Ok::<(), sci_core::Error>(())
    /// ```
    /// # References
    /// [SciPy LinearOperator.matvec](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.matvec.html).
    fn apply(&self, x: &[f64]) -> Result<Vec<f64>>;
    /// Compute `y[j] = Σ_i A[i,j] x[i]`, the real transpose action.
    ///
    /// # Parameters and conventions
    /// x has [`LinearMap::rows`] components; output uses logical column order.
    /// # Domain and special values
    /// Finite vectors; transpose callbacks must be adjoints of forward callbacks.
    /// # Returns
    /// A finite vector of length [`LinearMap::cols`].
    /// # Errors
    /// Invalid storage/length, callback failure, nonfinite evaluation or overflow.
    /// # Numerical behavior
    /// CSR inputs build a canonical transpose and multiply; callback accuracy
    /// and work depend on their implementation. No complex conjugation is needed.
    /// # Examples
    /// ```
    /// use sci_sparse::linalg::LinearMap;
    /// assert_eq!(sci_sparse::identity(1)?.transpose_apply(&[2.])?,[2.]);
    /// # Ok::<(), sci_core::Error>(())
    /// ```
    /// # References
    /// [SciPy LinearOperator.rmatvec](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.rmatvec.html).
    fn transpose_apply(&self, x: &[f64]) -> Result<Vec<f64>>;
}
impl LinearMap for SparseMatrix {
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn apply(&self, x: &[f64]) -> Result<Vec<f64>> {
        matvec(self, x)
    }
    fn transpose_apply(&self, x: &[f64]) -> Result<Vec<f64>> {
        matvec(&transpose(self)?, x)
    }
}
type Action = Arc<dyn Fn(&[f64]) -> Result<Vec<f64>> + Send + Sync>;
/// Owned matrix-free real linear operator, with private callback storage.
#[derive(Clone)]
pub struct LinearOperator {
    rows: usize,
    cols: usize,
    forward: Action,
    backward: Action,
}
impl std::fmt::Debug for LinearOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LinearOperator")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .finish_non_exhaustive()
    }
}
impl LinearOperator {
    /// Construct an operator from `A x` and `Aᵀ x` callbacks.
    ///
    /// # Parameters and conventions
    /// `rows,cols` declare shape; callbacks receive logical zero-based vectors.
    /// # Domain and special values
    /// Axes <=1,000,000; actions must return finite vectors of declared lengths.
    /// # Returns
    /// An owned clonable operator; callbacks are not evaluated during creation.
    /// # Errors
    /// Excessive shape; later applications report invalid inputs or callback output.
    /// # Numerical behavior
    /// Applications validate dimensions and finiteness, but cannot bound user
    /// callback work or verify that callbacks are linear or adjoints.
    /// # Examples
    /// ```
    /// use sci_sparse::linalg::{LinearOperator,LinearMap};
    /// let op=LinearOperator::new(2,2,|x|Ok(x.to_vec()),|x|Ok(x.to_vec()))?;
    /// assert_eq!(op.apply(&[1.,2.])?, [1.,2.]);
    /// # Ok::<(), sci_core::Error>(())
    /// ```
    /// # References
    /// [SciPy LinearOperator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html).
    pub fn new<F, G>(rows: usize, cols: usize, forward: F, backward: G) -> Result<Self>
    where
        F: Fn(&[f64]) -> Result<Vec<f64>> + Send + Sync + 'static,
        G: Fn(&[f64]) -> Result<Vec<f64>> + Send + Sync + 'static,
    {
        crate::shape(rows, cols)?;
        Ok(Self {
            rows,
            cols,
            forward: Arc::new(forward),
            backward: Arc::new(backward),
        })
    }
}
fn action(f: &Action, x: &[f64], input: usize, output: usize) -> Result<Vec<f64>> {
    if x.len() != input {
        return Err(Error::DimensionMismatch);
    }
    sci_core::finite_slice(x)?;
    let y = f(x)?;
    if y.len() != output {
        return Err(Error::DimensionMismatch);
    }
    sci_core::finite_slice(&y)?;
    Ok(y)
}
impl LinearMap for LinearOperator {
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn apply(&self, x: &[f64]) -> Result<Vec<f64>> {
        action(&self.forward, x, self.cols, self.rows)
    }
    fn transpose_apply(&self, x: &[f64]) -> Result<Vec<f64>> {
        action(&self.backward, x, self.rows, self.cols)
    }
}

/// Borrow-independent operator adaptation, `op(x)=A x`, `opᵀ(x)=Aᵀ x`.
///
/// # Parameters and conventions
/// `a` is copied into private shared callback storage; caller mutation is isolated.
/// # Domain and special values
/// Canonical finite CSR under the normal crate limits; empty axes are valid.
/// # Returns
/// An owned [`LinearOperator`] usable by all iterative solvers.
/// # Errors
/// Invalid CSR or storage limits.
/// # Numerical behavior
/// Copies A and its transpose once; each application performs a sparse product.
/// # Examples
/// ```
/// use sci_sparse::linalg::LinearMap;
/// let op=sci_sparse::linalg::aslinearoperator(&sci_sparse::identity(2)?)?;
/// assert_eq!(op.apply(&[3.,4.])?, [3.,4.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy aslinearoperator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.aslinearoperator.html).
pub fn aslinearoperator(a: &SparseMatrix) -> Result<LinearOperator> {
    validate(a)?;
    let t = transpose(a)?;
    let a = a.clone();
    LinearOperator::new(
        a.rows,
        a.cols,
        move |x| matvec(&a, x),
        move |x| matvec(&t, x),
    )
}

/// Common bounded iterative-solver controls.
#[derive(Clone, Debug)]
pub struct SolverOptions {
    /// Relative true-residual tolerance (default 1e-5).
    pub rtol: f64,
    /// Absolute true-residual tolerance (default zero).
    pub atol: f64,
    /// Maximum iterations or matrix-vector steps, in 0..=1024.
    pub maxiter: usize,
    /// Initial solution; None means zero. Length equals the number of columns.
    pub x0: Option<Vec<f64>>,
    /// Inner dimension for restarted methods, in 1..=256.
    pub restart: usize,
    /// Number of retained augmentation directions for LGMRES/GCROT, <=32.
    pub recycle: usize,
    /// Optional inverse preconditioner, acting on residuals.
    pub preconditioner: Option<LinearOperator>,
    /// MINRES solves `(A - shift I)x=b`; ignored by other methods.
    pub shift: f64,
}
impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-5,
            atol: 0.,
            maxiter: 1024,
            x0: None,
            restart: 30,
            recycle: 3,
            preconditioner: None,
            shift: 0.,
        }
    }
}
/// Final state of a bounded iterative solve.
#[derive(Clone, Debug)]
pub struct IterativeResult {
    /// Current solution in column order.
    pub x: Vec<f64>,
    /// True when the documented residual criterion is satisfied.
    pub converged: bool,
    /// Iterations or inner matrix-vector steps performed.
    pub iterations: usize,
    /// Euclidean norm of the final true residual.
    pub residual_norm: f64,
}
fn dot(x: &[f64], y: &[f64]) -> Result<f64> {
    finite(x.iter().zip(y).map(|(a, b)| a * b).sum())
}
fn length(x: &[f64]) -> Result<f64> {
    finite(x.iter().fold(0f64, |s, &v| s.hypot(v)))
}
fn axpy(y: &mut [f64], a: f64, x: &[f64]) -> Result<()> {
    for (y, &x) in y.iter_mut().zip(x) {
        *y = finite(*y + a * x)?;
    }
    Ok(())
}
fn times(x: &[f64], a: f64) -> Result<Vec<f64>> {
    x.iter().map(|v| finite(v * a)).collect()
}
fn ratio(a: f64, b: f64) -> Result<f64> {
    if b == 0. {
        return Err(Error::NumericalFailure("iterative solver breakdown"));
    }
    finite(a / b)
}
fn mv(a: &dyn LinearMap, x: &[f64], trans: bool) -> Result<Vec<f64>> {
    let y = if trans {
        a.transpose_apply(x)?
    } else {
        a.apply(x)?
    };
    if y.len() != if trans { a.cols() } else { a.rows() } {
        return Err(Error::DimensionMismatch);
    }
    sci_core::finite_slice(&y)?;
    Ok(y)
}
fn residual(a: &dyn LinearMap, b: &[f64], x: &[f64]) -> Result<Vec<f64>> {
    let mut r = b.to_vec();
    axpy(&mut r, -1., &mv(a, x, false)?)?;
    Ok(r)
}
fn pre(o: &SolverOptions, r: &[f64], trans: bool) -> Result<Vec<f64>> {
    match &o.preconditioner {
        Some(m) => mv(m, r, trans),
        None => Ok(r.to_vec()),
    }
}
fn setup(a: &dyn LinearMap, b: &[f64], o: &SolverOptions, square: bool) -> Result<(Vec<f64>, f64)> {
    if a.rows() == 0
        || a.cols() == 0
        || a.rows() > 256
        || a.cols() > 256
        || square && a.rows() != a.cols()
        || b.len() != a.rows()
    {
        return Err(Error::DimensionMismatch);
    }
    sci_core::finite_slice(b)?;
    if !o.rtol.is_finite()
        || !o.atol.is_finite()
        || !o.shift.is_finite()
        || o.rtol < 0.
        || o.atol < 0.
        || o.maxiter > 1024
        || o.restart == 0
        || o.restart > 256
        || o.recycle > 32
    {
        return Err(Error::InvalidInput("invalid iterative controls"));
    }
    if let Some(m) = &o.preconditioner
        && (m.rows() != a.rows() || m.cols() != a.rows())
    {
        return Err(Error::DimensionMismatch);
    }
    let x = o.x0.clone().unwrap_or_else(|| vec![0.; a.cols()]);
    if x.len() != a.cols() {
        return Err(Error::DimensionMismatch);
    }
    sci_core::finite_slice(&x)?;
    Ok((x, finite(o.atol.max(o.rtol * length(b)?))?))
}
fn finish(
    a: &dyn LinearMap,
    b: &[f64],
    x: Vec<f64>,
    iterations: usize,
    tol: f64,
) -> Result<IterativeResult> {
    let r = length(&residual(a, b, &x)?)?;
    Ok(IterativeResult {
        x,
        iterations,
        converged: r <= tol,
        residual_norm: r,
    })
}
fn operator_matrix(a: &dyn LinearMap) -> Result<faer::Mat<f64>> {
    let mut d = faer::Mat::<f64>::zeros(a.rows(), a.cols());
    for j in 0..a.cols() {
        let mut e = vec![0.; a.cols()];
        e[j] = 1.;
        let c = mv(a, &e, false)?;
        for i in 0..a.rows() {
            d[(i, j)] = c[i];
        }
    }
    Ok(d)
}
fn backend_short(
    a: &dyn LinearMap,
    b: &[f64],
    o: &SolverOptions,
    cg: bool,
) -> Result<IterativeResult> {
    let (initial, tol) = setup(a, b, o, true)?;
    if length(&residual(a, b, &initial)?)? <= tol || o.maxiter == 0 {
        return finish(a, b, initial, 0, tol);
    }
    let n = a.rows();
    let mat = operator_matrix(a)?;
    let pre = if let Some(m) = &o.preconditioner {
        operator_matrix(m)?
    } else {
        faer::Mat::identity(n, n)
    };
    let rhs = faer::Mat::from_fn(n, 1, |i, _| b[i]);
    let mut out = faer::Mat::from_fn(n, 1, |i, _| initial[i]);
    let mut iterations = 0;
    use faer::matrix_free::{IdentityPrecond, bicgstab as b, conjugate_gradient as c};
    if cg {
        let params = c::CgParams {
            abs_tolerance: o.atol,
            rel_tolerance: o.rtol,
            max_iters: o.maxiter,
            ..Default::default()
        };
        let mut mem = dyn_stack::MemBuffer::new(c::conjugate_gradient_scratch(
            pre.as_ref(),
            mat.as_ref(),
            1,
            faer::Par::Seq,
        ));
        match c::conjugate_gradient(
            out.as_mut(),
            pre.as_ref(),
            mat.as_ref(),
            rhs.as_ref(),
            params,
            |_| iterations += 1,
            faer::Par::Seq,
            dyn_stack::MemStack::new(&mut mem),
        ) {
            Ok(_) | Err(c::CgError::NoConvergence { .. }) => {}
            Err(_) => {
                return Err(Error::NumericalFailure(
                    "CG requires positive definite matrix and preconditioner",
                ));
            }
        }
    } else {
        let identity = IdentityPrecond { dim: n };
        let params = b::BicgParams {
            abs_tolerance: o.atol,
            rel_tolerance: o.rtol,
            max_iters: o.maxiter,
            ..Default::default()
        };
        let mut mem = dyn_stack::MemBuffer::new(b::bicgstab_scratch(
            pre.as_ref(),
            identity,
            mat.as_ref(),
            1,
            faer::Par::Seq,
        ));
        let _status = b::bicgstab(
            out.as_mut(),
            pre.as_ref(),
            identity,
            mat.as_ref(),
            rhs.as_ref(),
            params,
            |_| iterations += 1,
            faer::Par::Seq,
            dyn_stack::MemStack::new(&mut mem),
        );
    }
    let x = (0..n).map(|i| finite(out[(i, 0)])).collect::<Result<_>>()?;
    finish(a, b, x, iterations, tol)
}
#[derive(Clone, Copy)]
enum Short {
    Bicg,
    Cgs,
    Qmr,
    Tfqmr,
}
fn short(
    a: &dyn LinearMap,
    b: &[f64],
    o: &SolverOptions,
    method: Short,
) -> Result<IterativeResult> {
    let (mut x, tol) = setup(a, b, o, true)?;
    let n = b.len();
    let mut r = residual(a, b, &x)?;
    if length(&r)? <= tol {
        return finish(a, b, x, 0, tol);
    }
    if matches!(method, Short::Qmr) {
        return qmr_inner(a, b, o, x, r, tol);
    }
    if matches!(method, Short::Tfqmr) {
        return tfqmr_inner(a, b, o, x, r, tol);
    }
    let mut shadow = r.clone();
    let mut p = vec![0.; n];
    let mut pt = vec![0.; n];
    let mut u = vec![0.; n];
    let mut q = vec![0.; n];
    let mut v = vec![0.; n];
    let mut rho_prev = 1.;
    for iter in 0..o.maxiter {
        let z = pre(o, &r, false)?;
        let rho = if matches!(method, Short::Bicg) {
            dot(&shadow, &z)?
        } else {
            dot(&shadow, &r)?
        };
        if rho == 0. {
            return Err(Error::NumericalFailure("biorthogonal residual breakdown"));
        }
        match method {
            Short::Bicg => {
                let beta = if iter == 0 { 0. } else { ratio(rho, rho_prev)? };
                for i in 0..n {
                    p[i] = finite(z[i] + beta * p[i])?;
                }
                let ap = mv(a, &p, false)?;
                let zt = pre(o, &shadow, true)?;
                for i in 0..n {
                    pt[i] = finite(zt[i] + beta * pt[i])?;
                }
                let alpha = ratio(rho, dot(&pt, &ap)?)?;
                axpy(&mut shadow, -alpha, &mv(a, &pt, true)?)?;
                axpy(&mut x, alpha, &p)?;
                axpy(&mut r, -alpha, &ap)?;
            }
            Short::Cgs => {
                if iter == 0 {
                    u = r.clone();
                    p = r.clone();
                } else {
                    let beta = ratio(rho, rho_prev)?;
                    for i in 0..n {
                        u[i] = finite(r[i] + beta * q[i])?;
                        p[i] = finite(u[i] + beta * (q[i] + beta * p[i]))?;
                    }
                }
                v = mv(a, &pre(o, &p, false)?, false)?;
                let alpha = ratio(rho, dot(&shadow, &v)?)?;
                for i in 0..n {
                    q[i] = finite(u[i] - alpha * v[i])?;
                }
                let mut uq = u.clone();
                axpy(&mut uq, 1., &q)?;
                axpy(&mut x, alpha, &pre(o, &uq, false)?)?;
                r = residual(a, b, &x)?;
            }
            _ => unreachable!(),
        }
        rho_prev = rho;
        if length(&r)? <= tol {
            let report = finish(a, b, x.clone(), iter + 1, tol)?;
            if report.converged {
                return Ok(report);
            }
            r = residual(a, b, &x)?;
        }
    }
    finish(a, b, x, o.maxiter, tol)
}
// Real, unpreconditioned QMR recurrence adapted from SciPy's BSD implementation.
fn qmr_inner(
    a: &dyn LinearMap,
    b: &[f64],
    o: &SolverOptions,
    mut x: Vec<f64>,
    mut r: Vec<f64>,
    tol: f64,
) -> Result<IterativeResult> {
    if o.preconditioner.is_some() {
        return Err(Error::InvalidInput(
            "QMR currently requires identity preconditioning",
        ));
    }
    let n = b.len();
    let mut vt = r.clone();
    let mut wt = r.clone();
    let mut rho = length(&vt)?;
    let mut xi = rho;
    let (mut gamma, mut eta, mut theta, mut epsilon) = (1., -1., 0., 1.);
    let mut p = vec![0.; n];
    let mut q = p.clone();
    let mut d = p.clone();
    let mut s = p.clone();
    for iter in 0..o.maxiter {
        let v = times(&vt, ratio(1., rho)?)?;
        let w = times(&wt, ratio(1., xi)?)?;
        let delta = dot(&w, &v)?;
        if iter == 0 {
            p = v.clone();
            q = w.clone();
        } else {
            let cp = ratio(xi * delta, epsilon)?;
            let cq = ratio(rho * delta, epsilon)?;
            for i in 0..n {
                p[i] = finite(v[i] - cp * p[i])?;
                q[i] = finite(w[i] - cq * q[i])?;
            }
        }
        let ap = mv(a, &p, false)?;
        epsilon = dot(&q, &ap)?;
        let beta = ratio(epsilon, delta)?;
        vt = ap.clone();
        axpy(&mut vt, -beta, &v)?;
        let rho_prev = rho;
        rho = length(&vt)?;
        wt = mv(a, &q, true)?;
        axpy(&mut wt, -beta, &w)?;
        xi = length(&wt)?;
        let gamma_prev = gamma;
        let theta_prev = theta;
        theta = ratio(rho, gamma_prev * beta.abs())?;
        gamma = 1. / 1f64.hypot(theta);
        eta = finite(-eta * ratio(rho_prev, beta)? * (gamma / gamma_prev).powi(2))?;
        let c = (theta_prev * gamma).powi(2);
        for i in 0..n {
            d[i] = finite(if iter == 0 {
                eta * p[i]
            } else {
                c * d[i] + eta * p[i]
            })?;
            s[i] = finite(if iter == 0 {
                eta * ap[i]
            } else {
                c * s[i] + eta * ap[i]
            })?;
        }
        axpy(&mut x, 1., &d)?;
        axpy(&mut r, -1., &s)?;
        if length(&r)? <= tol {
            return finish(a, b, x, iter + 1, tol);
        }
    }
    finish(a, b, x, o.maxiter, tol)
}
fn tfqmr_inner(
    a: &dyn LinearMap,
    b: &[f64],
    o: &SolverOptions,
    mut x: Vec<f64>,
    r: Vec<f64>,
    tol: f64,
) -> Result<IterativeResult> {
    if o.preconditioner.is_some() {
        return Err(Error::InvalidInput(
            "TFQMR currently requires identity preconditioning",
        ));
    }
    let mut u = r.clone();
    let mut w = r.clone();
    let shadow = r;
    let mut v = mv(a, &u, false)?;
    let mut uh = v.clone();
    let mut d = vec![0.; b.len()];
    let mut rho = dot(&shadow, &shadow)?;
    let mut rho_last = rho;
    let mut tau = length(&shadow)?;
    let (mut theta, mut eta, mut alpha) = (0., 0., 1.);
    let mut next = u.clone();
    for iter in 0..o.maxiter {
        let even = iter % 2 == 0;
        if even {
            alpha = ratio(rho, dot(&shadow, &v)?)?;
            next = u.clone();
            axpy(&mut next, -alpha, &v)?;
        }
        axpy(&mut w, -alpha, &uh)?;
        let c = ratio(theta * theta * eta, alpha)?;
        for i in 0..d.len() {
            d[i] = finite(u[i] + c * d[i])?;
        }
        theta = ratio(length(&w)?, tau)?;
        let c = 1. / 1f64.hypot(theta);
        tau = finite(tau * theta * c)?;
        eta = finite(c * c * alpha)?;
        axpy(&mut x, eta, &d)?;
        if length(&residual(a, b, &x)?)? <= tol {
            return finish(a, b, x, iter + 1, tol);
        }
        if !even {
            rho = dot(&shadow, &w)?;
            let beta = ratio(rho, rho_last)?;
            for i in 0..u.len() {
                u[i] = finite(w[i] + beta * u[i])?;
                v[i] = finite(beta * uh[i] + beta * beta * v[i])?;
            }
            uh = mv(a, &u, false)?;
            axpy(&mut v, 1., &uh)?;
        } else {
            uh = mv(a, &next, false)?;
            u = next.clone();
            rho_last = rho;
        }
    }
    finish(a, b, x, o.maxiter, tol)
}

#[derive(Clone, Copy)]
enum Restarted {
    Gmres,
    Lgmres,
    Gcrot,
}
fn projected(
    a: &dyn LinearMap,
    b: &[f64],
    o: &SolverOptions,
    kind: Restarted,
) -> Result<IterativeResult> {
    let (mut x, tol) = setup(a, b, o, true)?;
    let mut used = 0;
    let mut recycle: Vec<(Vec<f64>, Vec<f64>)> = vec![];
    loop {
        let mut r = residual(a, b, &x)?;
        if length(&r)? <= tol || used == o.maxiter {
            return finish(a, b, x, used, tol);
        }
        if matches!(kind, Restarted::Gcrot) {
            for (u, c) in &recycle {
                let t = dot(c, &r)?;
                axpy(&mut x, t, u)?;
                axpy(&mut r, -t, c)?;
            }
        }
        let base = x.clone();
        let r = pre(o, &r, false)?;
        let beta = length(&r)?;
        if beta == 0. {
            return Err(Error::NumericalFailure(
                "preconditioner annihilated a nonzero residual",
            ));
        }
        let inner = o.restart
            + if matches!(kind, Restarted::Lgmres) {
                recycle.len()
            } else {
                0
            };
        let max = inner
            .min(255)
            .min(o.maxiter - used)
            .min(a.rows() + recycle.len());
        let mut q = vec![times(&r, 1. / beta)?];
        let mut z: Vec<Vec<f64>> = vec![];
        let mut h = vec![vec![0.; max]; max + 1];
        for j in 0..max {
            let mut direction = if matches!(kind, Restarted::Lgmres) && j < recycle.len() {
                recycle[j].0.clone()
            } else {
                q.last()
                    .ok_or(Error::NumericalFailure("missing Krylov vector"))?
                    .clone()
            };
            let mut w = mv(a, &direction, false)?;
            if matches!(kind, Restarted::Gcrot) {
                for (u, c) in &recycle {
                    let t = dot(c, &w)?;
                    axpy(&mut w, -t, c)?;
                    axpy(&mut direction, -t, u)?;
                }
            }
            w = pre(o, &w, false)?;
            for _ in 0..2 {
                for (i, qi) in q.iter().enumerate() {
                    let t = dot(qi, &w)?;
                    h[i][j] = finite(h[i][j] + t)?;
                    axpy(&mut w, -t, qi)?;
                }
            }
            let wn = length(&w)?;
            h[j + 1][j] = wn;
            z.push(direction);
            used += 1;
            let hj: Matrix = (0..j + 2).map(|i| h[i][..=j].to_vec()).collect();
            let mut rhs = vec![0.; j + 2];
            rhs[0] = beta;
            let y = sci_linalg::lstsq(&hj, &rhs, None)?.solution;
            x = base.clone();
            for (zi, &c) in z.iter().zip(&y) {
                axpy(&mut x, c, zi)?;
            }
            if length(&residual(a, b, &x)?)? <= tol {
                return finish(a, b, x, used, tol);
            }
            if wn <= 64. * f64::EPSILON || used == o.maxiter {
                break;
            }
            q.push(times(&w, 1. / wn)?);
        }
        if !matches!(kind, Restarted::Gmres) && o.recycle > 0 {
            let mut u = x.clone();
            axpy(&mut u, -1., &base)?;
            let mut c = mv(a, &u, false)?;
            if matches!(kind, Restarted::Gcrot) {
                for (ui, ci) in &recycle {
                    let t = dot(ci, &c)?;
                    axpy(&mut c, -t, ci)?;
                    axpy(&mut u, -t, ui)?;
                }
            }
            let norm = if matches!(kind, Restarted::Gcrot) {
                length(&c)?
            } else {
                length(&u)?
            };
            if norm > 64. * f64::EPSILON {
                if recycle.len() == o.recycle {
                    recycle.remove(0);
                }
                recycle.push((times(&u, 1. / norm)?, times(&c, 1. / norm)?));
            }
        }
    }
}
fn min_res(a: &dyn LinearMap, b: &[f64], o: &SolverOptions) -> Result<IterativeResult> {
    setup(a, b, o, true)?;
    if o.preconditioner.is_some() {
        return Err(Error::InvalidInput(
            "full-reorthogonalized MINRES requires identity preconditioning",
        ));
    }
    let n = a.rows();
    let mut dense = vec![vec![0.; n]; n];
    for j in 0..n {
        let mut e = vec![0.; n];
        e[j] = 1.;
        let col = mv(a, &e, false)?;
        for i in 0..n {
            dense[i][j] = col[i];
        }
    }
    for (i, row) in dense.iter().enumerate() {
        for (j, &v) in row.iter().enumerate().take(i) {
            if v != dense[j][i] {
                return Err(Error::InvalidInput("MINRES requires symmetric A"));
            }
        }
    }
    for (i, row) in dense.iter_mut().enumerate() {
        row[i] = finite(row[i] - o.shift)?;
    }
    // Full reorthogonalization of symmetric Arnoldi is the Lanczos subspace
    // minimum-residual method; retaining H avoids unstable short recurrences.
    let shifted = from_dense(&dense)?;
    let mut opts = o.clone();
    opts.restart = n.min(255);
    projected(&shifted, b, &opts, Restarted::Gmres)
}

/// Controls sparse least-squares solves.
#[derive(Clone, Debug)]
pub struct LeastSquaresOptions {
    /// Relative residual/normal-residual tolerance.
    pub rtol: f64,
    /// Absolute residual/normal-residual tolerance.
    pub atol: f64,
    /// Tikhonov damping, solving min `||Ax-b||² + damp² ||x||²`.
    pub damp: f64,
    /// Maximum iterations in 0..=1024.
    pub maxiter: usize,
}
impl Default for LeastSquaresOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-6,
            atol: 1e-6,
            damp: 0.,
            maxiter: 1024,
        }
    }
}
/// Result and true residual diagnostics for sparse least squares.
#[derive(Clone, Debug)]
pub struct LeastSquaresResult {
    /// Approximate minimum-norm solution in column order.
    pub x: Vec<f64>,
    /// True when residual or regularized normal residual meets tolerance.
    pub converged: bool,
    /// Iterations executed.
    pub iterations: usize,
    /// `||Ax-b||₂`, without the damping rows.
    pub residual_norm: f64,
    /// `||Aᵀ(Ax-b) + damp² x||₂`.
    pub normal_residual_norm: f64,
}
fn ls_setup(
    a: &dyn LinearMap,
    b: &[f64],
    o: &LeastSquaresOptions,
) -> Result<(Matrix, Vec<f64>, f64)> {
    let opts = SolverOptions {
        rtol: o.rtol,
        atol: o.atol,
        maxiter: o.maxiter,
        ..Default::default()
    };
    setup(a, b, &opts, false)?;
    if !o.damp.is_finite() || o.damp < 0. {
        return Err(Error::InvalidInput("damping must be finite nonnegative"));
    }
    let m = a.rows();
    let n = a.cols();
    let mut d = vec![vec![0.; n]; m + if o.damp > 0. { n } else { 0 }];
    for j in 0..n {
        let mut e = vec![0.; n];
        e[j] = 1.;
        let c = mv(a, &e, false)?;
        for i in 0..m {
            d[i][j] = c[i];
        }
        if o.damp > 0. {
            d[m + j][j] = o.damp;
        }
    }
    let mut rhs = b.to_vec();
    rhs.resize(d.len(), 0.);
    let tol = finite(o.atol.max(o.rtol * length(b)?))?;
    Ok((d, rhs, tol))
}
fn ls_finish(
    a: &dyn LinearMap,
    b: &[f64],
    x: Vec<f64>,
    iterations: usize,
    o: &LeastSquaresOptions,
    tol: f64,
) -> Result<LeastSquaresResult> {
    let r = residual(a, b, &x)?;
    let residual_norm = length(&r)?;
    let mut normal = mv(a, &r, true)?;
    axpy(&mut normal, -finite(o.damp * o.damp)?, &x)?;
    let normal_residual_norm = length(&normal)?;
    Ok(LeastSquaresResult {
        x,
        converged: residual_norm <= tol || normal_residual_norm <= tol,
        iterations,
        residual_norm,
        normal_residual_norm,
    })
}
fn lsqr_inner(a: &dyn LinearMap, b: &[f64], o: &LeastSquaresOptions) -> Result<LeastSquaresResult> {
    let (d, rhs, tol) = ls_setup(a, b, o)?;
    let m = d.len();
    let n = a.cols();
    let mut x = vec![0.; n];
    let forward = |x: &[f64]| -> Result<Vec<f64>> { d.iter().map(|r| dot(r, x)).collect() };
    let backward = |x: &[f64]| -> Result<Vec<f64>> {
        (0..n)
            .map(|j| finite((0..m).map(|i| d[i][j] * x[i]).sum()))
            .collect()
    };
    let mut beta = length(&rhs)?;
    if beta == 0. {
        return ls_finish(a, b, x, 0, o, tol);
    }
    let mut u = times(&rhs, 1. / beta)?;
    let mut v = backward(&u)?;
    let mut alpha = length(&v)?;
    if alpha == 0. {
        return ls_finish(a, b, x, 0, o, tol);
    }
    v = times(&v, 1. / alpha)?;
    let mut w = v.clone();
    let mut phibar = beta;
    let mut rhobar = alpha;
    for iter in 0..o.maxiter {
        let mut next = forward(&v)?;
        axpy(&mut next, -alpha, &u)?;
        beta = length(&next)?;
        u = if beta == 0. {
            vec![0.; m]
        } else {
            times(&next, 1. / beta)?
        };
        next = backward(&u)?;
        axpy(&mut next, -beta, &v)?;
        alpha = length(&next)?;
        v = if alpha == 0. {
            vec![0.; n]
        } else {
            times(&next, 1. / alpha)?
        };
        let rho = rhobar.hypot(beta);
        let c = ratio(rhobar, rho)?;
        let s = ratio(beta, rho)?;
        let theta = s * alpha;
        rhobar = -c * alpha;
        let phi = c * phibar;
        phibar *= s;
        axpy(&mut x, ratio(phi, rho)?, &w)?;
        for i in 0..n {
            w[i] = finite(v[i] - ratio(theta, rho)? * w[i])?;
        }
        let result = ls_finish(a, b, x.clone(), iter + 1, o, tol)?;
        if result.converged {
            return Ok(result);
        }
        if alpha == 0. && beta == 0. {
            return Ok(result);
        }
    }
    ls_finish(a, b, x, o.maxiter, o, tol)
}
fn lsmr_inner(a: &dyn LinearMap, b: &[f64], o: &LeastSquaresOptions) -> Result<LeastSquaresResult> {
    let (d, rhs, tol) = ls_setup(a, b, o)?;
    let n = a.cols();
    if o.maxiter == 0 || length(b)? == 0. {
        return ls_finish(a, b, vec![0.; n], 0, o, tol);
    }
    use faer::matrix_free::{
        IdentityPrecond,
        lsmr::{LsmrParams, lsmr, lsmr_scratch},
    };
    let mat = faer::Mat::from_fn(d.len(), n, |i, j| d[i][j]);
    let rhs = faer::Mat::from_fn(rhs.len(), 1, |i, _| rhs[i]);
    let mut out = faer::Mat::<f64>::zeros(n, 1);
    let pre = IdentityPrecond { dim: n };
    let params = LsmrParams {
        abs_tolerance: o.atol,
        rel_tolerance: o.rtol,
        max_iters: o.maxiter,
        ..Default::default()
    };
    let mut memory = dyn_stack::MemBuffer::new(lsmr_scratch(pre, mat.as_ref(), 1, faer::Par::Seq));
    let mut iterations = 0;
    let _status = lsmr(
        out.as_mut(),
        pre,
        mat.as_ref(),
        rhs.as_ref(),
        params,
        |_| iterations += 1,
        faer::Par::Seq,
        dyn_stack::MemStack::new(&mut memory),
    );
    let x = (0..n).map(|i| finite(out[(i, 0)])).collect::<Result<_>>()?;
    ls_finish(a, b, x, iterations, o, tol)
}

/// Solve `A x = b` using conjugate gradient iterations.
///
/// # Parameters and conventions
/// `a` may be CSR or a [`LinearOperator`]; b is a real vector. `options` controls
/// initial guess, tolerances, iteration count and inverse preconditioning.
/// Requires symmetric positive-definite A and inverse preconditioner; these algebraic properties are caller obligations.
/// # Domain and special values
/// Finite, nonempty square operators with order <=256. Zero right sides and
/// already-converged initial guesses are supported. NaN/infinity are rejected.
/// # Returns
/// [`IterativeResult`]; convergence means `||b-Ax||₂ <= max(atol, rtol ||b||₂)`.
/// MINRES applies this test to A-shift I. Exhausted iterations return the
/// current approximation with converged=false; no result is silently discarded.
/// # Errors
/// Invalid shape/options/output, unsupported preconditioning, arithmetic
/// breakdown or nonfinite evaluations. maxiter <=1024 bounds iteration work;
/// each callback must bound its own work. No SciPy callback/logging controls.
/// # Numerical behavior
/// Uses the private faer CG backend; operator columns and preconditioner
/// columns are materialized once (order <=256). With f64 arithmetic, roundoff can delay convergence
/// or trigger breakdown. Final diagnostics recompute the true residual.
/// Projected least-squares problems use private Sci/faer SVD kernels.
/// For built-in CSR inputs, each product has at most 65,536 entries.
/// # Examples
/// ```
/// let a=sci_sparse::identity(2)?;
/// let result=sci_sparse::linalg::cg(&a,&[2.,3.],&Default::default())?;
/// assert!(result.converged);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy cg](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html).
pub fn cg(a: &impl LinearMap, b: &[f64], options: &SolverOptions) -> Result<IterativeResult> {
    backend_short(a, b, options, true)
}

/// Solve `A x = b` using biconjugate gradient iterations.
///
/// # Parameters and conventions
/// `a` may be CSR or a [`LinearOperator`]; b is a real vector. `options` controls
/// initial guess, tolerances, iteration count and inverse preconditioning.
/// Uses forward and transpose actions, including inverse-preconditioner transpose actions.
/// # Domain and special values
/// Finite, nonempty square operators with order <=256. Zero right sides and
/// already-converged initial guesses are supported. NaN/infinity are rejected.
/// # Returns
/// [`IterativeResult`]; convergence means `||b-Ax||₂ <= max(atol, rtol ||b||₂)`.
/// MINRES applies this test to A-shift I. Exhausted iterations return the
/// current approximation with converged=false; no result is silently discarded.
/// # Errors
/// Invalid shape/options/output, unsupported preconditioning, arithmetic
/// breakdown or nonfinite evaluations. maxiter <=1024 bounds iteration work;
/// each callback must bound its own work. No SciPy callback/logging controls.
/// # Numerical behavior
/// Biconjugate gradient with f64 arithmetic; roundoff can delay convergence
/// or trigger breakdown. Final diagnostics recompute the true residual.
/// Projected least-squares problems use private Sci/faer SVD kernels.
/// For built-in CSR inputs, each product has at most 65,536 entries.
/// # Examples
/// ```
/// let a=sci_sparse::identity(2)?;
/// let result=sci_sparse::linalg::bicg(&a,&[2.,3.],&Default::default())?;
/// assert!(result.converged);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy bicg](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.bicg.html).
pub fn bicg(a: &impl LinearMap, b: &[f64], options: &SolverOptions) -> Result<IterativeResult> {
    short(a, b, options, Short::Bicg)
}

/// Solve `A x = b` using stabilized biconjugate gradient iterations.
///
/// # Parameters and conventions
/// `a` may be CSR or a [`LinearOperator`]; b is a real vector. `options` controls
/// initial guess, tolerances, iteration count and inverse preconditioning.
/// Supports nonsymmetric A and left inverse preconditioning.
/// # Domain and special values
/// Finite, nonempty square operators with order <=256. Zero right sides and
/// already-converged initial guesses are supported. NaN/infinity are rejected.
/// # Returns
/// [`IterativeResult`]; convergence means `||b-Ax||₂ <= max(atol, rtol ||b||₂)`.
/// MINRES applies this test to A-shift I. Exhausted iterations return the
/// current approximation with converged=false; no result is silently discarded.
/// # Errors
/// Invalid shape/options/output, unsupported preconditioning, arithmetic
/// breakdown or nonfinite evaluations. maxiter <=1024 bounds iteration work;
/// each callback must bound its own work. No SciPy callback/logging controls.
/// # Numerical behavior
/// Uses the private faer BiCGSTAB backend; operator columns and preconditioner
/// columns are materialized once (order <=256). With f64 arithmetic, roundoff can delay convergence
/// or trigger breakdown. Final diagnostics recompute the true residual.
/// Projected least-squares problems use private Sci/faer SVD kernels.
/// For built-in CSR inputs, each product has at most 65,536 entries.
/// # Examples
/// ```
/// let a=sci_sparse::identity(2)?;
/// let result=sci_sparse::linalg::bicgstab(&a,&[2.,3.],&Default::default())?;
/// assert!(result.converged);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy bicgstab](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.bicgstab.html).
pub fn bicgstab(a: &impl LinearMap, b: &[f64], options: &SolverOptions) -> Result<IterativeResult> {
    backend_short(a, b, options, false)
}

/// Solve `A x = b` using conjugate gradient squared iterations.
///
/// # Parameters and conventions
/// `a` may be CSR or a [`LinearOperator`]; b is a real vector. `options` controls
/// initial guess, tolerances, iteration count and inverse preconditioning.
/// Supports nonsymmetric A; recomputes true residuals after every update.
/// # Domain and special values
/// Finite, nonempty square operators with order <=256. Zero right sides and
/// already-converged initial guesses are supported. NaN/infinity are rejected.
/// # Returns
/// [`IterativeResult`]; convergence means `||b-Ax||₂ <= max(atol, rtol ||b||₂)`.
/// MINRES applies this test to A-shift I. Exhausted iterations return the
/// current approximation with converged=false; no result is silently discarded.
/// # Errors
/// Invalid shape/options/output, unsupported preconditioning, arithmetic
/// breakdown or nonfinite evaluations. maxiter <=1024 bounds iteration work;
/// each callback must bound its own work. No SciPy callback/logging controls.
/// # Numerical behavior
/// Conjugate gradient squared with f64 arithmetic; roundoff can delay convergence
/// or trigger breakdown. Final diagnostics recompute the true residual.
/// Projected least-squares problems use private Sci/faer SVD kernels.
/// For built-in CSR inputs, each product has at most 65,536 entries.
/// # Examples
/// ```
/// let a=sci_sparse::identity(2)?;
/// let result=sci_sparse::linalg::cgs(&a,&[2.,3.],&Default::default())?;
/// assert!(result.converged);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy cgs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cgs.html).
pub fn cgs(a: &impl LinearMap, b: &[f64], options: &SolverOptions) -> Result<IterativeResult> {
    short(a, b, options, Short::Cgs)
}

/// Solve `A x = b` using quasi-minimal residual iterations.
///
/// # Parameters and conventions
/// `a` may be CSR or a [`LinearOperator`]; b is a real vector. `options` controls
/// initial guess, tolerances, iteration count and inverse preconditioning.
/// Uses both A and Aᵀ. This implementation requires identity preconditioning.
/// # Domain and special values
/// Finite, nonempty square operators with order <=256. Zero right sides and
/// already-converged initial guesses are supported. NaN/infinity are rejected.
/// # Returns
/// [`IterativeResult`]; convergence means `||b-Ax||₂ <= max(atol, rtol ||b||₂)`.
/// MINRES applies this test to A-shift I. Exhausted iterations return the
/// current approximation with converged=false; no result is silently discarded.
/// # Errors
/// Invalid shape/options/output, unsupported preconditioning, arithmetic
/// breakdown or nonfinite evaluations. maxiter <=1024 bounds iteration work;
/// each callback must bound its own work. No SciPy callback/logging controls.
/// # Numerical behavior
/// Quasi-minimal residual with f64 arithmetic; roundoff can delay convergence
/// or trigger breakdown. Final diagnostics recompute the true residual.
/// Projected least-squares problems use private Sci/faer SVD kernels.
/// For built-in CSR inputs, each product has at most 65,536 entries.
/// # Examples
/// ```
/// let a=sci_sparse::identity(2)?;
/// let result=sci_sparse::linalg::qmr(&a,&[2.,3.],&Default::default())?;
/// assert!(result.converged);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy qmr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.qmr.html).
pub fn qmr(a: &impl LinearMap, b: &[f64], options: &SolverOptions) -> Result<IterativeResult> {
    short(a, b, options, Short::Qmr)
}

/// Solve `A x = b` using transpose-free quasi-minimal residual iterations.
///
/// # Parameters and conventions
/// `a` may be CSR or a [`LinearOperator`]; b is a real vector. `options` controls
/// initial guess, tolerances, iteration count and inverse preconditioning.
/// Uses forward actions only. This implementation requires identity preconditioning.
/// # Domain and special values
/// Finite, nonempty square operators with order <=256. Zero right sides and
/// already-converged initial guesses are supported. NaN/infinity are rejected.
/// # Returns
/// [`IterativeResult`]; convergence means `||b-Ax||₂ <= max(atol, rtol ||b||₂)`.
/// MINRES applies this test to A-shift I. Exhausted iterations return the
/// current approximation with converged=false; no result is silently discarded.
/// # Errors
/// Invalid shape/options/output, unsupported preconditioning, arithmetic
/// breakdown or nonfinite evaluations. maxiter <=1024 bounds iteration work;
/// each callback must bound its own work. No SciPy callback/logging controls.
/// # Numerical behavior
/// Transpose-free quasi-minimal residual with f64 arithmetic; roundoff can delay convergence
/// or trigger breakdown. Final diagnostics recompute the true residual.
/// Projected least-squares problems use private Sci/faer SVD kernels.
/// For built-in CSR inputs, each product has at most 65,536 entries.
/// # Examples
/// ```
/// let a=sci_sparse::identity(2)?;
/// let result=sci_sparse::linalg::tfqmr(&a,&[2.,3.],&Default::default())?;
/// assert!(result.converged);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy tfqmr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.tfqmr.html).
pub fn tfqmr(a: &impl LinearMap, b: &[f64], options: &SolverOptions) -> Result<IterativeResult> {
    short(a, b, options, Short::Tfqmr)
}

/// Solve `A x = b` using restarted generalized minimal residual iterations.
///
/// # Parameters and conventions
/// `a` may be CSR or a [`LinearOperator`]; b is a real vector. `options` controls
/// initial guess, tolerances, iteration count and inverse preconditioning.
/// Uses left inverse preconditioning. restart is capped at 255 to bound projected least-squares dimensions.
/// # Domain and special values
/// Finite, nonempty square operators with order <=256. Zero right sides and
/// already-converged initial guesses are supported. NaN/infinity are rejected.
/// # Returns
/// [`IterativeResult`]; convergence means `||b-Ax||₂ <= max(atol, rtol ||b||₂)`.
/// MINRES applies this test to A-shift I. Exhausted iterations return the
/// current approximation with converged=false; no result is silently discarded.
/// # Errors
/// Invalid shape/options/output, unsupported preconditioning, arithmetic
/// breakdown or nonfinite evaluations. maxiter <=1024 bounds iteration work;
/// each callback must bound its own work. No SciPy callback/logging controls.
/// # Numerical behavior
/// Restarted generalized minimal residual with f64 arithmetic; roundoff can delay convergence
/// or trigger breakdown. Final diagnostics recompute the true residual.
/// Projected least-squares problems use private Sci/faer SVD kernels.
/// For built-in CSR inputs, each product has at most 65,536 entries.
/// # Examples
/// ```
/// let a=sci_sparse::identity(2)?;
/// let result=sci_sparse::linalg::gmres(&a,&[2.,3.],&Default::default())?;
/// assert!(result.converged);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy gmres](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html).
pub fn gmres(a: &impl LinearMap, b: &[f64], options: &SolverOptions) -> Result<IterativeResult> {
    projected(a, b, options, Restarted::Gmres)
}

/// Solve `A x = b` using augmented restarted generalized minimal residual iterations.
///
/// # Parameters and conventions
/// `a` may be CSR or a [`LinearOperator`]; b is a real vector. `options` controls
/// initial guess, tolerances, iteration count and inverse preconditioning.
/// Retains up to recycle previous correction vectors between inner cycles; unlike SciPy, recycling state is local to the call. restart is capped at 255.
/// # Domain and special values
/// Finite, nonempty square operators with order <=256. Zero right sides and
/// already-converged initial guesses are supported. NaN/infinity are rejected.
/// # Returns
/// [`IterativeResult`]; convergence means `||b-Ax||₂ <= max(atol, rtol ||b||₂)`.
/// MINRES applies this test to A-shift I. Exhausted iterations return the
/// current approximation with converged=false; no result is silently discarded.
/// # Errors
/// Invalid shape/options/output, unsupported preconditioning, arithmetic
/// breakdown or nonfinite evaluations. maxiter <=1024 bounds iteration work;
/// each callback must bound its own work. No SciPy callback/logging controls.
/// # Numerical behavior
/// Augmented restarted generalized minimal residual with f64 arithmetic; roundoff can delay convergence
/// or trigger breakdown. Final diagnostics recompute the true residual.
/// Projected least-squares problems use private Sci/faer SVD kernels.
/// For built-in CSR inputs, each product has at most 65,536 entries.
/// # Examples
/// ```
/// let a=sci_sparse::identity(2)?;
/// let result=sci_sparse::linalg::lgmres(&a,&[2.,3.],&Default::default())?;
/// assert!(result.converged);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy lgmres](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lgmres.html).
pub fn lgmres(a: &impl LinearMap, b: &[f64], options: &SolverOptions) -> Result<IterativeResult> {
    projected(a, b, options, Restarted::Lgmres)
}

/// Solve `A x = b` using recycling generalized conjugate residual with inner orthogonalization iterations.
///
/// # Parameters and conventions
/// `a` may be CSR or a [`LinearOperator`]; b is a real vector. `options` controls
/// initial guess, tolerances, iteration count and inverse preconditioning.
/// Retains paired correction/image vectors, projects residuals and Arnoldi steps against recycled images, and discards oldest pairs. External CU state and flexible preconditioner mutation are not exposed. restart is capped at 255.
/// # Domain and special values
/// Finite, nonempty square operators with order <=256. Zero right sides and
/// already-converged initial guesses are supported. NaN/infinity are rejected.
/// # Returns
/// [`IterativeResult`]; convergence means `||b-Ax||₂ <= max(atol, rtol ||b||₂)`.
/// MINRES applies this test to A-shift I. Exhausted iterations return the
/// current approximation with converged=false; no result is silently discarded.
/// # Errors
/// Invalid shape/options/output, unsupported preconditioning, arithmetic
/// breakdown or nonfinite evaluations. maxiter <=1024 bounds iteration work;
/// each callback must bound its own work. No SciPy callback/logging controls.
/// # Numerical behavior
/// Recycling generalized conjugate residual with inner orthogonalization with f64 arithmetic; roundoff can delay convergence
/// or trigger breakdown. Final diagnostics recompute the true residual.
/// Projected least-squares problems use private Sci/faer SVD kernels.
/// For built-in CSR inputs, each product has at most 65,536 entries.
/// # Examples
/// ```
/// let a=sci_sparse::identity(2)?;
/// let result=sci_sparse::linalg::gcrotmk(&a,&[2.,3.],&Default::default())?;
/// assert!(result.converged);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy gcrotmk](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gcrotmk.html).
pub fn gcrotmk(a: &impl LinearMap, b: &[f64], options: &SolverOptions) -> Result<IterativeResult> {
    projected(a, b, options, Restarted::Gcrot)
}

/// Solve `A x = b` using symmetric minimum residual iterations.
///
/// # Parameters and conventions
/// `a` may be CSR or a [`LinearOperator`]; b is a real vector. `options` controls
/// initial guess, tolerances, iteration count and inverse preconditioning.
/// Checks exact symmetry through operator actions and applies options.shift. Uses full-reorthogonalized symmetric Arnoldi/Lanczos with restart at min(n,255); identity preconditioning is required.
/// # Domain and special values
/// Finite, nonempty square operators with order <=256. Zero right sides and
/// already-converged initial guesses are supported. NaN/infinity are rejected.
/// # Returns
/// [`IterativeResult`]; convergence means `||b-Ax||₂ <= max(atol, rtol ||b||₂)`.
/// MINRES applies this test to A-shift I. Exhausted iterations return the
/// current approximation with converged=false; no result is silently discarded.
/// # Errors
/// Invalid shape/options/output, unsupported preconditioning, arithmetic
/// breakdown or nonfinite evaluations. maxiter <=1024 bounds iteration work;
/// each callback must bound its own work. No SciPy callback/logging controls.
/// # Numerical behavior
/// Symmetric minimum residual with f64 arithmetic; roundoff can delay convergence
/// or trigger breakdown. Final diagnostics recompute the true residual.
/// Projected least-squares problems use private Sci/faer SVD kernels.
/// For built-in CSR inputs, each product has at most 65,536 entries.
/// # Examples
/// ```
/// let a=sci_sparse::identity(2)?;
/// let result=sci_sparse::linalg::minres(&a,&[2.,3.],&Default::default())?;
/// assert!(result.converged);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy minres](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.minres.html).
pub fn minres(a: &impl LinearMap, b: &[f64], options: &SolverOptions) -> Result<IterativeResult> {
    min_res(a, b, options)
}

/// Minimize `||A x - b||₂² + damp² ||x||₂²` using Golub–Kahan bidiagonalization with QR rotations.
///
/// # Parameters and conventions
/// `a` is a rectangular CSR or operator; b has a.rows components. Options
/// select damping, tolerances and maximum iterations. The initial guess is zero.
/// # Domain and special values
/// Finite nonempty axes <=256, finite b and nonnegative damping/tolerances.
/// Zero right sides, rank-deficient and inconsistent systems are supported.
/// # Returns
/// [`LeastSquaresResult`] with true primal and normal residual diagnostics.
/// Convergence means either residual or regularized normal residual is at most
/// `max(atol, rtol ||b||₂)`. This differs from SciPy's scaled stop-code tests.
/// # Errors
/// Invalid shape/options, nonfinite output, numerical breakdown; maxiter <=1024.
/// # Numerical behavior
/// Uses Golub–Kahan bidiagonalization with QR rotations. Operator columns are materialized once to bound
/// backend workspace; damping adds diagonal rows. Rank deficiency and poor
/// conditioning can reduce accuracy. No conlim, variance or logging options.
/// # Examples
/// ```
/// let r=sci_sparse::linalg::lsqr(&sci_sparse::identity(2)?,&[2.,3.],&Default::default())?;
/// assert!(r.converged);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy lsqr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html).
pub fn lsqr(
    a: &impl LinearMap,
    b: &[f64],
    options: &LeastSquaresOptions,
) -> Result<LeastSquaresResult> {
    lsqr_inner(a, b, options)
}

/// Minimize `||A x - b||₂² + damp² ||x||₂²` using private faer LSMR iteration.
///
/// # Parameters and conventions
/// `a` is a rectangular CSR or operator; b has a.rows components. Options
/// select damping, tolerances and maximum iterations. The initial guess is zero.
/// # Domain and special values
/// Finite nonempty axes <=256, finite b and nonnegative damping/tolerances.
/// Zero right sides, rank-deficient and inconsistent systems are supported.
/// # Returns
/// [`LeastSquaresResult`] with true primal and normal residual diagnostics.
/// Convergence means either residual or regularized normal residual is at most
/// `max(atol, rtol ||b||₂)`. This differs from SciPy's scaled stop-code tests.
/// # Errors
/// Invalid shape/options, nonfinite output, numerical breakdown; maxiter <=1024.
/// # Numerical behavior
/// Uses private faer LSMR iteration. Operator columns are materialized once to bound
/// backend workspace; damping adds diagonal rows. Rank deficiency and poor
/// conditioning can reduce accuracy. No conlim, variance or logging options.
/// # Examples
/// ```
/// let r=sci_sparse::linalg::lsmr(&sci_sparse::identity(2)?,&[2.,3.],&Default::default())?;
/// assert!(r.converged);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy lsmr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html).
pub fn lsmr(
    a: &impl LinearMap,
    b: &[f64],
    options: &LeastSquaresOptions,
) -> Result<LeastSquaresResult> {
    lsmr_inner(a, b, options)
}
