use super::design::polynomial;
use super::{Zpk, checked, nonempty, normalize, positive, samples, size, tf2zpk, work, zpk2tf};
use crate::{Error, Result, complex::Complex};
use nalgebra::{DMatrix, DVector};

/// Real continuous or discrete state-space model: x'=Ax+Bu, y=Cx+Du.
/// For a discrete model x' denotes the next state. Rows are row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct StateSpace {
    /// Square n-by-n state matrix; an empty matrix represents zero states.
    pub a: Vec<Vec<f64>>,
    /// n-by-m input matrix; empty when n=0.
    pub b: Vec<Vec<f64>>,
    /// p-by-n output matrix (p empty rows when n=0).
    pub c: Vec<Vec<f64>>,
    /// p-by-m direct feedthrough matrix, with p,m positive.
    pub d: Vec<Vec<f64>>,
}
/// Discretization of a continuous state-space model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Discretization {
    /// Piecewise constant input, exact matrix exponential.
    ZeroOrderHold,
    /// Trapezoidal generalized bilinear transform, alpha=1/2.
    Bilinear,
    /// Forward Euler, alpha=0.
    Euler,
    /// Backward Euler, alpha=1.
    BackwardDifference,
}
/// Simulation arrays, with one row per time sample.
#[derive(Clone, Debug, PartialEq)]
pub struct Simulation {
    /// Times in caller time units.
    pub time: Vec<f64>,
    /// Outputs \[time\]\[output\].
    pub output: Vec<Vec<f64>>,
    /// States \[time\]\[state\].
    pub state: Vec<Vec<f64>>,
}
fn validate_matrix(x: &[Vec<f64>], r: usize, c: usize) -> Result<()> {
    if x.len() != r || x.iter().any(|row| row.len() != c) {
        return Err(Error::DimensionMismatch);
    }
    work(r, c)?;
    for row in x {
        samples(row)?;
    }
    Ok(())
}
fn mat(x: &[Vec<f64>], r: usize, c: usize) -> Result<DMatrix<f64>> {
    validate_matrix(x, r, c)?;
    Ok(DMatrix::from_fn(r, c, |i, j| x[i][j]))
}
fn rows(x: &DMatrix<f64>) -> Result<Vec<Vec<f64>>> {
    (0..x.nrows())
        .map(|i| checked((0..x.ncols()).map(|j| x[(i, j)]).collect()))
        .collect()
}
fn dimensions(s: &StateSpace) -> Result<(usize, usize, usize)> {
    let n = s.a.len();
    let p = s.d.len();
    let m = s.d.first().map_or(0, Vec::len);
    if n > 32 || p == 0 || m == 0 || p > 32 || m > 32 {
        return Err(Error::InvalidInput(
            "state/input/output dimensions must be <=32; input/output nonempty",
        ));
    }
    validate_matrix(&s.a, n, n)?;
    validate_matrix(&s.b, n, m)?;
    validate_matrix(&s.c, p, n)?;
    validate_matrix(&s.d, p, m)?;
    Ok((n, m, p))
}
fn characteristic(a: &DMatrix<f64>) -> Result<Vec<f64>> {
    let n = a.nrows();
    let mut b = DMatrix::identity(n, n);
    let mut c = vec![1.];
    for k in 1..=n {
        let ab = a * &b;
        let ck = -ab.trace() / k as f64;
        b = ab + DMatrix::identity(n, n) * ck;
        c.push(ck);
    }
    checked(c)
}
fn time_grid(t: &[f64]) -> Result<f64> {
    nonempty(t)?;
    if t[0] < 0. {
        return Err(Error::InvalidInput("negative initial time"));
    }
    if t.len() == 1 {
        return Ok(0.);
    }
    let dt = t[1] - t[0];
    positive(dt)?;
    if t.windows(2).any(|v| ((v[1] - v[0]) - dt).abs() > 1e-9 * dt) {
        return Err(Error::InvalidInput(
            "times must be increasing and equally spaced",
        ));
    }
    Ok(dt)
}
fn exponential(a: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    // Reject unbounded scaling and squaring before entering the backend.
    if a.iter().any(|v| !v.is_finite()) || a.norm() > 1e6 {
        return Err(Error::InvalidInput(
            "matrix exponential argument exceeds work bound",
        ));
    }
    if a.nrows() == 0 {
        return Ok(DMatrix::zeros(0, 0));
    }
    let e = a.exp();
    if e.iter().any(|v| !v.is_finite()) {
        return Err(Error::NonFiniteEvaluation);
    }
    Ok(e)
}

/// Validates matrices for `x' = Ax + Bu`, `y = Cx + Du`.
///
/// ## Parameters
///
/// All four matrices are explicit, row-major; A is n-by-n, B n-by-m, C p-by-n, D p-by-m. n<=32; p,m in 1..32. Zero-state models are supported. SciPy missing-matrix inference is replaced by explicit dimensions. Returns an owned StateSpace.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic; cancellation and ill conditioning can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::systems::*;
/// let _ = abcd_normalize(&[vec![-1.]],&[vec![1.]],&[vec![1.]],&[vec![0.]])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy abcd_normalize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.abcd_normalize.html).
pub fn abcd_normalize(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    c: &[Vec<f64>],
    d: &[Vec<f64>],
) -> Result<StateSpace> {
    let (n, p, m) = (a.len(), d.len(), d.first().map_or(0, Vec::len));
    if n > 32 || p == 0 || p > 32 || m == 0 || m > 32 {
        return Err(Error::InvalidInput("invalid state-space dimensions"));
    }
    validate_matrix(a, n, n)?;
    validate_matrix(b, n, m)?;
    validate_matrix(c, p, n)?;
    validate_matrix(d, p, m)?;
    Ok(StateSpace {
        a: a.to_vec(),
        b: b.to_vec(),
        c: c.to_vec(),
        d: d.to_vec(),
    })
}

/// Converts `H(s)=B(s)/A(s)` to controllable canonical state space.
///
/// ## Parameters
///
/// b/a are descending-power polynomial coefficients for a proper SISO system. Order is <=32; a must have nonzero leading coefficient. Returns StateSpace in SciPy controller canonical ordering, including one zero dummy state for constant transfer functions. Digital delay coefficients must first be padded to equal polynomial length.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic; cancellation and ill conditioning can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::systems::*;
/// let _ = tf2ss(&[1.],&[1.,2.,1.])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy tf2ss](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.tf2ss.html).
pub fn tf2ss(b: &[f64], a: &[f64]) -> Result<StateSpace> {
    let (b, a) = normalize(b, a)?;
    if b.len() > a.len() || a.len() > 33 {
        return Err(Error::InvalidInput("proper transfer order <=32 required"));
    }
    let n = a.len() - 1;
    let mut bp = vec![0.; a.len()];
    bp[a.len() - b.len()..].copy_from_slice(&b);
    if n == 0 {
        return abcd_normalize(&[vec![0.]], &[vec![0.]], &[vec![0.]], &[vec![bp[0]]]);
    }
    let mut am = vec![vec![0.; n]; n];
    am[0] = a[1..].iter().map(|v| -v).collect();
    for (i, row) in am.iter_mut().enumerate().skip(1) {
        row[i - 1] = 1.;
    }
    let mut bm = vec![vec![0.]; n];
    bm[0][0] = 1.;
    abcd_normalize(
        &am,
        &bm,
        &[bp[1..]
            .iter()
            .zip(&a[1..])
            .map(|(b, a)| b - bp[0] * a)
            .collect()],
        &[vec![bp[0]]],
    )
}

/// Converts `H(s)=k product(s-z)/product(s-p)` to controllable canonical state space.
///
/// ## Parameters
///
/// Real proper ZPK with order<=32. Root ordering follows zpk2tf; returns StateSpace as tf2ss.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic; cancellation and ill conditioning can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::systems::*;
/// use sci_signal::design::buttap;
/// let _ = zpk2ss(&buttap(2)?)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy zpk2ss](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.zpk2ss.html).
pub fn zpk2ss(zpk: &Zpk) -> Result<StateSpace> {
    let (b, a) = zpk2tf(zpk)?;
    tf2ss(&b, &a)
}

/// Computes `H(s)=C(sI-A)^-1 B + D` as real polynomial coefficients.
///
/// ## Parameters
///
/// s is StateSpace and input selects one input column. Returns (numerator rows in output order, common denominator), with descending powers. All numerator rows have n+1 coefficients.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Delegates matrix arithmetic to private nalgebra and uses the determinant lemma with Faddeev-LeVerrier characteristic coefficients; high orders can suffer cancellation.
///
/// # Examples
///
/// ```
/// use sci_signal::systems::*;
/// let s=tf2ss(&[1.],&[1.,2.])?; let _ = ss2tf(&s,0)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy ss2tf](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ss2tf.html).
pub fn ss2tf(s: &StateSpace, input: usize) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    let (n, m, p) = dimensions(s)?;
    if input >= m {
        return Err(Error::InvalidInput("input column outside model"));
    }
    let a = mat(&s.a, n, n)?;
    let den = characteristic(&a)?;
    let b = DVector::from_fn(n, |i, _| s.b[i][input]);
    let mut num = vec![];
    for o in 0..p {
        if n == 0 {
            num.push(vec![s.d[o][input]]);
            continue;
        }
        let c = DVector::from_vec(s.c[o].clone());
        let alt = characteristic(&(&a - &b * c.transpose()))?;
        num.push(checked(
            den.iter()
                .zip(alt)
                .map(|(d, v)| v + (s.d[o][input] - 1.) * d)
                .collect(),
        )?);
    }
    Ok((num, den))
}

/// Factors the selected output of `C(sI-A)^-1B+D` into zeros, poles and gain.
///
/// ## Parameters
///
/// input and output select model columns/rows. Returns shared Zpk, with sorted roots as tf2zpk. SciPy multiple output dispatch is replaced by explicit output selection.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic; cancellation and ill conditioning can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::systems::*;
/// let s=tf2ss(&[1.],&[1.,2.])?; let _ = ss2zpk(&s,0,0)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy ss2zpk](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ss2zpk.html).
pub fn ss2zpk(s: &StateSpace, input: usize, output: usize) -> Result<Zpk> {
    let (b, a) = ss2tf(s, input)?;
    let b = b
        .get(output)
        .ok_or(Error::InvalidInput("output outside model"))?;
    tf2zpk(b, &a)
}

/// Discretizes `x'=Ax+Bu` with zero-order hold or a generalized bilinear transformation.
///
/// ## Parameters
///
/// s is a continuous StateSpace, dt is positive time/sample, method selects ZOH or alpha=0,1/2,1. Returns a discrete StateSpace; caller retains dt. First-order-hold and impulse-invariant dispatch are unsupported.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses private nalgebra scaling-and-squaring matrix exponentials or pivoted LU; exponential matrix norm must not exceed 1e6. Singular transformations are errors.
///
/// # Examples
///
/// ```
/// use sci_signal::systems::*;
/// let s=tf2ss(&[1.],&[1.,2.])?;let _ = cont2discrete(&s,0.1,Discretization::ZeroOrderHold)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy cont2discrete](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cont2discrete.html).
pub fn cont2discrete(s: &StateSpace, dt: f64, method: Discretization) -> Result<StateSpace> {
    let (n, m, p) = dimensions(s)?;
    positive(dt)?;
    let a = mat(&s.a, n, n)?;
    let b = mat(&s.b, n, m)?;
    let c = mat(&s.c, p, n)?;
    let d = mat(&s.d, p, m)?;
    let (ad, bd, cd, dd) = if method == Discretization::ZeroOrderHold {
        let mut aug = DMatrix::zeros(n + m, n + m);
        aug.view_mut((0, 0), (n, n)).copy_from(&(a * dt));
        aug.view_mut((0, n), (n, m)).copy_from(&(b * dt));
        let e = exponential(&aug)?;
        (
            e.view((0, 0), (n, n)).into_owned(),
            e.view((0, n), (n, m)).into_owned(),
            c,
            d,
        )
    } else {
        let alpha = match method {
            Discretization::Bilinear => 0.5,
            Discretization::Euler => 0.,
            _ => 1.,
        };
        let eye = DMatrix::identity(n, n);
        let lu = (&eye - &a * (alpha * dt)).lu();
        let ad = lu
            .solve(&(&eye + a * ((1. - alpha) * dt)))
            .ok_or(Error::NumericalFailure("singular discretization"))?;
        let bd = lu
            .solve(&(b * dt))
            .ok_or(Error::NumericalFailure("singular discretization"))?;
        let cd = (&eye - mat(&s.a, n, n)? * (alpha * dt))
            .transpose()
            .lu()
            .solve(&c.transpose())
            .ok_or(Error::NumericalFailure("singular discretization"))?
            .transpose();
        let dd = d + &c * &bd * alpha;
        (ad, bd, cd, dd)
    };
    abcd_normalize(&rows(&ad)?, &rows(&bd)?, &rows(&cd)?, &rows(&dd)?)
}

/// Simulates `x[k+1]=Ax[k]+Bu[k]`, `y[k]=Cx[k]+Du[k]`.
///
/// ## Parameters
///
/// u has one row per sample, m input columns. dt>0; initial has n states. Returns Simulation with times k*dt, outputs, and states before each update. Input-time interpolation is unsupported.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic; cancellation and ill conditioning can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::systems::*;
/// let s=tf2ss(&[1.],&[1.,-0.5])?;let _=dlsim(&s,&[vec![1.],vec![0.]],1.,&[0.])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy dlsim](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.dlsim.html).
pub fn dlsim(s: &StateSpace, u: &[Vec<f64>], dt: f64, initial: &[f64]) -> Result<Simulation> {
    let (n, m, p) = dimensions(s)?;
    positive(dt)?;
    size(u.len())?;
    work(u.len(), (n + m + p) * (n + m))?;
    let u = mat(u, u.len(), m)?;
    samples(initial)?;
    if initial.len() != n {
        return Err(Error::DimensionMismatch);
    }
    let a = mat(&s.a, n, n)?;
    let b = mat(&s.b, n, m)?;
    let c = mat(&s.c, p, n)?;
    let d = mat(&s.d, p, m)?;
    let mut x = DVector::from_column_slice(initial);
    let mut output = vec![];
    let mut state = vec![];
    for k in 0..u.nrows() {
        let uk = u.row(k).transpose();
        state.push(checked(x.as_slice().to_vec())?);
        output.push(checked((&c * &x + &d * &uk).as_slice().to_vec())?);
        x = &a * x + &b * uk;
    }
    Ok(Simulation {
        time: checked((0..u.nrows()).map(|k| k as f64 * dt).collect())?,
        output,
        state,
    })
}

/// Simulates the continuous system `x'=Ax+Bu`, `y=Cx+Du` exactly for each held or linearly interpolated input segment.
///
/// ## Parameters
///
/// t is nonempty, nonnegative, equally spaced; u has \[time\]\[input\] shape; initial has n states at time zero. linear=true linearly interpolates input, otherwise ZOH. Returns Simulation in caller time units.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic; cancellation and ill conditioning can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::systems::*;
/// let s=tf2ss(&[1.],&[1.,1.])?;let _=lsim(&s,&[vec![1.],vec![1.]],&[0.,1.],&[0.],true)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy lsim](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lsim.html).
pub fn lsim(
    s: &StateSpace,
    u: &[Vec<f64>],
    t: &[f64],
    initial: &[f64],
    linear: bool,
) -> Result<Simulation> {
    let (n, m, p) = dimensions(s)?;
    let dt = time_grid(t)?;
    mat(u, t.len(), m)?;
    samples(initial)?;
    if initial.len() != n {
        return Err(Error::DimensionMismatch);
    }
    work(t.len(), (n + m + p) * (n + m))?;
    let a = mat(&s.a, n, n)?;
    let b = mat(&s.b, n, m)?;
    let c = mat(&s.c, p, n)?;
    let d = mat(&s.d, p, m)?;
    let mut x = exponential(&(&a * t[0]))? * DVector::from_column_slice(initial);
    let mut output = vec![];
    let mut state = vec![];
    let extra = if linear { 2 * m } else { m };
    let mut aug = DMatrix::zeros(n + extra, n + extra);
    aug.view_mut((0, 0), (n, n)).copy_from(&(&a * dt));
    aug.view_mut((0, n), (n, m)).copy_from(&(b * dt));
    if linear {
        aug.view_mut((n, n + m), (m, m))
            .copy_from(&DMatrix::identity(m, m));
    }
    let e = exponential(&aug)?;
    for k in 0..t.len() {
        let uk = DVector::from_vec(u[k].clone());
        state.push(checked(x.as_slice().to_vec())?);
        output.push(checked((&c * &x + &d * &uk).as_slice().to_vec())?);
        if k + 1 < t.len() {
            let mut z = DVector::zeros(n + extra);
            z.rows_mut(0, n).copy_from(&x);
            z.rows_mut(n, m).copy_from(&uk);
            if linear {
                z.rows_mut(n + m, m)
                    .copy_from(&(DVector::from_vec(u[k + 1].clone()) - uk));
            }
            x = (&e * z).rows(0, n).into_owned();
        }
    }
    Ok(Simulation {
        time: t.to_vec(),
        output,
        state,
    })
}

/// Computes the zero-state continuous step response for a selected input.
///
/// ## Parameters
///
/// s is continuous, t follows lsim; input selects the unit-step column. Returns Simulation including outputs for all output rows and state trajectories.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic; cancellation and ill conditioning can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::systems::*;
/// let s=tf2ss(&[1.],&[1.,1.])?;let _=step(&s,&[0.,1.,2.],0)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy step](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.step.html).
pub fn step(s: &StateSpace, t: &[f64], input: usize) -> Result<Simulation> {
    let (n, m, _) = dimensions(s)?;
    if input >= m {
        return Err(Error::InvalidInput("input outside model"));
    }
    size(t.len())?;
    let mut u = vec![vec![0.; m]; t.len()];
    for row in &mut u {
        row[input] = 1.;
    }
    lsim(s, &u, t, &vec![0.; n], false)
}

/// Computes the ordinary continuous impulse response `y(t)=C exp(At) B`.
///
/// ## Parameters
///
/// s is continuous; t follows lsim; input selects one column. Nonzero direct feedthrough is rejected because its Dirac delta cannot be represented as samples. Returns Simulation.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic; cancellation and ill conditioning can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::systems::*;
/// let s=tf2ss(&[1.],&[1.,1.])?;let _=impulse(&s,&[0.,1.,2.],0)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy impulse](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.impulse.html).
pub fn impulse(s: &StateSpace, t: &[f64], input: usize) -> Result<Simulation> {
    let (n, m, _) = dimensions(s)?;
    if input >= m {
        return Err(Error::InvalidInput("input outside model"));
    }
    if s.d.iter().any(|r| r[input] != 0.) {
        return Err(Error::InvalidInput(
            "impulse direct feedthrough is a Dirac delta",
        ));
    }
    size(t.len())?;
    let initial = (0..n).map(|i| s.b[i][input]).collect::<Vec<_>>();
    lsim(s, &vec![vec![0.; m]; t.len()], t, &initial, false)
}

/// Computes the zero-state discrete unit step response using the state recurrence.
///
/// ## Parameters
///
/// s is discrete; dt>0; length is positive sample count; input selects one input column. Returns Simulation with time, all outputs, and states.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic; cancellation and ill conditioning can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::systems::*;
/// let s=tf2ss(&[1.],&[1.,-0.5])?;let _=dstep(&s,1.,5,0)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy dstep](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.dstep.html).
pub fn dstep(s: &StateSpace, dt: f64, length: usize, input: usize) -> Result<Simulation> {
    let (n, m, _) = dimensions(s)?;
    if input >= m || length == 0 {
        return Err(Error::InvalidInput("invalid input or response length"));
    }
    size(length)?;
    let mut u = vec![vec![0.; m]; length];
    for row in &mut u {
        row[input] = 1.;
    }
    dlsim(s, &u, dt, &vec![0.; n])
}

/// Computes the zero-state discrete unit impulse response using the state recurrence.
///
/// ## Parameters
///
/// s is discrete; dt>0; length is positive sample count; input selects one input column. Returns Simulation with time, all outputs, and states.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses f64 arithmetic; cancellation and ill conditioning can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::systems::*;
/// let s=tf2ss(&[1.],&[1.,-0.5])?;let _=dimpulse(&s,1.,5,0)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy dimpulse](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.dimpulse.html).
pub fn dimpulse(s: &StateSpace, dt: f64, length: usize, input: usize) -> Result<Simulation> {
    let (n, m, _) = dimensions(s)?;
    if input >= m || length == 0 {
        return Err(Error::InvalidInput("invalid input or response length"));
    }
    size(length)?;
    let mut u = vec![vec![0.; m]; length];
    u[0][input] = 1.;
    dlsim(s, &u, dt, &vec![0.; n])
}

/// Computes single-input feedback `u=-Kx` so that `det(sI-(A-BK))=product(s-p)`.
///
/// ## Parameters
///
/// a is n-by-n, b is length n, poles are n shared Complex values forming real conjugate pairs. n in 1..16. Returns real gain vector K in state order. Multi-input iterative placement is unsupported.
///
/// ## Domain
///
/// Inputs must be finite and satisfy the stated bounds. NaN and infinity are errors.
///
/// ## Returns
///
/// Returns the values in the ordering and normalization stated above.
///
/// ## Errors
///
/// Returns shared errors for invalid inputs, incompatible shapes, singular solves,
/// work limits, or nonfinite results. Sample lengths are at most 1048576 and
/// direct work at most 16777216 operations.
///
/// ## Numerical behavior
///
/// Uses Ackermann single-input placement and a private pivoted LU solver. Ill-conditioned controllability matrices can lose accuracy; singular matrices are errors.
///
/// # Examples
///
/// ```
/// use sci_signal::systems::*;
/// let _=place_poles(&[vec![0.,1.],vec![0.,0.]],&[0.,1.],&[sci_complex::Complex{re:-1.,im:0.},sci_complex::Complex{re:-2.,im:0.}])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy place_poles](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.place_poles.html).
pub fn place_poles(a: &[Vec<f64>], b: &[f64], poles: &[Complex]) -> Result<Vec<f64>> {
    let n = a.len();
    if n == 0 || n > 16 || b.len() != n || poles.len() != n {
        return Err(Error::InvalidInput(
            "single-input placement requires 1..16 states and n poles",
        ));
    }
    samples(b)?;
    let a = mat(a, n, n)?;
    let coeff = polynomial(poles)?;
    let b = DVector::from_column_slice(b);
    let mut ctrl = DMatrix::zeros(n, n);
    let mut v = b;
    for j in 0..n {
        ctrl.set_column(j, &v);
        v = &a * v;
    }
    let mut e = DVector::zeros(n);
    e[n - 1] = 1.;
    let row = ctrl
        .transpose()
        .lu()
        .solve(&e)
        .ok_or(Error::NumericalFailure("system is uncontrollable"))?;
    let mut phi = DMatrix::identity(n, n);
    for c in coeff.iter().skip(1) {
        phi = &a * phi + DMatrix::identity(n, n) * *c;
    }
    checked((row.transpose() * phi).iter().copied().collect())
}
