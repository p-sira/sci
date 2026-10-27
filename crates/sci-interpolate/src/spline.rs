use crate::backend::{self, MAX_DEGREE, MAX_KNOTS};
use crate::error::is_finite_slice;
use crate::{Error, Result, Spline};

/// Behavior outside a spline's closed base interval, matching SciPy `ext=0..3`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplineExtension {
    /// Extrapolate the first or last polynomial (`ext=0`).
    Extrapolate,
    /// Return zero (`ext=1`).
    Zero,
    /// Reject the query (`ext=2`).
    Error,
    /// Evaluate at the closest endpoint (`ext=3`).
    Clamp,
}

/// Evaluates the B-spline derivative `S^(d)(x)`, where `S(x)=Σ c[i] B[i,k](x)`.
///
/// ## Parameters and conventions
///
/// `x` is query order; `derivative` is a nonnegative derivative order. `extension` selects SciPy ext=0..3 behavior. Units are sample-units / coordinate-units^derivative.
///
/// ## Domain and special values
///
/// Finite samples, knots and queries; degree 0..=32, at most 512 knots. Derivative order cannot exceed degree. The first/last nonempty base polynomial spans define extrapolation, including for unclamped knots. Clamp applies to derivative queries as well; SciPy legacy splev may extrapolate derivatives with ext=3.
///
/// ## Returns
///
/// One value per query, including an empty vector for no queries.
///
/// ## Errors
///
/// Rejects invalid shapes, domains, nonfinite inputs/arithmetic, work limits, or backend failure as applicable. No NaN/infinity inputs are accepted except explicitly allowed fill values.
///
/// ## Numerical behavior
///
/// Cox–de Boor recursion uses bounded local storage. At most 10,000,000 query × (degree+1)^2 operations. Derivatives can amplify roundoff.
///
/// # Examples
///
/// ```
/// let s=sci_interpolate::make_interp_spline(&[0.,1.], &[0.,2.],1)?;
/// assert_eq!(sci_interpolate::splev(&[0.5],&s,0,sci_interpolate::SplineExtension::Error)?,vec![1.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy splev](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splev.html).
/// Carl de Boor, *A Practical Guide to Splines* (2001); Paul Dierckx, *Curve and Surface Fitting with Splines* (1993).
pub fn splev(
    x: &[f64],
    spline: &Spline,
    derivative: usize,
    extension: SplineExtension,
) -> Result<Vec<f64>> {
    backend::bound(&[
        x.len(),
        spline.degree.saturating_add(1),
        spline.degree.saturating_add(1),
    ])?;
    backend::validate(spline)?;
    is_finite_slice(x)?;
    let s = splder(spline, derivative)?;
    let lo = s.knots[s.degree];
    let hi = s.knots[s.coefficients.len()];
    x.iter()
        .map(|x| {
            if *x < lo || *x > hi {
                match extension {
                    SplineExtension::Zero => return Ok(0.0),
                    SplineExtension::Error => {
                        return Err(Error::InvalidInput("query outside spline domain"));
                    }
                    SplineExtension::Clamp => return backend::evaluate(&s, x.clamp(lo, hi)),
                    SplineExtension::Extrapolate => (),
                }
            }
            backend::evaluate(&s, *x)
        })
        .collect()
}

/// Constructs `D^n S`, using `c_new[i]=k(c[i+1]-c[i])/(t[i+k+1]-t[i+1])`.
///
/// ## Parameters and conventions
///
/// `order` is a derivative count; zero returns a clone. SciPy padded tuple coefficients become exact-length Sci coefficients.
///
/// ## Domain and special values
///
/// Finite valid splines, degree at most 32 and at most 512 knots; order <= degree. Repeated knots that prevent classical differentiation are errors.
///
/// ## Returns
///
/// Spline of degree k-order with the same base interval; units are divided by coordinate-units^order.
///
/// ## Errors
///
/// Rejects invalid shapes, domains, nonfinite inputs/arithmetic, work limits, or backend failure as applicable. No NaN/infinity inputs are accepted except explicitly allowed fill values.
///
/// ## Numerical behavior
///
/// Linear coefficient work per derivative. Close knots amplify differences; overflowing intermediate arithmetic fails.
///
/// # Examples
///
/// ```
/// let s=sci_interpolate::make_interp_spline(&[0.,1.], &[0.,2.],1)?;
/// let d=sci_interpolate::splder(&s,1)?;
/// assert_eq!(d.degree,0);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy splder](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splder.html).
/// Carl de Boor, *A Practical Guide to Splines* (2001); Paul Dierckx, *Curve and Surface Fitting with Splines* (1993).
pub fn splder(spline: &Spline, order: usize) -> Result<Spline> {
    backend::validate(spline)?;
    if order > spline.degree {
        return Err(Error::InvalidInput("derivative order exceeds degree"));
    }
    let mut s = spline.clone();
    for _ in 0..order {
        let k = s.degree;
        let mut c = Vec::with_capacity(s.coefficients.len() - 1);
        for i in 0..s.coefficients.len() - 1 {
            let width = s.knots[i + k + 1] - s.knots[i + 1];
            if width == 0.0 {
                return Err(Error::InvalidInput(
                    "spline is not differentiable at repeated knots",
                ));
            }
            c.push(
                match (s.coefficients[i + 1] - s.coefficients[i]) * (k as f64 / width) {
                    value if value.is_finite() => Ok(value),
                    _ => Err(Error::NonFiniteEvaluation),
                }?,
            );
        }
        s = Spline {
            knots: s.knots[1..s.knots.len() - 1].to_vec(),
            coefficients: c,
            degree: k - 1,
        };
    }
    backend::validate(&s)?;
    Ok(s)
}

/// Constructs an n-fold antiderivative using `C[i+1]=C[i]+c[i](t[i+k+1]-t[i])/(k+1)`.
///
/// ## Parameters and conventions
///
/// `order` is an integration count; zero clones the spline. Each integration sets the first coefficient to zero, matching SciPy. For clamped splines the antiderivative vanishes at the left endpoint.
///
/// ## Domain and special values
///
/// Finite valid splines; order <=16, resulting degree <=32 and knots <=512. Nonfinite inputs fail.
///
/// ## Returns
///
/// Spline of degree k+order. Units are multiplied by coordinate-units^order.
///
/// ## Errors
///
/// Rejects invalid shapes, domains, nonfinite inputs/arithmetic, work limits, or backend failure as applicable. No NaN/infinity inputs are accepted except explicitly allowed fill values.
///
/// ## Numerical behavior
///
/// Linear cumulative coefficient work per integration; accumulation and small knot widths can lose precision.
///
/// # Examples
///
/// ```
/// let s=sci_interpolate::make_interp_spline(&[0.,1.], &[2.,2.],1)?;
/// let a=sci_interpolate::splantider(&s,1)?;
/// assert_eq!(a.degree,2);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy splantider](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splantider.html).
/// Carl de Boor, *A Practical Guide to Splines* (2001); Paul Dierckx, *Curve and Surface Fitting with Splines* (1993).
pub fn splantider(spline: &Spline, order: usize) -> Result<Spline> {
    backend::validate(spline)?;
    if order > 16
        || spline.degree + order > MAX_DEGREE
        || spline.knots.len() + 2 * order > MAX_KNOTS
    {
        return Err(Error::InvalidInput(
            "antiderivative degree or storage limit exceeded",
        ));
    }
    let mut s = spline.clone();
    for _ in 0..order {
        let mut c = vec![0.0];
        for i in 0..s.coefficients.len() {
            c.push(match c[i]
                + s.coefficients[i]
                    * ((s.knots[i + s.degree + 1] - s.knots[i]) / (s.degree + 1) as f64)
            {
                value if value.is_finite() => Ok(value),
                _ => Err(Error::NonFiniteEvaluation),
            }?);
        }
        let mut t = vec![s.knots[0]];
        t.extend_from_slice(&s.knots);
        t.push(s.knots[s.knots.len() - 1]);
        s = Spline {
            knots: t,
            coefficients: c,
            degree: s.degree + 1,
        };
    }
    backend::validate(&s)?;
    Ok(s)
}

/// Returns all derivatives `[S(x), S'(x), ..., S^(k)(x)]`.
///
/// ## Parameters and conventions
///
/// `x` contains scalar queries; `spline` is a Sci B-spline. Output is query-major, unlike Python scalar-versus-array nesting.
///
/// ## Domain and special values
///
/// Finite valid splines; all queries must be in the closed base interval. Right-hand pieces are used at interior knots.
///
/// ## Returns
///
/// A matrix with x.len() rows and degree+1 columns, in ascending derivative order.
///
/// ## Errors
///
/// Rejects invalid shapes, domains, nonfinite inputs/arithmetic, work limits, or backend failure as applicable. No NaN/infinity inputs are accepted except explicitly allowed fill values.
///
/// ## Numerical behavior
///
/// Reuses splder and local evaluation, bounding query × (degree+1)^3 work by 10,000,000. Repeated-knot nondifferentiability fails.
///
/// # Examples
///
/// ```
/// let s=sci_interpolate::make_interp_spline(&[0.,1.], &[0.,2.],1)?;
/// assert_eq!(sci_interpolate::spalde(&[0.5],&s)?,vec![vec![1.,2.]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy spalde](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.spalde.html).
/// Carl de Boor, *A Practical Guide to Splines* (2001); Paul Dierckx, *Curve and Surface Fitting with Splines* (1993).
pub fn spalde(x: &[f64], spline: &Spline) -> Result<Vec<Vec<f64>>> {
    backend::validate(spline)?;
    backend::bound(&[x.len(), (spline.degree + 1).pow(3)])?;
    is_finite_slice(x)?;
    let mut result = vec![Vec::with_capacity(spline.degree + 1); x.len()];
    for j in 0..=spline.degree {
        let values = splev(x, spline, j, SplineExtension::Error)?;
        for (row, value) in result.iter_mut().zip(values) {
            row.push(value);
        }
    }
    Ok(result)
}

/// Computes the definite integral `I=∫_a^b S(x) dx`, taking S=0 outside its base interval.
///
/// ## Parameters and conventions
///
/// `a`, `b` are finite bounds. Reversed bounds negate the result; both bounds are clipped to the base interval, matching SciPy. Diagnostic basis-integral arrays are omitted.
///
/// ## Domain and special values
///
/// Finite valid splines; the extra antiderivative degree and two knots must fit the shared storage limits.
///
/// ## Returns
///
/// Scalar integral in sample-units × coordinate-units.
///
/// ## Errors
///
/// Rejects invalid shapes, domains, nonfinite inputs/arithmetic, work limits, or backend failure as applicable. No NaN/infinity inputs are accepted except explicitly allowed fill values.
///
/// ## Numerical behavior
///
/// Analytic antiderivative evaluation, with coefficient accumulation and endpoint subtraction roundoff. Equal bounds give zero.
///
/// # Examples
///
/// ```
/// let s=sci_interpolate::make_interp_spline(&[0.,1.], &[0.,2.],1)?;
/// assert!((sci_interpolate::splint(0.,1.,&s)?-1.).abs()<1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy splint](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splint.html).
/// Carl de Boor, *A Practical Guide to Splines* (2001); Paul Dierckx, *Curve and Surface Fitting with Splines* (1993).
pub fn splint(a: f64, b: f64, spline: &Spline) -> Result<f64> {
    backend::validate(spline)?;
    is_finite_slice(&[a, b])?;
    let s = splantider(spline, 1)?;
    let lo = spline.knots[spline.degree];
    let hi = spline.knots[spline.coefficients.len()];
    match backend::evaluate(&s, b.clamp(lo, hi))? - backend::evaluate(&s, a.clamp(lo, hi))? {
        value if value.is_finite() => Ok(value),
        _ => Err(Error::NonFiniteEvaluation),
    }
}

/// Refines a spline knot vector while preserving `S(x)=Σ c[i] B[i,k](x)`.
///
/// ## Parameters and conventions
///
/// `x` is an interior knot and `multiplicity` is the number of new copies. Periodic insertion is omitted. Coefficients use the Boehm convex recurrence with `α=(x-t[i])/(t[i+k]-t[i])`.
///
/// ## Domain and special values
///
/// Finite valid splines; x strictly inside the base interval; positive insertion count and resulting multiplicity <=degree+1. At most 512 resulting knots.
///
/// ## Returns
///
/// A new Sci spline of unchanged degree and function, with the requested extra knots.
///
/// ## Errors
///
/// Rejects invalid shapes, domains, nonfinite inputs/arithmetic, work limits, or backend failure as applicable. No NaN/infinity inputs are accepted except explicitly allowed fill values.
///
/// ## Numerical behavior
///
/// Boehm insertion takes linear coefficient work per copy, up to 33 copies; preserves the function up to roundoff.
///
/// # Examples
///
/// ```
/// let s=sci_interpolate::make_interp_spline(&[0.,1.], &[0.,2.],1)?;
/// assert_eq!(sci_interpolate::insert(0.5,&s,1)?.knots.len(),s.knots.len()+1);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy insert](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.insert.html).
/// Carl de Boor, *A Practical Guide to Splines* (2001); Paul Dierckx, *Curve and Surface Fitting with Splines* (1993).
pub fn insert(x: f64, spline: &Spline, multiplicity: usize) -> Result<Spline> {
    backend::validate(spline)?;
    is_finite_slice(&[x])?;
    if multiplicity == 0
        || multiplicity > spline.degree + 1
        || spline.knots.len() + multiplicity > MAX_KNOTS
        || x <= spline.knots[spline.degree]
        || x >= spline.knots[spline.coefficients.len()]
        || spline.knots.iter().filter(|v| **v == x).count() + multiplicity > spline.degree + 1
    {
        return Err(Error::InvalidInput("invalid interior knot or multiplicity"));
    }
    let mut s = spline.clone();
    for _ in 0..multiplicity {
        let k = s.degree;
        let n = s.coefficients.len();
        let span = s.knots.partition_point(|v| *v <= x) - 1;
        let repeated = s.knots.iter().filter(|v| **v == x).count();
        let mut c = vec![0.0; n + 1];
        c[..=span - k].copy_from_slice(&s.coefficients[..=span - k]);
        c[span - repeated + 1..].copy_from_slice(&s.coefficients[span - repeated..]);
        for (i, ci) in c
            .iter_mut()
            .enumerate()
            .take(span - repeated + 1)
            .skip(span - k + 1)
        {
            let alpha = (x - s.knots[i]) / (s.knots[i + k] - s.knots[i]);
            *ci = match alpha * s.coefficients[i] + (1.0 - alpha) * s.coefficients[i - 1] {
                value if value.is_finite() => Ok(value),
                _ => Err(Error::NonFiniteEvaluation),
            }?;
        }
        s.knots.insert(span + 1, x);
        s.coefficients = c;
    }
    backend::validate(&s)?;
    Ok(s)
}

/// Finds the zeros `x` satisfying `S(x)=0` within a cubic spline base interval.
///
/// ## Parameters and conventions
///
/// `spline` is cubic, with simple interior knots. SciPy mest is omitted: storage is bounded by the knot count.
///
/// ## Domain and special values
///
/// Finite valid cubic splines with at most 512 knots. An identically zero span does not produce an infinite root set. Near-multiple roots may be missed or inaccurate.
///
/// ## Returns
///
/// Ascending real roots in the closed base interval, in coordinate units. Roots shared by adjacent spans are deduplicated within 1e-12 times the base interval width.
///
/// ## Errors
///
/// Rejects invalid shapes, domains, nonfinite inputs/arithmetic, work limits, or backend failure as applicable. No NaN/infinity inputs are accepted except explicitly allowed fill values.
///
/// ## Numerical behavior
///
/// Local cubics in normalized span coordinates use private nalgebra companion-matrix Schur solves, limited to 256 iterations per span, followed by four Newton refinements. Leading coefficients below 64ε relative to the largest coefficient are discarded. Near-multiple roots are ill-conditioned; no accuracy estimate is returned. Unlike FITPACK, knot roots can be detected.
///
/// # Examples
///
/// ```
/// let s=sci_interpolate::make_interp_spline(&[0.,1.,2.,3.], &[-1.,0.,1.,2.],3)?;
/// let roots=sci_interpolate::sproot(&s)?;
/// assert!(roots.iter().any(|r|(*r-1.).abs()<1e-10));
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy sproot](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.sproot.html).
/// Carl de Boor, *A Practical Guide to Splines* (2001); Paul Dierckx, *Curve and Surface Fitting with Splines* (1993).
pub fn sproot(spline: &Spline) -> Result<Vec<f64>> {
    backend::validate(spline)?;
    if spline.degree != 3
        || spline.knots[3..spline.knots.len() - 3]
            .windows(2)
            .any(|p| p[0] >= p[1])
    {
        return Err(Error::InvalidInput(
            "roots require a cubic with simple interior knots",
        ));
    }
    let derivatives = (0..=3)
        .map(|d| splder(spline, d))
        .collect::<Result<Vec<_>>>()?;
    let mut roots = Vec::new();
    for pair in spline.knots[3..spline.knots.len() - 3].windows(2) {
        let width = pair[1] - pair[0];
        let mut c = Vec::new();
        let mut factor = 1.0;
        for (j, d) in derivatives.iter().enumerate() {
            if j > 0 {
                factor = match factor * (width / j as f64) {
                    value if value.is_finite() => Ok(value),
                    _ => Err(Error::NonFiniteEvaluation),
                }?;
            }
            c.push(match factor * backend::evaluate(d, pair[0])? {
                value if value.is_finite() => Ok(value),
                _ => Err(Error::NonFiniteEvaluation),
            }?);
        }
        for u in backend::polynomial_roots(&c)? {
            if (-1e-12..=1.0 + 1e-12).contains(&u) {
                roots.push(match pair[0] + width * u.clamp(0.0, 1.0) {
                    value if value.is_finite() => Ok(value),
                    _ => Err(Error::NonFiniteEvaluation),
                }?);
            }
        }
    }
    roots.sort_by(f64::total_cmp);
    let tolerance = 1e-12 * (spline.knots[spline.coefficients.len()] - spline.knots[3]);
    roots.dedup_by(|a, b| (*a - *b).abs() <= tolerance);
    Ok(roots)
}
