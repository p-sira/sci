use crate::backend::{self, MAX_FIT};
use crate::error::{finite, finite_slice};
use crate::{Error, ParametricSpline, Result, Spline, SurfaceSpline};
use nalgebra::{DMatrix, DVector};

/// Constructs `S(x)=Σ c[i] B[i,k](x)` such that `S(x[j])=y[j]`.
///
/// ## Parameters and conventions
///
/// `degree` is 1..=5. Uses SciPy not-a-knot conventions (midpoint sites for even degrees); custom boundary constraints, knots, batching and axes are omitted.
///
/// ## Domain and special values
///
/// 2..=128 strictly increasing finite nodes, more than degree; finite values with matching length. Extrapolation is selected later in splev.
///
/// ## Returns
///
/// A Sci-owned spline with complete knots and exact-length coefficients, in sample units.
///
/// ## Errors
///
/// Rejects invalid shapes, domains, nonfinite inputs/arithmetic, work limits, or backend failure as applicable. No NaN/infinity inputs are accepted except explicitly allowed fill values.
///
/// ## Numerical behavior
///
/// Bounded dense collocation delegates to private nalgebra SVD with a rank check. Ill-conditioned finite fits can lose accuracy; no condition estimate is exposed.
///
/// # Examples
///
/// ```
/// let s=sci_interpolate::make_interp_spline(&[0.,1.], &[0.,2.],1)?;
/// assert_eq!(s.degree,1);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy make_interp_spline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_interp_spline.html).
/// Carl de Boor, *A Practical Guide to Splines* (2001); Paul Dierckx, *Curve and Surface Fitting with Splines* (1993).
pub fn make_interp_spline(x: &[f64], y: &[f64], degree: usize) -> Result<Spline> {
    backend::data(x, y, degree, None)?;
    if degree == 0 {
        return Err(Error::InvalidInput("interpolation degree must be in 1..=5"));
    }
    make_lsq_spline(x, y, &backend::knots(x, degree), degree, None)
}

/// Minimizes `Σ_j (w[j](S(x[j])-y[j]))²` on a supplied B-spline knot vector.
///
/// ## Parameters and conventions
///
/// `knots` is the complete knot vector; degree 0..=5; optional positive weights default to one. Scalar data replaces axis and batch arguments. SciPy qr/normal-equation method selection is replaced by SVD.
///
/// ## Domain and special values
///
/// 2..=128 increasing finite samples, more than degree, inside the knot base interval. At most 512 knots; number of coefficients cannot exceed sample count. Rank-deficient designs fail.
///
/// ## Returns
///
/// A fitted Sci spline with coefficients in increasing basis order.
///
/// ## Errors
///
/// Rejects invalid shapes, domains, nonfinite inputs/arithmetic, work limits, or backend failure as applicable. No NaN/infinity inputs are accepted except explicitly allowed fill values.
///
/// ## Numerical behavior
///
/// Private SVD uses at most 1,000 iterations and checks rank using ε × matrix-size × largest-singular-value. Work is bounded by 10,000,000 row × coefficient² operations.
///
/// # Examples
///
/// ```
/// let s=sci_interpolate::make_lsq_spline(&[0.,1.,2.], &[0.,1.,2.], &[0.,0.,2.,2.],1,None)?;
/// assert_eq!(s.coefficients.len(),2);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy make_lsq_spline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_lsq_spline.html).
/// Carl de Boor, *A Practical Guide to Splines* (2001); Paul Dierckx, *Curve and Surface Fitting with Splines* (1993).
pub fn make_lsq_spline(
    x: &[f64],
    y: &[f64],
    knots: &[f64],
    degree: usize,
    weights: Option<&[f64]>,
) -> Result<Spline> {
    let w = backend::data(x, y, degree, weights)?;
    if knots.len() > backend::MAX_KNOTS || knots.len() < 2 * (degree + 1) {
        return Err(Error::InvalidInput("invalid knot count"));
    }
    let mut s = Spline {
        knots: knots.to_vec(),
        coefficients: vec![0.0; knots.len() - degree - 1],
        degree,
    };
    backend::validate(&s)?;
    if x[0] < knots[degree] || x[x.len() - 1] > knots[s.coefficients.len()] {
        return Err(Error::InvalidInput("data outside knot interval"));
    }
    let mut a = backend::design(x, knots, degree);
    let mut b = DVector::from_column_slice(y);
    for i in 0..x.len() {
        a.row_mut(i).scale_mut(w[i]);
        b[i] *= w[i];
    }
    s.coefficients = backend::least_squares(a, b)?;
    Ok(s)
}

/// Fits a smooth B-spline with `Σ_j(w[j](S(x[j])-y[j]))² <= smoothing`, minimizing derivative jumps.
///
/// ## Parameters and conventions
///
/// `degree` is 1, 3 or 5; explicit finite nonnegative smoothing replaces SciPy defaults. Positive smoothing supports only degrees 3 and 5; degree 1 is interpolation-only. Weights are positive inverse-error scales, defaulting to one. Bounds equal the sample endpoints. Periodic/custom knots and task-cache modes are omitted.
///
/// ## Domain and special values
///
/// 2..=128 increasing finite samples, more than degree, and finite values. Smoothing zero produces not-a-knot interpolation. The Dierckx residual criterion has a 0.1% relative stopping tolerance.
///
/// ## Returns
///
/// Sci spline, in sample units, with automatically selected knots.
///
/// ## Errors
///
/// Rejects invalid shapes, domains, nonfinite inputs/arithmetic, work limits, or backend failure as applicable. No NaN/infinity inputs are accepted except explicitly allowed fill values.
///
/// ## Numerical behavior
///
/// Private pure-Rust Dierckx fitting bounds smoothing iterations by 20. Backend failures are errors; small smoothing and extreme scaling can fail. Interpolation reuses make_interp_spline.
///
/// # Examples
///
/// ```
/// let s=sci_interpolate::make_splrep(&[0.,1.,2.,3.], &[0.,1.,4.,9.],3,0.,None)?;
/// assert_eq!(s.degree,3);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy make_splrep](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_splrep.html).
/// Carl de Boor, *A Practical Guide to Splines* (2001); Paul Dierckx, *Curve and Surface Fitting with Splines* (1993).
pub fn make_splrep(
    x: &[f64],
    y: &[f64],
    degree: usize,
    smoothing: f64,
    weights: Option<&[f64]>,
) -> Result<Spline> {
    let w = backend::data(x, y, degree, weights)?;
    if !smoothing.is_finite() || smoothing < 0.0 || ![1, 3, 5].contains(&degree) {
        return Err(Error::InvalidInput(
            "smoothing must be finite and nonnegative; FITPACK degree must be 1, 3 or 5",
        ));
    }
    if smoothing == 0.0 {
        return make_interp_spline(x, y, degree);
    }
    if degree == 1 {
        return Err(Error::InvalidInput(
            "positive smoothing supports cubic and quintic fits",
        ));
    }
    match degree {
        3 => backend::fit::<3>(x, y, &w, smoothing),
        5 => backend::fit::<5>(x, y, &w, smoothing),
        _ => unreachable!(),
    }
}

/// Fits a parametric curve `g(u)=(S_0(u), ..., S_(d-1)(u))`, constraining `Σ_i w[i]² ||g(u[i])-points[i]||² <= smoothing`.
///
/// ## Parameters and conventions
///
/// `points` is sample-major, with 1..=3 coordinates. Optional increasing parameters default to normalized cumulative Euclidean chord lengths in `[0,1]`. Degree is 1,3 or5; positive smoothing supports only 3 and 5. Weights default to one. Periodic fits, custom knots, bounds and batching are omitted.
///
/// ## Domain and special values
///
/// 2..=128 finite points, more than degree; positive finite weights; finite nonnegative smoothing. Automatic parameterization rejects repeated adjacent points and zero-length curves.
///
/// ## Returns
///
/// ParametricSpline containing coordinate splines in input coordinate order and the sample parameters. Each component uses the same knots.
///
/// ## Errors
///
/// Rejects invalid shapes, domains, nonfinite inputs/arithmetic, work limits, or backend failure as applicable. No NaN/infinity inputs are accepted except explicitly allowed fill values.
///
/// ## Numerical behavior
///
/// Interpolation uses shared collocation. Smoothing delegates to a private Dierckx constrained-curve fitter without endpoint constraints, with at most 20 smoothing iterations.
///
/// # Examples
///
/// ```
/// let s=sci_interpolate::make_splprep(&[vec![0.,0.],vec![1.,2.]],None,1,0.,None)?;
/// assert_eq!(s.components.len(),2);
/// assert_eq!(s.parameters,vec![0.,1.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy make_splprep](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_splprep.html).
/// Carl de Boor, *A Practical Guide to Splines* (2001); Paul Dierckx, *Curve and Surface Fitting with Splines* (1993).
pub fn make_splprep(
    points: &[Vec<f64>],
    parameters: Option<&[f64]>,
    degree: usize,
    smoothing: f64,
    weights: Option<&[f64]>,
) -> Result<ParametricSpline> {
    if points.len() < 2
        || points.len() > MAX_FIT
        || degree > 5
        || points.len() <= degree
        || !smoothing.is_finite()
        || smoothing < 0.0
        || ![1, 3, 5].contains(&degree)
    {
        return Err(Error::InvalidInput(
            "invalid parametric fit size, degree, or smoothing",
        ));
    }
    let dim = points[0].len();
    if !(1..=3).contains(&dim) || points.iter().any(|p| p.len() != dim) {
        return Err(Error::DimensionMismatch);
    }
    for p in points {
        finite_slice(p)?;
    }
    let u = if let Some(u) = parameters {
        if u.len() != points.len() {
            return Err(Error::DimensionMismatch);
        }
        u.to_vec()
    } else {
        let mut u = vec![0.0];
        for p in points.windows(2) {
            let distance = p[0]
                .iter()
                .zip(&p[1])
                .try_fold(0.0_f64, |d, (a, b)| finite(d.hypot(finite(b - a)?)))?;
            u.push(finite(u[u.len() - 1] + distance)?);
        }
        let total = u[u.len() - 1];
        if total == 0.0 {
            return Err(Error::InvalidInput("curve has zero chord length"));
        }
        for ui in &mut u {
            *ui /= total;
        }
        u
    };
    let y: Vec<_> = points.iter().map(|p| p[0]).collect();
    let w = backend::data(&u, &y, degree, weights)?;
    if smoothing > 0.0 && degree == 1 {
        return Err(Error::InvalidInput(
            "positive curve smoothing supports cubic and quintic fits",
        ));
    }
    let components = if smoothing == 0.0 {
        (0..dim)
            .map(|j| {
                make_interp_spline(&u, &points.iter().map(|p| p[j]).collect::<Vec<_>>(), degree)
            })
            .collect::<Result<_>>()?
    } else {
        match (degree, dim) {
            (3, 1) => backend::curve::<3, 1>(&u, points, &w, smoothing),
            (3, 2) => backend::curve::<3, 2>(&u, points, &w, smoothing),
            (3, 3) => backend::curve::<3, 3>(&u, points, &w, smoothing),
            (5, 1) => backend::curve::<5, 1>(&u, points, &w, smoothing),
            (5, 2) => backend::curve::<5, 2>(&u, points, &w, smoothing),
            (5, 3) => backend::curve::<5, 3>(&u, points, &w, smoothing),
            _ => unreachable!(),
        }?
    };
    Ok(ParametricSpline {
        components,
        parameters: u,
    })
}

/// Estimates `P(h)=Σ_(j=0)^degree f^(j)(x) h^j/j!` by polynomial interpolation near x.
///
/// ## Parameters and conventions
///
/// `f` is a scalar callback; scale is positive coordinate width. Fitting order defaults to degree; nodes are x+scale*cos(π i/(order+1)), matching SciPy. Closure captures replace extra callback state.
///
/// ## Domain and special values
///
/// Finite x and scale>0; degree<=order<=31. Callback results must be finite. Nonfinite or indistinguishable generated nodes fail.
///
/// ## Returns
///
/// Descending coefficients of a polynomial in the displacement h, not absolute x; exactly degree+1 entries.
///
/// ## Errors
///
/// Rejects invalid shapes, domains, nonfinite inputs/arithmetic, work limits, or backend failure as applicable. No NaN/infinity inputs are accepted except explicitly allowed fill values.
///
/// ## Numerical behavior
///
/// Exactly order+1 callback evaluations. Shared divided differences and monomial conversion fit the sampled polynomial, then truncate its Taylor coefficients. Scale trades truncation error against roundoff; high orders are ill-conditioned.
///
/// # Examples
///
/// ```
/// let c=sci_interpolate::approximate_taylor_polynomial(|x|x*x,0.,2,1.,None)?;
/// assert!((c[0]-1.).abs()<1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy approximate_taylor_polynomial](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.approximate_taylor_polynomial.html).
/// Carl de Boor, *A Practical Guide to Splines* (2001); Paul Dierckx, *Curve and Surface Fitting with Splines* (1993).
pub fn approximate_taylor_polynomial(
    f: impl Fn(f64) -> f64,
    x: f64,
    degree: usize,
    scale: f64,
    order: Option<usize>,
) -> Result<Vec<f64>> {
    let order = order.unwrap_or(degree);
    if degree > order || order > 31 || !x.is_finite() || !scale.is_finite() || scale <= 0.0 {
        return Err(Error::InvalidInput(
            "Taylor fit requires degree <= order <= 31 and positive finite scale",
        ));
    }
    let n = order + 1;
    let nodes: Vec<_> = (0..n)
        .map(|i| finite(x + scale * (std::f64::consts::PI * i as f64 / n as f64).cos()))
        .collect::<Result<_>>()?;
    let centered: Vec<_> = nodes.iter().map(|t| finite(t - x)).collect::<Result<_>>()?;
    if centered
        .iter()
        .enumerate()
        .any(|(i, node)| centered[..i].contains(node))
    {
        return Err(Error::InvalidInput(
            "Taylor sample nodes are indistinguishable",
        ));
    }
    let values = nodes
        .into_iter()
        .map(|t| finite(f(t)))
        .collect::<Result<Vec<_>>>()?;
    let c = crate::lagrange(&centered, &values)?;
    Ok(c[c.len() - degree - 1..].to_vec())
}

/// Generates successively refined knots until the weighted least-squares residual `Σ (w[i](S(x[i])-y[i]))²` meets the smoothing target.
///
/// ## Parameters and conventions
///
/// `degree` is 1..=5; weights default to one. Returns an owned sequence instead of a Python generator. One knot is added at the median data site of the eligible interval with greatest residual, replacing SciPy batched insertion. Custom bounds, periodic conditions and nest are omitted.
///
/// ## Domain and special values
///
/// 2..=64 increasing finite samples, more than degree; positive weights and finite smoothing>=0. Zero smoothing returns only the not-a-knot interpolation vector.
///
/// ## Returns
///
/// Knot vectors in increasing refinement order, ending at the residual target or not-a-knot interpolation. The sequence may differ from SciPy for positive smoothing.
///
/// ## Errors
///
/// Rejects invalid shapes, domains, nonfinite inputs/arithmetic, work limits, or backend failure as applicable. No NaN/infinity inputs are accepted except explicitly allowed fill values.
///
/// ## Numerical behavior
///
/// At most sample-count refinements; private SVD solves reuse make_lsq_spline. The target accepts 0.1% relative residual tolerance; finite residual overflow fails.
///
/// # Examples
///
/// ```
/// let t=sci_interpolate::generate_knots(&[0.,1.,2.,3.], &[0.,1.,4.,9.],3,0.,None)?;
/// assert_eq!(t.len(),1);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy generate_knots](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.generate_knots.html).
/// Carl de Boor, *A Practical Guide to Splines* (2001); Paul Dierckx, *Curve and Surface Fitting with Splines* (1993).
pub fn generate_knots(
    x: &[f64],
    y: &[f64],
    degree: usize,
    smoothing: f64,
    weights: Option<&[f64]>,
) -> Result<Vec<Vec<f64>>> {
    let w = backend::data(x, y, degree, weights)?;
    if degree == 0 || x.len() > 64 || !smoothing.is_finite() || smoothing < 0.0 {
        return Err(Error::InvalidInput(
            "knot refinement requires degree 1..=5, at most 64 samples and nonnegative smoothing",
        ));
    }
    if smoothing == 0.0 {
        return Ok(vec![backend::knots(x, degree)]);
    }
    let mut t = vec![x[0]; degree + 1];
    t.extend(std::iter::repeat_n(x[x.len() - 1], degree + 1));
    let mut sequence = Vec::new();
    for _ in 0..x.len() {
        sequence.push(t.clone());
        let fit = make_lsq_spline(x, y, &t, degree, Some(&w))?;
        let residuals: Vec<_> = x
            .iter()
            .zip(y)
            .zip(&w)
            .map(|((x, y), w)| {
                let r = finite(w * finite(backend::evaluate(&fit, *x)? - y)?)?;
                finite(r * r)
            })
            .collect::<Result<_>>()?;
        let residual = finite(residuals.iter().sum())?;
        if residual <= smoothing * 1.001 {
            return Ok(sequence);
        }
        if t.len() >= x.len() + degree {
            sequence.push(backend::knots(x, degree));
            return Ok(sequence);
        }
        // Refine the interval with the greatest residual using its median data site.
        // One knot per iteration deliberately replaces FITPACK's batched insertion.
        let mut choice = None;
        let mut largest = -1.0;
        for p in t.windows(2).filter(|p| p[0] < p[1]) {
            let indices: Vec<_> = (1..x.len() - 1)
                .filter(|i| x[*i] > p[0] && x[*i] < p[1])
                .collect();
            if indices.is_empty() {
                continue;
            }
            let sum: f64 = indices.iter().map(|i| residuals[*i]).sum();
            if sum > largest {
                largest = sum;
                choice = Some(x[indices[indices.len() / 2]]);
            }
        }
        let knot = choice.ok_or(Error::NumericalFailure("no knot refinement site"))?;
        t.insert(t.partition_point(|v| *v <= knot), knot);
    }
    Err(Error::NumericalFailure("knot refinement limit exceeded"))
}

/// Minimizes `Σ_i w[i](y[i]-S(x[i]))² + lambda ∫ (S''(x))² dx` with natural cubic boundaries.
///
/// ## Parameters and conventions
///
/// `lambda` is explicit and nonnegative; SciPy automatic GCV selection is omitted. Positive weights multiply residual squares directly, rather than being squared. Boundary second derivatives are zero. Scalar data replaces batching.
///
/// ## Domain and special values
///
/// 5..=128 increasing finite samples, finite values and lambda, positive finite weights. lambda=0 is natural cubic interpolation.
///
/// ## Returns
///
/// Natural cubic Sci spline in sample units. lambda has units coordinate-units³ when weights are dimensionless.
///
/// ## Errors
///
/// Rejects invalid shapes, domains, nonfinite inputs/arithmetic, work limits, or backend failure as applicable. No NaN/infinity inputs are accepted except explicitly allowed fill values.
///
/// ## Numerical behavior
///
/// Private nalgebra LU solves a constrained dense system. Two-point Gauss quadrature integrates the roughness penalty exactly up to roundoff. Work is bounded by the sample limit; ill-conditioning can cause finite inaccuracies.
///
/// # Examples
///
/// ```
/// let s=sci_interpolate::make_smoothing_spline(&[0.,1.,2.,3.,4.], &[0.,1.,2.,3.,4.],None,1.)?;
/// assert_eq!(s.degree,3);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy make_smoothing_spline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_smoothing_spline.html).
/// Carl de Boor, *A Practical Guide to Splines* (2001); Paul Dierckx, *Curve and Surface Fitting with Splines* (1993).
pub fn make_smoothing_spline(
    x: &[f64],
    y: &[f64],
    weights: Option<&[f64]>,
    lambda: f64,
) -> Result<Spline> {
    let w = backend::data(x, y, 3, weights)?;
    if x.len() < 5 || !lambda.is_finite() || lambda < 0.0 {
        return Err(Error::InvalidInput(
            "natural smoothing requires 5..=128 samples and finite lambda >= 0",
        ));
    }
    let mut t = vec![x[0]; 4];
    t.extend_from_slice(&x[1..x.len() - 1]);
    t.extend(std::iter::repeat_n(x[x.len() - 1], 4));
    let n = t.len() - 4;
    let a = backend::design(x, &t, 3);
    let mut second = Vec::new();
    for j in 0..n {
        let mut c = vec![0.0; n];
        c[j] = 1.0;
        second.push(crate::splder(
            &Spline {
                knots: t.clone(),
                coefficients: c,
                degree: 3,
            },
            2,
        )?);
    }
    let mut penalty = DMatrix::<f64>::zeros(n, n);
    // Two-point Gauss quadrature integrates products of piecewise linear second derivatives exactly.
    for p in x.windows(2) {
        let half = 0.5 * (p[1] - p[0]);
        let mid = 0.5 * p[0] + 0.5 * p[1];
        for sign in [-1.0, 1.0] {
            let z = mid + sign * half / 3.0_f64.sqrt();
            let d = second
                .iter()
                .map(|s| backend::evaluate(s, z))
                .collect::<Result<Vec<_>>>()?;
            for i in 0..n {
                for j in 0..n {
                    penalty[(i, j)] += half * d[i] * d[j];
                }
            }
        }
    }
    let mut matrix = DMatrix::zeros(n + 2, n + 2);
    let mut rhs = DVector::zeros(n + 2);
    for i in 0..n {
        for r in 0..x.len() {
            rhs[i] += w[r] * a[(r, i)] * y[r];
        }
        for j in 0..n {
            matrix[(i, j)] = (0..x.len())
                .map(|r| w[r] * a[(r, i)] * a[(r, j)])
                .sum::<f64>()
                + lambda * penalty[(i, j)];
        }
        for (j, z) in [x[0], x[x.len() - 1]].into_iter().enumerate() {
            let d = backend::evaluate(&second[i], z)?;
            matrix[(i, n + j)] = d;
            matrix[(n + j, i)] = d;
        }
    }
    finite_slice(matrix.as_slice())?;
    finite_slice(rhs.as_slice())?;
    let c = matrix
        .lu()
        .solve(&rhs)
        .ok_or(Error::NumericalFailure("singular natural smoothing system"))?;
    finite_slice(c.as_slice())?;
    Ok(Spline {
        knots: t,
        coefficients: c.as_slice()[..n].to_vec(),
        degree: 3,
    })
}

#[allow(clippy::too_many_arguments)]
/// Fits a tensor-product surface `S(x,y)=Σ c[i,j] B[i,kx](x) B[j,ky](y)` minimizing `Σ(w[r](S(x[r],y[r])-z[r]))²`.
///
/// ## Parameters and conventions
///
/// `x`, `y`, `z` are matching scattered samples; x_knots and y_knots are complete vectors. Degrees are 1..=5; optional positive weights default to one. Adapts SciPy task=-1 fixed-knot fitting; automatic SURFIT smoothing, task caches, eps and diagnostics are omitted.
///
/// ## Domain and special values
///
/// 1..=128 finite observations within the closed knot rectangle; finite nondecreasing knots, valid support and a full-rank design. At most 512 knots on either axis.
///
/// ## Returns
///
/// SurfaceSpline with flattened coefficients ordered by x basis index first and y index fastest.
///
/// ## Errors
///
/// Rejects invalid shapes, domains, nonfinite inputs/arithmetic, work limits, or backend failure as applicable. No NaN/infinity inputs are accepted except explicitly allowed fill values.
///
/// ## Numerical behavior
///
/// Private SVD rank-checks and solves the tensor collocation system. At most 10,000,000 sample × coefficient-count² operations; clustered knots and incomplete coverage can fail.
///
/// # Examples
///
/// ```
/// let s=sci_interpolate::bisplrep(&[0.,0.,1.,1.], &[0.,1.,0.,1.], &[0.,1.,1.,2.], &[0.,0.,1.,1.], &[0.,0.,1.,1.],1,1,None)?;
/// assert_eq!(s.coefficients.len(),4);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy bisplrep](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.bisplrep.html).
/// Carl de Boor, *A Practical Guide to Splines* (2001); Paul Dierckx, *Curve and Surface Fitting with Splines* (1993).
pub fn bisplrep(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    x_knots: &[f64],
    y_knots: &[f64],
    x_degree: usize,
    y_degree: usize,
    weights: Option<&[f64]>,
) -> Result<SurfaceSpline> {
    if x.len() != y.len() || x.len() != z.len() || weights.is_some_and(|w| w.len() != x.len()) {
        return Err(Error::DimensionMismatch);
    }
    if x.is_empty()
        || x.len() > MAX_FIT
        || !(1..=5).contains(&x_degree)
        || !(1..=5).contains(&y_degree)
        || x_knots.len() > backend::MAX_KNOTS
        || y_knots.len() > backend::MAX_KNOTS
        || x_knots.len() < 2 * (x_degree + 1)
        || y_knots.len() < 2 * (y_degree + 1)
    {
        return Err(Error::InvalidInput(
            "invalid surface fit size, degree or knots",
        ));
    }
    finite_slice(x)?;
    finite_slice(y)?;
    finite_slice(z)?;
    let nx = x_knots.len() - x_degree - 1;
    let ny = y_knots.len() - y_degree - 1;
    backend::bound(&[x.len(), nx, ny, nx, ny])?;
    for (knots, k, n, data) in [(x_knots, x_degree, nx, x), (y_knots, y_degree, ny, y)] {
        backend::validate(&Spline {
            knots: knots.to_vec(),
            degree: k,
            coefficients: vec![0.0; n],
        })?;
        if data.iter().any(|v| *v < knots[k] || *v > knots[n]) {
            return Err(Error::InvalidInput("surface data outside knot rectangle"));
        }
    }
    let w = weights.map_or_else(|| vec![1.0; x.len()], <[f64]>::to_vec);
    finite_slice(&w)?;
    if w.iter().any(|v| *v <= 0.0) {
        return Err(Error::InvalidInput("weights must be positive"));
    }
    let ax = backend::design(x, x_knots, x_degree);
    let ay = backend::design(y, y_knots, y_degree);
    let a = DMatrix::from_fn(x.len(), nx * ny, |r, c| {
        w[r] * ax[(r, c / ny)] * ay[(r, c % ny)]
    });
    let b = DVector::from_iterator(x.len(), z.iter().zip(&w).map(|(z, w)| z * w));
    let coefficients = backend::least_squares(a, b)?;
    Ok(SurfaceSpline {
        x_knots: x_knots.to_vec(),
        y_knots: y_knots.to_vec(),
        coefficients,
        x_degree,
        y_degree,
    })
}
