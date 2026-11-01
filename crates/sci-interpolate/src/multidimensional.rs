use crate::backend;
use crate::error::is_finite_slice;
use crate::{Error, Result, Spline, SurfaceSpline};

/// Interpolation method for scalar regular grids.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridMethod {
    /// Multilinear interpolation with up to 2^dimension corners.
    Linear,
    /// Closest grid site; midpoint ties choose the lower coordinate.
    Nearest,
}

/// Interpolation method for scattered scalar samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScatteredMethod {
    /// Linear interpolation in one dimension or on 2D Delaunay triangles.
    Linear,
    /// Closest sample in up to eight dimensions; input-order ties choose first.
    Nearest,
    /// Not-a-knot cubic interpolation, supported only in one dimension.
    Cubic,
}

/// Evaluates scalar rectilinear data by multilinear interpolation `S(x)=Σ_corners v[c] Π_j weight[j,c]`, or by nearest site.
///
/// ## Parameters and conventions
///
/// `points` contains one coordinate vector per axis. Flat values are row-major with the last axis fastest. Queries are rows. bounds_error rejects outside queries; otherwise Some(fill_value) fills them, while None extrapolates. Nearest midpoint ties select the lower physical coordinate. Higher-order spline methods and batching are omitted.
///
/// ## Domain and special values
///
/// 1..=8 axes, each strictly ascending or descending; singleton axes are constant under extrapolation. At most 1,000,000 values/queries; finite coordinates and data. IEEE special fill values are allowed.
///
/// ## Returns
///
/// One scalar per query in input order. No coordinate-unit conversion is performed.
///
/// ## Errors
///
/// Rejects invalid shapes, domains, nonfinite inputs/arithmetic, work limits, or backend failure as applicable. No NaN/infinity inputs are accepted except explicitly allowed fill values.
///
/// ## Numerical behavior
///
/// Binary interval lookup and at most 2^dimension corners per query. Query × corner × dimension work is limited to 10,000,000. Extrapolation and extreme scales can overflow.
///
/// # Examples
///
/// ```
/// let v=sci_interpolate::multidimensional::interpn(&[vec![0.,1.]], &[0.,2.], &[vec![0.5]],sci_interpolate::multidimensional::GridMethod::Linear,true,None)?;
/// assert_eq!(v,vec![1.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy interpn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html).
/// Carl de Boor, *A Practical Guide to Splines* (2001); Paul Dierckx, *Curve and Surface Fitting with Splines* (1993).
pub fn interpn(
    points: &[Vec<f64>],
    values: &[f64],
    queries: &[Vec<f64>],
    method: GridMethod,
    bounds_error: bool,
    fill_value: Option<f64>,
) -> Result<Vec<f64>> {
    let dim = points.len();
    if !(1..=8).contains(&dim) || values.len() > 1_000_000 || queries.len() > 1_000_000 {
        return Err(Error::InvalidInput(
            "grid requires 1..=8 axes and bounded values/queries",
        ));
    }
    backend::bound(&[queries.len(), 1 << dim, dim])?;
    let mut count = 1usize;
    for axis in points {
        if axis.is_empty() || axis.len() > 1_000_000 {
            return Err(Error::InvalidInput("invalid grid axis size"));
        }
        count = count
            .checked_mul(axis.len())
            .filter(|n| *n <= 1_000_000)
            .ok_or(Error::InvalidInput("grid size limit exceeded"))?;
        is_finite_slice(axis)?;
        if axis.windows(2).any(|p| p[1] == p[0])
            || (axis.len() > 1
                && axis
                    .windows(2)
                    .any(|p| (p[1] > p[0]) != (axis[1] > axis[0])))
        {
            return Err(Error::InvalidInput("grid axes must be strictly monotonic"));
        }
        match axis[axis.len() - 1] - axis[0] {
            value if value.is_finite() => Ok(value),
            _ => Err(Error::NonFiniteEvaluation),
        }?;
    }
    if values.len() != count || queries.iter().any(|q| q.len() != dim) {
        return Err(Error::DimensionMismatch);
    }
    is_finite_slice(values)?;
    for q in queries {
        is_finite_slice(q)?;
    }
    let mut strides = vec![1usize; dim];
    for j in (0..dim - 1).rev() {
        strides[j] = strides[j + 1] * points[j + 1].len();
    }
    queries
        .iter()
        .map(|q| {
            let outside = points
                .iter()
                .zip(q)
                .any(|(a, z)| *z < a[0].min(a[a.len() - 1]) || *z > a[0].max(a[a.len() - 1]));
            if outside {
                if bounds_error {
                    return Err(Error::InvalidInput("query outside grid"));
                }
                if let Some(fill) = fill_value {
                    return Ok(fill);
                }
            }
            let mut indices = [0usize; 8];
            let mut fractions = [0.0; 8];
            for j in 0..dim {
                let a = &points[j];
                if a.len() == 1 {
                    continue;
                }
                let ascending = a[1] > a[0];
                let i = a
                    .partition_point(|v| if ascending { *v <= q[j] } else { *v >= q[j] })
                    .saturating_sub(1)
                    .min(a.len() - 2);
                let u = match (q[j] - a[i]) / (a[i + 1] - a[i]) {
                    value if value.is_finite() => Ok(value),
                    _ => Err(Error::NonFiniteEvaluation),
                }?;
                if method == GridMethod::Nearest {
                    indices[j] = i + usize::from(if ascending { u > 0.5 } else { u >= 0.5 });
                } else {
                    indices[j] = i;
                    fractions[j] = u;
                }
            }
            if method == GridMethod::Nearest {
                let index = (0..dim).map(|j| indices[j] * strides[j]).sum::<usize>();
                return Ok(values[index]);
            }
            let mut sum = 0.0;
            for corner in 0..1 << dim {
                let mut weight = 1.0;
                let mut index = 0;
                for j in 0..dim {
                    let upper = (corner >> j) & 1;
                    if points[j].len() == 1 && upper == 1 {
                        weight = 0.0;
                        break;
                    }
                    weight *= if upper == 1 {
                        fractions[j]
                    } else {
                        1.0 - fractions[j]
                    };
                    index += (indices[j] + upper) * strides[j];
                }
                if weight != 0.0 {
                    sum = match sum + weight * values[index] {
                        value if value.is_finite() => Ok(value),
                        _ => Err(Error::NonFiniteEvaluation),
                    }?;
                }
            }
            Ok(sum)
        })
        .collect()
}

/// Interpolates scattered scalar data: nearest samples, piecewise affine `S=Σ barycentric[i] value[i]`, or a one-dimensional cubic spline.
///
/// ## Parameters and conventions
///
/// `points` and queries are sample-major rows. Linear supports 1D/2D; cubic supports only 1D; nearest supports 1..=8 dimensions. rescale maps each nonconstant coordinate to `[0,1]`, also changing nearest distances. fill_value is used outside the convex hull for linear/cubic; nearest ignores it.
///
/// ## Domain and special values
///
/// Finite coordinates and values; at most 10,000 samples and 1,000,000 queries. One-dimensional linear/cubic fitting is limited to 128 nodes. Linear 2D rejects duplicate/collinear sites. IEEE fill values are allowed. 2D Clough–Tocher cubic and higher-dimensional simplices are omitted.
///
/// ## Returns
///
/// One value per query, in sample units and input order. Nearest ties select the first input sample in multiple dimensions; in 1D midpoint ties select the lower coordinate, matching SciPy.
///
/// ## Errors
///
/// Rejects invalid shapes, domains, nonfinite inputs/arithmetic, work limits, or backend failure as applicable. No NaN/infinity inputs are accepted except explicitly allowed fill values.
///
/// ## Numerical behavior
///
/// Private Delaunator triangulation and bounded triangle scanning provide 2D linear interpolation. Barycentric membership uses 64ε tolerance. Point × query × dimension and triangle-index × query work each have a 10,000,000 limit. Cocircular triangulations can differ from SciPy.
///
/// # Examples
///
/// ```
/// let v=sci_interpolate::multidimensional::griddata(&[vec![0.],vec![1.]], &[0.,2.], &[vec![0.5]],sci_interpolate::multidimensional::ScatteredMethod::Linear,f64::NAN,false)?;
/// assert_eq!(v,vec![1.]);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy griddata](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html).
/// Carl de Boor, *A Practical Guide to Splines* (2001); Paul Dierckx, *Curve and Surface Fitting with Splines* (1993).
pub fn griddata(
    points: &[Vec<f64>],
    values: &[f64],
    queries: &[Vec<f64>],
    method: ScatteredMethod,
    fill_value: f64,
    rescale: bool,
) -> Result<Vec<f64>> {
    if points.is_empty()
        || points.len() > 10_000
        || queries.len() > 1_000_000
        || values.len() != points.len()
    {
        return Err(Error::InvalidInput("invalid scattered data size"));
    }
    let dim = points[0].len();
    if !(1..=8).contains(&dim) || points.iter().chain(queries).any(|p| p.len() != dim) {
        return Err(Error::DimensionMismatch);
    }
    backend::bound(&[points.len(), queries.len(), dim])?;
    is_finite_slice(values)?;
    for p in points.iter().chain(queries) {
        is_finite_slice(p)?;
    }
    if method == ScatteredMethod::Cubic && dim != 1 || method == ScatteredMethod::Linear && dim > 2
    {
        return Err(Error::InvalidInput(
            "linear scattered interpolation supports 1D/2D; cubic supports 1D",
        ));
    }
    // Shift before triangulation; optional rescaling also changes the Euclidean metric.
    let mut lo = vec![f64::INFINITY; dim];
    let mut hi = vec![f64::NEG_INFINITY; dim];
    for p in points {
        for j in 0..dim {
            lo[j] = lo[j].min(p[j]);
            hi[j] = hi[j].max(p[j]);
        }
    }
    let widths = lo
        .iter()
        .zip(&hi)
        .map(|(a, b)| match b - a {
            value if value.is_finite() => Ok(value),
            _ => Err(Error::NonFiniteEvaluation),
        })
        .collect::<Result<Vec<_>>>()?;
    let transform = |p: &Vec<f64>| -> Result<Vec<f64>> {
        (0..dim)
            .map(|j| {
                match match p[j] - lo[j] {
                    value if value.is_finite() => Ok(value),
                    _ => Err(Error::NonFiniteEvaluation),
                }? / if rescale && widths[j] != 0.0 {
                    widths[j]
                } else {
                    1.0
                } {
                    value if value.is_finite() => Ok(value),
                    _ => Err(Error::NonFiniteEvaluation),
                }
            })
            .collect()
    };
    let p = points.iter().map(transform).collect::<Result<Vec<_>>>()?;
    let q = queries.iter().map(transform).collect::<Result<Vec<_>>>()?;
    if method == ScatteredMethod::Nearest {
        if dim == 1 {
            let mut order: Vec<_> = (0..p.len()).collect();
            order.sort_by(|a, b| p[*a][0].total_cmp(&p[*b][0]));
            let midpoints: Vec<_> = order
                .windows(2)
                .map(|i| 0.5 * p[i[0]][0] + 0.5 * p[i[1]][0])
                .collect();
            return Ok(q
                .iter()
                .map(|q| values[order[midpoints.partition_point(|mid| *mid < q[0])]])
                .collect());
        }
        return q
            .iter()
            .map(|q| {
                let mut index = 0;
                let mut distance = f64::INFINITY;
                for (i, p) in p.iter().enumerate() {
                    let d = p.iter().zip(q).try_fold(0.0_f64, |d, (a, b)| {
                        match d.hypot(match a - b {
                            value if value.is_finite() => Ok(value),
                            _ => Err(Error::NonFiniteEvaluation),
                        }?) {
                            value if value.is_finite() => Ok(value),
                            _ => Err(Error::NonFiniteEvaluation),
                        }
                    })?;
                    if d < distance {
                        distance = d;
                        index = i;
                    }
                }
                Ok(values[index])
            })
            .collect();
    }
    if dim == 1 {
        if points.len() > backend::MAX_FIT {
            return Err(Error::InvalidInput(
                "1D scattered interpolation is limited to 128 nodes",
            ));
        }
        let mut order: Vec<_> = (0..p.len()).collect();
        order.sort_by(|a, b| p[*a][0].total_cmp(&p[*b][0]));
        let x: Vec<_> = order.iter().map(|i| p[*i][0]).collect();
        let y: Vec<_> = order.iter().map(|i| values[*i]).collect();
        let s = crate::fitting::make_interp_spline(
            &x,
            &y,
            if method == ScatteredMethod::Cubic {
                3
            } else {
                1
            },
        )?;
        return q
            .iter()
            .map(|q| {
                if q[0] < x[0] || q[0] > x[x.len() - 1] {
                    Ok(fill_value)
                } else {
                    backend::evaluate(&s, q[0])
                }
            })
            .collect();
    }
    // Established private 2D Delaunay backend; reject ambiguous duplicate observations.
    let mut order: Vec<_> = (0..p.len()).collect();
    order.sort_by(|a, b| {
        p[*a][0]
            .total_cmp(&p[*b][0])
            .then(p[*a][1].total_cmp(&p[*b][1]))
    });
    if order.windows(2).any(|i| p[i[0]] == p[i[1]]) {
        return Err(Error::InvalidInput("duplicate scattered sites"));
    }
    let sites: Vec<_> = p
        .iter()
        .map(|p| delaunator::Point { x: p[0], y: p[1] })
        .collect();
    let triangulation = delaunator::triangulate(&sites);
    if triangulation.triangles.is_empty() {
        return Err(Error::InvalidInput("sites do not span a 2D triangulation"));
    }
    backend::bound(&[triangulation.triangles.len(), q.len()])?;
    q.iter()
        .map(|q| {
            for triangle in triangulation.triangles.as_chunks::<3>().0 {
                let a = &p[triangle[0]];
                let b = &p[triangle[1]];
                let c = &p[triangle[2]];
                let scale = (b[0] - a[0])
                    .abs()
                    .max((b[1] - a[1]).abs())
                    .max((c[0] - a[0]).abs())
                    .max((c[1] - a[1]).abs());
                let bx = (b[0] - a[0]) / scale;
                let by = (b[1] - a[1]) / scale;
                let cx = (c[0] - a[0]) / scale;
                let cy = (c[1] - a[1]) / scale;
                let qx = (q[0] - a[0]) / scale;
                let qy = (q[1] - a[1]) / scale;
                let determinant = bx * cy - by * cx;
                if determinant == 0.0 || !determinant.is_finite() {
                    return Err(Error::NumericalFailure("degenerate triangle"));
                }
                let v = (qx * cy - qy * cx) / determinant;
                let w = (bx * qy - by * qx) / determinant;
                let u = 1.0 - v - w;
                let tolerance = 64.0 * f64::EPSILON;
                if u >= -tolerance && v >= -tolerance && w >= -tolerance {
                    return match u * values[triangle[0]]
                        + v * values[triangle[1]]
                        + w * values[triangle[2]]
                    {
                        value if value.is_finite() => Ok(value),
                        _ => Err(Error::NonFiniteEvaluation),
                    };
                }
            }
            Ok(fill_value)
        })
        .collect()
}

/// Evaluates tensor spline partial derivatives `∂^(dx+dy) S / ∂x^dx ∂y^dy` on a Cartesian query product.
///
/// ## Parameters and conventions
///
/// `x` and `y` are nondecreasing query vectors; dx and dy are derivative orders. Both coordinates clamp to the knot rectangle, matching FITPACK evaluation. Output always retains both axes, replacing Python scalar/vector squeezing.
///
/// ## Domain and special values
///
/// Finite valid surface knots and coefficients; degrees 0..=5, derivative orders <=degree. Repeated knots that prevent differentiation fail. Empty axes produce an empty matrix or empty rows.
///
/// ## Returns
///
/// Matrix with x.len() rows and y.len() columns. Units are sample-units / x-units^dx / y-units^dy.
///
/// ## Errors
///
/// Rejects invalid shapes, domains, nonfinite inputs/arithmetic, work limits, or backend failure as applicable. No NaN/infinity inputs are accepted except explicitly allowed fill values.
///
/// ## Numerical behavior
///
/// Reuses scalar derivative and local evaluation kernels. The product of query-axis lengths (each at least one for work accounting) and coefficient counts is bounded by 10,000,000.
///
/// # Examples
///
/// ```
/// let s=sci_interpolate::fitting::bisplrep(&[0.,0.,1.,1.], &[0.,1.,0.,1.], &[0.,1.,1.,2.], &[0.,0.,1.,1.], &[0.,0.,1.,1.],1,1,None)?;
/// let v=sci_interpolate::multidimensional::bisplev(&[0.5],&[0.5],&s,0,0)?;
/// assert!((v[0][0]-1.).abs()<1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy bisplev](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.bisplev.html).
/// Carl de Boor, *A Practical Guide to Splines* (2001); Paul Dierckx, *Curve and Surface Fitting with Splines* (1993).
pub fn bisplev(
    x: &[f64],
    y: &[f64],
    spline: &SurfaceSpline,
    dx: usize,
    dy: usize,
) -> Result<Vec<Vec<f64>>> {
    let kx = spline.x_degree;
    let ky = spline.y_degree;
    if kx > 5
        || ky > 5
        || spline.x_knots.len() > backend::MAX_KNOTS
        || spline.y_knots.len() > backend::MAX_KNOTS
        || spline.x_knots.len() < 2 * (kx + 1)
        || spline.y_knots.len() < 2 * (ky + 1)
        || dx > kx
        || dy > ky
    {
        return Err(Error::InvalidInput(
            "invalid surface spline shape or derivative order",
        ));
    }
    let nx = spline.x_knots.len() - kx - 1;
    let ny = spline.y_knots.len() - ky - 1;
    if spline.coefficients.len() != nx * ny {
        return Err(Error::DimensionMismatch);
    }
    backend::bound(&[x.len().max(1), y.len().max(1), nx, ny])?;
    is_finite_slice(x)?;
    is_finite_slice(y)?;
    is_finite_slice(&spline.coefficients)?;
    if x.windows(2).any(|p| p[1] < p[0]) || y.windows(2).any(|p| p[1] < p[0]) {
        return Err(Error::InvalidInput("surface queries must be nondecreasing"));
    }
    let mut sx = Spline {
        knots: spline.x_knots.clone(),
        coefficients: vec![0.0; nx],
        degree: kx,
    };
    let mut sy = Spline {
        knots: spline.y_knots.clone(),
        coefficients: vec![0.0; ny],
        degree: ky,
    };
    backend::validate(&sx)?;
    backend::validate(&sy)?;
    let mut along_x = Vec::with_capacity(ny);
    for j in 0..ny {
        for (i, c) in sx.coefficients.iter_mut().enumerate() {
            *c = spline.coefficients[i * ny + j];
        }
        along_x.push(crate::spline::splev(
            x,
            &sx,
            dx,
            crate::spline::SplineExtension::Clamp,
        )?);
    }
    (0..x.len())
        .map(|i| {
            for (c, row) in sy.coefficients.iter_mut().zip(&along_x) {
                *c = row[i];
            }
            crate::spline::splev(y, &sy, dy, crate::spline::SplineExtension::Clamp)
        })
        .collect()
}
