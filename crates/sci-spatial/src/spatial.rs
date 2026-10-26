use super::{MAX_ENTRIES, MAX_WORK, bound, evaluated, pair, points};
use crate::{Error, Result, error::is_finite_slice};
use nalgebra::{DMatrix, DVector};

/// Computes the Lp distance `(Σ|xᵢ-yᵢ|ᵖ)^(1/p)`.
///
/// ## Parameters
///
/// Equal-length vectors, with explicit p>0; positive infinity selects L∞.
/// This is SciPy's single-vector adaptation without broadcasting.
///
/// ## Domain
///
/// Coordinates must be finite. Empty vectors return zero; p<1 gives a quasinorm.
///
/// ## Returns
///
/// A distance in coordinate units.
///
/// ## Errors
///
/// Rejects shapes, invalid p, overflow, or over 1,000,000 coordinates.
///
/// ## Numerical behavior
///
/// Reuses the scaled [`super::minkowski`] kernel.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_spatial::minkowski_distance(&[0.0], &[2.0], 2.0)?, 2.0);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.minkowski_distance.html).
pub fn minkowski_distance(x: &[f64], y: &[f64], p: f64) -> Result<f64> {
    super::minkowski(x, y, p, None)
}

/// Computes `Σ|xᵢ-yᵢ|ᵖ`, except p=∞ returns `max|xᵢ-yᵢ|`.
///
/// ## Parameters
///
/// Equal-length vectors; p>0 or positive infinity. Broadcasting is unsupported.
///
/// ## Domain
///
/// Coordinates must be finite. Empty vectors return zero.
///
/// ## Returns
///
/// The pth power of the distance, in coordinate-units^p (coordinate units for infinity).
///
/// ## Errors
///
/// Rejects shapes, invalid p, overflow, or over 1,000,000 coordinates.
///
/// ## Numerical behavior
///
/// Directly accumulates powered differences; powers may underflow or overflow.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_spatial::minkowski_distance_p(&[0.0], &[2.0], 2.0)?, 4.0);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.minkowski_distance_p.html).
pub fn minkowski_distance_p(x: &[f64], y: &[f64], p: f64) -> Result<f64> {
    pair(x, y)?;
    if p.is_nan() || p <= 0.0 {
        return Err(Error::InvalidInput("p must be positive"));
    }
    if p == f64::INFINITY {
        return minkowski_distance(x, y, p);
    }
    let mut total = 0.0;
    for (&a, &b) in x.iter().zip(y) {
        let difference = (a - b).abs();
        if !difference.is_finite() {
            return Err(Error::NonFiniteEvaluation);
        }
        total += difference.powf(p);
    }
    evaluated(total)
}

/// Computes the cross-point matrix `D[i][j]=||x[i]-y[j]||p`.
///
/// ## Parameters
///
/// Row-major observations with equal coordinate counts and explicit p>0 or infinity.
/// SciPy's threshold argument is omitted; work and allocation have fixed limits.
///
/// ## Domain
///
/// Coordinates must be finite; an empty collection yields empty rows or an empty matrix.
///
/// ## Returns
///
/// A matrix with x.len() rows and y.len() columns in coordinate units.
///
/// ## Errors
///
/// Rejects shapes, domains, overflow, over 1,000,000 output entries or
/// 10,000,000 pair-coordinate operations.
///
/// ## Numerical behavior
///
/// Reuses [`super::cdist`] with a scaled Lp kernel; p<1 gives a quasinorm.
///
/// # Examples
///
/// ```
/// assert_eq!(sci_spatial::distance_matrix(&[vec![0.0]], &[vec![2.0]], 2.0)?, vec![vec![2.0]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance_matrix.html).
pub fn distance_matrix(x: &[Vec<f64>], y: &[Vec<f64>], p: f64) -> Result<Vec<Vec<f64>>> {
    super::cdist(x, y, super::Metric::Minkowski(p))
}

/// Interpolates a unit great circle `γ(t)=cos(tθ)a+sin(tθ)b`, b⊥a.
///
/// ## Parameters
///
/// `start` and `end` are unit vectors of equal dimension ≥2; `t` contains
/// dimensionless interpolation parameters in any order. Extrapolation is allowed.
/// `tol` is a finite absolute chord-distance tolerance in [0,1) for rejecting
/// antipodes: `| ||start-end||₂ - 2 | <= tol`. Unlike SciPy's warning and
/// arbitrary path, ambiguous antipodal inputs return an error.
///
/// ## Domain
///
/// Finite values only; vector norms must be within 1e-9 of one. Equal unit
/// endpoints return repeated start vectors. Empty t returns an empty matrix.
///
/// ## Returns
///
/// One interpolated vector per t, in original parameter order, on the unit sphere.
///
/// ## Errors
///
/// Rejects shapes, nonunit endpoints, ambiguous antipodes, invalid tolerance,
/// nonfinite angular evaluation, or more than 1,000,000 output entries.
///
/// ## Numerical behavior
///
/// Uses an orthogonal residual and atan2 to avoid division by sin(θ) near zero.
/// Near-antipodal directions are ill-conditioned. Very large extrapolation
/// parameters lose phase accuracy. Endpoints t=0 and t=1 are copied exactly.
///
/// # Examples
///
/// ```
/// let p = sci_spatial::geometric_slerp(&[1.0,0.0], &[0.0,1.0], &[0.5], 1e-7)?;
/// assert!((p[0][0]-0.5_f64.sqrt()).abs() < 1e-14);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.geometric_slerp.html).
pub fn geometric_slerp(start: &[f64], end: &[f64], t: &[f64], tol: f64) -> Result<Vec<Vec<f64>>> {
    pair(start, end)?;
    is_finite_slice(t)?;
    bound(t.len().checked_mul(start.len()), MAX_ENTRIES)?;
    if start.len() < 2 || !tol.is_finite() || !(0.0..1.0).contains(&tol) {
        return Err(Error::InvalidInput(
            "slerp requires dimension >= 2 and 0 <= tolerance < 1",
        ));
    }
    let norm = |x: &[f64]| x.iter().fold(0.0_f64, |n, &v| n.hypot(v));
    let (ns, ne) = (norm(start), norm(end));
    if (ns - 1.0).abs() > 1e-9 || (ne - 1.0).abs() > 1e-9 {
        return Err(Error::InvalidInput("slerp endpoints must have unit norm"));
    }
    if start == end {
        return Ok(vec![start.to_vec(); t.len()]);
    }
    if (super::euclidean(start, end, None)? - 2.0).abs() <= tol {
        return Err(Error::InvalidInput("antipodal interpolation is ambiguous"));
    }
    let a: Vec<_> = start.iter().map(|x| x / ns).collect();
    let e: Vec<_> = end.iter().map(|x| x / ne).collect();
    let dot = a
        .iter()
        .zip(&e)
        .map(|(&x, &y)| x * y)
        .sum::<f64>()
        .clamp(-1.0, 1.0);
    let mut b: Vec<_> = e.iter().zip(&a).map(|(&y, &x)| y - dot * x).collect();
    // Remove residual parallel roundoff before normalizing the tangent.
    let parallel = b.iter().zip(&a).map(|(&y, &x)| x * y).sum::<f64>();
    for (y, &x) in b.iter_mut().zip(&a) {
        *y -= parallel * x;
    }
    let nb = norm(&b);
    if nb == 0.0 {
        return Err(Error::NumericalFailure(
            "cannot resolve interpolation direction",
        ));
    }
    let theta = nb.atan2(dot);
    for y in &mut b {
        *y /= nb;
    }
    t.iter()
        .map(|&t| {
            if t == 0.0 {
                return Ok(start.to_vec());
            }
            if t == 1.0 {
                return Ok(end.to_vec());
            }
            let phase = t * theta;
            if !phase.is_finite() {
                return Err(Error::NonFiniteEvaluation);
            }
            let (s, c) = phase.sin_cos();
            Ok(a.iter().zip(&b).map(|(&a, &b)| c * a + s * b).collect())
        })
        .collect()
}

/// The standardized point sets and squared residual from [`procrustes`].
#[derive(Clone, Debug, PartialEq)]
pub struct ProcrustesResult {
    /// Centered reference, with unit Frobenius norm.
    pub reference: Vec<Vec<f64>>,
    /// Centered second set after optimal rotation/reflection and dilation.
    pub transformed: Vec<Vec<f64>>,
    /// Sum of squared differences of the output point sets.
    pub disparity: f64,
}

/// Fits centered, normalized B to A by minimizing `M²=||A-sBR||F²`.
///
/// ## Parameters
///
/// `data1`, `data2` are identically shaped row-major point sets, n×d.
/// Each is centered and normalized first. R may rotate or reflect; s is the
/// optimal nonnegative dilation. Point correspondence is row by row.
///
/// ## Domain
///
/// Nonempty finite inputs with at least two distinct points per set are required.
/// Duplicate points are allowed and retain their multiplicity as fitting weight.
///
/// ## Returns
///
/// Standardized data1, fitted data2, and squared dimensionless disparity, in
/// SciPy's output order represented by named fields. Only reference has unit norm.
///
/// ## Errors
///
/// Rejects shapes/domains/overflow, dimension >64, more than 1,000,000 entries,
/// or work `nd²+d³ > 10,000,000`; bounded private SVD failure is an error.
///
/// ## Numerical behavior
///
/// Uses scaled centering and private nalgebra SVD, capped at 10,000 iterations.
/// Rank-deficient alignments are allowed but optimal rotation is not unique.
/// Extremely clustered coordinates can lose information during centering.
///
/// # Examples
///
/// ```
/// let a = vec![vec![0.0,0.0],vec![1.0,0.0],vec![0.0,1.0]];
/// let b = vec![vec![2.0,3.0],vec![2.0,5.0],vec![4.0,3.0]];
/// assert!(sci_spatial::procrustes(&a,&b)?.disparity < 1e-25);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.procrustes.html);
/// Gower (1975), Generalized Procrustes Analysis, Psychometrika 40, 33–51.
pub fn procrustes(data1: &[Vec<f64>], data2: &[Vec<f64>]) -> Result<ProcrustesResult> {
    let (d, e) = (points(data1)?, points(data2)?);
    let n = data1.len();
    if d != e || n != data2.len() {
        return Err(Error::DimensionMismatch);
    }
    if n == 0 || d == 0 {
        return Err(Error::InvalidInput("point sets must be nonempty"));
    }
    bound(Some(d), 64)?;
    bound(
        n.checked_mul(d)
            .and_then(|x| x.checked_mul(d))
            .and_then(|x| x.checked_add(d * d * d)),
        MAX_WORK,
    )?;
    let normalize = |data: &[Vec<f64>]| -> Result<DMatrix<f64>> {
        let scale = data
            .iter()
            .flatten()
            .copied()
            .map(f64::abs)
            .fold(0.0, f64::max);
        if scale == 0.0 {
            return Err(Error::InvalidInput(
                "point set must contain distinct points",
            ));
        }
        let mut a = DMatrix::from_fn(n, d, |i, j| data[i][j] / scale);
        for j in 0..d {
            let mean = (0..n).map(|i| a[(i, j)] / n as f64).sum::<f64>();
            for i in 0..n {
                a[(i, j)] -= mean;
            }
        }
        let norm = a.iter().fold(0.0_f64, |n, &x| n.hypot(x));
        if norm == 0.0 || data.iter().all(|row| row == &data[0]) {
            return Err(Error::InvalidInput(
                "point set must contain distinct points",
            ));
        }
        Ok(a / norm)
    };
    let (a, b) = (normalize(data1)?, normalize(data2)?);
    let cross = b.transpose() * &a;
    let svd = nalgebra::linalg::SVD::try_new(cross, true, true, f64::EPSILON, 10_000)
        .ok_or(Error::NumericalFailure("Procrustes SVD did not converge"))?;
    let scale = svd.singular_values.iter().sum::<f64>();
    let rotation = svd.u.unwrap() * svd.v_t.unwrap();
    let fitted = b * rotation * scale;
    let disparity = (&a - &fitted).norm_squared();
    if !disparity.is_finite() {
        return Err(Error::NonFiniteEvaluation);
    }
    let rows = |m: &DMatrix<f64>| {
        (0..n)
            .map(|i| (0..d).map(|j| m[(i, j)]).collect())
            .collect()
    };
    Ok(ProcrustesResult {
        reference: rows(&a),
        transformed: rows(&fitted),
        disparity,
    })
}

/// Finds containing simplices by solving `q = v₀ + Σ λⱼ(vⱼ-v₀)`.
///
/// ## Parameters
///
/// `vertices` contains finite d-dimensional points; each `simplices` row has
/// d+1 vertex indices. `queries` has one point per row. `tol` is a finite
/// nonnegative absolute barycentric tolerance. SciPy's Delaunay object is replaced
/// by explicit simplices; triangulation construction is outside this function.
///
/// ## Domain
///
/// Dimension is 1..=64; all simplices must be nondegenerate. Empty simplex or
/// query lists are allowed. Coordinates are finite; simplex indices must exist.
///
/// ## Returns
///
/// One `Option<usize>` per query: first containing simplex index, or None outside
/// (SciPy uses -1). Boundary ties follow supplied simplex order.
///
/// ## Errors
///
/// Rejects shapes, indices, invalid tolerance, singular simplices, overflow,
/// input collections exceeding 1,000,000 entries, or work
/// `s d³ + q s d² > 10,000,000`, with dimension at least one in work counts.
///
/// ## Numerical behavior
///
/// Uses private nalgebra LU inverses and exhaustive barycentric tests. Nearly
/// singular simplices and distant coordinates can lose accuracy. Supplied cells
/// need not be Delaunay and are not checked for overlap or tessellation validity.
///
/// # Examples
///
/// ```
/// let v = vec![vec![0.0,0.0],vec![1.0,0.0],vec![0.0,1.0]];
/// assert_eq!(sci_spatial::tsearch(&v,&[vec![0,1,2]],&[vec![0.2,0.2]],1e-12)?,vec![Some(0)]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.tsearch.html).
pub fn tsearch(
    vertices: &[Vec<f64>],
    simplices: &[Vec<usize>],
    queries: &[Vec<f64>],
    tol: f64,
) -> Result<Vec<Option<usize>>> {
    let (d, e) = (points(vertices)?, points(queries)?);
    if !queries.is_empty() && d != e {
        return Err(Error::DimensionMismatch);
    }
    if d == 0 || d > 64 || !tol.is_finite() || tol < 0.0 {
        return Err(Error::InvalidInput(
            "invalid simplex dimension or tolerance",
        ));
    }
    bound(simplices.len().checked_mul(d + 1), MAX_ENTRIES)?;
    bound(
        simplices
            .len()
            .checked_mul(d * d)
            .and_then(|x| x.checked_mul(d.checked_add(queries.len())?)),
        MAX_WORK,
    )?;
    let mut cells = Vec::with_capacity(simplices.len());
    for cell in simplices {
        if cell.len() != d + 1 {
            return Err(Error::DimensionMismatch);
        }
        if cell.iter().any(|&i| i >= vertices.len()) {
            return Err(Error::InvalidInput("simplex vertex index out of bounds"));
        }
        let origin = &vertices[cell[0]];
        let a = DMatrix::from_fn(d, d, |i, j| vertices[cell[j + 1]][i] - origin[i]);
        if a.iter().any(|x| !x.is_finite()) {
            return Err(Error::NonFiniteEvaluation);
        }
        let inverse = a
            .lu()
            .try_inverse()
            .ok_or(Error::NumericalFailure("degenerate simplex"))?;
        if inverse.iter().any(|x| !x.is_finite()) {
            return Err(Error::NonFiniteEvaluation);
        }
        cells.push((origin, inverse));
    }
    queries
        .iter()
        .map(|q| {
            for (i, (origin, inverse)) in cells.iter().enumerate() {
                let delta = DVector::from_iterator(d, q.iter().zip(*origin).map(|(&x, &a)| x - a));
                let bary = inverse * delta;
                if bary.iter().any(|x| !x.is_finite()) {
                    return Err(Error::NonFiniteEvaluation);
                }
                let last = 1.0 - bary.iter().sum::<f64>();
                if !last.is_finite() {
                    return Err(Error::NonFiniteEvaluation);
                }
                if last >= -tol
                    && last <= 1.0 + tol
                    && bary.iter().all(|&x| x >= -tol && x <= 1.0 + tol)
                {
                    return Ok(Some(i));
                }
            }
            Ok(None)
        })
        .collect()
}
