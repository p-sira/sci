//! Private numerical adapters; backend types never cross the public API.
use crate::error::{finite, finite_slice};
use crate::{Error, Result, Spline};
use nalgebra::{DMatrix, DVector};

pub(crate) const MAX_FIT: usize = 128;
pub(crate) const MAX_KNOTS: usize = 512;
pub(crate) const MAX_DEGREE: usize = 32;
pub(crate) const MAX_WORK: usize = 10_000_000;

pub(crate) fn bound(factors: &[usize]) -> Result<()> {
    let work = factors.iter().try_fold(1usize, |a, b| a.checked_mul(*b));
    if work.is_none_or(|n| n > MAX_WORK) {
        return Err(Error::InvalidInput("interpolation work limit exceeded"));
    }
    Ok(())
}

pub(crate) fn validate(s: &Spline) -> Result<()> {
    let k = s.degree;
    let n = s.coefficients.len();
    if k > MAX_DEGREE || s.knots.len() > MAX_KNOTS || n <= k || s.knots.len() != n + k + 1 {
        return Err(Error::InvalidInput("invalid spline shape or degree"));
    }
    finite_slice(&s.knots)?;
    finite_slice(&s.coefficients)?;
    if s.knots.windows(2).any(|p| p[1] < p[0])
        || s.knots[k] >= s.knots[n]
        || (0..n).any(|i| s.knots[i] >= s.knots[i + k + 1])
    {
        return Err(Error::InvalidInput("invalid spline knot support"));
    }
    finite(s.knots[s.knots.len() - 1] - s.knots[0])?;
    Ok(())
}

pub(crate) fn data(x: &[f64], y: &[f64], k: usize, w: Option<&[f64]>) -> Result<Vec<f64>> {
    if k > 5 || x.len() <= k || x.len() > MAX_FIT || x.len() < 2 {
        return Err(Error::InvalidInput(
            "fit requires degree 0..=5 and 2..=128 samples, more than degree",
        ));
    }
    if x.len() != y.len() || w.is_some_and(|w| w.len() != x.len()) {
        return Err(Error::DimensionMismatch);
    }
    finite_slice(x)?;
    finite_slice(y)?;
    if x.windows(2).any(|p| p[1] <= p[0]) {
        return Err(Error::InvalidInput(
            "sample coordinates must strictly increase",
        ));
    }
    finite(x[x.len() - 1] - x[0])?;
    let weights = w.map_or_else(|| vec![1.0; x.len()], <[f64]>::to_vec);
    finite_slice(&weights)?;
    if weights.iter().any(|w| *w <= 0.0) {
        return Err(Error::InvalidInput("weights must be positive"));
    }
    Ok(weights)
}

pub(crate) fn knots(x: &[f64], k: usize) -> Vec<f64> {
    if k == 0 {
        let mut t = x.to_vec();
        t.push(x[x.len() - 1]);
        return t;
    }
    if k == 1 {
        let mut t = vec![x[0]];
        t.extend_from_slice(x);
        t.push(x[x.len() - 1]);
        return t;
    }
    let sites: Vec<_> = if k % 2 == 1 {
        x.to_vec()
    } else {
        x.windows(2).map(|p| 0.5 * p[0] + 0.5 * p[1]).collect()
    };
    let trim = k.div_ceil(2);
    let mut t = vec![x[0]; k + 1];
    t.extend_from_slice(&sites[trim..sites.len() - trim]);
    t.extend(std::iter::repeat_n(x[x.len() - 1], k + 1));
    t
}

// Cox–de Boor basis recursion on one nonempty span (also extrapolates it).
pub(crate) fn basis(t: &[f64], k: usize, x: f64) -> (usize, [f64; 33]) {
    let n = t.len() - k - 1;
    let span = if x <= t[k] {
        t.partition_point(|v| *v <= t[k])
            .saturating_sub(1)
            .clamp(k, n - 1)
    } else if x >= t[n] {
        t.partition_point(|v| *v < t[n])
            .saturating_sub(1)
            .clamp(k, n - 1)
    } else {
        t.partition_point(|v| *v <= x)
            .saturating_sub(1)
            .clamp(k, n - 1)
    };
    let mut b = [0.0; 33];
    let mut left = [0.0; 33];
    let mut right = [0.0; 33];
    b[0] = 1.0;
    for j in 1..=k {
        left[j] = x - t[span + 1 - j];
        right[j] = t[span + j] - x;
        let mut saved = 0.0;
        for r in 0..j {
            let denominator = right[r + 1] + left[j - r];
            let term = if denominator == 0.0 {
                0.0
            } else {
                b[r] / denominator
            };
            b[r] = saved + right[r + 1] * term;
            saved = left[j - r] * term;
        }
        b[j] = saved;
    }
    (span - k, b)
}

pub(crate) fn evaluate(s: &Spline, x: f64) -> Result<f64> {
    let (start, b) = basis(&s.knots, s.degree, x);
    finite(
        (0..=s.degree)
            .map(|j| b[j] * s.coefficients[start + j])
            .sum(),
    )
}

pub(crate) fn design(x: &[f64], t: &[f64], k: usize) -> DMatrix<f64> {
    let n = t.len() - k - 1;
    let mut a = DMatrix::zeros(x.len(), n);
    for (i, x) in x.iter().enumerate() {
        let (start, b) = basis(t, k, *x);
        for j in 0..=k {
            a[(i, start + j)] = b[j];
        }
    }
    a
}

pub(crate) fn least_squares(a: DMatrix<f64>, b: DVector<f64>) -> Result<Vec<f64>> {
    bound(&[a.nrows(), a.ncols(), a.ncols()])?;
    if a.nrows() < a.ncols() {
        return Err(Error::InvalidInput("underdetermined spline fit"));
    }
    finite_slice(a.as_slice())?;
    finite_slice(b.as_slice())?;
    // SVD avoids squaring the condition number and provides a rank check.
    let svd = nalgebra::linalg::SVD::try_new(a, true, true, f64::EPSILON, 1000).ok_or(
        Error::NumericalFailure("spline SVD iteration limit exceeded"),
    )?;
    let largest = svd.singular_values.iter().copied().fold(0.0, f64::max);
    let tolerance = f64::EPSILON * (b.len().max(svd.singular_values.len()) as f64) * largest;
    if svd.singular_values.iter().any(|v| *v <= tolerance) {
        return Err(Error::NumericalFailure("rank-deficient spline fit"));
    }
    let c = svd
        .solve(&b, tolerance)
        .map_err(|_| Error::NumericalFailure("spline SVD solve failed"))?;
    finite_slice(c.as_slice())?;
    Ok(c.as_slice().to_vec())
}

pub(crate) fn fit<const K: usize>(x: &[f64], y: &[f64], w: &[f64], s: f64) -> Result<Spline> {
    let fitter = splinefit::SplineCurveFit::<K>::new(x.to_vec(), y.to_vec())
        .set_weights(w.to_vec())
        .map_err(|_| Error::NumericalFailure("FITPACK weights failed"))?;
    let result = fitter
        .smoothing_spline((s / x.len() as f64).sqrt())
        .map_err(|_| Error::NumericalFailure("FITPACK fit did not converge"))?;
    let result = Spline {
        knots: result.t,
        coefficients: result.c,
        degree: K,
    };
    validate(&result)?;
    check_residual(
        x,
        &y.iter().map(|v| vec![*v]).collect::<Vec<_>>(),
        w,
        s,
        std::slice::from_ref(&result),
    )?;
    Ok(result)
}

pub(crate) fn curve<const K: usize, const N: usize>(
    u: &[f64],
    points: &[Vec<f64>],
    w: &[f64],
    s: f64,
) -> Result<Vec<Spline>> {
    let fitter = splinefit::ParameterSplineCurveFit::<K, N>::new(
        u.to_vec(),
        points.iter().flatten().copied().collect(),
    )
    .and_then(|f| f.weights(w.to_vec()))
    .map_err(|_| Error::NumericalFailure("FITPACK curve initialization failed"))?;
    let result = fitter
        .smoothing_spline((s / u.len() as f64).sqrt())
        .map_err(|_| Error::NumericalFailure("FITPACK curve fit did not converge"))?;
    if result.t.len() < 2 * (K + 1)
        || result.t.len() > MAX_KNOTS
        || result.c.len() != N * (result.t.len() - K - 1)
    {
        return Err(Error::NumericalFailure(
            "FITPACK returned an invalid curve representation",
        ));
    }
    let n = result.t.len() - K - 1;
    let components: Vec<_> = (0..N)
        .map(|j| {
            let s = Spline {
                knots: result.t.clone(),
                coefficients: result.c[j * n..(j + 1) * n].to_vec(),
                degree: K,
            };
            validate(&s)?;
            Ok(s)
        })
        .collect::<Result<_>>()?;
    check_residual(u, points, w, s, &components)?;
    Ok(components)
}

fn check_residual(
    x: &[f64],
    points: &[Vec<f64>],
    w: &[f64],
    target: f64,
    components: &[Spline],
) -> Result<()> {
    let mut residual = 0.0;
    for ((x, point), w) in x.iter().zip(points).zip(w) {
        for (s, value) in components.iter().zip(point) {
            let r = finite(w * finite(evaluate(s, *x)? - value)?)?;
            residual = finite(residual + finite(r * r)?)?;
        }
    }
    if residual > target && (residual - target) / target > 0.0011 {
        return Err(Error::NumericalFailure(
            "spline fit missed its residual target",
        ));
    }
    Ok(())
}

pub(crate) fn polynomial_roots(coefficients: &[f64]) -> Result<Vec<f64>> {
    let scale = coefficients.iter().map(|v| v.abs()).fold(0.0, f64::max);
    if scale == 0.0 {
        return Ok(Vec::new());
    }
    let mut c: Vec<_> = coefficients.iter().map(|v| v / scale).collect();
    while c.len() > 1 && c[c.len() - 1].abs() <= 64.0 * f64::EPSILON {
        c.pop();
    }
    let n = c.len() - 1;
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        return Ok(vec![finite(-c[0] / c[1])?]);
    }
    let mut companion = DMatrix::zeros(n, n);
    for i in 1..n {
        companion[(i, i - 1)] = 1.0;
    }
    for i in 0..n {
        companion[(i, n - 1)] = -c[i] / c[n];
    }
    let schur = nalgebra::linalg::Schur::try_new(companion, 64.0 * f64::EPSILON, 256).ok_or(
        Error::NumericalFailure("polynomial root iteration limit exceeded"),
    )?;
    let eigenvalues = schur.complex_eigenvalues();
    let mut roots = Vec::new();
    for z in eigenvalues.iter() {
        if z.im.abs() <= 1e-10 * (1.0 + z.re.abs()) {
            let mut root = finite(z.re)?;
            // Refine with the polynomial itself; repeated roots may remain ill-conditioned.
            for _ in 0..4 {
                let mut value = c[n];
                let mut derivative = 0.0;
                for j in (0..n).rev() {
                    derivative = derivative * root + value;
                    value = value * root + c[j];
                }
                if derivative == 0.0 {
                    break;
                }
                let next = root - value / derivative;
                if !next.is_finite() {
                    break;
                }
                root = next;
            }
            roots.push(root);
        }
    }
    Ok(roots)
}
