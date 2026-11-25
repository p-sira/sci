//! Real cosine-sine decomposition through SVD and orthogonal complements.
use super::{Matrix, backend::*};
use crate::{Error, Result};
use faer::{Mat, Scale};
/// Cosine-sine factorization `X = U CS VH`.
#[derive(Clone, Debug)]
pub struct CsResult {
    /// Block diagonal left orthogonal factor, partitions p and n-p.
    pub u: Matrix,
    /// Central cosine-sine factor; see cossin for ordering conventions.
    pub cs: Matrix,
    /// Block diagonal right orthogonal transpose, partitions q and n-q.
    pub vh: Matrix,
    /// Ascending principal angles in radians, length min(p,n-p,q,n-q).
    pub theta: Vec<f64>,
}
fn complete(columns: Vec<Mat<f64>>, n: usize) -> Result<(Mat<f64>, usize)> {
    let k = columns.len();
    if k > n {
        return Err(Error::NumericalFailure("too many CS basis columns"));
    }
    if k == 0 {
        return Ok((Mat::<f64>::identity(n, n), 0));
    }
    let basis = Mat::from_fn(n, k, |i, j| columns[j][(i, 0)]);
    if (basis.transpose() * &basis - Mat::<f64>::identity(k, k)).norm_l2() > 1e-8 {
        return Err(Error::NumericalFailure("CS cross basis lost orthogonality"));
    }
    let complement = if k == n {
        vec![vec![]; n]
    } else {
        super::null_space(&rows(&basis.transpose())?, Some(1e-12))?
    };
    if complement.first().map_or(0, Vec::len) != n - k {
        return Err(Error::NumericalFailure("CS complement has unexpected rank"));
    }
    Ok((
        Mat::from_fn(n, n, |i, j| {
            if j < k {
                basis[(i, j)]
            } else {
                complement[i][j - k]
            }
        }),
        k,
    ))
}
/// Computes `X = U CS VH`, the real cosine-sine decomposition.
///
/// ## Parameters
///
/// `x` is an orthogonal n × n matrix; its top-left block is p × q.
/// U is block diagonal with sizes p,n-p; VH with sizes q,n-q.
/// Central rows/columns use SVD order: coupled sine directions precede
/// uncoupled identity directions in the lower/right blocks. This is a
/// permutation-equivalent CS layout rather than SciPy identity-block ordering.
/// Both factors are always computed; swap_sign and separate options are omitted.
///
/// ## Domain
///
/// Finite orthogonal matrices, order 2..=256, 0<p,q<n.
/// Orthogonality is checked at absolute Frobenius tolerance 1e-10.
/// NaN/infinity are rejected.
///
/// ## Returns
///
/// Owned CsResult with orthogonal U, central CS, VH, and ascending angles
/// in radians. Upper-left cosines are descending. Central cross blocks
/// have paired sine entries; nonunique factors may differ from SciPy.
///
/// ## Errors
///
/// Invalid partitions, nonorthogonal data, failed SVD/complement,
/// loss of orthogonality, or nonfinite output.
///
/// ## Numerical behavior
///
/// Uses private faer full SVD of X11, shared faer SVD null spaces,
/// and SVD of the remaining X22 block. Native assembly completes the
/// block factors; work is O(n³). Cross-column norms below 1e-12 are treated
/// as zero. Angles use atan2(sine,cosine) to retain small-angle accuracy.
/// Near zero/π/2, factor directions are sensitive to perturbations.
///
/// # Examples
///
/// ```
/// let f = sci_linalg::decompositions::cossin(&[vec![0.6,-0.8],vec![0.8,0.6]],1,1)?;
/// assert!((f.theta[0]-0.8_f64.atan2(0.6)).abs()<1e-12);
/// # Ok::<(), sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy `cossin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cossin.html).
pub fn cossin(x: &(impl super::MatrixInput + ?Sized), p: usize, q: usize) -> Result<CsResult> {
    let x = square(x)?;
    let n = x.nrows();
    if p == 0 || q == 0 || p >= n || q >= n {
        return Err(Error::InvalidInput(
            "CS partitions must be strictly inside matrix axes",
        ));
    }
    if (x.transpose() * &x - Mat::<f64>::identity(n, n)).norm_l2() > 1e-10 {
        return Err(Error::InvalidInput("CS input must be orthogonal"));
    }
    let x11 = faer::Mat::from_fn(p, q, |i, j| x[(i, j)]);
    let f = x11
        .svd()
        .map_err(|_| Error::NumericalFailure("CS SVD did not converge"))?;
    let u1 = Mat::from_fn(p, p, |i, j| f.U()[(i, j)]);
    let v1 = Mat::from_fn(q, q, |i, j| f.V()[(i, j)]);
    let lower = x.as_ref().submatrix(p, 0, n - p, q) * &v1;
    let right = -x.as_ref().submatrix(0, q, p, n - q).transpose() * &u1;
    let mut ucolumns = Vec::new();
    let mut vcolumns = Vec::new();
    for j in 0..q {
        let column = lower.col(j);
        let norm = column.norm_l2();
        if norm > 1e-12 {
            ucolumns.push(column.as_mat() * Scale(1.0 / norm));
        }
    }
    for j in 0..p {
        let column = right.col(j);
        let norm = column.norm_l2();
        if norm > 1e-12 {
            vcolumns.push(column.as_mat() * Scale(1.0 / norm));
        }
    }
    let (mut u2, ku) = complete(ucolumns, n - p)?;
    let (mut v2, kv) = complete(vcolumns, n - q)?;
    let ru = n - p - ku;
    let rv = n - q - kv;
    if ru != rv {
        return Err(Error::NumericalFailure("CS uncoupled dimensions disagree"));
    }
    if ru > 0 {
        let remainder = u2.subcols(ku, ru).transpose()
            * x.as_ref().submatrix(p, q, n - p, n - q)
            * v2.subcols(kv, rv);
        let rem = super::svd(&rows(&remainder)?)?;
        let newu = u2.subcols(ku, ru) * dense(&rem.u)?;
        u2.as_mut().subcols_mut(ku, ru).copy_from(&newu);
        let newv = v2.subcols(kv, rv) * dense(&rem.vh)?.transpose();
        v2.as_mut().subcols_mut(kv, rv).copy_from(&newv);
    }
    let u = Mat::from_fn(n, n, |i, j| {
        if i < p && j < p {
            u1[(i, j)]
        } else if i >= p && j >= p {
            u2[(i - p, j - p)]
        } else {
            0.
        }
    });
    let v = Mat::from_fn(n, n, |i, j| {
        if i < q && j < q {
            v1[(i, j)]
        } else if i >= q && j >= q {
            v2[(i - q, j - q)]
        } else {
            0.
        }
    });
    let cs = u.transpose() * &x * &v;
    let r = p.min(n - p).min(q).min(n - q);
    let skip = p.min(q) - r;
    let theta = (skip..skip + r)
        .map(|j| lower.col(j).norm_l2().atan2(f.S()[j].clamp(0., 1.)))
        .collect();
    Ok(CsResult {
        u: rows(&u)?,
        cs: rows(&cs)?,
        vh: rows(&v.transpose())?,
        theta,
    })
}
