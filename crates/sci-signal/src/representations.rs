use super::design::{roots, validate};
use super::{
    TransferFunction, Zpk, checked, complex_checked, nonempty, positive, samples, size, tf2zpk,
    work,
};
use crate::{
    Error, Result,
    complex::{Complex, native, owned},
};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64 as C;

/// Representative chosen for a cluster of nearby roots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootRepresentative {
    /// Arithmetic mean of the cluster.
    Mean,
    /// Lexicographic minimum (real, imaginary).
    Minimum,
    /// Lexicographic maximum (real, imaginary).
    Maximum,
}
/// Partial-fraction expansion with consecutive ascending powers for repeated poles.
#[derive(Clone, Debug, PartialEq)]
pub struct PartialFractions {
    /// Residues matching the poles and their consecutive powers.
    pub residues: Vec<Complex>,
    /// Poles sorted by magnitude, then real and imaginary components.
    pub poles: Vec<Complex>,
    /// Polynomial direct term, descending powers for residue, ascending delays for residuez.
    pub direct: Vec<f64>,
}
fn finite_complex(x: &[Complex]) -> Result<()> {
    size(x.len())?;
    if x.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return Err(Error::InvalidInput("complex entries must be finite"));
    }
    Ok(())
}
fn poly_mul(a: &[C], b: &[C]) -> Vec<C> {
    let mut c = vec![C::new(0., 0.); a.len() + b.len() - 1];
    for (i, a) in a.iter().enumerate() {
        for (j, b) in b.iter().enumerate() {
            c[i + j] += a * b;
        }
    }
    c
}
fn real(x: Vec<C>) -> Result<Vec<f64>> {
    let scale = x.iter().map(|z| z.norm()).fold(1., f64::max);
    if x.iter().any(|z| z.im.abs() > 1e-9 * scale) {
        return Err(Error::InvalidInput("expansion must have real coefficients"));
    }
    checked(x.into_iter().map(|z| z.re).collect())
}
fn root_factors(poles: &[C], skip: usize, power: usize) -> Vec<C> {
    let mut a = vec![C::new(1., 0.)];
    for (i, p) in poles.iter().enumerate() {
        if i >= skip && i < skip + power {
            continue;
        }
        a = poly_mul(&a, &[C::new(1., 0.), -*p]);
    }
    a
}
fn fraction(b: &[f64], a: &[f64], tol: f64, digital: bool) -> Result<PartialFractions> {
    nonempty(b)?;
    nonempty(a)?;
    positive(tol)?;
    let (mut b, mut a) = (b.to_vec(), a.to_vec());
    if digital {
        while a.len() > 1 && a.last() == Some(&0.) {
            a.pop();
        }
        while b.len() > 1 && b.last() == Some(&0.) {
            b.pop();
        }
    } else {
        let a_start = a.iter().position(|v| *v != 0.).unwrap_or(a.len() - 1);
        let b_start = b.iter().position(|v| *v != 0.).unwrap_or(b.len() - 1);
        a.drain(..a_start);
        b.drain(..b_start);
    }
    if a[0] == 0. || a.len() > 33 || b.len() > 65 {
        return Err(Error::InvalidInput(
            "nonzero a[0], pole order<=32 and numerator degree<=64 required",
        ));
    }
    let gain = a[0];
    for v in &mut a {
        *v /= gain;
    }
    for v in &mut b {
        *v /= gain;
    }
    let ps = roots(&a)?;
    let (up, mult) = unique_roots(&ps, tol, RootRepresentative::Mean)?;
    let mut groups: Vec<_> = up.into_iter().zip(mult).collect();
    groups.sort_by(|(a, _), (b, _)| {
        native(*a)
            .norm()
            .total_cmp(&native(*b).norm())
            .then(a.re.total_cmp(&b.re))
            .then(a.im.total_cmp(&b.im))
    });
    let poles: Vec<C> = groups
        .iter()
        .flat_map(|(p, m)| std::iter::repeat_n(native(*p), *m))
        .collect();
    let mut direct = vec![];
    if b.len() >= a.len() {
        let len = b.len() - a.len() + 1;
        direct = vec![0.; len];
        if digital {
            for i in (0..len).rev() {
                let v = b[i + a.len() - 1] / a[a.len() - 1];
                direct[i] = v;
                for (j, a) in a.iter().enumerate() {
                    b[i + j] -= v * a;
                }
            }
            b.truncate(a.len() - 1);
        } else {
            for i in 0..len {
                let v = b[i];
                direct[i] = v;
                for (j, a) in a.iter().enumerate() {
                    b[i + j] -= v * a;
                }
            }
            b.drain(..len);
        }
    }
    let n = poles.len();
    if n == 0 {
        return Ok(PartialFractions {
            residues: vec![],
            poles: vec![],
            direct: checked(direct)?,
        });
    }
    let mut matrix = DMatrix::<C>::zeros(n, n);
    let mut col = 0;
    for (_, m) in &groups {
        for power in 1..=*m {
            let basis = root_factors(&poles, col, power);
            let target = col + power - 1;
            let offset = if digital { 0 } else { n - basis.len() };
            for (i, &v) in basis.iter().enumerate() {
                matrix[(offset + i, target)] = v;
            }
        }
        col += m;
    }
    let mut target = DVector::zeros(n);
    let offset = if digital { 0 } else { n - b.len() };
    for (i, v) in b.into_iter().enumerate() {
        target[offset + i] = C::new(v, 0.);
    }
    let r = matrix
        .lu()
        .solve(&target)
        .ok_or(Error::NumericalFailure("singular partial fractions"))?;
    Ok(PartialFractions {
        residues: complex_checked(r.iter().copied().map(owned).collect())?,
        poles: poles.into_iter().map(owned).collect(),
        direct: checked(direct)?,
    })
}
fn inverse(r: &[Complex], p: &[Complex], k: &[f64], digital: bool) -> Result<TransferFunction> {
    finite_complex(r)?;
    finite_complex(p)?;
    samples(k)?;
    if r.len() != p.len() || p.len() > 32 || k.len() > 65 {
        return Err(Error::DimensionMismatch);
    }
    let poles: Vec<_> = p.iter().copied().map(native).collect();
    let a = root_factors(&poles, 0, 0);
    let mut b = vec![C::new(0., 0.); a.len()];
    if !k.is_empty() {
        b = poly_mul(&a, &k.iter().map(|&x| C::new(x, 0.)).collect::<Vec<_>>());
    }
    let mut start = 0;
    for (i, residue) in r.iter().enumerate() {
        if i == 0 || native(p[i]) - native(p[i - 1]) != C::new(0., 0.) {
            start = i;
        }
        let power = i - start + 1;
        let basis = root_factors(&poles, start, power);
        let offset = if digital { 0 } else { b.len() - basis.len() };
        for (j, &v) in basis.iter().enumerate() {
            b[offset + j] += native(*residue) * v;
        }
    }
    let mut b = real(b)?;
    if digital {
        while b.len() > 1 && b.last() == Some(&0.) {
            b.pop();
        }
    } else {
        while b.len() > 1 && b[0] == 0. {
            b.remove(0);
        }
    }
    Ok((b, real(a)?))
}
fn pairs(mut roots: Vec<C>) -> Result<Vec<[f64; 3]>> {
    let mut out = vec![];
    roots.sort_by(|a, b| {
        a.norm()
            .total_cmp(&b.norm())
            .then(a.re.total_cmp(&b.re))
            .then(a.im.total_cmp(&b.im))
    });
    while let Some(a) = roots.pop() {
        let j = if a.im.abs() > 1e-10 {
            roots
                .iter()
                .enumerate()
                .min_by(|(_, x), (_, y)| {
                    (**x - a.conj()).norm().total_cmp(&(**y - a.conj()).norm())
                })
                .map(|(j, _)| j)
        } else {
            roots.iter().rposition(|b| b.im.abs() <= 1e-10)
        };
        let j = j.ok_or(Error::InvalidInput("roots must occur in conjugate pairs"))?;
        let b = roots.remove(j);
        if (a + b).im.abs() > 1e-8 || (a * b).im.abs() > 1e-8 {
            return Err(Error::InvalidInput(
                "roots must form real polynomial factors",
            ));
        }
        out.push([1., -(a + b).re, (a * b).re]);
    }
    Ok(out)
}

/// Groups roots within Euclidean distance tol of each first unused root.
///
/// ## Parameters
///
/// Input order determines greedy clusters, so grouping is not transitive. representative selects mean or lexicographic minimum/maximum. Returns (representative roots, multiplicities) in cluster discovery order; tol>0.
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
/// use sci_signal::representations::*;
/// let _=unique_roots(&[sci_complex::Complex{re:1.,im:0.}],1e-3,RootRepresentative::Mean)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy unique_roots](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.unique_roots.html).
pub fn unique_roots(
    p: &[Complex],
    tol: f64,
    representative: RootRepresentative,
) -> Result<(Vec<Complex>, Vec<usize>)> {
    finite_complex(p)?;
    positive(tol)?;
    work(p.len(), p.len())?;
    let mut used = vec![false; p.len()];
    let mut roots = vec![];
    let mut multiplicity = vec![];
    for i in 0..p.len() {
        if used[i] {
            continue;
        }
        let mut cluster = vec![];
        for j in i..p.len() {
            if !used[j] && (native(p[i]) - native(p[j])).norm() <= tol {
                used[j] = true;
                cluster.push(native(p[j]));
            }
        }
        let z = match representative {
            RootRepresentative::Mean => cluster.iter().sum::<C>() / cluster.len() as f64,
            RootRepresentative::Minimum => *cluster
                .iter()
                .min_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)))
                .unwrap(),
            RootRepresentative::Maximum => *cluster
                .iter()
                .max_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)))
                .unwrap(),
        };
        roots.push(owned(z));
        multiplicity.push(cluster.len());
    }
    Ok((complex_checked(roots)?, multiplicity))
}

/// Expands `B/A = K + sum r[j]/(s-p[j])^power`.
///
/// ## Parameters
///
/// b/a are descending-power coefficients. tol>0 clusters nearby poles by their mean. Poles are in ascending magnitude order; repeated poles occupy consecutive entries with powers starting at one. Returns PartialFractions; denominator degree<=32 and numerator degree<=64.
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
/// Uses bounded Schur roots and private pivoted complex LU to solve the coefficient identity. Near-coincident roots are replaced by cluster means; repeated and ill-conditioned poles can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::representations::*;
/// let _=residue(&[1.],&[1.,-0.5],1e-3)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy residue](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.residue.html).
pub fn residue(b: &[f64], a: &[f64], tol: f64) -> Result<PartialFractions> {
    fraction(b, a, tol, false)
}

/// Expands `B/A = K + sum r[j]/(1-p[j] z^-1)^power`.
///
/// ## Parameters
///
/// b/a are ascending-delay coefficients. tol>0 clusters nearby poles by their mean. Poles are in ascending magnitude order; repeated poles occupy consecutive entries with powers starting at one. Returns PartialFractions; denominator degree<=32 and numerator degree<=64.
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
/// Uses bounded Schur roots and private pivoted complex LU to solve the coefficient identity. Near-coincident roots are replaced by cluster means; repeated and ill-conditioned poles can reduce accuracy.
///
/// # Examples
///
/// ```
/// use sci_signal::representations::*;
/// let _=residuez(&[1.],&[1.,-0.5],1e-3)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy residuez](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.residuez.html).
pub fn residuez(b: &[f64], a: &[f64], tol: f64) -> Result<PartialFractions> {
    fraction(b, a, tol, true)
}

/// Reconstructs `B/A = K + sum r[j]/(s-p[j])^power`.
///
/// ## Parameters
///
/// Residues r and poles p have identical lengths <=32; exactly repeated poles must be consecutive with ascending powers. k is the direct polynomial in descending powers. Returns real (b,a) with a\[0\]=1. Complex coefficients and approximate repeated-pole grouping are unsupported.
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
/// use sci_signal::representations::*;
/// let p=residue(&[1.],&[1.,-0.5],1e-3)?;let _=invres(&p.residues,&p.poles,&p.direct)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy invres](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.invres.html).
pub fn invres(r: &[Complex], p: &[Complex], k: &[f64]) -> Result<TransferFunction> {
    inverse(r, p, k, false)
}

/// Reconstructs `B/A = K + sum r[j]/(1-p[j] z^-1)^power`.
///
/// ## Parameters
///
/// Residues r and poles p have identical lengths <=32; exactly repeated poles must be consecutive with ascending powers. k is the direct polynomial in ascending delays. Returns real (b,a) with a\[0\]=1. Complex coefficients and approximate repeated-pole grouping are unsupported.
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
/// use sci_signal::representations::*;
/// let p=residuez(&[1.],&[1.,-0.5],1e-3)?;let _=invresz(&p.residues,&p.poles,&p.direct)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy invresz](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.invresz.html).
pub fn invresz(r: &[Complex], p: &[Complex], k: &[f64]) -> Result<TransferFunction> {
    inverse(r, p, k, true)
}

/// Factors `H(z)=k product(z-zero)/product(z-pole)` into second-order digital sections.
///
/// ## Parameters
///
/// Real proper ZPK with at most 128 poles. Returns rows \[b0,b1,b2,1,a1,a2\]. Roots are paired with their conjugates, real roots by magnitude; section order is descending root magnitude and gain is placed in the first section. Zero padding adds canceling roots at zero for odd order. Analog/pairing keyword dispatch is unsupported.
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
/// Pairs roots deterministically rather than optimizing SciPy peak gain. Prefer designing directly in ZPK to avoid polynomial root conditioning; section scaling may be less favorable than SciPy nearest pairing.
///
/// # Examples
///
/// ```
/// use sci_signal::representations::*;
/// use sci_signal::design::buttap;
/// let _=zpk2sos(&buttap(3)?)?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy zpk2sos](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.zpk2sos.html).
pub fn zpk2sos(zpk: &Zpk) -> Result<Vec<[f64; 6]>> {
    validate(zpk)?;
    let n = zpk.poles.len();
    if n == 0 {
        return Ok(vec![[zpk.gain, 0., 0., 1., 0., 0.]]);
    }
    let len = n + n % 2;
    let mut p: Vec<_> = zpk.poles.iter().copied().map(native).collect();
    let mut z: Vec<_> = zpk.zeros.iter().copied().map(native).collect();
    p.resize(len, C::new(0., 0.));
    z.resize(len, C::new(0., 0.));
    let a = pairs(p)?;
    let b = pairs(z)?;
    let mut sos: Vec<_> = b
        .into_iter()
        .zip(a)
        .map(|(b, a)| [b[0], b[1], b[2], a[0], a[1], a[2]])
        .collect();
    for x in &mut sos[0][..3] {
        *x *= zpk.gain;
    }
    for row in &sos {
        samples(row)?;
    }
    Ok(sos)
}

/// Factors a real digital transfer function into second-order sections.
///
/// ## Parameters
///
/// b/a are ascending-delay digital coefficients. Returns SOS rows as zpk2sos. Unlike tf2zpk alone, numerator is padded with trailing zeros to preserve digital relative degree.
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
/// use sci_signal::representations::*;
/// let _=tf2sos(&[1.],&[1.,-0.5])?;
/// # Ok::<(),sci_core::Error>(())
/// ```
///
/// # References
///
/// [SciPy tf2sos](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.tf2sos.html).
pub fn tf2sos(b: &[f64], a: &[f64]) -> Result<Vec<[f64; 6]>> {
    nonempty(b)?;
    nonempty(a)?;
    let mut b = b.to_vec();
    let mut a = a.to_vec();
    let len = b.len().max(a.len());
    b.resize(len, 0.);
    a.resize(len, 0.);
    zpk2sos(&tf2zpk(&b, &a)?)
}
