//! Agglomerative clustering and validated linkage-tree utilities.
//!
//! Linkages use shared row matrices: row `i` is `[left, right, height, count]`,
//! creates node `n+i`, and refers only to earlier nodes. Leaves are `0..n`.
//! All functions reject nonfinite linkages and bound trees to 512 leaves.
use crate::error::finite;
use crate::{Error, Result, linalg::Matrix};
use std::collections::HashMap;

const MAX_LEAVES: usize = 512;

fn validate(z: &[Vec<f64>]) -> Result<usize> {
    let n = z.len() + 1;
    if z.is_empty() || n > MAX_LEAVES {
        return Err(Error::InvalidInput("linkage requires 2..=512 leaves"));
    }
    let mut used = vec![false; 2 * n - 1];
    let mut sizes = vec![1; 2 * n - 1];
    for (i, row) in z.iter().enumerate() {
        if row.len() != 4 {
            return Err(Error::DimensionMismatch);
        }
        if row.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(Error::InvalidInput("invalid linkage values"));
        }
        let mut count = 0;
        for &v in &row[..2] {
            if v.fract() != 0.0 || v >= (n + i) as f64 {
                return Err(Error::InvalidInput("invalid linkage child"));
            }
            let child = v as usize;
            if used[child] {
                return Err(Error::InvalidInput("linkage reuses a child"));
            }
            used[child] = true;
            count += sizes[child];
        }
        if row[3] != count as f64 {
            return Err(Error::InvalidInput("incorrect linkage count"));
        }
        sizes[n + i] = count;
    }
    Ok(n)
}

fn subtree_max(z: &[Vec<f64>], values: &[f64]) -> Vec<f64> {
    let n = z.len() + 1;
    let mut out = values.to_vec();
    for (i, row) in z.iter().enumerate() {
        for &v in &row[..2] {
            if v as usize >= n {
                out[i] = out[i].max(out[v as usize - n]);
            }
        }
    }
    out
}

fn descendants(z: &[Vec<f64>], node: usize) -> Vec<usize> {
    let n = z.len() + 1;
    let mut out = Vec::new();
    let mut stack = vec![node];
    while let Some(v) = stack.pop() {
        if v < n {
            out.push(v);
        } else {
            stack.push(z[v - n][1] as usize);
            stack.push(z[v - n][0] as usize);
        }
    }
    out
}

/// The criterion for updating inter-cluster distances after a merge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Minimum pairwise distance.
    Single,
    /// Maximum pairwise distance.
    Complete,
    /// Mean distance across all observation pairs.
    Average,
    /// Equal-weight mean of the two previous cluster distances.
    Weighted,
    /// Euclidean distance between cluster centroids.
    Centroid,
    /// Euclidean distance between equally weighted previous centroids.
    Median,
    /// Square root of twice the increase in within-cluster sum of squares.
    Ward,
}

/// Builds an agglomerative binary tree by repeatedly merging the closest clusters.
///
/// ## Parameters
///
/// `y` is upper-triangle condensed distance order, matching `spatial::distance::pdist`. `method` selects the distance recurrence. Centroid, median, and Ward require Euclidean distances; this geometric condition is the caller’s responsibility. Observations and optimal ordering are adapted to separate calls.
/// ## Domain
///
/// Finite, nonnegative distances for 2..=512 observations; condensed length must be n(n−1)/2.
/// ## Returns
///
/// An `(n-1) × 4` linkage matrix with zero-based cluster IDs.
/// ## Errors
///
/// Returns a shared error for invalid linkage structure, values, counts, dimensions, or work limits.
/// ## Numerical behavior
///
/// Private Kodama backend: O(n²) storage; centroid/median have O(n³) worst-case time. Equal distances may produce a different valid merge order from SciPy. Squared-distance methods reject inputs too large for bounded intermediate arithmetic.
/// # Examples
///
/// ```
/// let z = sci_cluster::hierarchy::linkage(&[2.0], sci_cluster::hierarchy::Method::Single)?;
/// assert_eq!(z, vec![vec![0.0, 1.0, 2.0, 2.0]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html).
pub fn linkage(y: &[f64], method: Method) -> Result<Matrix> {
    let n = crate::spatial::distance::num_obs_y(y)?;
    if !(2..=MAX_LEAVES).contains(&n) || y.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(Error::InvalidInput(
            "invalid condensed clustering distances",
        ));
    }
    if matches!(method, Method::Centroid | Method::Median | Method::Ward)
        && y.iter().any(|v| *v > (f64::MAX / (8.0 * n as f64)).sqrt())
    {
        return Err(Error::InvalidInput("clustering distance is too large"));
    }
    let backend = match method {
        Method::Single => kodama::Method::Single,
        Method::Complete => kodama::Method::Complete,
        Method::Average => kodama::Method::Average,
        Method::Weighted => kodama::Method::Weighted,
        Method::Centroid => kodama::Method::Centroid,
        Method::Median => kodama::Method::Median,
        Method::Ward => kodama::Method::Ward,
    };
    let result = kodama::linkage(&mut y.to_vec(), n, backend)
        .steps()
        .iter()
        .map(|s| {
            vec![
                s.cluster1 as f64,
                s.cluster2 as f64,
                s.dissimilarity,
                s.size as f64,
            ]
        })
        .collect::<Matrix>();
    validate(&result)?;
    Ok(result)
}

/// Clusters using single linkage, with distance `min d(a,b)`.
///
/// ## Parameters
///
/// Condensed `y`, with exactly the conventions and geometric requirements of [`linkage`].
/// ## Domain
///
/// The domain and finite-value constraints of [`linkage`].
/// ## Returns
///
/// The shared linkage matrix returned by [`linkage`].
/// ## Errors
///
/// Returns a shared error for invalid linkage structure, values, counts, dimensions, or work limits.
/// ## Numerical behavior
///
/// Delegates to [`linkage`], sharing its precision, tie handling, and work limits.
/// # Examples
///
/// ```
/// let z = sci_cluster::hierarchy::single(&[2.0])?;
/// assert_eq!(z.len(),1);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy single](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.single.html).
pub fn single(y: &[f64]) -> Result<Matrix> {
    linkage(y, Method::Single)
}

/// Clusters using complete linkage, with distance `max d(a,b)`.
///
/// ## Parameters
///
/// Condensed `y`, with exactly the conventions and geometric requirements of [`linkage`].
/// ## Domain
///
/// The domain and finite-value constraints of [`linkage`].
/// ## Returns
///
/// The shared linkage matrix returned by [`linkage`].
/// ## Errors
///
/// Returns a shared error for invalid linkage structure, values, counts, dimensions, or work limits.
/// ## Numerical behavior
///
/// Delegates to [`linkage`], sharing its precision, tie handling, and work limits.
/// # Examples
///
/// ```
/// let z = sci_cluster::hierarchy::complete(&[2.0])?;
/// assert_eq!(z.len(),1);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy complete](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.complete.html).
pub fn complete(y: &[f64]) -> Result<Matrix> {
    linkage(y, Method::Complete)
}

/// Clusters using average linkage, with distance `Σ d(a,b)/(|A||B|)`.
///
/// ## Parameters
///
/// Condensed `y`, with exactly the conventions and geometric requirements of [`linkage`].
/// ## Domain
///
/// The domain and finite-value constraints of [`linkage`].
/// ## Returns
///
/// The shared linkage matrix returned by [`linkage`].
/// ## Errors
///
/// Returns a shared error for invalid linkage structure, values, counts, dimensions, or work limits.
/// ## Numerical behavior
///
/// Delegates to [`linkage`], sharing its precision, tie handling, and work limits.
/// # Examples
///
/// ```
/// let z = sci_cluster::hierarchy::average(&[2.0])?;
/// assert_eq!(z.len(),1);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy average](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.average.html).
pub fn average(y: &[f64]) -> Result<Matrix> {
    linkage(y, Method::Average)
}

/// Clusters using weighted linkage, with distance `(d(A,C)+d(B,C))/2`.
///
/// ## Parameters
///
/// Condensed `y`, with exactly the conventions and geometric requirements of [`linkage`].
/// ## Domain
///
/// The domain and finite-value constraints of [`linkage`].
/// ## Returns
///
/// The shared linkage matrix returned by [`linkage`].
/// ## Errors
///
/// Returns a shared error for invalid linkage structure, values, counts, dimensions, or work limits.
/// ## Numerical behavior
///
/// Delegates to [`linkage`], sharing its precision, tie handling, and work limits.
/// # Examples
///
/// ```
/// let z = sci_cluster::hierarchy::weighted(&[2.0])?;
/// assert_eq!(z.len(),1);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy weighted](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.weighted.html).
pub fn weighted(y: &[f64]) -> Result<Matrix> {
    linkage(y, Method::Weighted)
}

/// Clusters using centroid linkage, with distance `‖mean(A)−mean(B)‖₂`.
///
/// ## Parameters
///
/// Condensed `y`, with exactly the conventions and geometric requirements of [`linkage`].
/// ## Domain
///
/// The domain and finite-value constraints of [`linkage`].
/// ## Returns
///
/// The shared linkage matrix returned by [`linkage`].
/// ## Errors
///
/// Returns a shared error for invalid linkage structure, values, counts, dimensions, or work limits.
/// ## Numerical behavior
///
/// Delegates to [`linkage`], sharing its precision, tie handling, and work limits.
/// # Examples
///
/// ```
/// let z = sci_cluster::hierarchy::centroid(&[2.0])?;
/// assert_eq!(z.len(),1);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy centroid](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.centroid.html).
pub fn centroid(y: &[f64]) -> Result<Matrix> {
    linkage(y, Method::Centroid)
}

/// Clusters using median linkage, with distance `‖(centroid(A)+centroid(B))/2−centroid(C)‖₂`.
///
/// ## Parameters
///
/// Condensed `y`, with exactly the conventions and geometric requirements of [`linkage`].
/// ## Domain
///
/// The domain and finite-value constraints of [`linkage`].
/// ## Returns
///
/// The shared linkage matrix returned by [`linkage`].
/// ## Errors
///
/// Returns a shared error for invalid linkage structure, values, counts, dimensions, or work limits.
/// ## Numerical behavior
///
/// Delegates to [`linkage`], sharing its precision, tie handling, and work limits.
/// # Examples
///
/// ```
/// let z = sci_cluster::hierarchy::median(&[2.0])?;
/// assert_eq!(z.len(),1);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy median](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.median.html).
pub fn median(y: &[f64]) -> Result<Matrix> {
    linkage(y, Method::Median)
}

/// Clusters using ward linkage, with distance `sqrt(2 Δ within-cluster sum of squares)`.
///
/// ## Parameters
///
/// Condensed `y`, with exactly the conventions and geometric requirements of [`linkage`].
/// ## Domain
///
/// The domain and finite-value constraints of [`linkage`].
/// ## Returns
///
/// The shared linkage matrix returned by [`linkage`].
/// ## Errors
///
/// Returns a shared error for invalid linkage structure, values, counts, dimensions, or work limits.
/// ## Numerical behavior
///
/// Delegates to [`linkage`], sharing its precision, tie handling, and work limits.
/// # Examples
///
/// ```
/// let z = sci_cluster::hierarchy::ward(&[2.0])?;
/// assert_eq!(z.len(),1);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy ward](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.ward.html).
pub fn ward(y: &[f64]) -> Result<Matrix> {
    linkage(y, Method::Ward)
}

/// Counts the original observations: `n = rows(Z) + 1`.
///
/// ## Parameters
///
/// `z` is a validated linkage matrix.
/// ## Domain
///
/// Linkage matrices follow the module convention, have 2..=512 leaves, and must contain finite nonnegative heights and exact subtree counts. NaN and infinity are rejected.
/// ## Returns
///
/// The leaf count as `usize`.
/// ## Errors
///
/// Returns a shared error for invalid linkage structure, values, counts, dimensions, or work limits.
/// ## Numerical behavior
///
/// Linear-time structural validation; exact integer arithmetic.
/// # Examples
///
/// ```
/// let z = vec![vec![0.0, 1.0, 2.0, 2.0]];
/// let _ = sci_cluster::hierarchy::num_obs_linkage(&z)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy num_obs_linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.num_obs_linkage.html).
pub fn num_obs_linkage(z: &[Vec<f64>]) -> Result<usize> {
    validate(z)
}

/// Tests whether child indices and counts describe a binary merge tree.
///
/// ## Parameters
///
/// `z` contains linkage rows. SciPy warning/throw/name controls are omitted.
/// ## Domain
///
/// Linkage matrices follow the module convention, have 2..=512 leaves, and must contain finite nonnegative heights and exact subtree counts. NaN and infinity are rejected.
/// ## Returns
///
/// `true` for a valid supported tree; `false` otherwise.
/// ## Errors
///
/// Infallible; invalid data or trees above the work limit return false.
/// ## Numerical behavior
///
/// Linear-time validation. Additionally checks integral indices and exact counts, stricter than SciPy for malformed matrices.
/// # Examples
///
/// ```
/// assert!(sci_cluster::hierarchy::is_valid_linkage(&[vec![0.0,1.0,2.0,2.0]]));
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy is_valid_linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.is_valid_linkage.html).
pub fn is_valid_linkage(z: &[Vec<f64>]) -> bool {
    validate(z).is_ok()
}

/// Tests monotonic merge heights: `Z[i+1,2] ≥ Z[i,2]`.
///
/// ## Parameters
///
/// `z` is a linkage matrix; inversions are allowed in its input.
/// ## Domain
///
/// Linkage matrices follow the module convention, have 2..=512 leaves, and must contain finite nonnegative heights and exact subtree counts. NaN and infinity are rejected.
/// ## Returns
///
/// Whether heights are nondecreasing in merge order.
/// ## Errors
///
/// Returns a shared error for invalid linkage structure, values, counts, dimensions, or work limits.
/// ## Numerical behavior
///
/// Exact comparisons after structural validation.
/// # Examples
///
/// ```
/// let z = vec![vec![0.0, 1.0, 2.0, 2.0]];
/// let _ = sci_cluster::hierarchy::is_monotonic(&z)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy is_monotonic](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.is_monotonic.html).
pub fn is_monotonic(z: &[Vec<f64>]) -> Result<bool> {
    validate(z)?;
    Ok(z.windows(2).all(|w| w[0][2] <= w[1][2]))
}

/// Lists leaves in left-to-right depth-first tree order.
///
/// ## Parameters
///
/// `z` uses ordered left and right children.
/// ## Domain
///
/// Linkage matrices follow the module convention, have 2..=512 leaves, and must contain finite nonnegative heights and exact subtree counts. NaN and infinity are rejected.
/// ## Returns
///
/// A permutation of `0..n`.
/// ## Errors
///
/// Returns a shared error for invalid linkage structure, values, counts, dimensions, or work limits.
/// ## Numerical behavior
///
/// O(n) time and space; uses an explicit stack to avoid recursion depth limits.
/// # Examples
///
/// ```
/// let z = vec![vec![0.0, 1.0, 2.0, 2.0]];
/// let _ = sci_cluster::hierarchy::leaves_list(&z)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy leaves_list](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.leaves_list.html).
pub fn leaves_list(z: &[Vec<f64>]) -> Result<Vec<usize>> {
    let n = validate(z)?;
    Ok(descendants(z, 2 * n - 2))
}

/// Computes `max { height(v) : v in subtree(n+i) }` for each merge.
///
/// ## Parameters
///
/// `z` may contain height inversions.
/// ## Domain
///
/// Linkage matrices follow the module convention, have 2..=512 leaves, and must contain finite nonnegative heights and exact subtree counts. NaN and infinity are rejected.
/// ## Returns
///
/// One maximum height per row, in merge order and input distance units.
/// ## Errors
///
/// Returns a shared error for invalid linkage structure, values, counts, dimensions, or work limits.
/// ## Numerical behavior
///
/// Linear-time bottom-up maxima, preserving finite heights exactly.
/// # Examples
///
/// ```
/// let z = vec![vec![0.0, 1.0, 2.0, 2.0]];
/// let _ = sci_cluster::hierarchy::maxdists(&z)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy maxdists](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.maxdists.html).
pub fn maxdists(z: &[Vec<f64>]) -> Result<Vec<f64>> {
    validate(z)?;
    Ok(subtree_max(z, &z.iter().map(|r| r[2]).collect::<Vec<_>>()))
}

/// Converts `[a,b,h,count]` to MATLAB `[a+1,b+1,h]`.
///
/// ## Parameters
///
/// `z` uses zero-based IDs; output uses one-based IDs.
/// ## Domain
///
/// Linkage matrices follow the module convention, have 2..=512 leaves, and must contain finite nonnegative heights and exact subtree counts. NaN and infinity are rejected.
/// ## Returns
///
/// An `(n-1) × 3` matrix.
/// ## Errors
///
/// Returns a shared error for invalid linkage structure, values, counts, dimensions, or work limits.
/// ## Numerical behavior
///
/// Exact index conversion; heights are copied without rounding.
/// # Examples
///
/// ```
/// let z = vec![vec![0.0, 1.0, 2.0, 2.0]];
/// let _ = sci_cluster::hierarchy::to_mlab_linkage(&z)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy to_mlab_linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.to_mlab_linkage.html).
pub fn to_mlab_linkage(z: &[Vec<f64>]) -> Result<Matrix> {
    validate(z)?;
    Ok(z.iter()
        .map(|r| vec![r[0] + 1.0, r[1] + 1.0, r[2]])
        .collect())
}

/// Converts MATLAB `[a,b,h]` to `[a−1,b−1,h,subtree_count]`.
///
/// ## Parameters
///
/// `z` contains three-column rows with one-based child IDs.
/// ## Domain
///
/// Finite three-column MATLAB trees with 2..=512 leaves.
/// ## Returns
///
/// An `(n-1) × 4` shared linkage matrix.
/// ## Errors
///
/// Returns a shared error for invalid linkage structure, values, counts, dimensions, or work limits.
/// ## Numerical behavior
///
/// Exact integer count accumulation in O(n) time. Malformed MATLAB trees are rejected before indexing.
/// # Examples
///
/// ```
/// assert_eq!(sci_cluster::hierarchy::from_mlab_linkage(&[vec![1.0,2.0,3.0]])?,vec![vec![0.0,1.0,3.0,2.0]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy from_mlab_linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.from_mlab_linkage.html).
pub fn from_mlab_linkage(z: &[Vec<f64>]) -> Result<Matrix> {
    let n = z.len() + 1;
    if z.is_empty() || n > MAX_LEAVES {
        return Err(Error::InvalidInput("unsupported MATLAB tree size"));
    }
    let mut counts = vec![1; 2 * n - 1];
    let mut out = Vec::with_capacity(z.len());
    for (i, r) in z.iter().enumerate() {
        if r.len() != 3 {
            return Err(Error::DimensionMismatch);
        }
        if r[..2]
            .iter()
            .any(|v| !v.is_finite() || *v < 1.0 || *v > (n + i) as f64 || v.fract() != 0.0)
        {
            return Err(Error::InvalidInput("invalid MATLAB child"));
        }
        let a = r[0] as usize - 1;
        let b = r[1] as usize - 1;
        counts[n + i] = counts[a] + counts[b];
        out.push(vec![a as f64, b as f64, r[2], counts[n + i] as f64]);
    }
    validate(&out)?;
    Ok(out)
}

/// Tests whether condensed distances and a tree have the same observation count.
///
/// ## Parameters
///
/// `z` is a linkage; `y` is condensed distance order. This tests sizes, not agreement of merge heights.
/// ## Domain
///
/// Linkage matrices follow the module convention, have 2..=512 leaves, and must contain finite nonnegative heights and exact subtree counts. NaN and infinity are rejected.
/// ## Returns
///
/// Whether `len(y)=n(n−1)/2`.
/// ## Errors
///
/// Returns a shared error for invalid linkage structure, values, counts, dimensions, or work limits.
/// ## Numerical behavior
///
/// Linear validation of tree and distance inputs.
/// # Examples
///
/// ```
/// assert!(sci_cluster::hierarchy::correspond(&[vec![0.0,1.0,2.0,2.0]],&[2.0])?);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy correspond](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.correspond.html).
pub fn correspond(z: &[Vec<f64>], y: &[f64]) -> Result<bool> {
    let n = validate(z)?;
    let m = crate::spatial::distance::num_obs_y(y)?;
    if y.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(Error::InvalidInput("invalid distances"));
    }
    Ok(n == m)
}

/// Tests whether two labelings define the same partition up to a bijection of labels.
///
/// ## Parameters
///
/// `a` and `b` are nonnegative integer cluster labels, with arbitrary label values.
/// ## Domain
///
/// Equal-length integer slices with at most 1,000,000 entries; empty partitions agree.
/// ## Returns
///
/// Whether `a[i]=a[j]` if and only if `b[i]=b[j]` for every pair.
/// ## Errors
///
/// Dimension mismatch or exceeded work limit.
/// ## Numerical behavior
///
/// Expected O(n) time using two label maps; no floating-point operations.
/// # Examples
///
/// ```
/// assert!(sci_cluster::hierarchy::is_isomorphic(&[1,1,2],&[9,9,0])?);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy is_isomorphic](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.is_isomorphic.html).
pub fn is_isomorphic(a: &[usize], b: &[usize]) -> Result<bool> {
    if a.len() != b.len() {
        return Err(Error::DimensionMismatch);
    }
    if a.len() > 1_000_000 {
        return Err(Error::InvalidInput("label work limit"));
    }
    let mut ab = HashMap::new();
    let mut ba = HashMap::new();
    for (&x, &y) in a.iter().zip(b) {
        if ab.insert(x, y).is_some_and(|v| v != y) || ba.insert(y, x).is_some_and(|v| v != x) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Computes local link mean, sample standard deviation, count, and `(height−mean)/std`.
///
/// ## Parameters
///
/// `depth` counts levels including the current merge; depth zero behaves as one, matching SciPy.
/// ## Domain
///
/// Linkage matrices follow the module convention, have 2..=512 leaves, and must contain finite nonnegative heights and exact subtree counts. NaN and infinity are rejected.
/// ## Returns
///
/// An `(n-1) × 4` matrix `[mean, sample_std, count, coefficient]`. Zero standard deviation gives coefficient zero.
/// ## Errors
///
/// Returns a shared error for invalid linkage structure, values, counts, dimensions, or work limits.
/// ## Numerical behavior
///
/// O(n²) bounded tree traversal; scaled two-pass moments reduce overflow. Small cancellation differences from SciPy’s raw moments are possible.
/// # Examples
///
/// ```
/// let r=sci_cluster::hierarchy::inconsistent(&[vec![0.0,1.0,2.0,2.0]],2)?;
/// assert_eq!(r,vec![vec![2.0,0.0,1.0,0.0]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy inconsistent](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.inconsistent.html).
pub fn inconsistent(z: &[Vec<f64>], depth: usize) -> Result<Matrix> {
    let n = validate(z)?;
    let mut out = Vec::with_capacity(z.len());
    for (i, r) in z.iter().enumerate() {
        let mut stack = vec![(n + i, 1)];
        let mut values = Vec::new();
        while let Some((v, d)) = stack.pop() {
            values.push(z[v - n][2]);
            if d < depth {
                for &c in &z[v - n][..2] {
                    if c as usize >= n {
                        stack.push((c as usize, d + 1));
                    }
                }
            }
        }
        let count = values.len();
        let scale = values.iter().copied().fold(0.0, f64::max);
        let mean = values.iter().map(|v| v / count as f64).sum::<f64>();
        let sd = if count == 1 || scale == 0.0 {
            0.0
        } else {
            scale
                * (values
                    .iter()
                    .map(|v| (v / scale - mean / scale).powi(2) / (count - 1) as f64)
                    .sum::<f64>())
                .sqrt()
        };
        let coeff = if sd == 0.0 { 0.0 } else { (r[2] - mean) / sd };
        out.push(vec![
            finite(mean)?,
            finite(sd)?,
            count as f64,
            finite(coeff)?,
        ]);
    }
    Ok(out)
}

/// Checks inconsistency rows `[mean,std,count,coefficient]`.
///
/// ## Parameters
///
/// `r` is an inconsistency matrix. SciPy warning/throw/name options are omitted.
/// ## Domain
///
/// Nonempty four-column matrices with at most 511 rows; nonfinite values return false.
/// ## Returns
///
/// True for supported finite rows with nonnegative means/std/count.
/// ## Errors
///
/// Infallible; invalid matrices return false.
/// ## Numerical behavior
///
/// Checks actual SciPy nonnegative-column behavior; does not impose the count upper bound stated in some SciPy documentation. Adds finite-value validation.
/// # Examples
///
/// ```
/// assert!(sci_cluster::hierarchy::is_valid_im(&[vec![2.0,0.0,1.0,0.0]]));
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy is_valid_im](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.is_valid_im.html).
pub fn is_valid_im(r: &[Vec<f64>]) -> bool {
    !r.is_empty()
        && r.len() < MAX_LEAVES
        && r.iter().all(|v| {
            v.len() == 4 && v.iter().all(|x| x.is_finite()) && v[..3].iter().all(|x| *x >= 0.0)
        })
}

/// Computes `max { R[v,column] : v in subtree(n+i) }`.
///
/// ## Parameters
///
/// `z` is a linkage; `r` is a same-row-count valid inconsistency matrix; `column` is zero-based in `0..4`.
/// ## Domain
///
/// Linkage matrices follow the module convention, have 2..=512 leaves, and must contain finite nonnegative heights and exact subtree counts. NaN and infinity are rejected.
/// ## Returns
///
/// A subtree maximum per merge, in the selected column’s units.
/// ## Errors
///
/// Returns a shared error for invalid linkage structure, values, counts, dimensions, or work limits.
/// ## Numerical behavior
///
/// O(n) bottom-up recurrence. Rust exports the idiomatic `max_rstat` spelling.
/// # Examples
///
/// ```
/// let z=vec![vec![0.0,1.0,2.0,2.0]];
/// let r=sci_cluster::hierarchy::inconsistent(&z,2)?;
/// assert_eq!(sci_cluster::hierarchy::max_rstat(&z,&r,0)?,vec![2.0]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy maxRstat](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.maxRstat.html).
pub fn max_rstat(z: &[Vec<f64>], r: &[Vec<f64>], column: usize) -> Result<Vec<f64>> {
    validate(z)?;
    if z.len() != r.len() {
        return Err(Error::DimensionMismatch);
    }
    if !is_valid_im(r) || column >= 4 {
        return Err(Error::InvalidInput(
            "invalid inconsistency matrix or column",
        ));
    }
    Ok(subtree_max(
        z,
        &r.iter().map(|v| v[column]).collect::<Vec<_>>(),
    ))
}

/// Computes the maximum inconsistency coefficient over each subtree.
///
/// ## Parameters
///
/// `z` and `r` follow [`max_rstat`], selecting column three.
/// ## Domain
///
/// Linkage matrices follow the module convention, have 2..=512 leaves, and must contain finite nonnegative heights and exact subtree counts. NaN and infinity are rejected.
/// ## Returns
///
/// One maximum coefficient per merge.
/// ## Errors
///
/// Returns a shared error for invalid linkage structure, values, counts, dimensions, or work limits.
/// ## Numerical behavior
///
/// Reuses [`max_rstat`] with identical validation and O(n) bounds.
/// # Examples
///
/// ```
/// let z=vec![vec![0.0,1.0,2.0,2.0]];
/// let r=sci_cluster::hierarchy::inconsistent(&z,2)?;
/// assert_eq!(sci_cluster::hierarchy::maxinconsts(&z,&r)?,vec![0.0]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy maxinconsts](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.maxinconsts.html).
pub fn maxinconsts(z: &[Vec<f64>], r: &[Vec<f64>]) -> Result<Vec<f64>> {
    max_rstat(z, r, 3)
}

/// Computes cophenetic distances `d(i,j)=height(lowest_common_ancestor(i,j))`.
///
/// ## Parameters
///
/// `z` is a linkage; optional condensed `y` requests Pearson correlation against original distances.
/// ## Domain
///
/// Linkage matrices follow the module convention, have 2..=512 leaves, and must contain finite nonnegative heights and exact subtree counts. NaN and infinity are rejected.
/// ## Returns
///
/// `(correlation, distances)` in SciPy condensed order. Correlation is `None` without `y`, NaN for zero variance.
/// ## Errors
///
/// Returns a shared error for invalid linkage structure, values, counts, dimensions, or work limits.
/// ## Numerical behavior
///
/// O(n²) tree traversal and storage; shared Pearson correlation uses scaled reductions.
/// # Examples
///
/// ```
/// let (_,d)=sci_cluster::hierarchy::cophenet(&[vec![0.0,1.0,2.0,2.0]],None)?;
/// assert_eq!(d,vec![2.0]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy cophenet](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.cophenet.html).
pub fn cophenet(z: &[Vec<f64>], y: Option<&[f64]>) -> Result<(Option<f64>, Vec<f64>)> {
    let n = validate(z)?;
    let mut out = vec![0.0; n * (n - 1) / 2];
    for row in z {
        let a = descendants(z, row[0] as usize);
        let b = descendants(z, row[1] as usize);
        for &i in &a {
            for &j in &b {
                let (i, j) = (i.min(j), i.max(j));
                out[n * i - i * (i + 1) / 2 + j - i - 1] = row[2];
            }
        }
    }
    let corr = if let Some(y) = y {
        if y.len() != out.len() {
            return Err(Error::DimensionMismatch);
        }
        if y.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(Error::InvalidInput("invalid distances"));
        }
        let scale = y.iter().chain(&out).copied().fold(0.0, f64::max);
        if scale == 0.0 {
            Some(f64::NAN)
        } else {
            let m = y.iter().map(|v| v / scale / y.len() as f64).sum::<f64>();
            let c = out
                .iter()
                .map(|v| v / scale / out.len() as f64)
                .sum::<f64>();
            let (mut xy, mut xx, mut yy) = (0.0, 0.0, 0.0);
            for (&a, &b) in y.iter().zip(&out) {
                let a = a / scale - m;
                let b = b / scale - c;
                xy += a * b;
                xx += a * a;
                yy += b * b;
            }
            Some(if xx == 0.0 || yy == 0.0 {
                f64::NAN
            } else {
                (xy / xx.sqrt() / yy.sqrt()).clamp(-1.0, 1.0)
            })
        }
    } else {
        None
    };
    Ok((corr, out))
}

/// Flat clustering criterion; all thresholds include equality.
#[derive(Clone, Copy, Debug)]
pub enum Criterion<'a> {
    /// Maximum subtree merge height at most the supplied distance threshold.
    Distance,
    /// Maximum depth-limited inconsistency coefficient at most the threshold.
    Inconsistent(usize),
    /// At most the threshold number of clusters, using subtree merge heights.
    MaxClust,
    /// User-supplied monotone statistic per merge, compared to the threshold.
    Monocrit(&'a [f64]),
    /// At most the threshold number of clusters using a monotone statistic.
    MaxClustMonocrit(&'a [f64]),
}

fn flat(z: &[Vec<f64>], stats: &[f64], threshold: f64) -> Vec<usize> {
    let n = z.len() + 1;
    let mut labels = vec![0; n];
    let mut next = 1;
    // SciPy traverses internal children first, then singleton children.
    let mut stack = vec![2 * n - 2];
    while let Some(v) = stack.pop() {
        if v < n {
            labels[v] = next;
            next += 1;
        } else if stats[v - n] <= threshold {
            for leaf in descendants(z, v) {
                labels[leaf] = next;
            }
            next += 1;
        } else {
            let a = z[v - n][0] as usize;
            let b = z[v - n][1] as usize;
            let children = if a < n && b >= n { [b, a] } else { [a, b] };
            stack.push(children[1]);
            stack.push(children[0]);
        }
    }
    labels
}

/// Partitions leaves by a monotone subtree statistic or a maximum cluster count.
///
/// ## Parameters
///
/// `threshold` is inclusive. `criterion` selects distances, depth-limited inconsistency, or supplied monotone values. Labels are one-based; tied heights may produce fewer clusters than requested.
/// ## Domain
///
/// Linkage matrices follow the module convention, have 2..=512 leaves, and must contain finite nonnegative heights and exact subtree counts. NaN and infinity are rejected.
/// ## Returns
///
/// One cluster label per original observation, in input order.
/// ## Errors
///
/// Returns a shared error for invalid linkage structure, values, counts, dimensions, or work limits.
/// ## Numerical behavior
///
/// O(n²) bounded work for inconsistency or threshold search. Supplied statistics must be finite and monotone along the tree. Label numbers follow internal-child-first traversal.
/// # Examples
///
/// ```
/// let labels=sci_cluster::hierarchy::fcluster(&[vec![0.0,1.0,2.0,2.0]],2.0,sci_cluster::hierarchy::Criterion::Distance)?;
/// assert_eq!(labels,vec![1,1]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy fcluster](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html).
pub fn fcluster(z: &[Vec<f64>], threshold: f64, criterion: Criterion<'_>) -> Result<Vec<usize>> {
    validate(z)?;
    if !threshold.is_finite() {
        return Err(Error::InvalidInput("nonfinite clustering threshold"));
    }
    let count = matches!(
        criterion,
        Criterion::MaxClust | Criterion::MaxClustMonocrit(_)
    );
    let stats = match criterion {
        Criterion::Distance | Criterion::MaxClust => maxdists(z)?,
        Criterion::Inconsistent(depth) => maxinconsts(z, &inconsistent(z, depth)?)?,
        Criterion::Monocrit(v) | Criterion::MaxClustMonocrit(v) => {
            if v.len() != z.len() {
                return Err(Error::DimensionMismatch);
            }
            if v.iter().any(|x| !x.is_finite()) || subtree_max(z, v) != v {
                return Err(Error::InvalidInput("criterion must be finite and monotone"));
            }
            v.to_vec()
        }
    };
    if !count {
        return Ok(flat(z, &stats, threshold));
    }
    if threshold < 1.0 || threshold.fract() != 0.0 {
        return Err(Error::InvalidInput(
            "cluster count must be a positive integer",
        ));
    }
    if threshold >= (z.len() + 1) as f64 {
        return Ok((1..=z.len() + 1).collect());
    }
    let mut levels = stats.clone();
    levels.sort_by(f64::total_cmp);
    levels.dedup();
    let target = threshold as usize;
    let index =
        levels.partition_point(|t| flat(z, &stats, *t).iter().copied().max().unwrap_or(0) > target);
    Ok(flat(z, &stats, levels[index]))
}

/// Clusters Euclidean observations through `pdist`, linkage, and flat clustering.
///
/// ## Parameters
///
/// `x` has observation rows; `method`, `threshold`, and `criterion` follow [`linkage`] and [`fcluster`]. SciPy non-Euclidean metrics are available by computing distances separately.
/// ## Domain
///
/// Finite rectangular observations, 2..=512 rows and 1..=512 features. Threshold and criterion follow [`fcluster`].
/// ## Returns
///
/// One one-based label per observation.
/// ## Errors
///
/// Returns a shared error for invalid linkage structure, values, counts, dimensions, or work limits.
/// ## Numerical behavior
///
/// Reuses shared spatial distances, the private linkage backend, and [`fcluster`]. Checks observation size before computing pairwise distances.
/// # Examples
///
/// ```
/// let labels=sci_cluster::hierarchy::fclusterdata(&[vec![0.0],vec![2.0]],2.0,sci_cluster::hierarchy::Method::Single,sci_cluster::hierarchy::Criterion::Distance)?;
/// assert_eq!(labels,vec![1,1]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy fclusterdata](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fclusterdata.html).
pub fn fclusterdata(
    x: &[Vec<f64>],
    threshold: f64,
    method: Method,
    criterion: Criterion<'_>,
) -> Result<Vec<usize>> {
    if !(2..=MAX_LEAVES).contains(&x.len())
        || x.first().is_none_or(|r| r.is_empty() || r.len() > 512)
    {
        return Err(Error::InvalidInput("observation work limit"));
    }
    let y = crate::spatial::distance::pdist(x, crate::spatial::distance::Metric::Euclidean)?;
    fcluster(&linkage(&y, method)?, threshold, criterion)
}

/// Finds the tree node whose descendant leaves equal each flat cluster.
///
/// ## Parameters
///
/// `labels` contains one arbitrary integer cluster ID per leaf. Every cluster must correspond to a complete subtree.
/// ## Domain
///
/// Linkage matrices follow the module convention, have 2..=512 leaves, and must contain finite nonnegative heights and exact subtree counts. NaN and infinity are rejected.
/// ## Returns
///
/// `(node_ids, cluster_ids)` ordered by ascending node ID. SciPy’s traversal ordering is adapted explicitly.
/// ## Errors
///
/// Returns a shared error for invalid linkage structure, values, counts, dimensions, or work limits.
/// ## Numerical behavior
///
/// O(n²) bounded traversal. Detects partitions which split subtrees inconsistently.
/// # Examples
///
/// ```
/// assert_eq!(sci_cluster::hierarchy::leaders(&[vec![0.0,1.0,2.0,2.0]],&[7,7])?,(vec![2],vec![7]));
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy leaders](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.leaders.html).
pub fn leaders(z: &[Vec<f64>], labels: &[usize]) -> Result<(Vec<usize>, Vec<usize>)> {
    let n = validate(z)?;
    if labels.len() != n {
        return Err(Error::DimensionMismatch);
    }
    let mut counts = HashMap::new();
    for &v in labels {
        *counts.entry(v).or_insert(0) += 1;
    }
    let mut nodes = Vec::new();
    let mut ids = Vec::new();
    for v in 0..2 * n - 1 {
        let leaves = descendants(z, v);
        let label = labels[leaves[0]];
        if leaves.len() == counts[&label] && leaves.iter().all(|i| labels[*i] == label) {
            nodes.push(v);
            ids.push(label);
        }
    }
    if nodes.len() != counts.len() {
        return Err(Error::InvalidInput("flat clusters are not subtrees"));
    }
    Ok((nodes, ids))
}

/// Requested cuts of a hierarchical tree.
#[derive(Clone, Copy, Debug)]
pub enum Cut<'a> {
    /// All n stages, from n singleton clusters through one root cluster.
    All,
    /// Exact numbers of clusters in the requested order, each in 1..=n.
    Counts(&'a [usize]),
    /// Cuts strictly below each finite height; requires monotone merge heights.
    Heights(&'a [f64]),
}

/// Returns the partitions after selected numbers of agglomerative merges.
///
/// ## Parameters
///
/// `cut` requests all stages, exact cluster counts, or strict height cuts. Counts split tied merge heights in row order; all cuts require monotonicity. Columns preserve request order; labels are zero-based and ordered by smallest original leaf.
/// ## Domain
///
/// Linkage matrices follow the module convention, have 2..=512 leaves, and must contain finite nonnegative heights and exact subtree counts. NaN and infinity are rejected.
/// ## Returns
///
/// An `n × requests` matrix of integer labels, one row per leaf.
/// ## Errors
///
/// Returns a shared error for invalid linkage structure, values, counts, dimensions, or work limits.
/// ## Numerical behavior
///
/// O(n²) work and up to 1,000,000 output labels. Exact label operations with finite height comparisons.
/// # Examples
///
/// ```
/// let labels=sci_cluster::hierarchy::cut_tree(&[vec![0.0,1.0,2.0,2.0]],sci_cluster::hierarchy::Cut::All)?;
/// assert_eq!(labels,vec![vec![0,0],vec![1,0]]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy cut_tree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.cut_tree.html).
pub fn cut_tree(z: &[Vec<f64>], cut: Cut<'_>) -> Result<Vec<Vec<usize>>> {
    let n = validate(z)?;
    if !is_monotonic(z)? {
        return Err(Error::InvalidInput("cuts require a monotone tree"));
    }
    let stages: Vec<usize> = match cut {
        Cut::All => (0..n).collect(),
        Cut::Counts(v) => {
            if v.len() > 1_000_000 / n || v.iter().any(|k| *k == 0 || *k > n) {
                return Err(Error::InvalidInput("invalid cut counts"));
            }
            v.iter().map(|k| n - k).collect()
        }
        Cut::Heights(v) => {
            if v.len() > 1_000_000 / n || v.iter().any(|h| !h.is_finite()) || !is_monotonic(z)? {
                return Err(Error::InvalidInput(
                    "height cuts require finite heights and a monotone tree",
                ));
            }
            v.iter().map(|h| z.partition_point(|r| r[2] < *h)).collect()
        }
    };
    let mut states = Vec::with_capacity(n);
    let mut labels: Vec<usize> = (0..n).collect();
    states.push(labels.clone());
    for r in z {
        let leaves = descendants(z, r[0] as usize)
            .into_iter()
            .chain(descendants(z, r[1] as usize))
            .collect::<Vec<_>>();
        let label = leaves.iter().map(|i| labels[*i]).min().unwrap();
        for i in leaves {
            labels[i] = label;
        }
        let mut unique = labels.clone();
        unique.sort_unstable();
        unique.dedup();
        for label in &mut labels {
            *label = unique.binary_search(label).unwrap();
        }
        states.push(labels.clone());
    }
    Ok((0..n)
        .map(|i| stages.iter().map(|s| states[*s][i]).collect())
        .collect())
}

/// A shared, nonrecursive tree node; child indices refer to the node vector.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterNode {
    /// Zero-based node ID; also its index in the returned vector.
    pub id: usize,
    /// Left child ID; absent for a leaf.
    pub left: Option<usize>,
    /// Right child ID; absent for a leaf.
    pub right: Option<usize>,
    /// Merge height, or zero for a leaf.
    pub distance: f64,
    /// Number of descendant leaves, including a leaf itself.
    pub count: usize,
}

/// Represents the linkage as nodes of its binary merge tree.
///
/// ## Parameters
///
/// `z` is a linkage. Adapts SciPy’s recursive objects and optional `rd` flag to one flat node vector.
/// ## Domain
///
/// Linkage matrices follow the module convention, have 2..=512 leaves, and must contain finite nonnegative heights and exact subtree counts. NaN and infinity are rejected.
/// ## Returns
///
/// All `2n−1` nodes ordered by ID. The last node is the root; leaf children are `None`.
/// ## Errors
///
/// Returns a shared error for invalid linkage structure, values, counts, dimensions, or work limits.
/// ## Numerical behavior
///
/// O(n) allocation and validation; indexed children avoid recursive allocation and destruction.
/// # Examples
///
/// ```
/// let z = vec![vec![0.0, 1.0, 2.0, 2.0]];
/// let _ = sci_cluster::hierarchy::to_tree(&z)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
///
/// [SciPy to_tree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.to_tree.html).
pub fn to_tree(z: &[Vec<f64>]) -> Result<Vec<ClusterNode>> {
    let n = validate(z)?;
    let mut nodes = (0..n)
        .map(|id| ClusterNode {
            id,
            left: None,
            right: None,
            distance: 0.0,
            count: 1,
        })
        .collect::<Vec<_>>();
    nodes.extend(z.iter().enumerate().map(|(i, r)| ClusterNode {
        id: n + i,
        left: Some(r[0] as usize),
        right: Some(r[1] as usize),
        distance: r[2],
        count: r[3] as usize,
    }));
    Ok(nodes)
}
