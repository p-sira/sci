//! Weighted graph algorithms on shared CSR adjacency matrices.
//!
//! An absent entry means no edge; an explicitly stored zero is an edge. For
//! undirected paths, asymmetric edges use the smaller available weight.
//! Graphs are bounded at 256 vertices (matching allows 256 on either side),
//! with additional per-operation work limits. Weights are dimensionless unless
//! callers supply a consistent application unit. Distances retain that unit;
//! normalized Laplacians are dimensionless. Backend graph types stay private.
use super::*;
use petgraph::algo;
use petgraph::{Directed, Graph, Undirected, graph::NodeIndex, visit::EdgeRef};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

const N: usize = 256;
type Adj = Vec<BTreeMap<usize, f64>>;
fn check(a: &SparseMatrix) -> Result<()> {
    validate(a)?;
    if a.rows != a.cols || a.rows > N {
        return Err(Error::InvalidInput(
            "graph must be square with at most 256 vertices",
        ));
    }
    Ok(())
}
fn adjacency(a: &SparseMatrix, directed: bool, unweighted: bool) -> Result<Adj> {
    check(a)?;
    let mut adj = vec![BTreeMap::<usize, f64>::new(); a.rows];
    for (r, c, v) in entries(a) {
        let v = if unweighted { 1. } else { v };
        adj[r].entry(c).and_modify(|w| *w = w.min(v)).or_insert(v);
        if !directed {
            adj[c].entry(r).and_modify(|w| *w = w.min(v)).or_insert(v);
        }
    }
    Ok(adj)
}
fn backend(adj: &Adj) -> Graph<(), f64, Directed> {
    let mut g = Graph::new();
    for _ in adj {
        g.add_node(());
    }
    for (r, row) in adj.iter().enumerate() {
        for (&c, &v) in row.iter().rev() {
            g.add_edge(NodeIndex::new(r), NodeIndex::new(c), v);
        }
    }
    g
}
fn node(i: usize, n: usize) -> Result<()> {
    if i >= n {
        Err(Error::InvalidInput("graph vertex out of bounds"))
    } else {
        Ok(())
    }
}
fn budget(factors: [usize; 3]) -> Result<()> {
    let v = factors
        .into_iter()
        .try_fold(1usize, usize::checked_mul)
        .ok_or(Error::InvalidInput("graph work count overflow"))?;
    if v > WORK {
        Err(Error::InvalidInput(
            "graph work exceeds 20,000,000 operations",
        ))
    } else {
        Ok(())
    }
}

/// Controls shortest paths; predecessor arrays are always returned.
#[derive(Clone, Debug)]
pub struct PathOptions {
    /// Follow edge orientation; false uses the minimum of opposite edges.
    pub directed: bool,
    /// Replace every stored edge weight, including zero, by one.
    pub unweighted: bool,
    /// Source vertices in output row order; None means all vertices.
    pub indices: Option<Vec<usize>>,
    /// Inclusive distance cutoff for Dijkstra; positive infinity is unbounded.
    pub limit: f64,
    /// Dijkstra only: return one row containing the nearest source distances.
    pub min_only: bool,
}
impl Default for PathOptions {
    fn default() -> Self {
        Self {
            directed: true,
            unweighted: false,
            indices: None,
            limit: f64::INFINITY,
            min_only: false,
        }
    }
}
/// Distances and predecessor vertices for shortest paths.
#[derive(Clone, Debug)]
pub struct ShortestPaths {
    /// Shape sources×vertices (one row in min-only mode); unreachable is +∞.
    pub distances: Matrix,
    /// Same shape; None for source, unreachable, or cutoff-excluded vertices.
    pub predecessors: Vec<Vec<Option<usize>>>,
    /// Requested source vertices, in input order.
    pub sources: Vec<usize>,
    /// In min-only mode, the nearest source for each vertex, or None.
    pub nearest_sources: Option<Vec<Option<usize>>>,
}
#[derive(Clone, Copy)]
enum Method {
    Auto,
    Dijkstra,
    Bellman,
    Floyd,
    Johnson,
}
fn paths(a: &SparseMatrix, o: &PathOptions, method: Method) -> Result<ShortestPaths> {
    let adj = adjacency(a, o.directed, o.unweighted)?;
    let n = adj.len();
    if o.limit.is_nan() || o.limit < 0. {
        return Err(Error::InvalidInput("distance limit must be nonnegative"));
    }
    let sources = o.indices.clone().unwrap_or_else(|| (0..n).collect());
    if sources.len() > N {
        return Err(Error::InvalidInput("too many source vertices"));
    }
    for &s in &sources {
        node(s, n)?;
    }
    let negative = adj.iter().any(|r| r.values().any(|&w| w < 0.));
    let method = match method {
        Method::Auto => {
            if negative {
                Method::Bellman
            } else {
                Method::Dijkstra
            }
        }
        m => m,
    };
    if !matches!(method, Method::Dijkstra) && (o.min_only || o.limit != f64::INFINITY) {
        return Err(Error::InvalidInput("limit/min_only are Dijkstra options"));
    }
    if matches!(method, Method::Dijkstra) && negative {
        return Err(Error::InvalidInput(
            "Dijkstra requires nonnegative edge weights",
        ));
    }
    if matches!(method, Method::Bellman) {
        budget([
            sources.len(),
            n,
            adj.iter().map(BTreeMap::len).sum::<usize>(),
        ])?;
    }
    if adj.iter().any(|r| {
        r.values()
            .any(|w| w.abs() > f64::MAX / (4. * n as f64 + 1.))
    }) {
        return Err(Error::InvalidInput(
            "edge weights exceed shortest-path arithmetic bound",
        ));
    }
    let g = backend(&adj);
    let mut dist = vec![];
    let mut pred = vec![];
    let all = if matches!(method, Method::Floyd) {
        Some(
            algo::floyd_warshall(&g, |e| *e.weight())
                .map_err(|_| Error::NumericalFailure("negative graph cycle"))?,
        )
    } else if matches!(method, Method::Johnson) {
        Some(
            algo::johnson(&g, |e| *e.weight())
                .map_err(|_| Error::NumericalFailure("negative graph cycle"))?,
        )
    } else {
        None
    };
    for &s in &sources {
        let mut d = vec![f64::INFINITY; n];
        if let Some(all) = &all {
            for (j, v) in d.iter_mut().enumerate() {
                let w = all
                    .get(&(NodeIndex::new(s), NodeIndex::new(j)))
                    .copied()
                    .unwrap_or(f64::MAX);
                if w != f64::MAX {
                    *v = w;
                }
            }
        } else if matches!(method, Method::Bellman) {
            d = algo::bellman_ford(&g, NodeIndex::new(s))
                .map_err(|_| Error::NumericalFailure("negative graph cycle reachable from source"))?
                .distances;
        } else {
            for (j, v) in algo::dijkstra(&g, NodeIndex::new(s), None, |e| *e.weight()) {
                d[j.index()] = v;
            }
        }
        // Build an acyclic tree on tight edges, including zero-cost cycles.
        let mut p = vec![None; n];
        let mut seen = vec![false; n];
        let mut queue = VecDeque::from([s]);
        seen[s] = true;
        while let Some(u) = queue.pop_front() {
            for (&v, &w) in &adj[u] {
                let candidate = d[u] + w;
                if !candidate.is_finite() && d[u].is_finite() {
                    return Err(Error::NonFiniteEvaluation);
                }
                let tol = 8. * f64::EPSILON * (candidate.abs() + d[v].abs());
                if !seen[v] && d[v].is_finite() && (candidate - d[v]).abs() <= tol {
                    seen[v] = true;
                    p[v] = Some(u);
                    queue.push_back(v);
                }
            }
        }
        for j in 0..n {
            if d[j] > o.limit {
                d[j] = f64::INFINITY;
                p[j] = None;
            }
        }
        dist.push(d);
        pred.push(p);
    }
    let nearest_sources = if o.min_only {
        let mut d = vec![f64::INFINITY; n];
        let mut p = vec![None; n];
        let mut roots = vec![None; n];
        for (i, &s) in sources.iter().enumerate() {
            for j in 0..n {
                if dist[i][j] < d[j] {
                    d[j] = dist[i][j];
                    p[j] = pred[i][j];
                    roots[j] = Some(s);
                }
            }
        }
        dist = vec![d];
        pred = vec![p];
        Some(roots)
    } else {
        None
    };
    Ok(ShortestPaths {
        distances: dist,
        predecessors: pred,
        sources,
        nearest_sources,
    })
}

/// Compute `d(s,v) = min_path Σ edge_weight` using automatic method selection.
///
/// # Parameters and conventions
/// `a` is adjacency CSR, with stored zeros representing edges. `options`
/// controls direction, unweighted mode and source/output ordering. All methods
/// return predecessor trees; ties may choose different valid paths than SciPy.
/// `limit` and `min_only` are supported by Dijkstra and automatic nonnegative paths.
/// # Domain and special values
/// Finite graphs of order 0..256, with |weight| <= f64::MAX/(4n+1)
/// to bound backend path arithmetic. Unreachable distances are positive infinity;
/// source distances are zero and source predecessors are None.
/// # Returns
/// [`ShortestPaths`] with source-major rows and zero-based predecessors.
/// # Errors
/// Invalid graph/options, negative cycles, nonfinite arithmetic, or work limits.
/// Dijkstra rejects negative edges; Bellman–Ford bounds source×vertices×edges
/// at 20,000,000. Floyd–Warshall and Johnson detect cycles anywhere in the graph.
/// # Numerical behavior
/// Private petgraph kernels; O(n³) worst-case dense work for all-pairs paths.
/// Floating cancellation can affect path ties; exact output ordering is stable
/// within a fixed backend version, but is not a cross-backend contract.
/// # Examples
/// ```
/// let g=sci_sparse::from_coo(2,2,&[(0,1,3.)])?;
/// let p=sci_sparse::csgraph::shortest_path(&g, &Default::default())?;
/// assert_eq!(p.distances[0][1],3.);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy shortest_path](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.shortest_path.html).
pub fn shortest_path(a: &SparseMatrix, options: &PathOptions) -> Result<ShortestPaths> {
    paths(a, options, Method::Auto)
}

/// Compute shortest paths on nonnegative weighted graphs using Dijkstra's algorithm.
///
/// # Parameters and conventions
/// `a` is adjacency CSR, with stored zeros representing edges. `options`
/// controls direction, unweighted mode and source/output ordering. All methods
/// return predecessor trees; ties may choose different valid paths than SciPy.
/// `limit` and `min_only` are supported by Dijkstra and automatic nonnegative paths.
/// # Domain and special values
/// Finite graphs of order 0..256, with |weight| <= f64::MAX/(4n+1)
/// to bound backend path arithmetic. Unreachable distances are positive infinity;
/// source distances are zero and source predecessors are None.
/// # Returns
/// [`ShortestPaths`] with source-major rows and zero-based predecessors.
/// # Errors
/// Invalid graph/options, negative cycles, nonfinite arithmetic, or work limits.
/// Dijkstra rejects negative edges; Bellman–Ford bounds source×vertices×edges
/// at 20,000,000. Floyd–Warshall and Johnson detect cycles anywhere in the graph.
/// # Numerical behavior
/// Private petgraph kernels; O(n³) worst-case dense work for all-pairs paths.
/// Floating cancellation can affect path ties; exact output ordering is stable
/// within a fixed backend version, but is not a cross-backend contract.
/// # Examples
/// ```
/// let g=sci_sparse::from_coo(2,2,&[(0,1,3.)])?;
/// let p=sci_sparse::csgraph::dijkstra(&g, &Default::default())?;
/// assert_eq!(p.distances[0][1],3.);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy dijkstra](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.dijkstra.html).
pub fn dijkstra(a: &SparseMatrix, options: &PathOptions) -> Result<ShortestPaths> {
    paths(a, options, Method::Dijkstra)
}

/// Compute shortest paths with negative edges using Bellman–Ford relaxation.
///
/// # Parameters and conventions
/// `a` is adjacency CSR, with stored zeros representing edges. `options`
/// controls direction, unweighted mode and source/output ordering. All methods
/// return predecessor trees; ties may choose different valid paths than SciPy.
/// `limit` and `min_only` are supported by Dijkstra and automatic nonnegative paths.
/// # Domain and special values
/// Finite graphs of order 0..256, with |weight| <= f64::MAX/(4n+1)
/// to bound backend path arithmetic. Unreachable distances are positive infinity;
/// source distances are zero and source predecessors are None.
/// # Returns
/// [`ShortestPaths`] with source-major rows and zero-based predecessors.
/// # Errors
/// Invalid graph/options, negative cycles, nonfinite arithmetic, or work limits.
/// Dijkstra rejects negative edges; Bellman–Ford bounds source×vertices×edges
/// at 20,000,000. Floyd–Warshall and Johnson detect cycles anywhere in the graph.
/// # Numerical behavior
/// Private petgraph kernels; O(n³) worst-case dense work for all-pairs paths.
/// Floating cancellation can affect path ties; exact output ordering is stable
/// within a fixed backend version, but is not a cross-backend contract.
/// # Examples
/// ```
/// let g=sci_sparse::from_coo(2,2,&[(0,1,3.)])?;
/// let p=sci_sparse::csgraph::bellman_ford(&g, &Default::default())?;
/// assert_eq!(p.distances[0][1],3.);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy bellman_ford](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.bellman_ford.html).
pub fn bellman_ford(a: &SparseMatrix, options: &PathOptions) -> Result<ShortestPaths> {
    paths(a, options, Method::Bellman)
}

/// Compute all-pairs shortest paths by Floyd–Warshall dynamic programming.
///
/// # Parameters and conventions
/// `a` is adjacency CSR, with stored zeros representing edges. `options`
/// controls direction, unweighted mode and source/output ordering. All methods
/// return predecessor trees; ties may choose different valid paths than SciPy.
/// `limit` and `min_only` are supported by Dijkstra and automatic nonnegative paths.
/// # Domain and special values
/// Finite graphs of order 0..256, with |weight| <= f64::MAX/(4n+1)
/// to bound backend path arithmetic. Unreachable distances are positive infinity;
/// source distances are zero and source predecessors are None.
/// # Returns
/// [`ShortestPaths`] with source-major rows and zero-based predecessors.
/// # Errors
/// Invalid graph/options, negative cycles, nonfinite arithmetic, or work limits.
/// Dijkstra rejects negative edges; Bellman–Ford bounds source×vertices×edges
/// at 20,000,000. Floyd–Warshall and Johnson detect cycles anywhere in the graph.
/// # Numerical behavior
/// Private petgraph kernels; O(n³) worst-case dense work for all-pairs paths.
/// Floating cancellation can affect path ties; exact output ordering is stable
/// within a fixed backend version, but is not a cross-backend contract.
/// # Examples
/// ```
/// let g=sci_sparse::from_coo(2,2,&[(0,1,3.)])?;
/// let p=sci_sparse::csgraph::floyd_warshall(&g, &Default::default())?;
/// assert_eq!(p.distances[0][1],3.);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy floyd_warshall](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.floyd_warshall.html).
pub fn floyd_warshall(a: &SparseMatrix, options: &PathOptions) -> Result<ShortestPaths> {
    paths(a, options, Method::Floyd)
}

/// Compute shortest paths with Johnson reweighting and repeated Dijkstra searches.
///
/// # Parameters and conventions
/// `a` is adjacency CSR, with stored zeros representing edges. `options`
/// controls direction, unweighted mode and source/output ordering. All methods
/// return predecessor trees; ties may choose different valid paths than SciPy.
/// `limit` and `min_only` are supported by Dijkstra and automatic nonnegative paths.
/// # Domain and special values
/// Finite graphs of order 0..256, with |weight| <= f64::MAX/(4n+1)
/// to bound backend path arithmetic. Unreachable distances are positive infinity;
/// source distances are zero and source predecessors are None.
/// # Returns
/// [`ShortestPaths`] with source-major rows and zero-based predecessors.
/// # Errors
/// Invalid graph/options, negative cycles, nonfinite arithmetic, or work limits.
/// Dijkstra rejects negative edges; Bellman–Ford bounds source×vertices×edges
/// at 20,000,000. Floyd–Warshall and Johnson detect cycles anywhere in the graph.
/// # Numerical behavior
/// Private petgraph kernels; O(n³) worst-case dense work for all-pairs paths.
/// Floating cancellation can affect path ties; exact output ordering is stable
/// within a fixed backend version, but is not a cross-backend contract.
/// # Examples
/// ```
/// let g=sci_sparse::from_coo(2,2,&[(0,1,3.)])?;
/// let p=sci_sparse::csgraph::johnson(&g, &Default::default())?;
/// assert_eq!(p.distances[0][1],3.);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy johnson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.johnson.html).
pub fn johnson(a: &SparseMatrix, options: &PathOptions) -> Result<ShortestPaths> {
    paths(a, options, Method::Johnson)
}

/// Connected-component labels of a graph.
#[derive(Clone, Debug)]
pub struct Components {
    /// Number of components, including isolated vertices.
    pub count: usize,
    /// One label per vertex; labels enumerate components by smallest vertex.
    pub labels: Vec<usize>,
}
/// Search visitation order and its predecessor tree.
#[derive(Clone, Debug)]
pub struct Traversal {
    /// Reachable vertices in discovery order.
    pub order: Vec<usize>,
    /// One predecessor per input vertex; None for root or unreachable vertices.
    pub predecessors: Vec<Option<usize>>,
}
fn traverse(a: &SparseMatrix, start: usize, directed: bool, depth: bool) -> Result<Traversal> {
    let adj = adjacency(a, directed, false)?;
    node(start, a.rows)?;
    let mut seen = vec![false; a.rows];
    let mut predecessors = vec![None; a.rows];
    let mut order = vec![start];
    seen[start] = true;
    if depth {
        // Stack frames preserve recursive ascending-neighbor discovery order.
        let mut stack = vec![(start, adj[start].keys().copied().collect::<Vec<_>>(), 0)];
        while let Some((u, neighbors, pos)) = stack.last_mut() {
            if *pos == neighbors.len() {
                stack.pop();
                continue;
            }
            let v = neighbors[*pos];
            *pos += 1;
            if !seen[v] {
                seen[v] = true;
                predecessors[v] = Some(*u);
                order.push(v);
                stack.push((v, adj[v].keys().copied().collect(), 0));
            }
        }
    } else {
        let mut q = VecDeque::from([start]);
        while let Some(u) = q.pop_front() {
            for &v in adj[u].keys() {
                if !seen[v] {
                    seen[v] = true;
                    predecessors[v] = Some(u);
                    order.push(v);
                    q.push_back(v);
                }
            }
        }
    }
    Ok(Traversal {
        order,
        predecessors,
    })
}
fn tree(a: &SparseMatrix, pred: &[Option<usize>], directed: bool) -> Result<SparseMatrix> {
    let adj = adjacency(a, directed, false)?;
    if pred.len() != a.rows {
        return Err(Error::DimensionMismatch);
    }
    let mut triples = vec![];
    for (v, &p) in pred.iter().enumerate() {
        if let Some(u) = p {
            node(u, a.rows)?;
            if u == v {
                return Err(Error::InvalidInput("self predecessor"));
            }
            let &w = adj[u]
                .get(&v)
                .ok_or(Error::InvalidInput("predecessor does not describe an edge"))?;
            triples.push((u, v, w));
        }
    }
    // Reject predecessor cycles rather than returning a non-tree graph.
    for v in 0..a.rows {
        let mut u = v;
        for step in 0..=a.rows {
            match pred[u] {
                None => break,
                Some(p) => {
                    if step == a.rows {
                        return Err(Error::InvalidInput("cyclic predecessors"));
                    }
                    u = p;
                }
            }
        }
    }
    from_coo(a.rows, a.cols, &triples)
}
fn components(a: &SparseMatrix, directed: bool, strong: bool) -> Result<Components> {
    let adj = adjacency(a, directed && strong, false)?;
    let mut groups = algo::kosaraju_scc(&backend(&adj));
    groups.sort_by_key(|c| c.iter().map(|i| i.index()).min().unwrap_or(0));
    let mut labels = vec![0; a.rows];
    for (label, c) in groups.iter().enumerate() {
        for v in c {
            labels[v.index()] = label;
        }
    }
    Ok(Components {
        count: groups.len(),
        labels,
    })
}
fn mst(a: &SparseMatrix) -> Result<SparseMatrix> {
    let cleaned = eliminate_zeros(a)?;
    let adj = adjacency(&cleaned, false, false)?;
    let mut g = Graph::<(), f64, Undirected>::new_undirected();
    for _ in &adj {
        g.add_node(());
    }
    for (r, row) in adj.iter().enumerate() {
        for (&c, &v) in row {
            if r < c && v != 0. {
                g.add_edge(NodeIndex::new(r), NodeIndex::new(c), v);
            }
        }
    }
    use petgraph::data::FromElements;
    let t = Graph::<(), f64, Undirected>::from_elements(algo::min_spanning_tree(&g));
    from_coo(
        a.rows,
        a.cols,
        &t.edge_references()
            .map(|e| (e.source().index(), e.target().index(), *e.weight()))
            .collect::<Vec<_>>(),
    )
}
fn rcm(a: &SparseMatrix, symmetric: bool) -> Result<Vec<usize>> {
    let adj = adjacency(a, symmetric, false)?;
    let degree: Vec<_> = adj
        .iter()
        .enumerate()
        .map(|(i, r)| r.len() + usize::from(r.contains_key(&i)))
        .collect();
    let mut seen = vec![false; a.rows];
    let mut order = vec![];
    while let Some(start) = (0..a.rows)
        .filter(|&i| !seen[i])
        .min_by_key(|&i| (degree[i], i))
    {
        seen[start] = true;
        let mut q = VecDeque::from([start]);
        while let Some(u) = q.pop_front() {
            order.push(u);
            let mut ns: Vec<_> = adj[u].keys().copied().filter(|&v| !seen[v]).collect();
            ns.sort_by_key(|&v| (degree[v], v));
            for v in ns {
                if !seen[v] {
                    seen[v] = true;
                    q.push_back(v);
                }
            }
        }
    }
    order.reverse();
    Ok(order)
}
fn matching(a: &SparseMatrix, rows: bool) -> Result<Vec<Option<usize>>> {
    validate(a)?;
    if a.rows > N || a.cols > N {
        return Err(Error::InvalidInput("matching axis exceeds 256"));
    }
    let mut g = Graph::<(), (), Undirected>::new_undirected();
    for _ in 0..a.rows + a.cols {
        g.add_node(());
    }
    for (r, c, _) in entries(a) {
        g.add_edge(NodeIndex::new(r), NodeIndex::new(a.rows + c), ());
    }
    let m = algo::maximum_matching(&g);
    Ok(if rows {
        (0..a.cols)
            .map(|c| m.mate(NodeIndex::new(a.rows + c)).map(|r| r.index()))
            .collect()
    } else {
        (0..a.rows)
            .map(|r| m.mate(NodeIndex::new(r)).map(|c| c.index() - a.rows))
            .collect()
    })
}
fn weighted_matching(a: &SparseMatrix, maximize: bool) -> Result<(Vec<usize>, Vec<usize>)> {
    validate(a)?;
    let n = a.rows.max(a.cols);
    if n > N {
        return Err(Error::InvalidInput("matching axis exceeds 256"));
    }
    if a.rows == 0 || a.cols == 0 {
        return Ok((vec![], vec![]));
    }
    let scale = a.data.iter().fold(0f64, |s, v| s.max(v.abs())).max(1.);
    let mut costs = ndarray::Array2::from_shape_fn((n, n), |(r, c)| {
        if r >= a.rows || c >= a.cols {
            0.
        } else {
            2. * n as f64 + 1.
        }
    });
    for (r, c, v) in entries(a) {
        if v != 0. {
            costs[(r, c)] = if maximize { -v / scale } else { v / scale };
        }
    }
    let (assignment, _) =
        lapjv::lapjv(&costs).map_err(|_| Error::NumericalFailure("weighted matching failed"))?;
    let mut rows = vec![];
    let mut cols = vec![];
    for (r, &c) in assignment.iter().take(a.rows).enumerate() {
        if c < a.cols {
            if costs[(r, c)] > 2. {
                return Err(Error::NumericalFailure("no full bipartite matching"));
            }
            rows.push(r);
            cols.push(c);
        }
    }
    if rows.len() != a.rows.min(a.cols) {
        return Err(Error::NumericalFailure("no full bipartite matching"));
    }
    Ok((rows, cols))
}

/// Laplacian matrix and the diagonal normalization factors.
#[derive(Clone, Debug)]
pub struct Laplacian {
    /// `D - A` or `D^(-1/2) (D-A) D^(-1/2)`; self edges are ignored.
    pub matrix: SparseMatrix,
    /// Degrees for unnormalized output; square roots of degrees otherwise.
    /// In normalized mode isolated-vertex factors are one.
    pub diagonal: Vec<f64>,
}
fn lap(a: &SparseMatrix, normed: bool, out: bool, symmetrized: bool) -> Result<Laplacian> {
    check(a)?;
    let mut weights = vec![vec![0.; a.rows]; a.rows];
    for (r, c, v) in entries(a) {
        if r != c {
            weights[r][c] = v;
        }
    }
    if symmetrized {
        for r in 0..a.rows {
            let (head, tail) = weights.split_at_mut(r + 1);
            for (forward, reverse) in head[r][r + 1..].iter_mut().zip(tail) {
                let v = match *forward + reverse[r] {
                    value if value.is_finite() => Ok(value),
                    _ => Err(crate::Error::NonFiniteEvaluation),
                }?;
                *forward = v;
                reverse[r] = v;
            }
        }
    }
    let mut d = vec![0.; a.rows];
    for (r, row) in weights.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            let i = if out { r } else { c };
            d[i] = match d[i] + v {
                value if value.is_finite() => Ok(value),
                _ => Err(crate::Error::NonFiniteEvaluation),
            }?;
        }
    }
    let diagonal = if normed {
        d.iter()
            .map(|&v| {
                if v < 0. {
                    Err(Error::InvalidInput(
                        "normalized Laplacian requires nonnegative degrees",
                    ))
                } else {
                    Ok(if v == 0. { 1. } else { v.sqrt() })
                }
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        d.clone()
    };
    let mut triples = vec![];
    for r in 0..a.rows {
        for c in 0..a.rows {
            let v = if r == c {
                if normed { f64::from(d[r] != 0.) } else { d[r] }
            } else if normed {
                -weights[r][c] / diagonal[r] / diagonal[c]
            } else {
                -weights[r][c]
            };
            if v != 0. {
                triples.push((
                    r,
                    c,
                    match v {
                        value if value.is_finite() => Ok(value),
                        _ => Err(crate::Error::NonFiniteEvaluation),
                    }?,
                ));
            }
        }
    }
    Ok(Laplacian {
        matrix: from_coo(a.rows, a.cols, &triples)?,
        diagonal,
    })
}
/// Integral maximum flow, represented with an antisymmetric flow matrix.
#[derive(Clone, Debug)]
pub struct MaximumFlow {
    /// Maximum total flow from source to sink, exactly representable as f64.
    pub value: f64,
    /// Net edge flows; `flow[u,v] = -flow[v,u]`.
    pub flow: SparseMatrix,
}
fn flow(a: &SparseMatrix, source: usize, sink: usize) -> Result<MaximumFlow> {
    check(a)?;
    node(source, a.rows)?;
    node(sink, a.rows)?;
    if source == sink {
        return Err(Error::InvalidInput("source equals sink"));
    }
    let mut g = Graph::<(), u64, Directed>::new();
    for _ in 0..a.rows {
        g.add_node(());
    }
    let mut total = 0.;
    for (r, c, v) in entries(a) {
        if v < 0. || v.fract() != 0. {
            return Err(Error::InvalidInput(
                "flow capacities must be nonnegative integers",
            ));
        }
        total += v;
        if total > 9_007_199_254_740_991. {
            return Err(Error::InvalidInput(
                "capacity sum exceeds exact f64 integer range",
            ));
        }
        g.add_edge(NodeIndex::new(r), NodeIndex::new(c), v as u64);
    }
    let (value, flows) = algo::dinics(&g, NodeIndex::new(source), NodeIndex::new(sink));
    let mut net = BTreeMap::<(usize, usize), f64>::new();
    for e in g.edge_references() {
        let f = flows[e.id().index()] as f64;
        *net.entry((e.source().index(), e.target().index()))
            .or_default() += f;
        *net.entry((e.target().index(), e.source().index()))
            .or_default() -= f;
    }
    Ok(MaximumFlow {
        value: value as f64,
        flow: from_coo(
            a.rows,
            a.cols,
            &net.into_iter()
                .filter(|(_, v)| *v != 0.)
                .map(|((r, c), v)| (r, c, v))
                .collect::<Vec<_>>(),
        )?,
    })
}

/// Dense graph with an explicit non-edge mask.
#[derive(Clone, Debug, PartialEq)]
pub struct MaskedGraph {
    /// Square row-major values; masked entries need not be finite.
    pub values: Matrix,
    /// `true` means no edge; same shape as values.
    pub mask: Vec<Vec<bool>>,
}
fn dense_shape(values: &Matrix) -> Result<usize> {
    let n = values.len();
    if n > N || values.iter().any(|r| r.len() != n) {
        return Err(Error::InvalidInput(
            "dense graph must be square with order <=256",
        ));
    }
    Ok(n)
}
fn from_masked(a: &MaskedGraph) -> Result<SparseMatrix> {
    let n = dense_shape(&a.values)?;
    if a.mask.len() != n || a.mask.iter().any(|r| r.len() != n) {
        return Err(Error::DimensionMismatch);
    }
    let mut t = vec![];
    for r in 0..n {
        for c in 0..n {
            if !a.mask[r][c] {
                t.push((
                    r,
                    c,
                    match a.values[r][c] {
                        value if value.is_finite() => Ok(value),
                        _ => Err(crate::Error::NonFiniteEvaluation),
                    }?,
                ));
            }
        }
    }
    from_coo(n, n, &t)
}
fn masked_dense(a: &Matrix, null: f64, nan_null: bool, infinity_null: bool) -> Result<MaskedGraph> {
    dense_shape(a)?;
    let mask = a
        .iter()
        .map(|row| {
            row.iter()
                .map(|&v| {
                    v == null
                        || (v.is_nan() && (nan_null || null.is_nan()))
                        || (v.is_infinite() && infinity_null)
                })
                .collect()
        })
        .collect();
    Ok(MaskedGraph {
        values: a.clone(),
        mask,
    })
}
fn masked(a: &SparseMatrix) -> Result<MaskedGraph> {
    check(a)?;
    let mut values = vec![vec![0.; a.cols]; a.rows];
    let mut mask = vec![vec![true; a.cols]; a.rows];
    for (r, c, v) in entries(a) {
        values[r][c] = v;
        mask[r][c] = false;
    }
    Ok(MaskedGraph { values, mask })
}
fn graph_dense(a: &SparseMatrix, null: f64) -> Result<Matrix> {
    check(a)?;
    let mut v = vec![vec![null; a.cols]; a.rows];
    for (r, c, w) in entries(a) {
        v[r][c] = w;
    }
    Ok(v)
}
fn distances_from_pred(
    a: &SparseMatrix,
    pred: &[Vec<Option<usize>>],
    directed: bool,
    null: f64,
) -> Result<Matrix> {
    let adj = adjacency(a, directed, false)?;
    if pred.len() != a.rows || pred.iter().any(|p| p.len() != a.rows) {
        return Err(Error::DimensionMismatch);
    }
    let mut d = vec![vec![null; a.rows]; a.rows];
    for s in 0..a.rows {
        d[s][s] = 0.;
        tree(a, &pred[s], directed)?;
        for v in 0..a.rows {
            if v == s || pred[s][v].is_none() {
                continue;
            }
            let mut sum = 0.;
            let mut u = v;
            let mut steps = 0;
            while u != s {
                let p = pred[s][u].ok_or(Error::InvalidInput(
                    "predecessor chain does not reach source",
                ))?;
                sum = match sum + adj[p][&u] {
                    value if value.is_finite() => Ok(value),
                    _ => Err(crate::Error::NonFiniteEvaluation),
                }?;
                u = p;
                steps += 1;
                if steps > a.rows {
                    return Err(Error::InvalidInput("cyclic predecessors"));
                }
            }
            d[s][v] = sum;
        }
    }
    Ok(d)
}

/// Simple paths and their lengths, in nondecreasing length order.
#[derive(Clone, Debug)]
pub struct KShortestPaths {
    /// Total weight (or edge count in unweighted mode) for each path.
    pub distances: Vec<f64>,
    /// Full zero-based vertex sequences, each without repeated vertices.
    pub paths: Vec<Vec<usize>>,
    /// One predecessor row per path, None outside the path and at its root.
    pub predecessors: Vec<Vec<Option<usize>>>,
}
fn shortest_single(
    adj: &Adj,
    source: usize,
    sink: usize,
    banned_nodes: &BTreeSet<usize>,
    banned_edges: &BTreeSet<(usize, usize)>,
) -> Option<(f64, Vec<usize>)> {
    let mut g = Graph::<(), f64, Directed>::new();
    for _ in adj {
        g.add_node(());
    }
    for (r, row) in adj.iter().enumerate() {
        if banned_nodes.contains(&r) {
            continue;
        }
        for (&c, &w) in row {
            if !banned_nodes.contains(&c) && !banned_edges.contains(&(r, c)) {
                g.add_edge(NodeIndex::new(r), NodeIndex::new(c), w);
            }
        }
    }
    algo::astar(
        &g,
        NodeIndex::new(source),
        |n| n.index() == sink,
        |e| *e.weight(),
        |_| 0.,
    )
    .map(|(w, p)| (w, p.into_iter().map(|i| i.index()).collect()))
}
fn k_paths(
    a: &SparseMatrix,
    source: usize,
    sink: usize,
    k: usize,
    directed: bool,
    unweighted: bool,
) -> Result<KShortestPaths> {
    let adj = adjacency(a, directed, unweighted)?;
    node(source, a.rows)?;
    node(sink, a.rows)?;
    if k > 64 {
        return Err(Error::InvalidInput("Yen path count exceeds 64"));
    }
    if adj.iter().any(|r| {
        r.values()
            .any(|w| w.abs() > f64::MAX / (4. * a.rows as f64 + 1.))
    }) {
        return Err(Error::InvalidInput(
            "edge weights exceed path arithmetic bound",
        ));
    }
    if adj.iter().any(|r| r.values().any(|&w| w < 0.)) {
        return Err(Error::InvalidInput("Yen requires nonnegative weights"));
    }
    budget([k, a.rows, adj.iter().map(BTreeMap::len).sum::<usize>()])?;
    let mut result = KShortestPaths {
        distances: vec![],
        paths: vec![],
        predecessors: vec![],
    };
    if k == 0 {
        return Ok(result);
    }
    if let Some((w, p)) = shortest_single(&adj, source, sink, &BTreeSet::new(), &BTreeSet::new()) {
        result.distances.push(match w {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        }?);
        result.paths.push(p);
    } else {
        return Ok(result);
    }
    let mut candidates: Vec<(f64, Vec<usize>)> = vec![];
    while result.paths.len() < k {
        let last = result
            .paths
            .last()
            .ok_or(Error::NumericalFailure("missing Yen path"))?;
        let mut root_weight = 0.;
        for spur in 0..last.len().saturating_sub(1) {
            let root = &last[..=spur];
            let banned_nodes = last[..spur].iter().copied().collect();
            let mut banned_edges = BTreeSet::new();
            for p in &result.paths {
                if p.len() > spur + 1 && p[..=spur] == *root {
                    banned_edges.insert((p[spur], p[spur + 1]));
                }
            }
            if let Some((weight, path)) =
                shortest_single(&adj, last[spur], sink, &banned_nodes, &banned_edges)
            {
                let mut joined = last[..spur].to_vec();
                joined.extend(path);
                if !result.paths.contains(&joined) && !candidates.iter().any(|(_, p)| p == &joined)
                {
                    candidates.push((
                        match root_weight + weight {
                            value if value.is_finite() => Ok(value),
                            _ => Err(crate::Error::NonFiniteEvaluation),
                        }?,
                        joined,
                    ));
                }
            }
            root_weight = match root_weight + adj[last[spur]][&last[spur + 1]] {
                value if value.is_finite() => Ok(value),
                _ => Err(crate::Error::NonFiniteEvaluation),
            }?;
        }
        if candidates.is_empty() {
            break;
        }
        let i = (0..candidates.len())
            .min_by(|&i, &j| {
                candidates[i]
                    .0
                    .total_cmp(&candidates[j].0)
                    .then_with(|| candidates[i].1.cmp(&candidates[j].1))
            })
            .ok_or(Error::NumericalFailure("missing Yen candidate"))?;
        let (w, p) = candidates.swap_remove(i);
        result.distances.push(match w {
            value if value.is_finite() => Ok(value),
            _ => Err(crate::Error::NonFiniteEvaluation),
        }?);
        result.paths.push(p);
    }
    result.predecessors = result
        .paths
        .iter()
        .map(|p| {
            let mut row = vec![None; a.rows];
            for e in p.windows(2) {
                row[e[1]] = Some(e[0]);
            }
            row
        })
        .collect();
    Ok(result)
}

/// Partition vertices into components; `u ~ v` means mutual reachability (strong) or connectivity after forgetting direction (weak).
///
/// # Parameters and conventions
/// `strong` selects strong versus weak connection; ignored for undirected graphs. Labels are normalized by smallest vertex; SciPy may number strong components differently.
/// # Domain and special values
/// Stored edges, including zeros, define connectivity; weights are ignored.
/// Graph inputs are square, order 0..256, unless rectangular matching is specified.
/// # Returns
/// Owned output with zero-based indices and the ordering described above.
/// # Errors
/// Invalid storage, incompatible shapes/indices, violated domain/work limits,
/// missing full matchings, or nonfinite numerical arithmetic.
/// # Numerical behavior
/// At most 256 vertices and 65,536 dense cells. Traversals and conversion use linear storage scans; path validation is at most cubic. Floating sums can round or overflow.
/// # Examples
/// ```
/// let a=sci_sparse::from_coo(2,2,&[(0,1,3.)])?;
/// let _ = sci_sparse::csgraph::connected_components(&a,true,false)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy connected_components](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.connected_components.html).
pub fn connected_components(a: &SparseMatrix, directed: bool, strong: bool) -> Result<Components> {
    components(a, directed, strong)
}

/// Visit vertices in breadth-first discovery order from `start`.
///
/// # Parameters and conventions
/// Neighbors are visited in ascending vertex order. Predecessors are always returned.
/// # Domain and special values
/// Stored zeros are edges; only vertices reachable from start appear in the order.
/// Graph inputs are square, order 0..256, unless rectangular matching is specified.
/// # Returns
/// Owned output with zero-based indices and the ordering described above.
/// # Errors
/// Invalid storage, incompatible shapes/indices, violated domain/work limits,
/// missing full matchings, or nonfinite numerical arithmetic.
/// # Numerical behavior
/// At most 256 vertices and 65,536 dense cells. Traversals and conversion use linear storage scans; path validation is at most cubic. Floating sums can round or overflow.
/// # Examples
/// ```
/// let a=sci_sparse::from_coo(2,2,&[(0,1,3.)])?;
/// let _ = sci_sparse::csgraph::breadth_first_order(&a,0,true)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy breadth_first_order](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.breadth_first_order.html).
pub fn breadth_first_order(a: &SparseMatrix, start: usize, directed: bool) -> Result<Traversal> {
    traverse(a, start, directed, false)
}

/// Visit vertices in depth-first discovery order from `start`.
///
/// # Parameters and conventions
/// Neighbors are visited in ascending vertex order using an explicit stack.
/// # Domain and special values
/// Stored zeros are edges; only vertices reachable from start appear in the order.
/// Graph inputs are square, order 0..256, unless rectangular matching is specified.
/// # Returns
/// Owned output with zero-based indices and the ordering described above.
/// # Errors
/// Invalid storage, incompatible shapes/indices, violated domain/work limits,
/// missing full matchings, or nonfinite numerical arithmetic.
/// # Numerical behavior
/// At most 256 vertices and 65,536 dense cells. Traversals and conversion use linear storage scans; path validation is at most cubic. Floating sums can round or overflow.
/// # Examples
/// ```
/// let a=sci_sparse::from_coo(2,2,&[(0,1,3.)])?;
/// let _ = sci_sparse::csgraph::depth_first_order(&a,0,true)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy depth_first_order](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.depth_first_order.html).
pub fn depth_first_order(a: &SparseMatrix, start: usize, directed: bool) -> Result<Traversal> {
    traverse(a, start, directed, true)
}

/// Return the weighted breadth-first predecessor tree rooted at `start`.
///
/// # Parameters and conventions
/// The result has the input shape and stores original weights on parent-to-child edges.
/// # Domain and special values
/// Zero-weight tree edges are retained; unreachable vertices have no tree edges.
/// Graph inputs are square, order 0..256, unless rectangular matching is specified.
/// # Returns
/// Owned output with zero-based indices and the ordering described above.
/// # Errors
/// Invalid storage, incompatible shapes/indices, violated domain/work limits,
/// missing full matchings, or nonfinite numerical arithmetic.
/// # Numerical behavior
/// At most 256 vertices and 65,536 dense cells. Traversals and conversion use linear storage scans; path validation is at most cubic. Floating sums can round or overflow.
/// # Examples
/// ```
/// let a=sci_sparse::from_coo(2,2,&[(0,1,3.)])?;
/// let _ = sci_sparse::csgraph::breadth_first_tree(&a,0,true)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy breadth_first_tree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.breadth_first_tree.html).
pub fn breadth_first_tree(a: &SparseMatrix, start: usize, directed: bool) -> Result<SparseMatrix> {
    tree(
        a,
        &traverse(a, start, directed, false)?.predecessors,
        directed,
    )
}

/// Return the weighted depth-first predecessor tree rooted at `start`.
///
/// # Parameters and conventions
/// The result has the input shape and stores original weights on parent-to-child edges.
/// # Domain and special values
/// Zero-weight tree edges are retained; unreachable vertices have no tree edges.
/// Graph inputs are square, order 0..256, unless rectangular matching is specified.
/// # Returns
/// Owned output with zero-based indices and the ordering described above.
/// # Errors
/// Invalid storage, incompatible shapes/indices, violated domain/work limits,
/// missing full matchings, or nonfinite numerical arithmetic.
/// # Numerical behavior
/// At most 256 vertices and 65,536 dense cells. Traversals and conversion use linear storage scans; path validation is at most cubic. Floating sums can round or overflow.
/// # Examples
/// ```
/// let a=sci_sparse::from_coo(2,2,&[(0,1,3.)])?;
/// let _ = sci_sparse::csgraph::depth_first_tree(&a,0,true)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy depth_first_tree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.depth_first_tree.html).
pub fn depth_first_tree(a: &SparseMatrix, start: usize, directed: bool) -> Result<SparseMatrix> {
    tree(
        a,
        &traverse(a, start, directed, true)?.predecessors,
        directed,
    )
}

/// Reconstruct tree edges `(predecessors[v],v)` with their original weights.
///
/// # Parameters and conventions
/// The predecessor vector has one entry per vertex; None marks roots/unreachable vertices.
/// # Domain and special values
/// Acyclic predecessor forests are accepted; nonexistent edges and cycles return errors.
/// Graph inputs are square, order 0..256, unless rectangular matching is specified.
/// # Returns
/// Owned output with zero-based indices and the ordering described above.
/// # Errors
/// Invalid storage, incompatible shapes/indices, violated domain/work limits,
/// missing full matchings, or nonfinite numerical arithmetic.
/// # Numerical behavior
/// At most 256 vertices and 65,536 dense cells. Traversals and conversion use linear storage scans; path validation is at most cubic. Floating sums can round or overflow.
/// # Examples
/// ```
/// let a=sci_sparse::from_coo(2,2,&[(0,1,3.)])?;
/// let _ = sci_sparse::csgraph::reconstruct_path(&a,&[None,Some(0)],true)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy reconstruct_path](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.reconstruct_path.html).
pub fn reconstruct_path(
    a: &SparseMatrix,
    predecessors: &[Option<usize>],
    directed: bool,
) -> Result<SparseMatrix> {
    tree(a, predecessors, directed)
}

/// Compute `D[s,v] = Σ edge weights` along predecessor chains from `s` to `v`.
///
/// # Parameters and conventions
/// Predecessors have shape n×n with row s rooted at s. Diagonal distances are zero.
/// # Domain and special values
/// Unreachable entries use null_value, which may be finite, NaN or infinity.
/// Graph inputs are square, order 0..256, unless rectangular matching is specified.
/// # Returns
/// Owned output with zero-based indices and the ordering described above.
/// # Errors
/// Invalid storage, incompatible shapes/indices, violated domain/work limits,
/// missing full matchings, or nonfinite numerical arithmetic.
/// # Numerical behavior
/// At most 256 vertices and 65,536 dense cells. Traversals and conversion use linear storage scans; path validation is at most cubic. Floating sums can round or overflow.
/// # Examples
/// ```
/// let a=sci_sparse::from_coo(2,2,&[(0,1,3.)])?;
/// let _ = sci_sparse::csgraph::construct_dist_matrix(&a,&[vec![None,Some(0)],vec![None,None]],true,f64::INFINITY)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy construct_dist_matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.construct_dist_matrix.html).
pub fn construct_dist_matrix(
    a: &SparseMatrix,
    predecessors: &[Vec<Option<usize>>],
    directed: bool,
    null_value: f64,
) -> Result<Matrix> {
    distances_from_pred(a, predecessors, directed, null_value)
}

/// Find a spanning forest minimizing `Σ tree edge weights`.
///
/// # Parameters and conventions
/// Opposite edges use their minimum nonzero weight; output edges are oriented from smaller to larger vertex.
/// # Domain and special values
/// SciPy MST semantics treat explicit zeros as absent. Negative weights and disconnected graphs are supported.
/// Graph inputs are square, order 0..256, unless rectangular matching is specified.
/// # Returns
/// Owned output with zero-based indices and the ordering described above.
/// # Errors
/// Invalid storage, incompatible shapes/indices, violated domain/work limits,
/// missing full matchings, or nonfinite numerical arithmetic.
/// # Numerical behavior
/// Private petgraph Kruskal kernel, O(edges log edges). Floating ties may produce a different optimal forest.
/// # Examples
/// ```
/// let a=sci_sparse::from_coo(2,2,&[(0,1,3.)])?;
/// let _ = sci_sparse::csgraph::minimum_spanning_tree(&a)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy minimum_spanning_tree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.minimum_spanning_tree.html).
pub fn minimum_spanning_tree(a: &SparseMatrix) -> Result<SparseMatrix> {
    mst(a)
}

/// Return a permutation obtained by reversing degree-ordered breadth-first search.
///
/// # Parameters and conventions
/// When symmetric_mode is false the graph is symmetrized structurally; otherwise input adjacency is used as supplied.
/// # Domain and special values
/// Stored zeros affect structure. Degree ties use vertex order, so valid permutations can differ from SciPy.
/// Graph inputs are square, order 0..256, unless rectangular matching is specified.
/// # Returns
/// Owned output with zero-based indices and the ordering described above.
/// # Errors
/// Invalid storage, incompatible shapes/indices, violated domain/work limits,
/// missing full matchings, or nonfinite numerical arithmetic.
/// # Numerical behavior
/// At most 256 vertices and 65,536 dense cells. Traversals and conversion use linear storage scans; path validation is at most cubic. Floating sums can round or overflow.
/// # Examples
/// ```
/// let a=sci_sparse::from_coo(2,2,&[(0,1,3.)])?;
/// let _ = sci_sparse::csgraph::reverse_cuthill_mckee(&a,false)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy reverse_cuthill_mckee](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.reverse_cuthill_mckee.html).
pub fn reverse_cuthill_mckee(a: &SparseMatrix, symmetric_mode: bool) -> Result<Vec<usize>> {
    rcm(a, symmetric_mode)
}

/// Form `L = D - A` or `D^(-1/2) L D^(-1/2)`.
///
/// # Parameters and conventions
/// Incoming degrees are default; use_out_degree selects row sums. symmetrized uses A+Aᵀ without a half factor. Self edges are ignored.
/// # Domain and special values
/// Negative weights are allowed; normalized output requires nonnegative degrees. Isolated vertices have zero diagonal and normalization factor one.
/// Graph inputs are square, order 0..256, unless rectangular matching is specified.
/// # Returns
/// Owned output with zero-based indices and the ordering described above.
/// # Errors
/// Invalid storage, incompatible shapes/indices, violated domain/work limits,
/// missing full matchings, or nonfinite numerical arithmetic.
/// # Numerical behavior
/// At most 256 vertices and 65,536 dense cells. Traversals and conversion use linear storage scans; path validation is at most cubic. Floating sums can round or overflow.
/// # Examples
/// ```
/// let a=sci_sparse::from_coo(2,2,&[(0,1,3.)])?;
/// let _ = sci_sparse::csgraph::laplacian(&a,false,false,false)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy laplacian](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.laplacian.html).
pub fn laplacian(
    a: &SparseMatrix,
    normed: bool,
    use_out_degree: bool,
    symmetrized: bool,
) -> Result<Laplacian> {
    lap(a, normed, use_out_degree, symmetrized)
}

/// Find a maximum-cardinality matching with no shared endpoints.
///
/// # Parameters and conventions
/// Rows and columns are the bipartition. true returns a row index per column; false returns a column per row. None replaces SciPy -1.
/// # Domain and special values
/// Rectangular axes <=256; all stored entries, including zeros, are edges.
/// Graph inputs are square, order 0..256, unless rectangular matching is specified.
/// # Returns
/// Owned output with zero-based indices and the ordering described above.
/// # Errors
/// Invalid storage, incompatible shapes/indices, violated domain/work limits,
/// missing full matchings, or nonfinite numerical arithmetic.
/// # Numerical behavior
/// Private petgraph maximum matching or lapjv assignment backend; cubic work bounded by axis limits. Nonunique optima can choose different pairs than SciPy.
/// # Examples
/// ```
/// let a=sci_sparse::from_coo(2,2,&[(0,1,3.)])?;
/// let _ = sci_sparse::csgraph::maximum_bipartite_matching(&a,true)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy maximum_bipartite_matching](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.maximum_bipartite_matching.html).
pub fn maximum_bipartite_matching(
    a: &SparseMatrix,
    perm_type_row: bool,
) -> Result<Vec<Option<usize>>> {
    matching(a, perm_type_row)
}

/// Compute structural rank as the cardinality of a maximum bipartite matching.
///
/// # Parameters and conventions
/// The result is the maximum rank possible for the given sparsity pattern.
/// # Domain and special values
/// Rectangular axes <=256; values, including stored zeros, do not affect the structural pattern.
/// Graph inputs are square, order 0..256, unless rectangular matching is specified.
/// # Returns
/// Owned output with zero-based indices and the ordering described above.
/// # Errors
/// Invalid storage, incompatible shapes/indices, violated domain/work limits,
/// missing full matchings, or nonfinite numerical arithmetic.
/// # Numerical behavior
/// Private petgraph maximum matching or lapjv assignment backend; cubic work bounded by axis limits. Nonunique optima can choose different pairs than SciPy.
/// # Examples
/// ```
/// let a=sci_sparse::from_coo(2,2,&[(0,1,3.)])?;
/// let _ = sci_sparse::csgraph::structural_rank(&a)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy structural_rank](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.structural_rank.html).
pub fn structural_rank(a: &SparseMatrix) -> Result<usize> {
    Ok(matching(a, false)?.iter().filter(|v| v.is_some()).count())
}

/// Minimize (or maximize) `Σ A[row[i],col[i]]` over full bipartite matchings.
///
/// # Parameters and conventions
/// Returns row and column indices for min(rows,cols) pairs, sorted by row.
/// # Domain and special values
/// Rectangular axes <=256. As in SciPy, stored zero weights are absent; negative nonzero weights are allowed.
/// Graph inputs are square, order 0..256, unless rectangular matching is specified.
/// # Returns
/// Owned output with zero-based indices and the ordering described above.
/// # Errors
/// Invalid storage, incompatible shapes/indices, violated domain/work limits,
/// missing full matchings, or nonfinite numerical arithmetic.
/// # Numerical behavior
/// Private petgraph maximum matching or lapjv assignment backend; cubic work bounded by axis limits. Nonunique optima can choose different pairs than SciPy.
/// # Examples
/// ```
/// let a=sci_sparse::from_coo(2,2,&[(0,1,3.)])?;
/// let _ = sci_sparse::csgraph::min_weight_full_bipartite_matching(&sci_sparse::identity(2)?,false)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy min_weight_full_bipartite_matching](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.min_weight_full_bipartite_matching.html).
pub fn min_weight_full_bipartite_matching(
    a: &SparseMatrix,
    maximize: bool,
) -> Result<(Vec<usize>, Vec<usize>)> {
    weighted_matching(a, maximize)
}

/// Maximize source-to-sink flow subject to capacity constraints and interior conservation.
///
/// # Parameters and conventions
/// Directed capacities are integral. The output is the antisymmetric net flow matrix.
/// # Domain and special values
/// Distinct source and sink; nonnegative integer capacities with total <=2^53-1.
/// Graph inputs are square, order 0..256, unless rectangular matching is specified.
/// # Returns
/// Owned output with zero-based indices and the ordering described above.
/// # Errors
/// Invalid storage, incompatible shapes/indices, violated domain/work limits,
/// missing full matchings, or nonfinite numerical arithmetic.
/// # Numerical behavior
/// Private petgraph Dinic kernel; exact integer arithmetic under the capacity sum bound. Work is bounded by 256 vertices and at most 65,536 edges.
/// # Examples
/// ```
/// let a=sci_sparse::from_coo(2,2,&[(0,1,3.)])?;
/// let _ = sci_sparse::csgraph::maximum_flow(&a,0,1)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy maximum_flow](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.maximum_flow.html).
pub fn maximum_flow(a: &SparseMatrix, source: usize, sink: usize) -> Result<MaximumFlow> {
    flow(a, source, sink)
}

/// Find up to k distinct loopless paths, ordered by `Σ edge weights`.
///
/// # Parameters and conventions
/// Returns explicit paths, distances and predecessor rows. k=0 gives no paths; source=sink includes its zero-length path.
/// # Domain and special values
/// Nonnegative weights bounded by f64::MAX/(4n+1); 0<=k<=64. k×vertices×edges <=20,000,000. Equal-cost path order may differ from SciPy.
/// Graph inputs are square, order 0..256, unless rectangular matching is specified.
/// # Returns
/// Owned output with zero-based indices and the ordering described above.
/// # Errors
/// Invalid storage, incompatible shapes/indices, violated domain/work limits,
/// missing full matchings, or nonfinite numerical arithmetic.
/// # Numerical behavior
/// Yen spur-path assembly uses private petgraph A* with a zero heuristic. Storage is bounded by 64 paths and their candidates; work is checked before searches.
/// # Examples
/// ```
/// let a=sci_sparse::from_coo(2,2,&[(0,1,3.)])?;
/// let _ = sci_sparse::csgraph::yen(&a,0,1,2,true,false)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy yen](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.yen.html).
pub fn yen(
    a: &SparseMatrix,
    source: usize,
    sink: usize,
    k: usize,
    directed: bool,
    unweighted: bool,
) -> Result<KShortestPaths> {
    k_paths(a, source, sink, k, directed, unweighted)
}

/// Convert dense adjacency to CSR, omitting designated null entries.
///
/// # Parameters and conventions
/// Equality to null_value masks an edge; nan_null and infinity_null additionally mask NaN and both infinities.
/// # Domain and special values
/// Finite unmasked values only; null_value may be NaN or infinity. Zero edges survive when zero is not null.
/// Graph inputs are square, order 0..256, unless rectangular matching is specified.
/// # Returns
/// Owned output with zero-based indices and the ordering described above.
/// # Errors
/// Invalid storage, incompatible shapes/indices, violated domain/work limits,
/// missing full matchings, or nonfinite numerical arithmetic.
/// # Numerical behavior
/// At most 256 vertices and 65,536 dense cells. Traversals and conversion use linear storage scans; path validation is at most cubic. Floating sums can round or overflow.
/// # Examples
/// ```
/// let a=sci_sparse::from_coo(2,2,&[(0,1,3.)])?;
/// let _ = sci_sparse::csgraph::csgraph_from_dense(&vec![vec![0.]],0.,true,true)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy csgraph_from_dense](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.csgraph_from_dense.html).
pub fn csgraph_from_dense(
    a: &Matrix,
    null_value: f64,
    nan_null: bool,
    infinity_null: bool,
) -> Result<SparseMatrix> {
    from_masked(&masked_dense(a, null_value, nan_null, infinity_null)?)
}

/// Construct a mask identifying non-edges in dense adjacency.
///
/// # Parameters and conventions
/// null_value, nan_null and infinity_null determine missing edges; stored values are copied.
/// # Domain and special values
/// Masked entries may contain NaN/infinity; this conversion also permits unmasked nonfinite values, rejected by numerical graph algorithms.
/// Graph inputs are square, order 0..256, unless rectangular matching is specified.
/// # Returns
/// Owned output with zero-based indices and the ordering described above.
/// # Errors
/// Invalid storage, incompatible shapes/indices, violated domain/work limits,
/// missing full matchings, or nonfinite numerical arithmetic.
/// # Numerical behavior
/// At most 256 vertices and 65,536 dense cells. Traversals and conversion use linear storage scans; path validation is at most cubic. Floating sums can round or overflow.
/// # Examples
/// ```
/// let a=sci_sparse::from_coo(2,2,&[(0,1,3.)])?;
/// let _ = sci_sparse::csgraph::csgraph_masked_from_dense(&vec![vec![0.]],0.,true,true)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy csgraph_masked_from_dense](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.csgraph_masked_from_dense.html).
pub fn csgraph_masked_from_dense(
    a: &Matrix,
    null_value: f64,
    nan_null: bool,
    infinity_null: bool,
) -> Result<MaskedGraph> {
    masked_dense(a, null_value, nan_null, infinity_null)
}

/// Convert unmasked entries to explicitly stored CSR graph edges.
///
/// # Parameters and conventions
/// True mask entries are absent edges, independent of their underlying values.
/// # Domain and special values
/// Finite unmasked values only; zero edges and empty graphs are preserved.
/// Graph inputs are square, order 0..256, unless rectangular matching is specified.
/// # Returns
/// Owned output with zero-based indices and the ordering described above.
/// # Errors
/// Invalid storage, incompatible shapes/indices, violated domain/work limits,
/// missing full matchings, or nonfinite numerical arithmetic.
/// # Numerical behavior
/// At most 256 vertices and 65,536 dense cells. Traversals and conversion use linear storage scans; path validation is at most cubic. Floating sums can round or overflow.
/// # Examples
/// ```
/// let a=sci_sparse::csgraph::MaskedGraph{values:vec![vec![0.]],mask:vec![vec![true]]};
/// let _ = sci_sparse::csgraph::csgraph_from_masked(&a)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy csgraph_from_masked](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.csgraph_from_masked.html).
pub fn csgraph_from_masked(a: &MaskedGraph) -> Result<SparseMatrix> {
    from_masked(a)
}

/// Fill absent graph edges with null_value and copy stored edge weights.
///
/// # Parameters and conventions
/// Explicitly stored zero edges remain zero. No implicit diagonal entries are inserted.
/// # Domain and special values
/// null_value may be finite, NaN or infinity; stored CSR data must be finite.
/// Graph inputs are square, order 0..256, unless rectangular matching is specified.
/// # Returns
/// Owned output with zero-based indices and the ordering described above.
/// # Errors
/// Invalid storage, incompatible shapes/indices, violated domain/work limits,
/// missing full matchings, or nonfinite numerical arithmetic.
/// # Numerical behavior
/// At most 256 vertices and 65,536 dense cells. Traversals and conversion use linear storage scans; path validation is at most cubic. Floating sums can round or overflow.
/// # Examples
/// ```
/// let a=sci_sparse::from_coo(2,2,&[(0,1,3.)])?;
/// let _ = sci_sparse::csgraph::csgraph_to_dense(&a,f64::INFINITY)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy csgraph_to_dense](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.csgraph_to_dense.html).
pub fn csgraph_to_dense(a: &SparseMatrix, null_value: f64) -> Result<Matrix> {
    graph_dense(a, null_value)
}

/// Mark every unstored edge as masked, retaining all stored edge weights.
///
/// # Parameters and conventions
/// True means no edge; masked entries have underlying value zero.
/// # Domain and special values
/// Explicit zeros are unmasked; finite canonical CSR input is required.
/// Graph inputs are square, order 0..256, unless rectangular matching is specified.
/// # Returns
/// Owned output with zero-based indices and the ordering described above.
/// # Errors
/// Invalid storage, incompatible shapes/indices, violated domain/work limits,
/// missing full matchings, or nonfinite numerical arithmetic.
/// # Numerical behavior
/// At most 256 vertices and 65,536 dense cells. Traversals and conversion use linear storage scans; path validation is at most cubic. Floating sums can round or overflow.
/// # Examples
/// ```
/// let a=sci_sparse::from_coo(2,2,&[(0,1,3.)])?;
/// let _ = sci_sparse::csgraph::csgraph_to_masked(&a)?;
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy csgraph_to_masked](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.csgraph_to_masked.html).
pub fn csgraph_to_masked(a: &SparseMatrix) -> Result<MaskedGraph> {
    masked(a)
}
