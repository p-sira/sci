use l::LinearMap;
use sci::sparse::{self as s, csgraph as g, linalg as l};
use std::collections::BTreeMap;

fn nums(s: &str) -> Vec<f64> {
    if s.is_empty() {
        vec![]
    } else {
        s.split(';').map(|v| v.parse().unwrap()).collect()
    }
}
fn matrix(rows: usize, cols: usize, triples: &str) -> s::SparseMatrix {
    let t = if triples.is_empty() {
        vec![]
    } else {
        triples
            .split(':')
            .map(|v| {
                let v = nums(v);
                (v[0] as usize, v[1] as usize, v[2])
            })
            .collect()
    };
    s::from_coo(rows, cols, &t).unwrap()
}
fn flat(a: &s::Matrix) -> Vec<f64> {
    a.iter().flatten().copied().collect()
}
fn close(actual: &[f64], expected: &[f64], tol: f64, id: &str) {
    assert_eq!(actual.len(), expected.len(), "{id} shape");
    for (i, (&v, &e)) in actual.iter().zip(expected).enumerate() {
        if e.is_nan() {
            assert!(v.is_nan(), "{id}[{i}]");
        } else if e.is_infinite() {
            assert_eq!(v, e, "{id}[{i}]");
        } else {
            assert!(
                (v - e).abs() <= tol * (1. + e.abs()),
                "{id}[{i}]: {v} != {e}"
            );
        }
    }
}
fn array_dense(a: &s::SparseArray) -> Vec<f64> {
    let n = a.shape.iter().product();
    let mut v = vec![0.; n];
    for (c, &x) in a.coordinates.iter().zip(&a.data) {
        let i = c.iter().zip(&a.shape).fold(0, |i, (&j, &n)| i * n + j);
        v[i] = x;
    }
    v
}
fn pred(v: &[f64]) -> Vec<Option<usize>> {
    v.iter()
        .map(|&v| if v < 0. { None } else { Some(v as usize) })
        .collect()
}

#[test]
fn scipy_public_api_references() {
    let mut expected: BTreeMap<(String, String), (usize, usize, Vec<f64>)> = BTreeMap::new();
    for line in include_str!("fixtures/sparse_parity_expected.csv")
        .lines()
        .skip(1)
    {
        let r: Vec<_> = line.split(',').collect();
        expected.insert(
            (r[0].to_owned(), r[1].to_owned()),
            (r[2].parse().unwrap(), r[3].parse().unwrap(), nums(r[4])),
        );
    }
    let mut tested = 0;
    for line in include_str!("fixtures/sparse_parity_inputs.csv")
        .lines()
        .skip(1)
    {
        let r: Vec<_> = line.split(',').collect();
        let (id, api) = (r[0], r[1]);
        let (m, n) = (
            r[2].parse::<usize>().unwrap(),
            r[3].parse::<usize>().unwrap(),
        );
        let a = matrix(m, n, r[4]);
        let p = nums(r[5]);
        let x = if api == "spdiags" { vec![] } else { nums(r[6]) };
        let e = |key: &str| &expected[&(id.to_owned(), key.to_owned())].2;
        let assert_v = |key: &str, v: &[f64], tol: f64| close(v, e(key), tol, id);
        let assert_m = |key: &str, v: &s::Matrix, tol: f64| {
            let (rows, cols, _) = &expected[&(id.to_owned(), key.to_owned())];
            assert_eq!(v.len(), *rows, "{id} rows");
            assert!(v.iter().all(|r| r.len() == *cols), "{id} cols");
            assert_v(key, &flat(v), tol);
        };
        match api {
            "shortest_path" | "dijkstra" | "bellman_ford" | "floyd_warshall" | "johnson" => {
                let o = g::PathOptions {
                    directed: p[0] != 0.,
                    unweighted: p[1] != 0.,
                    indices: if x.is_empty() {
                        None
                    } else {
                        Some(x.iter().map(|v| *v as usize).collect())
                    },
                    min_only: p.get(2) == Some(&1.),
                    limit: p.get(3).copied().unwrap_or(f64::INFINITY),
                };
                let f = match api {
                    "shortest_path" => g::shortest_path,
                    "dijkstra" => g::dijkstra,
                    "bellman_ford" => g::bellman_ford,
                    "floyd_warshall" => g::floyd_warshall,
                    _ => g::johnson,
                };
                let out = f(&a, &o).unwrap();
                assert_m("d", &out.distances, 3e-12);
                assert_eq!(out.predecessors.len(), out.distances.len());
                for (i, row) in out.predecessors.iter().enumerate() {
                    assert_eq!(row.len(), n);
                    for (v, &parent) in row.iter().enumerate() {
                        if let Some(u) = parent {
                            assert!(u < n);
                            let adj = g::csgraph_to_dense(&a, f64::INFINITY).unwrap();
                            assert!(adj[u][v].is_finite() || !o.directed && adj[v][u].is_finite());
                        }
                    }
                    if !o.min_only {
                        assert_eq!(row[out.sources[i]], None);
                    }
                }
            }
            "breadth_first_order" | "depth_first_order" => {
                let out = if api == "breadth_first_order" {
                    g::breadth_first_order(&a, p[0] as usize, p[1] != 0.)
                } else {
                    g::depth_first_order(&a, p[0] as usize, p[1] != 0.)
                }
                .unwrap();
                assert_v(
                    "order",
                    &out.order.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                    0.,
                );
                assert_v(
                    "pred",
                    &out.predecessors
                        .iter()
                        .map(|v| v.map_or(-9999., |v| v as f64))
                        .collect::<Vec<_>>(),
                    0.,
                );
            }
            "breadth_first_tree" | "depth_first_tree" => {
                let out = if api == "breadth_first_tree" {
                    g::breadth_first_tree(&a, p[0] as usize, p[1] != 0.)
                } else {
                    g::depth_first_tree(&a, p[0] as usize, p[1] != 0.)
                }
                .unwrap();
                assert_m("a", &s::to_dense(&out).unwrap(), 0.);
            }
            "connected_components" => {
                let out = g::connected_components(&a, p[0] != 0., p[1] != 0.).unwrap();
                assert_v("count", &[out.count as f64], 0.);
                assert_v(
                    "labels",
                    &out.labels.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                    0.,
                );
            }
            "minimum_spanning_tree" => {
                let out = g::minimum_spanning_tree(&a).unwrap();
                assert_v("weight", &[out.data.iter().sum()], 3e-12);
                assert!(out.data.len() < m);
            }
            "reverse_cuthill_mckee" => {
                let out = g::reverse_cuthill_mckee(&a, p[0] != 0.).unwrap();
                let mut sorted = out.clone();
                sorted.sort_unstable();
                assert_eq!(sorted, (0..m).collect::<Vec<_>>());
                assert_eq!(out.len(), e("permutation").len());
            }
            "laplacian" => {
                let out = g::laplacian(&a, p[0] != 0., p[1] != 0., p[2] != 0.).unwrap();
                assert_m("a", &s::to_dense(&out.matrix).unwrap(), 3e-12);
                assert_v("diag", &out.diagonal, 3e-12);
            }
            "maximum_bipartite_matching" => {
                let out = g::maximum_bipartite_matching(&a, p[0] != 0.).unwrap();
                assert_eq!(out.len(), if p[0] != 0. { n } else { m });
                let mut seen = std::collections::BTreeSet::new();
                for (i, v) in out.iter().enumerate() {
                    if let Some(j) = v {
                        assert!(seen.insert(*j));
                        let (r, c) = if p[0] != 0. { (*j, i) } else { (i, *j) };
                        assert!(a.indices[a.indptr[r]..a.indptr[r + 1]].contains(&c));
                    }
                }
                assert_v(
                    "count",
                    &[out.iter().filter(|v| v.is_some()).count() as f64],
                    0.,
                );
            }
            "structural_rank" => assert_v("count", &[g::structural_rank(&a).unwrap() as f64], 0.),
            "min_weight_full_bipartite_matching" => {
                let (rows, cols) = g::min_weight_full_bipartite_matching(&a, p[0] != 0.).unwrap();
                assert_v(
                    "rows",
                    &rows.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                    0.,
                );
                assert_v(
                    "cols",
                    &cols.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                    0.,
                );
            }
            "maximum_flow" => {
                let out = g::maximum_flow(&a, p[0] as usize, p[1] as usize).unwrap();
                assert_v("value", &[out.value], 0.);
                let d = s::to_dense(&out.flow).unwrap();
                for (i, row) in d.iter().enumerate() {
                    for (j, &v) in row.iter().enumerate() {
                        assert_eq!(v, -d[j][i]);
                    }
                    let sum: f64 = row.iter().sum();
                    assert_eq!(
                        sum,
                        if i == p[0] as usize {
                            out.value
                        } else if i == p[1] as usize {
                            -out.value
                        } else {
                            0.
                        }
                    );
                }
            }
            "yen" => {
                let out = g::yen(
                    &a,
                    p[0] as usize,
                    p[1] as usize,
                    p[2] as usize,
                    p[3] != 0.,
                    p[4] != 0.,
                )
                .unwrap();
                assert_v("d", &out.distances, 3e-12);
                assert_eq!(out.paths.len(), out.distances.len());
                for path in out.paths {
                    let unique: std::collections::BTreeSet<_> = path.iter().collect();
                    assert_eq!(unique.len(), path.len());
                    assert_eq!(path.first(), Some(&(p[0] as usize)));
                    assert_eq!(path.last(), Some(&(p[1] as usize)));
                }
            }
            "csgraph_from_dense" | "csgraph_masked_from_dense" | "csgraph_from_masked" => {
                let d = s::to_dense(&a).unwrap();
                let masked = g::csgraph_masked_from_dense(&d, p[0], true, true).unwrap();
                let out = if api == "csgraph_from_dense" {
                    g::csgraph_from_dense(&d, p[0], true, true).unwrap()
                } else {
                    g::csgraph_from_masked(&masked).unwrap()
                };
                assert_m("a", &s::to_dense(&out).unwrap(), 0.);
                if api == "csgraph_masked_from_dense" {
                    assert_v(
                        "mask",
                        &masked
                            .mask
                            .iter()
                            .flatten()
                            .map(|v| f64::from(*v))
                            .collect::<Vec<_>>(),
                        0.,
                    );
                }
            }
            "csgraph_to_dense" => assert_m("a", &g::csgraph_to_dense(&a, p[0]).unwrap(), 0.),
            "csgraph_to_masked" => {
                let out = g::csgraph_to_masked(&a).unwrap();
                assert_m("a", &out.values, 0.);
                assert_v(
                    "mask",
                    &out.mask
                        .iter()
                        .flatten()
                        .map(|v| f64::from(*v))
                        .collect::<Vec<_>>(),
                    0.,
                );
            }
            "reconstruct_path" => {
                let out = g::reconstruct_path(&a, &pred(e("pred")), p[0] != 0.).unwrap();
                assert_m("a", &s::to_dense(&out).unwrap(), 0.);
            }
            "construct_dist_matrix" => {
                let pr = e("pred").chunks(m).map(pred).collect::<Vec<_>>();
                assert_m(
                    "d",
                    &g::construct_dist_matrix(&a, &pr, p[0] != 0., f64::INFINITY).unwrap(),
                    3e-12,
                );
            }
            "cg" | "bicg" | "bicgstab" | "cgs" | "qmr" | "tfqmr" | "gmres" | "lgmres"
            | "gcrotmk" | "minres" => {
                let opts = l::SolverOptions {
                    rtol: p.get(3).copied().unwrap_or(1e-11),
                    atol: p.get(1).copied().unwrap_or(1e-13),
                    maxiter: p.get(2).copied().unwrap_or(512.) as usize,
                    restart: p.first().copied().unwrap_or(30.) as usize,
                    ..Default::default()
                };
                let f = match api {
                    "cg" => l::cg,
                    "bicg" => l::bicg,
                    "bicgstab" => l::bicgstab,
                    "cgs" => l::cgs,
                    "qmr" => l::qmr,
                    "tfqmr" => l::tfqmr,
                    "gmres" => l::gmres,
                    "lgmres" => l::lgmres,
                    "gcrotmk" => l::gcrotmk,
                    _ => l::minres,
                };
                let out = f(&a, &x, &opts).unwrap_or_else(|e| panic!("{id}: {e}"));
                assert!(out.converged, "{id} residual {}", out.residual_norm);
                if p.len() > 1 {
                    let bnorm = x.iter().fold(0f64, |n, &v| n.hypot(v));
                    let ax = s::matvec(&a, &out.x).unwrap();
                    let residual = ax.iter().zip(&x).fold(0f64, |n, (a, b)| n.hypot(a - b));
                    assert!(residual / bnorm <= opts.rtol, "{id}: relative residual");
                    assert!(out.iterations <= opts.maxiter, "{id}: work limit");
                }
                assert_v("x", &out.x, 3e-9);
            }
            "lsqr" | "lsmr" => {
                let opts = l::LeastSquaresOptions {
                    rtol: 1e-12,
                    atol: 1e-12,
                    damp: p.first().copied().unwrap_or(0.),
                    ..Default::default()
                };
                let out = if api == "lsqr" {
                    l::lsqr(&a, &x, &opts)
                } else {
                    l::lsmr(&a, &x, &opts)
                }
                .unwrap();
                assert!(
                    out.converged,
                    "{id}: {} {}",
                    out.residual_norm, out.normal_residual_norm
                );
                assert_v("x", &out.x, 3e-9);
            }
            "eigs" => {
                let out = l::eigs(&a, p[0] as usize, l::Spectrum::LargestMagnitude).unwrap();
                let mut got: Vec<_> = out.eigenvalues.iter().map(|v| (v.re, v.im)).collect();
                let mut want: Vec<_> = e("wre")
                    .iter()
                    .copied()
                    .zip(e("wim").iter().copied())
                    .collect();
                let sort = |v: &mut Vec<(f64, f64)>| {
                    v.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)))
                };
                sort(&mut got);
                sort(&mut want);
                close(
                    &got.iter().flat_map(|v| [v.0, v.1]).collect::<Vec<_>>(),
                    &want.iter().flat_map(|v| [v.0, v.1]).collect::<Vec<_>>(),
                    3e-11,
                    id,
                );
                assert_eq!(out.eigenvectors.len(), m);
                for (j, w) in out.eigenvalues.iter().enumerate() {
                    let re: Vec<_> = out.eigenvectors.iter().map(|r| r[j].re).collect();
                    let im: Vec<_> = out.eigenvectors.iter().map(|r| r[j].im).collect();
                    let ar = s::matvec(&a, &re).unwrap();
                    let ai = s::matvec(&a, &im).unwrap();
                    for i in 0..m {
                        assert!((ar[i] - (w.re * re[i] - w.im * im[i])).abs() < 1e-10);
                        assert!((ai[i] - (w.re * im[i] + w.im * re[i])).abs() < 1e-10);
                    }
                }
            }
            "eigsh" | "lobpcg" => {
                let out = if api == "eigsh" {
                    l::eigsh(&a, p[0] as usize, l::Spectrum::LargestReal)
                } else {
                    l::lobpcg(&a, &vec![vec![1.; p[0] as usize]; m], true)
                }
                .unwrap();
                assert_v("w", &out.eigenvalues, 3e-11);
                assert_eq!(out.eigenvectors.len(), m);
                for (j, w) in out.eigenvalues.iter().enumerate() {
                    let v: Vec<_> = out.eigenvectors.iter().map(|r| r[j]).collect();
                    let av = s::matvec(&a, &v).unwrap();
                    close(&av, &v.iter().map(|v| v * w).collect::<Vec<_>>(), 3e-11, id);
                }
            }
            "svds" => {
                let out = l::svds(&a, p[0] as usize, true).unwrap();
                assert_v("s", &out.singular_values, 3e-11);
                assert_eq!(out.u.len(), m);
                assert_eq!(out.vh.len(), p[0] as usize);
                for (j, &sigma) in out.singular_values.iter().enumerate() {
                    let av = s::matvec(&a, &out.vh[j]).unwrap();
                    close(
                        &av,
                        &out.u.iter().map(|r| r[j] * sigma).collect::<Vec<_>>(),
                        3e-11,
                        id,
                    );
                }
            }
            "expm" => assert_m("a", &s::to_dense(&l::expm(&a).unwrap()).unwrap(), 3e-11),
            "expm_multiply" => assert_m(
                "a",
                &l::expm_multiply(&a, &x, &[0., 0.5, 1.]).unwrap(),
                3e-11,
            ),
            "funm_multiply_krylov" => {
                let out = l::funm_multiply_krylov(
                    &a,
                    &x,
                    |mut z| {
                        let re = z.re.exp() * z.im.cos();
                        z.im = z.re.exp() * z.im.sin();
                        z.re = re;
                        z
                    },
                    l::KrylovOptions {
                        rtol: 1e-12,
                        ..Default::default()
                    },
                )
                .unwrap();
                assert_v("x", &out.iter().map(|v| v.re).collect::<Vec<_>>(), 3e-11);
                assert!(out.iter().all(|v| v.im.abs() < 1e-11));
            }
            "onenormest" => assert_v("value", &[l::onenormest(&a).unwrap()], 3e-12),
            "spectral_norm" => assert_v(
                "value",
                &[l::norm(&a, l::MatrixNorm::Spectral).unwrap()],
                3e-11,
            ),
            "norm_axis" => {
                assert_v("x", &l::norm_axis(&a, 2., 0).unwrap(), 3e-12);
                assert_v("xt", &l::norm_axis(&a, -1., 1).unwrap(), 3e-12);
            }
            "spilu" => assert_v(
                "x",
                &l::spilu(&a, 0., 20.).unwrap().solve(&x).unwrap(),
                3e-11,
            ),
            "aslinearoperator" => {
                let op = l::aslinearoperator(&a).unwrap();
                assert_v("x", &op.apply(&x).unwrap(), 3e-12);
                assert_v("xt", &op.transpose_apply(&x).unwrap(), 3e-12);
            }
            "as_array" | "expand_dims" | "permute_dims" | "swapaxes" => {
                let a = s::as_array(&a).unwrap();
                let out = match api {
                    "expand_dims" => s::expand_dims(&a, p[0] as isize),
                    "permute_dims" => {
                        s::permute_dims(&a, &p.iter().map(|v| *v as isize).collect::<Vec<_>>())
                    }
                    "swapaxes" => s::swapaxes(&a, p[0] as isize, p[1] as isize),
                    _ => Ok(a),
                }
                .unwrap();
                assert_v(
                    "shape",
                    &out.shape.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                    0.,
                );
                assert_v("data", &array_dense(&out), 0.);
            }
            "issparse" => assert_v(
                "value",
                &[f64::from(if p[0] != 0. {
                    s::issparse(&a)
                } else {
                    s::issparse(&s::to_dense(&a).unwrap())
                })],
                0.,
            ),
            "get_index_dtype" => {
                let v: Vec<_> = x.iter().map(|v| *v as u64).collect();
                let dtype = s::get_index_dtype(&[&v], None).unwrap();
                assert_v(
                    "value",
                    &[if dtype == s::IndexDtype::I32 {
                        32.
                    } else {
                        64.
                    }],
                    0.,
                );
            }
            "safely_cast_index_arrays" => {
                let (ptr, idx) = s::safely_cast_index_arrays(
                    &a,
                    if p[0] == 32. {
                        s::IndexDtype::I32
                    } else {
                        s::IndexDtype::I64
                    },
                )
                .unwrap();
                assert_v(
                    "indptr",
                    &ptr.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                    0.,
                );
                assert_v(
                    "indices",
                    &idx.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                    0.,
                );
            }
            "random" | "random_array" => {
                let out = if api == "random" {
                    s::as_array(&s::random(m, n, p[0], p[1] as u64).unwrap()).unwrap()
                } else {
                    s::random_array(&[m, n], p[0], p[1] as u64).unwrap()
                };
                assert_v("nnz", &[out.data.len() as f64], 0.);
                assert_v(
                    "shape",
                    &out.shape.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                    0.,
                );
                assert!(out.data.iter().all(|v| (0.0..1.0).contains(v)));
                assert_eq!(out, s::random_array(&[m, n], p[0], p[1] as u64).unwrap());
            }
            "spdiags" => {
                let data: Vec<Vec<f64>> = r[6].split(':').map(nums).collect();
                let offsets = p.iter().map(|v| *v as isize).collect::<Vec<_>>();
                assert_m(
                    "a",
                    &s::to_dense(&s::spdiags(&data, &offsets, m, n).unwrap()).unwrap(),
                    0.,
                );
            }
            "load_npz" | "save_npz" => {
                let bytes = std::fs::read(format!(
                    "{}/tests/fixtures/{id}.npz",
                    env!("CARGO_MANIFEST_DIR")
                ))
                .unwrap();
                let loaded = s::load_npz(std::io::Cursor::new(bytes)).unwrap();
                assert_m("a", &s::to_dense(&loaded).unwrap(), 0.);
                let mut dest = std::io::Cursor::new(vec![]);
                s::save_npz(&mut dest, &a, p[0] != 0.).unwrap();
                dest.set_position(0);
                assert_eq!(s::load_npz(&mut dest).unwrap(), a);
                if id == "save_npz_0" {
                    std::fs::create_dir_all(format!("{}/target", env!("CARGO_MANIFEST_DIR")))
                        .unwrap();
                    std::fs::write(
                        format!("{}/target/sparse-rust.npz", env!("CARGO_MANIFEST_DIR")),
                        dest.into_inner(),
                    )
                    .unwrap();
                }
            }
            _ => panic!("untested API {api}"),
        }
        tested += 1;
    }
    assert_eq!(tested, 173);
}

#[test]
fn scipy_npz_format_compatibility() {
    let expected = vec![1., 0., 2., 0., -3., 0., 4., 0., 5.];
    for format in ["csr", "csc", "coo", "dia", "bsr"] {
        for dtype in ["float64", "int32"] {
            let bytes = std::fs::read(format!(
                "{}/tests/fixtures/sparse_{format}_{dtype}.npz",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap();
            let out = s::load_npz(std::io::Cursor::new(bytes)).unwrap();
            close(&flat(&s::to_dense(&out).unwrap()), &expected, 0., format);
        }
    }
    for name in [
        "bad_pointer",
        "negative_shape",
        "object",
        "complex",
        "nonfinite",
        "oversize",
    ] {
        let bytes = std::fs::read(format!(
            "{}/tests/fixtures/sparse_invalid_{name}.npz",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        assert!(s::load_npz(std::io::Cursor::new(bytes)).is_err(), "{name}");
    }
    for bytes in [vec![], b"not a zip archive".to_vec()] {
        assert!(s::load_npz(std::io::Cursor::new(bytes)).is_err());
    }
}

#[test]
fn graph_edges_cycles_and_validation() {
    let zero = s::from_coo(3, 3, &[(0, 1, 0.), (1, 2, 2.)]).unwrap();
    let opts = g::PathOptions::default();
    for f in [
        g::shortest_path,
        g::dijkstra,
        g::bellman_ford,
        g::floyd_warshall,
        g::johnson,
    ] {
        let out = f(&zero, &opts).unwrap();
        assert_eq!(out.distances[0], [0., 0., 2.]);
        assert_eq!(out.predecessors[0], [None, Some(0), Some(1)]);
    }
    let tri = g::breadth_first_tree(&zero, 0, true).unwrap();
    assert_eq!(tri.data, [0., 2.]);
    assert_eq!(g::structural_rank(&zero).unwrap(), 2);
    let masked = g::csgraph_to_masked(&zero).unwrap();
    assert!(!masked.mask[0][1]);
    assert_eq!(g::csgraph_from_masked(&masked).unwrap(), zero);
    let neg = s::from_coo(2, 2, &[(0, 1, -1.), (1, 0, -2.)]).unwrap();
    for f in [
        g::shortest_path,
        g::bellman_ford,
        g::floyd_warshall,
        g::johnson,
    ] {
        assert!(f(&neg, &opts).is_err());
    }
    assert!(g::dijkstra(&neg, &opts).is_err());
    assert!(g::yen(&neg, 0, 1, 3, true, false).is_err());
    assert!(g::reconstruct_path(&zero, &[Some(1), Some(0), None], true).is_err());
    assert!(g::construct_dist_matrix(&zero, &[vec![None; 3]], true, 0.).is_err());
    assert!(g::breadth_first_order(&zero, 3, true).is_err());
    assert!(g::depth_first_order(&zero, usize::MAX, false).is_err());
    assert!(g::yen(&zero, 0, 2, 65, true, false).is_err());
    assert!(g::maximum_flow(&zero, 0, 0).is_err());
    assert!(g::maximum_flow(&neg, 0, 1).is_err());
    let fractional = s::from_coo(2, 2, &[(0, 1, 1.5)]).unwrap();
    assert!(g::maximum_flow(&fractional, 0, 1).is_err());
    let missing = s::from_coo(2, 3, &[(0, 0, 2.), (1, 0, 3.)]).unwrap();
    assert!(g::min_weight_full_bipartite_matching(&missing, false).is_err());
    assert_eq!(g::structural_rank(&missing).unwrap(), 1);
    let bad = g::PathOptions {
        limit: f64::NAN,
        ..Default::default()
    };
    assert!(g::dijkstra(&zero, &bad).is_err());
    let bad = g::PathOptions {
        indices: Some(vec![3]),
        ..Default::default()
    };
    assert!(g::shortest_path(&zero, &bad).is_err());
    let overflow = s::from_coo(2, 2, &[(0, 1, f64::MAX)]).unwrap();
    assert!(g::shortest_path(&overflow, &opts).is_err());
    let rect = s::from_coo(1, 2, &[]).unwrap();
    for f in [
        g::shortest_path,
        g::dijkstra,
        g::bellman_ford,
        g::floyd_warshall,
        g::johnson,
    ] {
        assert!(f(&rect, &opts).is_err());
    }
    assert!(g::connected_components(&rect, true, false).is_err());
    assert!(g::laplacian(&rect, false, false, false).is_err());
    assert!(g::minimum_spanning_tree(&rect).is_err());
    assert!(g::reverse_cuthill_mckee(&rect, false).is_err());
    assert!(g::csgraph_from_dense(&vec![vec![1., 2.]], 0., true, true).is_err());
    assert!(g::csgraph_from_dense(&vec![vec![f64::NAN]], 0., false, false).is_err());
    let a = g::csgraph_from_dense(
        &vec![vec![f64::INFINITY, 0.], vec![f64::NAN, 2.]],
        f64::INFINITY,
        true,
        true,
    )
    .unwrap();
    assert_eq!(a.data, [0., 2.]);
    let mst_zero = s::from_coo(2, 2, &[(0, 1, 0.), (1, 0, 3.)]).unwrap();
    assert_eq!(g::minimum_spanning_tree(&mst_zero).unwrap().data, [3.]);
    let empty = s::identity(0).unwrap();
    assert_eq!(
        g::connected_components(&empty, true, true).unwrap().count,
        0
    );
    assert!(
        g::shortest_path(&empty, &opts)
            .unwrap()
            .distances
            .is_empty()
    );
}

#[test]
fn iterative_properties_and_errors() {
    let a = s::from_dense(&[vec![4., 1.], vec![1., 3.]]).unwrap();
    let b = [1., 2.];
    let op = l::aslinearoperator(&a).unwrap();
    for f in [
        l::cg,
        l::bicg,
        l::bicgstab,
        l::cgs,
        l::qmr,
        l::tfqmr,
        l::gmres,
        l::lgmres,
        l::gcrotmk,
        l::minres,
    ] {
        let opts = l::SolverOptions {
            rtol: 1e-12,
            ..Default::default()
        };
        let out = f(&op, &b, &opts).unwrap();
        assert!(out.converged);
        close(&s::matvec(&a, &out.x).unwrap(), &b, 1e-11, "operator solve");
        assert!(f(&op, &[1.], &opts).is_err());
        assert!(f(&op, &[f64::NAN, 2.], &opts).is_err());
        let bad = l::SolverOptions {
            rtol: f64::NAN,
            ..opts.clone()
        };
        assert!(f(&op, &b, &bad).is_err());
        let limited = l::SolverOptions {
            maxiter: 0,
            ..opts.clone()
        };
        let out = f(&op, &b, &limited).unwrap();
        assert!(!out.converged);
        assert_eq!(out.iterations, 0);
        let exact = l::SolverOptions {
            x0: Some(vec![1., 2.]),
            ..opts.clone()
        };
        assert_eq!(f(&op, &[6., 7.], &exact).unwrap().iterations, 0);
    }
    let identity = l::aslinearoperator(&s::identity(2).unwrap()).unwrap();
    let opts = l::SolverOptions {
        preconditioner: Some(identity),
        rtol: 1e-12,
        ..Default::default()
    };
    for f in [
        l::cg,
        l::bicg,
        l::bicgstab,
        l::cgs,
        l::gmres,
        l::lgmres,
        l::gcrotmk,
    ] {
        assert!(f(&a, &b, &opts).unwrap().converged);
    }
    let opts = l::SolverOptions {
        shift: 1.,
        rtol: 1e-12,
        ..Default::default()
    };
    let out = l::minres(&a, &b, &opts).unwrap();
    let shifted = s::add(&a, &s::scale(&s::identity(2).unwrap(), -1.).unwrap()).unwrap();
    close(
        &s::matvec(&shifted, &out.x).unwrap(),
        &b,
        1e-11,
        "shifted minres",
    );
    let invalid = l::LinearOperator::new(2, 2, |_| Ok(vec![1.]), |_| Ok(vec![1., 2.])).unwrap();
    assert!(l::cg(&invalid, &b, &Default::default()).is_err());
    let nonfinite =
        l::LinearOperator::new(2, 2, |_| Ok(vec![f64::NAN; 2]), |_| Ok(vec![0.; 2])).unwrap();
    assert!(l::cg(&nonfinite, &b, &Default::default()).is_err());
    let singular = s::from_coo(2, 2, &[]).unwrap();
    assert!(l::bicg(&singular, &b, &Default::default()).is_err());
    for f in [l::lsqr, l::lsmr] {
        assert!(
            f(
                &a,
                &b,
                &l::LeastSquaresOptions {
                    damp: -1.,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(f(&a, &[f64::INFINITY, 1.], &Default::default()).is_err());
        let out = f(&singular, &b, &Default::default()).unwrap();
        assert!(out.converged);
        assert_eq!(out.x, [0., 0.]);
        let out = f(&a, &[0., 0.], &Default::default()).unwrap();
        assert_eq!(out.iterations, 0);
        assert!(out.converged);
        let out = f(
            &a,
            &b,
            &l::LeastSquaresOptions {
                maxiter: 0,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(!out.converged);
    }
    let ilu = l::spilu(&a, 0., 20.).unwrap();
    close(
        &ilu.apply(&b).unwrap(),
        &l::spsolve(&a, &b).unwrap(),
        1e-12,
        "ILU inverse",
    );
    close(
        &ilu.transpose_apply(&b).unwrap(),
        &l::spsolve(&s::transpose(&a).unwrap(), &b).unwrap(),
        1e-12,
        "ILU transpose inverse",
    );
    assert!(l::spilu(&a, f64::NAN, 2.).is_err());
    assert!(l::spilu(&a, 0., 0.5).is_err());
    assert!(l::spilu(&singular, 0., 2.).is_err());
}

#[test]
fn spectral_and_array_domains() {
    let a = s::from_dense(&[vec![2., 1.], vec![1., 3.]]).unwrap();
    for spectrum in [
        l::Spectrum::LargestMagnitude,
        l::Spectrum::SmallestMagnitude,
        l::Spectrum::LargestReal,
        l::Spectrum::SmallestReal,
        l::Spectrum::BothEnds,
    ] {
        let e = l::eigsh(&a, 1, spectrum).unwrap();
        assert_eq!(e.eigenvalues.len(), 1);
    }
    assert!(l::eigs(&a, 0, l::Spectrum::LargestMagnitude).is_err());
    assert!(l::eigsh(&a, 3, l::Spectrum::SmallestReal).is_err());
    assert!(l::eigsh(&a, 1, l::Spectrum::LargestImaginary).is_err());
    assert!(l::svds(&a, 0, true).is_err());
    assert_eq!(l::svds(&a, 1, false).unwrap().singular_values.len(), 1);
    let asymmetric = s::from_coo(2, 2, &[(0, 1, 1.)]).unwrap();
    assert!(l::eigsh(&asymmetric, 1, l::Spectrum::LargestReal).is_err());
    assert!(l::lobpcg(&a, &vec![vec![1.]], true).is_err());
    assert!(l::expm(&s::from_coo(1, 2, &[]).unwrap()).is_err());
    assert!(l::expm_multiply(&a, &[1.], &[1.]).is_err());
    assert!(l::expm_multiply(&a, &[1., 2.], &[f64::NAN]).is_err());
    assert!(l::norm_axis(&a, f64::NAN, 0).is_err());
    assert!(l::norm_axis(&a, 2., 2).is_err());
    let invalid = l::KrylovOptions {
        max_basis: 0,
        ..Default::default()
    };
    assert!(l::funm_multiply_krylov(&a, &[1., 2.], |z| z, invalid).is_err());
    let y = l::funm_multiply_krylov(&a, &[1., 2.], |z| z, Default::default()).unwrap();
    close(
        &y.iter().map(|v| v.re).collect::<Vec<_>>(),
        &s::matvec(&a, &[1., 2.]).unwrap(),
        1e-12,
        "function identity",
    );
    let nd = s::expand_dims(&s::as_array(&a).unwrap(), 0).unwrap();
    assert_eq!(s::permute_dims(&nd, &[2, 0, 1]).unwrap().shape, [2, 1, 2]);
    assert!(s::expand_dims(&nd, 4).is_err());
    assert!(s::permute_dims(&nd, &[0, 0, 2]).is_err());
    assert!(s::swapaxes(&nd, isize::MIN, 0).is_err());
    let malformed = s::SparseArray {
        shape: vec![1],
        coordinates: vec![vec![2]],
        data: vec![1.],
    };
    assert!(s::expand_dims(&malformed, 0).is_err());
    assert!(s::issparse(&malformed));
    assert!(s::get_index_dtype(&[&[u64::MAX]], None).is_err());
    assert_eq!(
        s::get_index_dtype(&[], Some(i32::MAX as u64 + 1)).unwrap(),
        s::IndexDtype::I64
    );
    assert!(s::random_array(&[3, 3], f64::NAN, 1).is_err());
    assert!(s::random_array(&[3, 3], -1., 1).is_err());
    assert!(s::random_array(&[], 0., 1).is_err());
    assert!(s::random_array(&[1_000_000, 1_000_000], 0., 1).is_err());
    let mut shape = vec![1_000_000; 32];
    shape[31] = 0;
    assert!(s::random_array(&shape, 1., 1).unwrap().data.is_empty());
    assert!(s::spdiags(&[vec![1.], vec![2.]], &[0, 0], 2, 2).is_err());
    assert!(s::spdiags(&[vec![f64::INFINITY]], &[0], 1, 1).is_err());
    assert!(
        s::spdiags(&[vec![1.]], &[isize::MIN], 2, 2)
            .unwrap()
            .data
            .is_empty()
    );
}

#[test]
fn zero_axis_norms_and_polynomial_actions() {
    let zero = s::from_coo(2, 2, &[(0, 0, 0.)]).unwrap();
    assert_eq!(l::norm_axis(&zero, 2., 1).unwrap(), [0., 0.]);
    assert_eq!(l::norm_axis(&zero, -1., 1).unwrap(), [0., f64::INFINITY]);
    let a = s::from_dense(&[
        vec![2., 1., 0., 0.],
        vec![1., 3., 1., 0.],
        vec![0., 1., 4., 1.],
        vec![0., 0., 1., 5.],
    ])
    .unwrap();
    let b = [1., 0., 0., 0.];
    let y = l::funm_multiply_krylov(
        &a,
        &b,
        |mut z| {
            let re = z.re;
            z.re = 1. + re.powi(4);
            z.im = 0.;
            z
        },
        Default::default(),
    )
    .unwrap();
    let expected = s::matvec(&l::matrix_power(&a, 4).unwrap(), &b).unwrap();
    let mut expected = expected;
    expected[0] += 1.;
    close(
        &y.iter().map(|v| v.re).collect::<Vec<_>>(),
        &expected,
        1e-10,
        "polynomial Arnoldi action",
    );
}

#[test]
fn npz_numeric_types_and_block_duplicates() {
    for line in include_str!("fixtures/sparse_npz_expected.csv")
        .lines()
        .skip(1)
    {
        let r: Vec<_> = line.split(',').collect();
        let bytes = std::fs::read(format!(
            "{}/tests/fixtures/{}.npz",
            env!("CARGO_MANIFEST_DIR"),
            r[0]
        ))
        .unwrap();
        let a = s::load_npz(std::io::Cursor::new(bytes)).unwrap();
        assert_eq!(
            (a.rows, a.cols),
            (r[1].parse().unwrap(), r[2].parse().unwrap())
        );
        close(&flat(&s::to_dense(&a).unwrap()), &nums(r[3]), 0., r[0]);
    }
}

#[test]
fn graph_work_counts_are_checked_before_search() {
    // On 32-bit targets the all-pairs Bellman-Ford count is exactly 2^32.
    // It must return an error before multiplication can wrap or search begins.
    let a = s::from_dense(&vec![vec![1.; 256]; 256]).unwrap();
    assert!(g::bellman_ford(&a, &Default::default()).is_err());
    assert!(g::yen(&a, 0, 255, 64, true, false).is_err());
}

#[test]
fn automatic_paths_validate_selected_method() {
    let a = s::from_coo(3, 3, &[(0, 1, 10.), (1, 2, -9.)]).unwrap();
    let base = g::PathOptions {
        indices: Some(vec![0]),
        ..Default::default()
    };
    let paths = g::shortest_path(&a, &base).unwrap();
    assert_eq!(paths.distances, [vec![0., 10., 1.]]);
    assert_eq!(paths.predecessors, [vec![None, Some(0), Some(1)]]);
    for opts in [
        g::PathOptions {
            limit: 5.,
            ..base.clone()
        },
        g::PathOptions {
            min_only: true,
            ..base.clone()
        },
        g::PathOptions {
            limit: 5.,
            min_only: true,
            ..base.clone()
        },
    ] {
        assert!(matches!(
            g::shortest_path(&a, &opts),
            Err(sci_core::Error::InvalidInput(_))
        ));
        // Unweighted conversion removes negative costs before method selection.
        let unweighted = g::PathOptions {
            unweighted: true,
            ..opts
        };
        let out = g::shortest_path(&a, &unweighted).unwrap();
        assert_eq!(out.distances, [vec![0., 1., 2.]]);
        assert_eq!(out.predecessors, [vec![None, Some(0), Some(1)]]);
    }
}

#[test]
fn transpose_action_rectangular_validation_and_overflow() {
    let a = s::from_dense(&[vec![1., 0., 2.], vec![0., 3., -1.]]).unwrap();
    let x = [4., 5.];
    assert_eq!(a.transpose_apply(&x).unwrap(), [4., 15., 3.]);
    assert_eq!(
        a.transpose_apply(&x).unwrap(),
        s::matvec(&s::transpose(&a).unwrap(), &x).unwrap()
    );
    assert!(a.transpose_apply(&[1.]).is_err());
    assert!(a.transpose_apply(&[f64::NAN, 1.]).is_err());
    let mut invalid = a.clone();
    invalid.indices[0] = a.cols;
    assert!(invalid.transpose_apply(&x).is_err());
    let product = s::from_coo(1, 1, &[(0, 0, f64::MAX)]).unwrap();
    assert!(product.transpose_apply(&[2.]).is_err());
    let sum = s::from_coo(2, 1, &[(0, 0, f64::MAX), (1, 0, f64::MAX)]).unwrap();
    assert!(sum.transpose_apply(&[1., 1.]).is_err());
    assert_eq!(
        s::from_coo(0, 3, &[])
            .unwrap()
            .transpose_apply(&[])
            .unwrap(),
        [0.; 3]
    );
    assert!(
        s::from_coo(2, 0, &[])
            .unwrap()
            .transpose_apply(&[1., 2.])
            .unwrap()
            .is_empty()
    );
}

#[test]
fn projected_rank_loss_checks_true_residual() {
    for f in [l::gmres, l::lgmres, l::gcrotmk, l::minres] {
        let a = s::from_dense(&[vec![1., 0.], vec![0., 0.]]).unwrap();
        let opts = l::SolverOptions {
            atol: 0.,
            rtol: 1e-10,
            maxiter: 2,
            restart: 2,
            ..Default::default()
        };
        let out = f(&a, &[1., 1.], &opts).unwrap();
        assert!(!out.converged);
        assert_eq!(out.iterations, 2);
        // Independent exact least-squares solution, with unavoidable residual 1.
        close(&out.x, &[1., 0.], 1e-12, "deficient projection");
        assert!((out.residual_norm - 1.).abs() < 1e-12);
        let zero = s::from_coo(2, 2, &[]).unwrap();
        let out = f(&zero, &[1., 1.], &opts).unwrap();
        assert!(!out.converged);
        assert_eq!(out.x, [0., 0.]);
        assert_eq!(out.iterations, 2);
    }
}

#[test]
fn exponential_action_time_buffer_is_reinitialized() {
    let a = s::from_dense(&[vec![0., 1.], vec![0., 0.]]).unwrap();
    // A²=0 implies exp(t A)b = b + t A b exactly.
    let actual = l::expm_multiply(&a, &[1., 2.], &[1., -2., 0., 1.]).unwrap();
    assert_eq!(actual.len(), 4);
    // Allow a few ulps for the dense Pade implementation's roundoff.
    for (row, expected) in actual.iter().zip([[3., 2.], [-3., 2.], [1., 2.], [3., 2.]]) {
        close(row, &expected, 4e-15, "nilpotent exponential action");
    }
    assert!(l::expm_multiply(&a, &[1., 2.], &[]).unwrap().is_empty());
    let overflow = s::from_dense(&[vec![f64::MAX]]).unwrap();
    assert!(l::expm_multiply(&overflow, &[1.], &[2.]).is_err());
    assert!(l::expm_multiply(&a, &[1., 2.], &[0.; 65]).is_err());
}

#[test]
fn givens_projection_with_nontrivial_preconditioning() {
    let n = 24;
    let expected: Vec<_> = (0..n).map(|i| (i + 1) as f64).collect();
    let mut entries = vec![];
    let mut b = vec![0.; n];
    for i in 0..n {
        let diagonal = 2. + i as f64;
        entries.push((i, i, diagonal));
        b[i] = diagonal * expected[i];
        if i > 0 {
            entries.push((i, i - 1, -0.5));
            b[i] -= 0.5 * expected[i - 1];
        }
        if i + 1 < n {
            entries.push((i, i + 1, -0.5));
            b[i] -= 0.5 * expected[i + 1];
        }
    }
    let a = s::from_coo(n, n, &entries).unwrap();
    let diagonal_inverse = |x: &[f64]| {
        Ok(x.iter()
            .enumerate()
            .map(|(i, v)| v / (2. + i as f64))
            .collect())
    };
    let m = l::LinearOperator::new(n, n, diagonal_inverse, diagonal_inverse).unwrap();
    for f in [l::gmres, l::lgmres, l::gcrotmk] {
        for preconditioner in [None, Some(m.clone())] {
            let opts = l::SolverOptions {
                atol: 0.,
                rtol: 1e-12,
                maxiter: n,
                restart: n,
                x0: Some(vec![0.25; n]),
                preconditioner,
                ..Default::default()
            };
            let out = f(&a, &b, &opts).unwrap();
            assert!(out.converged);
            close(&out.x, &expected, 2e-10, "preconditioned projected solve");
            let norm = b.iter().fold(0f64, |n, &v| n.hypot(v));
            let residual = s::matvec(&a, &out.x)
                .unwrap()
                .iter()
                .zip(&b)
                .fold(0f64, |n, (a, b)| n.hypot(a - b));
            assert!(residual / norm <= opts.rtol);
        }
    }
}
