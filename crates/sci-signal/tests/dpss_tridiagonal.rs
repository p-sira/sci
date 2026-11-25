use sci_signal::windows::dpss;
use std::f64::consts::TAU;

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
        "actual {actual:.17e}, expected {expected:.17e}, tolerance {tolerance:.2e}"
    );
}

fn compare_scipy(m: usize, nw: f64, symmetric: bool, expected: &[&[f64]]) {
    let actual = dpss(m, nw, expected.len(), symmetric).unwrap();
    assert_eq!(actual.len(), expected.len());
    for (taper, expected) in actual.iter().zip(expected) {
        assert_eq!(taper.len(), m);
        assert_eq!(taper.len(), expected.len());
        for (&actual, &expected) in taper.iter().zip(*expected) {
            close(actual, expected, 2e-11);
        }
    }
}

#[test]
fn dpss_matches_independent_scipy_reference_tapers() {
    // SciPy windows.dpss(..., Kmax=3, norm=2), with values from the checked-in
    // Python reference fixtures tests/fixtures/signal_expected.csv (188-193).
    compare_scipy(
        5,
        1.5,
        true,
        &[
            &[
                0.17450710526715613,
                0.49567491774329436,
                0.6691093275862416,
                0.49567491774329436,
                0.17450710526715613,
            ],
            &[
                0.43994933484723864,
                0.5535743696807796,
                0.0,
                -0.5535743696807796,
                -0.43994933484723864,
            ],
            &[
                0.6314527564981198,
                0.07328023879083398,
                -0.4379438843360748,
                0.07328023879083397,
                0.6314527564981198,
            ],
        ],
    );
    compare_scipy(
        5,
        1.5,
        false,
        &[
            &[
                0.1329448709959633,
                0.37656498596981974,
                0.5835447477421214,
                0.5835447477421214,
                0.37656498596981974,
            ],
            &[
                0.36936475106520456,
                0.5414804240761459,
                0.2652708634826988,
                -0.2652708634826989,
                -0.5414804240761459,
            ],
            &[
                0.5881286780513311,
                0.25494658594275504,
                -0.2985077827624051,
                -0.2985077827624051,
                0.25494658594275504,
            ],
        ],
    );
    compare_scipy(
        6,
        1.5,
        true,
        &[
            &[
                0.1329448709959633,
                0.37656498596981974,
                0.5835447477421214,
                0.5835447477421214,
                0.37656498596981974,
                0.1329448709959633,
            ],
            &[
                0.36936475106520456,
                0.5414804240761459,
                0.2652708634826988,
                -0.2652708634826989,
                -0.5414804240761459,
                -0.36936475106520456,
            ],
            &[
                0.5881286780513311,
                0.25494658594275504,
                -0.2985077827624051,
                -0.2985077827624051,
                0.25494658594275504,
                0.5881286780513311,
            ],
        ],
    );
    compare_scipy(
        6,
        1.5,
        false,
        &[
            &[
                0.10714176817722658,
                0.29489073899828905,
                0.4883183198029346,
                0.5711485132778329,
                0.4883183198029346,
                0.29489073899828905,
            ],
            &[
                0.31971185119340145,
                0.49876856686761145,
                0.38602363776240106,
                0.0,
                -0.38602363776240106,
                -0.49876856686761145,
            ],
            &[
                0.5471005659694719,
                0.3478828081923798,
                -0.1181249614337886,
                -0.36250521642131833,
                -0.1181249614337886,
                0.3478828081923798,
            ],
        ],
    );
}

fn legacy_dense_nalgebra(m: usize, nw: f64, k: usize, symmetric: bool) -> Vec<Vec<f64>> {
    // Independent copy of the former dense solver: normalize the matrix,
    // tridiagonalize with nalgebra, sort descending, orient, then truncate.
    // Keep this reference in tests to catch changes in the public convention.
    let n = m + usize::from(!symmetric);
    let w = nw / n as f64;
    let mut matrix = nalgebra::DMatrix::zeros(n, n);
    for i in 0..n {
        matrix[(i, i)] = ((n as f64 - 1.0 - 2.0 * i as f64) / 2.0).powi(2) * (TAU * w).cos();
        if i > 0 {
            let subdiag = (i * (n - i)) as f64 / 2.0;
            matrix[(i, i - 1)] = subdiag;
            matrix[(i - 1, i)] = subdiag;
        }
    }
    let eig = nalgebra::linalg::SymmetricEigen::try_new(matrix, 1e-14, 100_000).unwrap();
    let mut indices: Vec<_> = (0..n).collect();
    indices.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));
    indices
        .into_iter()
        .take(k)
        .enumerate()
        .map(|(order, col)| {
            let mut taper: Vec<_> = eig.eigenvectors.column(col).iter().copied().collect();
            let sign = if order.is_multiple_of(2) {
                taper.iter().sum::<f64>()
            } else {
                taper
                    .iter()
                    .copied()
                    .find(|x| x.abs() > 1e-7)
                    .unwrap_or(1.0)
            };
            if sign < 0.0 {
                taper.iter_mut().for_each(|x| *x = -*x);
            }
            taper.truncate(m);
            taper
        })
        .collect()
}

#[test]
fn dpss_matches_original_dense_eigensolver_on_regular_inputs() {
    for (m, nw, k, symmetric) in [
        (2, 0.6, 2, true),
        (5, 1.5, 3, true),
        (6, 1.5, 3, false),
        (12, 1.5, 3, true),
        (24, 3.2, 8, true),
        (48, 2.5, 4, true),
        (48, 2.5, 4, false),
        (126, 2.5, 4, false),
        (127, 2.5, 4, false),
    ] {
        let actual = dpss(m, nw, k, symmetric).unwrap();
        let reference = legacy_dense_nalgebra(m, nw, k, symmetric);
        for (i, (taper, expected)) in actual.iter().zip(reference.iter()).enumerate() {
            for (&value, &want) in taper.iter().zip(expected.iter()) {
                assert!(
                    (value - want).abs() < 3e-9,
                    "m={m}, NW={nw}, k={k}, sym={symmetric}, order={i}: {value} vs {want}"
                );
            }
        }
    }
}

fn tridiagonal(n: usize, nw: f64) -> (Vec<f64>, Vec<f64>) {
    let c = (TAU * nw / n as f64).cos();
    let diagonal = (0..n)
        .map(|i| {
            let x = (n as f64 - 1.0 - 2.0 * i as f64) / 2.0;
            x * x * c
        })
        .collect();
    let off_diagonal = (1..n).map(|i| (i * (n - i)) as f64 / 2.0).collect();
    (diagonal, off_diagonal)
}

fn full_taper(taper: &[f64], order: usize, symmetric: bool) -> Vec<f64> {
    let mut full = taper.to_vec();
    if !symmetric {
        // The original DPSS Jacobi matrix commutes with reversal; its simple
        // descending eigenpairs alternate reflection parity. Reconstruct the
        // discarded final sample independently of the implementation.
        full.push(if order.is_multiple_of(2) {
            taper[0]
        } else {
            -taper[0]
        });
    }
    full
}

fn check_eigenpairs(m: usize, nw: f64, k: usize, symmetric: bool) {
    let n = m + usize::from(!symmetric);
    let tapers = dpss(m, nw, k, symmetric).unwrap();
    let full: Vec<_> = tapers
        .iter()
        .enumerate()
        .map(|(order, v)| {
            assert_eq!(v.len(), m);
            full_taper(v, order, symmetric)
        })
        .collect();
    let (d, e) = tridiagonal(n, nw);
    let scale = d
        .iter()
        .chain(&e)
        .fold(1.0_f64, |a, &value| a.max(value.abs()));
    for (order, v) in full.iter().enumerate() {
        assert_eq!(v.len(), n);
        let norm = v.iter().map(|x| x * x).sum::<f64>();
        close(norm, 1.0, 2e-10);
        for i in 0..n {
            let mirrored = if order.is_multiple_of(2) {
                v[n - 1 - i]
            } else {
                -v[n - 1 - i]
            };
            close(v[i], mirrored, 3e-9);
        }
        let av: Vec<f64> = (0..n)
            .map(|i| {
                d[i] * v[i]
                    + if i > 0 { e[i - 1] * v[i - 1] } else { 0.0 }
                    + if i + 1 < n { e[i] * v[i + 1] } else { 0.0 }
            })
            .collect();
        let eigenvalue = av.iter().zip(v).map(|(a, x)| a * x).sum::<f64>() / norm;
        let residual = av
            .iter()
            .zip(v)
            .map(|(a, x)| (a - eigenvalue * x).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            residual / scale < 2e-10,
            "m={m} NW={nw} k={k} sym={symmetric} order={order} residual={residual:.3e}"
        );

        let expected_sign = if order.is_multiple_of(2) {
            v.iter().sum::<f64>()
        } else {
            v.iter().copied().find(|x| x.abs() > 1e-7).unwrap_or(1.0)
        };
        assert!(
            expected_sign >= 0.0,
            "m={m} order={order} sign={expected_sign}"
        );
    }
    for (i, v) in full.iter().enumerate() {
        for (j, w) in full.iter().enumerate() {
            let dot = v.iter().zip(w).map(|(a, b)| a * b).sum::<f64>();
            close(dot, if i == j { 1.0 } else { 0.0 }, 2e-10);
        }
    }
}

#[test]
fn dpss_eigenvectors_satisfy_independent_residual_and_orthogonality() {
    for (m, nw, k, symmetric) in [
        (2, 0.6, 2, true),
        (3, 1.2, 3, true),
        (5, 1.5, 3, true),
        (6, 1.5, 3, false),
        (47, 2.5, 4, true),
        (48, 2.5, 4, true),
        (48, 2.5, 4, false),
        (49, 3.25, 6, true),
        (24, 3.2, 24, true),
        (126, 2.5, 4, false), // faer QR at n=127
        (127, 2.5, 4, false), // nalgebra fallback at n=128
    ] {
        check_eigenpairs(m, nw, k, symmetric);
    }
}

#[test]
fn dpss_near_bandwidth_endpoints_remains_orthogonal() {
    for nw in [1e-8, 1e-4, 2.5, 23.9999, 24.0 - 1e-8] {
        check_eigenpairs(48, nw, 4, true);
    }
}

#[test]
fn dpss_periodic_256_preserves_full_order_257_eigenvectors() {
    // The original fallback must handle n=m+1=257, above the linalg crate's
    // separate 256-order bound; this also checks periodic sign before truncation.
    check_eigenpairs(256, 2.5, 4, false);
}

#[test]
fn dpss_retains_original_input_domain() {
    for (m, nw, k, symmetric) in [
        (1, 0.1, 1, true),
        (257, 1.0, 1, true),
        (48, 0.0, 2, true),
        (48, 24.0, 2, true),
        (48, f64::NAN, 2, true),
        (48, f64::INFINITY, 2, false),
        (48, 2.5, 0, true),
        (48, 2.5, 49, false),
    ] {
        assert!(dpss(m, nw, k, symmetric).is_err());
    }
}
