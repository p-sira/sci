use s::linalg as l;
use sci::{Error, SparseMatrix, sparse as s};

fn numbers(s: &str) -> Vec<f64> {
    if s.is_empty() {
        Vec::new()
    } else {
        s.split(';').map(|x| x.parse().unwrap()).collect()
    }
}
fn triples(s: &str) -> Vec<(usize, usize, f64)> {
    if s.is_empty() {
        Vec::new()
    } else {
        s.split(':')
            .map(|row| {
                let v = numbers(row);
                (v[0] as usize, v[1] as usize, v[2])
            })
            .collect()
    }
}
fn raw_dense(rows: usize, cols: usize, t: &[(usize, usize, f64)]) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; cols]; rows];
    for &(r, c, v) in t {
        out[r][c] += v;
    }
    out
}
fn assert_close(actual: f64, expected: f64, context: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{context}");
    } else if expected.is_infinite() {
        assert_eq!(actual, expected, "{context}");
    } else {
        // Relative-only comparisons near zero also exercise subnormal-scale data.
        let abs = if expected != 0.0 && expected.abs() < 1e-100 {
            0.0
        } else {
            3e-12
        };
        assert!(
            (actual - expected).abs() <= abs + 3e-12 * expected.abs(),
            "{context}: actual {actual}, expected {expected}"
        );
    }
}

#[test]
fn scipy_fixtures_cover_public_functions() {
    let inputs: Vec<_> = include_str!("fixtures/sparse_inputs.csv")
        .lines()
        .skip(1)
        .collect();
    let expected: Vec<_> = include_str!("fixtures/sparse_expected.csv")
        .lines()
        .skip(1)
        .collect();
    assert_eq!(inputs.len(), expected.len());
    let mut count = 0;
    for (line, reference) in inputs.into_iter().zip(&expected) {
        let f: Vec<_> = line.split(',').collect();
        let e: Vec<_> = reference.split(',').collect();
        assert_eq!(f[0], e[0]);
        let api = f[1];
        let rows = f[2].parse().unwrap();
        let cols = f[3].parse().unwrap();
        let t = triples(f[4]);
        let a = s::from_coo(rows, cols, &t).unwrap();
        let b = s::from_coo(f[5].parse().unwrap(), f[6].parse().unwrap(), &triples(f[7])).unwrap();
        let k = numbers(f[9]);
        let scalar = k.first().copied().unwrap_or(0.0);
        let x = if matches!(api, "diags" | "diags_array") {
            Vec::new()
        } else {
            numbers(f[8])
        };
        let matrix: Option<SparseMatrix> = match api {
            "from_coo" => Some(s::from_coo(rows, cols, &t).unwrap()),
            "from_csr" => Some(s::from_csr(rows, cols, &a.indptr, &a.indices, &a.data).unwrap()),
            "from_csc" => {
                // Build CSC directly from the input coordinates, independently of Sci transpose.
                let mut t = t.clone();
                t.sort_by_key(|&(r, c, _)| (c, r));
                let mut p = vec![0; cols + 1];
                let mut indices = Vec::new();
                let mut data = Vec::new();
                for (r, c, v) in t {
                    p[c + 1] += 1;
                    indices.push(r);
                    data.push(v);
                }
                for j in 0..cols {
                    p[j + 1] += p[j];
                }
                Some(s::from_csc(rows, cols, &p, &indices, &data).unwrap())
            }
            "from_dense" => Some(s::from_dense(&raw_dense(rows, cols, &t)).unwrap()),
            "to_dense" | "validate" => {
                s::validate(&a).unwrap();
                Some(a.clone())
            }
            "eye" => Some(s::eye(rows, cols, scalar as isize).unwrap()),
            "eye_array" => Some(s::eye_array(rows, cols, scalar as isize).unwrap()),
            "identity" => Some(s::identity(rows).unwrap()),
            "diags" | "diags_array" => {
                let diagonals: Vec<_> = f[8].split(':').map(numbers).collect();
                let offsets: Vec<_> = k.iter().map(|&v| v as isize).collect();
                Some(
                    if api == "diags" {
                        s::diags(rows, cols, &diagonals, &offsets)
                    } else {
                        s::diags_array(rows, cols, &diagonals, &offsets)
                    }
                    .unwrap(),
                )
            }
            "transpose" => Some(s::transpose(&a).unwrap()),
            "matrix_transpose" => Some(s::matrix_transpose(&a).unwrap()),
            "eliminate_zeros" => Some(s::eliminate_zeros(&a).unwrap()),
            "tril" => Some(s::tril(&a, scalar as isize).unwrap()),
            "triu" => Some(s::triu(&a, scalar as isize).unwrap()),
            "scale" => Some(s::scale(&a, scalar).unwrap()),
            "add" => Some(s::add(&a, &b).unwrap()),
            "multiply" => Some(s::multiply(&a, &b).unwrap()),
            "matmul" => Some(s::matmul(&a, &b).unwrap()),
            "hstack" => Some(s::hstack(&[&a, &b]).unwrap()),
            "vstack" => Some(s::vstack(&[&a, &b]).unwrap()),
            "block_diag" => Some(s::block_diag(&[&a, &b]).unwrap()),
            "block_array" => Some(s::block_array(&[vec![&a, &b], vec![&b, &a]]).unwrap()),
            "bmat" => Some(s::bmat(&[vec![&a, &b], vec![&b, &a]]).unwrap()),
            "kron" => Some(s::kron(&a, &b).unwrap()),
            "kronsum" => Some(s::kronsum(&a, &b).unwrap()),
            "inv" => Some(l::inv(&a).unwrap()),
            "matrix_power" => Some(l::matrix_power(&a, scalar as u32).unwrap()),
            _ => None,
        };
        let (actual_rows, actual_cols, values) = if let Some(matrix) = matrix {
            let dense = s::to_dense(&matrix).unwrap();
            (
                matrix.rows,
                matrix.cols,
                dense.into_iter().flatten().collect::<Vec<_>>(),
            )
        } else {
            let values = match api {
                "find" => s::find(&a)
                    .unwrap()
                    .into_iter()
                    .flat_map(|(r, c, v)| [r as f64, c as f64, v])
                    .collect(),
                "matvec" => s::matvec(&a, &x).unwrap(),
                "spsolve" => l::spsolve(&a, &x).unwrap(),
                "splu" | "lu_solve" => l::splu(&a).unwrap().solve(&x).unwrap(),
                "factorized" => l::factorized(&a).unwrap().solve(&x).unwrap(),
                "spsolve_triangular" => l::spsolve_triangular(
                    &a,
                    &x,
                    matches!(scalar as i32, 0 | 2),
                    matches!(scalar as i32, 2 | 3),
                )
                .unwrap(),
                "norm" => vec![
                    l::norm(
                        &a,
                        match scalar as i32 {
                            0 => l::MatrixNorm::Frobenius,
                            1 => l::MatrixNorm::One,
                            2 => l::MatrixNorm::Infinity,
                            3 => l::MatrixNorm::NegOne,
                            4 => l::MatrixNorm::NegInfinity,
                            _ => unreachable!(),
                        },
                    )
                    .unwrap(),
                ],
                "spbandwidth" => {
                    let (l, u) = l::spbandwidth(&a).unwrap();
                    vec![l as f64, u as f64]
                }
                "is_sptriangular" => {
                    let (l, u) = l::is_sptriangular(&a).unwrap();
                    vec![u8::from(l) as f64, u8::from(u) as f64]
                }
                _ => panic!("unknown fixture API: {api}"),
            };
            if api == "find" {
                (values.len() / 3, 3, values)
            } else {
                (1, values.len(), values)
            }
        };
        assert_eq!(
            (actual_rows, actual_cols),
            (e[1].parse().unwrap(), e[2].parse().unwrap()),
            "{} shape",
            f[0]
        );
        let wanted = numbers(e[3]);
        assert_eq!(values.len(), wanted.len());
        for (i, (v, w)) in values.into_iter().zip(wanted).enumerate() {
            if matches!(api, "spbandwidth" | "is_sptriangular") || (api == "find" && i % 3 != 2) {
                assert_eq!(v, w, "{} index/structure", f[0]);
            } else {
                assert_close(v, w, f[0]);
            }
        }
        count += 1;
    }
    assert_eq!(count, expected.len());
    assert!(count >= 100);
}

#[test]
fn storage_validation_and_canonicalization() {
    let a = s::from_csr(2, 3, &[0, 4, 4], &[2, 0, 2, 1], &[3.0, 2.0, -3.0, -0.0]).unwrap();
    assert_eq!(a.indptr, vec![0, 3, 3]);
    assert_eq!(a.indices, vec![0, 1, 2]);
    assert_eq!(a.data, vec![2.0, -0.0, 0.0]);
    assert!(a.data[1].is_sign_negative());
    assert_eq!(s::eliminate_zeros(&a).unwrap().indices, vec![0]);
    let mut bad = a.clone();
    bad.indptr[0] = 1;
    assert!(s::validate(&bad).is_err());
    let mut bad = a.clone();
    bad.indptr[1] = 99;
    assert!(s::validate(&bad).is_err());
    let mut bad = a.clone();
    bad.indptr[1] = 2;
    bad.indptr[2] = 1;
    assert!(s::validate(&bad).is_err());
    let mut bad = a.clone();
    bad.indices[1] = 0;
    assert!(s::validate(&bad).is_err());
    let mut bad = a.clone();
    bad.indices[2] = 3;
    assert!(s::validate(&bad).is_err());
    for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut bad = a.clone();
        bad.data[0] = v;
        assert!(s::validate(&bad).is_err());
        assert!(s::from_coo(1, 1, &[(0, 0, v)]).is_err());
        assert!(s::from_dense(&[vec![v]]).is_err());
        assert!(s::scale(&a, v).is_err());
        assert!(s::matvec(&a, &[v, 0.0, 0.0]).is_err());
    }
    assert!(s::from_coo(1, 1, &[(1, 0, 1.0)]).is_err());
    assert!(s::from_coo(1, 1, &[(0, 1, 1.0)]).is_err());
    assert!(s::from_coo(1, 1, &[(0, 0, f64::MAX), (0, 0, f64::MAX)]).is_err());
    for p in [&[0, 2][..], &[1, 1], &[0, 1, 1], &[0, 0]] {
        assert!(s::from_csr(1, 1, p, &[0], &[1.0]).is_err());
        assert!(s::from_csc(1, 1, p, &[0], &[1.0]).is_err());
    }
    assert!(s::from_csr(1, 1, &[0, 1], &[], &[1.0]).is_err());
    assert!(s::from_dense(&[vec![1.0], vec![]]).is_err());
}

#[test]
fn empty_axes_offsets_and_shapes() {
    for (rows, cols) in [(0, 0), (0, 4), (3, 0)] {
        let a = s::from_coo(rows, cols, &[]).unwrap();
        s::validate(&a).unwrap();
        assert_eq!(s::transpose(&s::transpose(&a).unwrap()).unwrap(), a);
        assert_eq!(s::matvec(&a, &vec![0.0; cols]).unwrap(), vec![0.0; rows]);
        assert_eq!(l::norm(&a, l::MatrixNorm::Frobenius).unwrap(), 0.0);
        assert!(l::norm(&a, l::MatrixNorm::One).is_err());
        if rows == 0 && cols > 0 {
            assert!(s::to_dense(&a).is_err());
        } else {
            assert_eq!(s::to_dense(&a).unwrap().len(), rows);
        }
    }
    for offset in [isize::MIN, isize::MAX] {
        assert!(s::eye(2, 3, offset).unwrap().data.is_empty());
        assert!(
            s::diags(2, 3, &[vec![]], &[offset])
                .unwrap()
                .data
                .is_empty()
        );
    }
    assert_eq!(s::diags(0, 0, &[], &[]).unwrap(), s::identity(0).unwrap());
    assert!(s::diags(2, 2, &[vec![1.0]], &[0]).is_err());
    assert!(s::diags(2, 2, &[vec![1.0; 2], vec![2.0; 2]], &[0, 0]).is_err());
    assert!(s::diags(2, 2, &[], &[0]).is_err());
    assert!(s::hstack(&[]).is_err());
    assert!(s::vstack(&[]).is_err());
    assert!(s::block_diag(&[]).is_err());
    assert!(s::block_array(&[]).is_err());
    assert!(s::block_array(&[vec![]]).is_err());
    let a = s::identity(2).unwrap();
    let b = s::eye(1, 3, 0).unwrap();
    assert!(s::hstack(&[&a, &b]).is_err());
    assert!(s::vstack(&[&a, &b]).is_err());
    assert!(s::block_array(&[vec![&a], vec![&a, &a]]).is_err());
    assert!(s::block_array(&[vec![&a, &b]]).is_err());
    assert_eq!(s::add(&a, &b), Err(Error::DimensionMismatch));
    assert!(s::multiply(&a, &b).is_err());
    assert!(s::matmul(&a, &b).is_err());
    assert!(s::kronsum(&a, &b).is_err());
    assert!(s::matvec(&a, &[1.0]).is_err());
}

#[test]
fn numerical_failures_and_work_limits() {
    let max = s::from_coo(1, 1, &[(0, 0, f64::MAX)]).unwrap();
    assert!(s::scale(&max, 2.0).is_err());
    assert!(s::add(&max, &max).is_err());
    assert!(s::multiply(&max, &max).is_err());
    assert!(s::matmul(&max, &max).is_err());
    assert!(s::matvec(&max, &[2.0]).is_err());
    assert!(s::kron(&max, &max).is_err());
    let a = s::identity(2).unwrap();
    let zero = s::eye(2, 2, 9).unwrap();
    assert!(l::splu(&zero).is_err());
    assert!(l::inv(&zero).is_err());
    assert!(l::spsolve(&zero, &[1.0, 2.0]).is_err());
    assert!(l::spsolve(&a, &[1.0]).is_err());
    assert!(l::spsolve(&a, &[f64::NAN, 1.0]).is_err());
    let lu = l::splu(&a).unwrap();
    assert!(lu.solve(&[]).is_err());
    assert!(lu.solve(&[f64::INFINITY, 0.0]).is_err());
    assert!(l::splu(&s::identity(0).unwrap()).is_err());
    assert!(l::splu(&s::identity(1025).unwrap()).is_err());
    assert!(l::inv(&s::identity(1001).unwrap()).is_err());
    assert!(l::matrix_power(&a, 1025).is_err());
    assert!(l::matrix_power(&s::eye(2, 3, 0).unwrap(), 2).is_err());
    assert!(s::eye(1_000_001, 0, 0).is_err());
    assert!(s::to_dense(&s::eye(1001, 1000, 0).unwrap()).is_err());
    assert!(s::kron(&s::eye(1001, 1, 0).unwrap(), &s::eye(1001, 1, 0).unwrap()).is_err());
    let column = s::from_coo(5000, 1, &(0..5000).map(|i| (i, 0, 1.0)).collect::<Vec<_>>()).unwrap();
    let row = s::transpose(&column).unwrap();
    assert!(s::matmul(&column, &row).is_err()); // 25 million products, rejected before numerical work.
    assert!(s::kron(&column, &row).is_err()); // Stored pair limit, despite legal axes.
    let both = s::from_coo(1, 2, &[(0, 0, f64::MAX), (0, 1, f64::MAX)]).unwrap();
    assert!(l::norm(&both, l::MatrixNorm::Infinity).is_err());
}

#[test]
fn triangular_conventions_and_failures() {
    let a = s::from_coo(2, 2, &[(0, 1, 2.0)]).unwrap();
    assert_eq!(
        l::spsolve_triangular(&a, &[3.0, 1.0], false, true).unwrap(),
        vec![1.0, 1.0]
    );
    assert!(l::spsolve_triangular(&a, &[3.0, 1.0], true, true).is_err());
    assert!(l::spsolve_triangular(&a, &[3.0, 1.0], false, false).is_err());
    assert!(l::spsolve_triangular(&a, &[1.0], false, true).is_err());
    assert!(l::spsolve_triangular(&a, &[f64::NAN, 1.0], false, true).is_err());
    let z = s::from_coo(2, 2, &[(0, 1, 0.0), (0, 0, 1.0), (1, 1, 1.0)]).unwrap();
    assert_eq!(l::is_sptriangular(&z).unwrap(), (true, true));
    assert_eq!(
        l::spsolve_triangular(&z, &[1.0, 2.0], true, false).unwrap(),
        vec![1.0, 2.0]
    );
    let empty = s::identity(0).unwrap();
    assert!(
        l::spsolve_triangular(&empty, &[], true, false)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn algebraic_properties_and_dense_interoperability() {
    let dense = vec![
        vec![4.0, 0.0, 1.0],
        vec![2.0, 5.0, 0.0],
        vec![0.0, -1.0, 3.0],
    ];
    let a = s::from_dense(&dense).unwrap();
    let b = s::transpose(&a).unwrap();
    assert_eq!(s::transpose(&b).unwrap(), a);
    assert_eq!(
        s::transpose(&s::matmul(&a, &b).unwrap()).unwrap(),
        s::matmul(&a, &b).unwrap()
    );
    let actual = s::to_dense(&s::matmul(&a, &b).unwrap()).unwrap();
    assert_eq!(
        actual,
        sci::linalg::matmul(&dense, &s::to_dense(&b).unwrap()).unwrap()
    );
    let rhs = vec![1.0, 2.0, 3.0];
    let x = l::spsolve(&a, &rhs).unwrap();
    for (v, w) in s::matvec(&a, &x).unwrap().into_iter().zip(&rhs) {
        assert_close(v, *w, "solve residual");
    }
    for (v, w) in x.into_iter().zip(sci::linalg::solve(&dense, &rhs).unwrap()) {
        assert_close(v, w, "dense solve interop");
    }
    let inverse = l::inv(&a).unwrap();
    let product = s::to_dense(&s::matmul(&a, &inverse).unwrap()).unwrap();
    for (i, row) in product.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert_close(v, if i == j { 1.0 } else { 0.0 }, "inverse residual");
        }
    }
    let p2 = l::matrix_power(&a, 2).unwrap();
    let p3 = l::matrix_power(&a, 3).unwrap();
    assert_eq!(s::matmul(&p2, &a).unwrap(), p3);
    let tiny = s::from_coo(1, 1, &[(0, 0, 1e-300)]).unwrap();
    assert_eq!(l::norm(&tiny, l::MatrixNorm::Frobenius).unwrap(), 1e-300);
}

#[test]
fn sparse_work_is_bounded_for_empty_rows() {
    let tall = s::from_coo(1_000_000, 0, &[]).unwrap();
    let blocks = vec![&tall; 21];
    assert!(s::hstack(&blocks).is_err());
    assert!(s::block_array(&[blocks]).is_err());
    // Kronecker traversal must scan empty rows once, rather than per stored pair.
    let a = s::from_coo(1, 1000, &(0..1000).map(|c| (0, c, 1.0)).collect::<Vec<_>>()).unwrap();
    let b = s::from_coo(100_000, 1, &[(99_999, 0, 2.0)]).unwrap();
    let product = s::kron(&a, &b).unwrap();
    assert_eq!((product.rows, product.cols), (100_000, 1000));
    assert_eq!(product.data, vec![2.0; 1000]);
    assert_eq!(product.indptr[99_999], 0);
    assert_eq!(product.indptr[100_000], 1000);
}
