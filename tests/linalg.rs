mod common;
use common::finite_close;
use common::parsed_call;
use common::{matrix, number as boolean, optional, vector};

use sci::{Error, linalg as l, signal::Mode};

type Matrix = Vec<Vec<f64>>;

fn complex_matrix(a: &[Vec<sci::complex::Complex>]) -> (Matrix, Matrix) {
    (
        a.iter().map(|r| r.iter().map(|z| z.re).collect()).collect(),
        a.iter().map(|r| r.iter().map(|z| z.im).collect()).collect(),
    )
}
fn cmul(a: &l::ComplexMatrix, b: &l::ComplexMatrix) -> l::ComplexMatrix {
    use sci::complex::{Complex, add, multiply};
    (0..a.len())
        .map(|i| {
            (0..b[0].len())
                .map(|j| {
                    (0..b.len()).fold(Complex::default(), |s, k| {
                        add(s, multiply(a[i][k], b[k][j]))
                    })
                })
                .collect()
        })
        .collect()
}
fn cadjoint(a: &l::ComplexMatrix) -> l::ComplexMatrix {
    (0..a[0].len())
        .map(|j| {
            (0..a.len())
                .map(|i| sci::complex::conjugate(a[i][j]))
                .collect()
        })
        .collect()
}
fn cfrom(a: &[Vec<f64>]) -> l::ComplexMatrix {
    a.iter()
        .map(|r| {
            r.iter()
                .map(|&re| sci::complex::Complex { re, im: 0. })
                .collect()
        })
        .collect()
}
fn ccheck(a: &l::ComplexMatrix, b: &l::ComplexMatrix) {
    let (ar, ai) = complex_matrix(a);
    let (br, bi) = complex_matrix(b);
    check_matrix(&ar, &br);
    check_matrix(&ai, &bi);
}
fn eigenvalues(mut w: Vec<sci::complex::Complex>, re: &str, im: &str) {
    w.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));
    check_vector(&w.iter().map(|z| z.re).collect::<Vec<_>>(), &vector(re));
    check_vector(&w.iter().map(|z| z.im).collect::<Vec<_>>(), &vector(im));
}

#[track_caller]
fn check_vector(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        finite_close(a, b, 3e-11, 3e-11);
    }
}
#[track_caller]
fn check_matrix(a: &[Vec<f64>], b: &[Vec<f64>]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        check_vector(a, b);
    }
}
fn mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Matrix {
    l::matmul(a, b).unwrap()
}
fn trans(a: &[Vec<f64>]) -> Matrix {
    l::transpose(a).unwrap()
}
fn projector(a: &[Vec<f64>]) -> Matrix {
    if a.first().is_none_or(Vec::is_empty) {
        vec![vec![0.; a.len()]; a.len()]
    } else {
        mul(a, &trans(a))
    }
}
fn identity(n: usize) -> Matrix {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1. } else { 0. }).collect())
        .collect()
}
fn eigen(e: l::EigenResult, a: Option<&[Vec<f64>]>, expected: &[&str]) {
    check_vector(&e.eigenvalues, &vector(expected[1]));
    let squares: Matrix = e
        .eigenvectors
        .iter()
        .map(|row| row.iter().map(|v| v * v).collect())
        .collect();
    check_matrix(&squares, &matrix(expected[2]));
    check_matrix(
        &mul(&trans(&e.eigenvectors), &e.eigenvectors),
        &identity(e.eigenvalues.len()),
    );
    if let Some(a) = a {
        let diagonal = l::diagsvd(&e.eigenvalues, a.len(), a.len()).unwrap();
        check_matrix(&mul(a, &e.eigenvectors), &mul(&e.eigenvectors, &diagonal));
    }
}
fn kind(s: &str) -> l::PascalKind {
    match s {
        "lower" => l::PascalKind::Lower,
        "upper" => l::PascalKind::Upper,
        "symmetric" => l::PascalKind::Symmetric,
        _ => panic!("{s}"),
    }
}
fn selected_symmetric(a: &[Vec<f64>], lower: bool) -> Matrix {
    (0..a.len())
        .map(|i| {
            (0..a.len())
                .map(|j| {
                    if (lower && i >= j) || (!lower && i <= j) {
                        a[i][j]
                    } else {
                        a[j][i]
                    }
                })
                .collect()
        })
        .collect()
}

fn check_scipy_references<M: l::MatrixInput, V: l::VectorInput>(
    convert_matrix: impl Fn(&[Vec<f64>]) -> M,
    convert_vector: impl Fn(&[f64]) -> V,
) {
    let inputs: Vec<_> = common::lines(include_str!("fixtures/linalg_inputs.csv")).collect();
    let expected: Vec<_> = common::lines(include_str!("fixtures/linalg_expected.csv")).collect();
    assert_eq!(inputs.len(), expected.len());
    let mut covered = std::collections::BTreeSet::new();
    for (input, expected) in inputs.iter().zip(expected) {
        let c: Vec<_> = common::fields(input);
        let e: Vec<_> = common::fields(expected);
        assert_eq!(c[0], e[0]);
        covered.insert(c[1]);
        println!("reference {}: {}", c[0], c[1]);
        let a = || matrix(c[2]);
        let b = || vector(c[3]);
        let vm = |v: Vec<f64>| check_vector(&v, &vector(e[1]));
        let mm = |m: Matrix| check_matrix(&m, &matrix(e[1]));
        let col = || vector(c[2]);
        let row = || {
            if c[3] == "none" {
                None
            } else {
                Some(vector(c[3]))
            }
        };
        match c[1] {
            "toeplitz" => mm(l::toeplitz(&convert_vector(&col()), row().as_deref()).unwrap()),
            "hankel" => mm(l::hankel(&convert_vector(&col()), row().as_deref()).unwrap()),
            "circulant" => mm(l::circulant(&convert_vector(&col())).unwrap()),
            "hilbert" => mm(parsed_call!(l::hilbert; c[2]).unwrap()),
            "invhilbert" => mm(parsed_call!(l::invhilbert; c[2]).unwrap()),
            "pascal" => mm(l::pascal(c[2].parse().unwrap(), kind(c[3])).unwrap()),
            "invpascal" => mm(l::invpascal(c[2].parse().unwrap(), kind(c[3])).unwrap()),
            "hadamard" => mm(parsed_call!(l::hadamard; c[2]).unwrap()),
            "helmert" => mm(l::helmert(c[2].parse().unwrap(), boolean(c[3])).unwrap()),
            "fiedler" => mm(l::fiedler(&convert_vector(&col())).unwrap()),
            "companion" => mm(l::companion(&convert_vector(&col())).unwrap()),
            "fiedler_companion" => mm(l::fiedler_companion(&convert_vector(&col())).unwrap()),
            "leslie" => mm(l::leslie(&convert_vector(&col()), &convert_vector(&b())).unwrap()),
            "block_diag" => {
                let blocks: Vec<_> = if c[2].is_empty() {
                    vec![]
                } else {
                    c[2].split('|').map(matrix).collect()
                };
                let blocks: Vec<_> = blocks.iter().map(|a| convert_matrix(a)).collect();
                mm(l::block_diag(&blocks).unwrap());
            }
            "convolution_matrix" => {
                let mode = match c[4] {
                    "full" => Mode::Full,
                    "same" => Mode::Same,
                    "valid" => Mode::Valid,
                    _ => panic!(),
                };
                mm(
                    l::convolution_matrix(&convert_vector(&col()), c[3].parse().unwrap(), mode)
                        .unwrap(),
                );
            }
            "dft" => {
                let scale = match c[3] {
                    "none" => l::DftScale::None,
                    "sqrtn" => l::DftScale::SqrtN,
                    "n" => l::DftScale::N,
                    _ => panic!(),
                };
                let f = l::dft(c[2].parse().unwrap(), scale).unwrap();
                mm(f.iter().map(|r| r.iter().map(|v| v.re).collect()).collect());
                check_matrix(
                    &f.iter()
                        .map(|r| r.iter().map(|v| v.im).collect())
                        .collect::<Matrix>(),
                    &matrix(e[2]),
                );
            }
            "diagsvd" => mm(l::diagsvd(
                &convert_vector(&col()),
                c[3].parse().unwrap(),
                c[4].parse().unwrap(),
            )
            .unwrap()),
            "solve" => vm(l::solve(&convert_matrix(&a()), &convert_vector(&b())).unwrap()),
            "inv" => mm(l::inv(&convert_matrix(&a())).unwrap()),
            "det" => finite_close(
                l::det(&convert_matrix(&a())).unwrap(),
                e[1].parse().unwrap(),
                3e-11,
                3e-11,
            ),
            "solve_triangular" => vm(l::solve_triangular(
                &convert_matrix(&a()),
                &convert_vector(&b()),
                boolean(c[4]),
                boolean(c[5]),
                boolean(c[6]),
            )
            .unwrap()),
            "solve_toeplitz" => vm(l::solve_toeplitz(
                &convert_vector(&col()),
                row().as_deref(),
                &convert_vector(&vector(c[4])),
            )
            .unwrap()),
            "matmul_toeplitz" => mm(l::matmul_toeplitz(
                &convert_vector(&col()),
                row().as_deref(),
                &convert_matrix(&matrix(c[4])),
            )
            .unwrap()),
            "solve_circulant" => {
                vm(l::solve_circulant(&convert_vector(&col()), &convert_vector(&b())).unwrap())
            }
            "cholesky" | "cho_factor" => {
                let lower = boolean(c[3]);
                let f = if c[1] == "cholesky" {
                    l::cholesky(&convert_matrix(&a()), lower).unwrap()
                } else {
                    let f = l::cho_factor(&convert_matrix(&a()), lower).unwrap();
                    assert_eq!(f.lower, lower);
                    f.factor
                };
                mm(f.clone());
                let reconstruction = if lower {
                    mul(&f, &trans(&f))
                } else {
                    mul(&trans(&f), &f)
                };
                check_matrix(&reconstruction, &selected_symmetric(&a(), lower));
            }
            "cho_solve" => vm(l::cho_solve(
                &l::cho_factor(&convert_matrix(&a()), boolean(c[4])).unwrap(),
                &convert_vector(&b()),
            )
            .unwrap()),
            "lu" => {
                let f = l::lu(&convert_matrix(&a())).unwrap();
                mm(f.p.clone());
                check_matrix(&f.l, &matrix(e[2]));
                check_matrix(&f.u, &matrix(e[3]));
                check_matrix(&mul(&mul(&f.p, &f.l), &f.u), &a());
            }
            "lu_factor" => {
                let f = l::lu_factor(&convert_matrix(&a())).unwrap();
                mm(f.packed);
                check_vector(
                    &f.row_permutation
                        .iter()
                        .map(|&p| p as f64)
                        .collect::<Vec<_>>(),
                    &vector(e[2]),
                );
            }
            "lu_solve" => vm(l::lu_solve(
                &l::lu_factor(&convert_matrix(&a())).unwrap(),
                &convert_vector(&b()),
                boolean(c[4]),
            )
            .unwrap()),
            "khatri_rao" => {
                mm(l::khatri_rao(&convert_matrix(&a()), &convert_matrix(&matrix(c[3]))).unwrap())
            }
            "rq" => {
                let (r, q) = l::rq(&convert_matrix(&a())).unwrap();
                mm(projector(&r));
                check_matrix(&mul(&trans(&q), &q), &matrix(e[2]));
                check_matrix(&mul(&r, &q), &matrix(e[3]));
                check_matrix(&mul(&q, &trans(&q)), &identity(q.len()));
            }
            "qr_multiply" => {
                let (product, r) = l::qr_multiply(
                    &convert_matrix(&a()),
                    &convert_matrix(&matrix(c[3])),
                    boolean(c[4]),
                )
                .unwrap();
                mm(projector(&product));
                check_matrix(&mul(&trans(&r), &r), &matrix(e[2]));
            }
            "qr_update" | "qr_insert" | "qr_delete" => {
                let f = match c[1] {
                    "qr_update" => l::qr_update(
                        &convert_matrix(&a()),
                        &convert_matrix(&matrix(c[3])),
                        &convert_vector(&vector(c[4])),
                        &convert_vector(&vector(c[5])),
                    )
                    .unwrap(),
                    "qr_insert" => l::qr_insert(
                        &convert_matrix(&a()),
                        &convert_matrix(&matrix(c[3])),
                        &convert_matrix(&matrix(c[4])),
                        c[5].parse().unwrap(),
                        boolean(c[6]),
                    )
                    .unwrap(),
                    _ => l::qr_delete(
                        &convert_matrix(&a()),
                        &convert_matrix(&matrix(c[3])),
                        c[4].parse().unwrap(),
                        c[5].parse().unwrap(),
                        boolean(c[6]),
                    )
                    .unwrap(),
                };
                mm(mul(&f.q, &f.r));
                check_matrix(&mul(&trans(&f.q), &f.q), &identity(f.r.len()));
            }
            "qr" => {
                let f = l::qr(&convert_matrix(&a())).unwrap();
                mm(projector(&f.q));
                check_matrix(&mul(&trans(&f.r), &f.r), &matrix(e[2]));
                check_matrix(&mul(&f.q, &f.r), &matrix(e[3]));
                check_matrix(&mul(&trans(&f.q), &f.q), &identity(f.r.len()));
            }
            "svd" => {
                let f = l::svd(&convert_matrix(&a())).unwrap();
                mm(projector(&f.u));
                check_vector(&f.singular_values, &vector(e[2]));
                check_matrix(&projector(&trans(&f.vh)), &matrix(e[3]));
                let s = l::diagsvd(
                    &convert_vector(&f.singular_values),
                    f.singular_values.len(),
                    f.singular_values.len(),
                )
                .unwrap();
                check_matrix(&mul(&mul(&f.u, &s), &f.vh), &matrix(e[4]));
            }
            "svdvals" => vm(l::svdvals(&convert_matrix(&a())).unwrap()),
            "pinv" | "pinvh" => {
                let (p, r) = if c[1] == "pinv" {
                    l::pinv(&convert_matrix(&a()), c[3].parse().unwrap(), optional(c[4])).unwrap()
                } else {
                    l::pinvh(
                        &convert_matrix(&a()),
                        c[3].parse().unwrap(),
                        optional(c[4]),
                        boolean(c[5]),
                    )
                    .unwrap()
                };
                mm(p);
                assert_eq!(r, e[2].parse().unwrap());
            }
            "lstsq" => {
                let r =
                    l::lstsq(&convert_matrix(&a()), &convert_vector(&b()), optional(c[4])).unwrap();
                vm(r.solution);
                check_vector(&r.residuals, &vector(e[2]));
                assert_eq!(r.rank, e[3].parse().unwrap());
                check_vector(&r.singular_values, &vector(e[4]));
            }
            "orth" | "null_space" => {
                let q = if c[1] == "orth" {
                    l::orth(&convert_matrix(&a()), optional(c[3])).unwrap()
                } else {
                    l::null_space(&convert_matrix(&a()), optional(c[3])).unwrap()
                };
                mm(projector(&q));
                let k = q[0].len();
                assert_eq!(k, e[2].parse().unwrap());
                if k > 0 {
                    check_matrix(&mul(&trans(&q), &q), &identity(k));
                    if c[1] == "null_space" {
                        let cutoff = optional(c[3])
                            .unwrap_or(f64::EPSILON * a().len().max(q.len()) as f64)
                            * l::norm(&convert_matrix(&a()), l::MatrixNorm::Spectral).unwrap();
                        let residual = mul(&a(), &q);
                        for j in 0..k {
                            let n = residual.iter().fold(0.0_f64, |n, r| n.hypot(r[j]));
                            assert!(n <= cutoff + 3e-11);
                        }
                    }
                }
            }
            "polar" => {
                let (u, p) = l::polar(&convert_matrix(&a()), boolean(c[3])).unwrap();
                mm(if u.len() >= u[0].len() {
                    mul(&trans(&u), &u)
                } else {
                    mul(&u, &trans(&u))
                });
                check_matrix(&p, &matrix(e[2]));
                check_matrix(
                    &if boolean(c[3]) {
                        mul(&u, &p)
                    } else {
                        mul(&p, &u)
                    },
                    &a(),
                );
            }
            "orthogonal_procrustes" => {
                let (r, s) =
                    l::orthogonal_procrustes(&convert_matrix(&a()), &convert_matrix(&matrix(c[3])))
                        .unwrap();
                mm(r.clone());
                finite_close(s, e[2].parse().unwrap(), 3e-11, 3e-11);
                check_matrix(&mul(&trans(&r), &r), &identity(r.len()));
            }
            "eigh" => eigen(
                l::eigh(&convert_matrix(&a()), boolean(c[3])).unwrap(),
                Some(&selected_symmetric(&a(), boolean(c[3]))),
                &e,
            ),
            "eigvalsh" => vm(l::eigvalsh(&convert_matrix(&a()), boolean(c[3])).unwrap()),
            "eigh_tridiagonal" => eigen(
                l::eigh_tridiagonal(&convert_vector(&col()), &convert_vector(&b())).unwrap(),
                None,
                &e,
            ),
            "eigvalsh_tridiagonal" => vm(l::eigvalsh_tridiagonal(
                &convert_vector(&col()),
                &convert_vector(&b()),
            )
            .unwrap()),
            "hessenberg" | "schur" => {
                let (t, z) = if c[1] == "hessenberg" {
                    l::hessenberg(&convert_matrix(&a())).unwrap()
                } else {
                    l::schur(&convert_matrix(&a())).unwrap()
                };
                mm(mul(&mul(&z, &t), &trans(&z)));
                check_matrix(&mul(&trans(&z), &z), &matrix(e[2]));
                for (i, row) in t.iter().enumerate() {
                    for &v in row.iter().take(i.saturating_sub(1)) {
                        finite_close(v, 0., 3e-11, 3e-11);
                    }
                }
                if c[1] == "schur" {
                    for i in 2..t.len() {
                        assert!(t[i][i - 1] == 0. || t[i - 1][i - 2] == 0.);
                    }
                }
            }
            "eig" => {
                let f = l::eig(&convert_matrix(&a())).unwrap();
                let mut order: Vec<usize> = (0..f.eigenvalues.len()).collect();
                order.sort_by(|&i, &j| {
                    f.eigenvalues[i]
                        .re
                        .total_cmp(&f.eigenvalues[j].re)
                        .then(f.eigenvalues[i].im.total_cmp(&f.eigenvalues[j].im))
                });
                eigenvalues(f.eigenvalues.clone(), e[1], e[2]);
                let norm2: Matrix = f
                    .eigenvectors
                    .iter()
                    .map(|r| {
                        order
                            .iter()
                            .map(|&j| r[j].re.powi(2) + r[j].im.powi(2))
                            .collect()
                    })
                    .collect();
                check_matrix(&norm2, &matrix(e[3]));
                let av = cmul(&cfrom(&a()), &f.eigenvectors);
                let vd: l::ComplexMatrix = f
                    .eigenvectors
                    .iter()
                    .map(|r| {
                        r.iter()
                            .zip(&f.eigenvalues)
                            .map(|(&v, &w)| sci::complex::multiply(v, w))
                            .collect()
                    })
                    .collect();
                ccheck(&av, &vd);
            }
            "ldl" => {
                let lower = boolean(c[3]);
                let f = l::ldl(&convert_matrix(&a()), lower).unwrap();
                mm(mul(&mul(&f.outer, &f.d), &trans(&f.outer)));
                for i in 0..f.outer.len() {
                    finite_close(f.outer[f.permutation[i]][i], 1., 3e-11, 3e-11);
                    for j in 0..f.outer.len() {
                        if (lower && j > i) || (!lower && j < i) {
                            finite_close(f.outer[f.permutation[i]][j], 0., 3e-11, 3e-11);
                        }
                    }
                }
            }
            "sqrtm" | "logm" | "signm" | "fractional_matrix_power" | "funm" => {
                let f = match c[1] {
                    "sqrtm" => l::sqrtm(&convert_matrix(&a())),
                    "logm" => l::logm(&convert_matrix(&a())),
                    "signm" => l::signm(&convert_matrix(&a())),
                    "fractional_matrix_power" => {
                        l::fractional_matrix_power(&convert_matrix(&a()), c[3].parse().unwrap())
                    }
                    _ => l::funm(&convert_matrix(&a()), |z| {
                        if c[3] == "exp" {
                            sci::complex::exp(z)
                        } else {
                            sci::complex::sin(z)
                        }
                    }),
                }
                .unwrap();
                let (re, im) = complex_matrix(&f);
                mm(re);
                check_matrix(&im, &matrix(e[2]));
                if c[1] == "sqrtm" {
                    ccheck(&cmul(&f, &f), &cfrom(&a()));
                }
                if c[1] == "signm" {
                    ccheck(&cmul(&f, &f), &cfrom(&identity(f.len())));
                }
            }
            "rsf2csf" => {
                let (t, z) =
                    l::rsf2csf(&convert_matrix(&a()), &convert_matrix(&matrix(c[3]))).unwrap();
                let f = cmul(&cmul(&z, &t), &cadjoint(&z));
                let (re, im) = complex_matrix(&f);
                mm(re);
                check_matrix(&im, &matrix(e[2]));
                ccheck(&cmul(&cadjoint(&z), &z), &cfrom(&identity(z.len())));
                for (i, row) in t.iter().enumerate() {
                    for z in row.iter().take(i) {
                        finite_close(z.re, 0., 3e-11, 3e-11);
                        finite_close(z.im, 0., 3e-11, 3e-11);
                    }
                }
            }
            "cdf2rdf" => {
                let w = matrix(c[2]);
                let vv: Vec<_> = c[3].split('|').map(matrix).collect();
                let w: Vec<_> = w[0]
                    .iter()
                    .zip(&w[1])
                    .map(|(&re, &im)| sci::complex::Complex { re, im })
                    .collect();
                let v: l::ComplexMatrix = vv[0]
                    .iter()
                    .zip(&vv[1])
                    .map(|(r, s)| {
                        r.iter()
                            .zip(s)
                            .map(|(&re, &im)| sci::complex::Complex { re, im })
                            .collect()
                    })
                    .collect();
                let (d, v) = l::cdf2rdf(&w, &v).unwrap();
                mm(d);
                check_matrix(&v, &matrix(e[2]));
            }
            "qz" | "ordqz" => {
                let f = if c[1] == "qz" {
                    l::qz(&convert_matrix(&a()), &convert_matrix(&matrix(c[3])))
                } else {
                    l::ordqz(
                        &convert_matrix(&a()),
                        &convert_matrix(&matrix(c[3])),
                        |a, b| {
                            if c[4] == "lhp" {
                                a.re * b.re + a.im * b.im < 0.
                            } else {
                                a.re.hypot(a.im) < b.re.hypot(b.im)
                            }
                        },
                    )
                }
                .unwrap();
                let aa = cmul(&cmul(&f.q, &f.s), &cadjoint(&f.z));
                let bb = cmul(&cmul(&f.q, &f.t), &cadjoint(&f.z));
                ccheck(&aa, &cfrom(&matrix(e[1])));
                ccheck(&bb, &cfrom(&matrix(e[2])));
                ccheck(&cmul(&cadjoint(&f.q), &f.q), &cfrom(&identity(f.q.len())));
                ccheck(&cmul(&cadjoint(&f.z), &f.z), &cfrom(&identity(f.z.len())));
                for i in 0..f.s.len() {
                    for j in 0..i {
                        finite_close(f.s[i][j].re, 0., 3e-11, 3e-11);
                        finite_close(f.s[i][j].im, 0., 3e-11, 3e-11);
                        finite_close(f.t[i][j].re, 0., 3e-11, 3e-11);
                        finite_close(f.t[i][j].im, 0., 3e-11, 3e-11);
                    }
                }
                let w: Vec<_> = f
                    .alpha
                    .iter()
                    .zip(&f.beta)
                    .filter(|(_, b)| b.re.hypot(b.im) > 1e-14)
                    .map(|(&a, &b)| sci::complex::divide(a, b))
                    .collect();
                eigenvalues(w, e[3], e[4]);
                if c[1] == "ordqz" {
                    for i in 0..f.alpha.len() {
                        let (a, b) = (f.alpha[i], f.beta[i]);
                        let selected = if c[4] == "lhp" {
                            a.re * b.re + a.im * b.im < 0.
                        } else {
                            a.re.hypot(a.im) < b.re.hypot(b.im)
                        };
                        assert_eq!(selected, i < f.selected);
                    }
                }
            }
            "solve_continuous_are" | "solve_discrete_are" => {
                let f = if c[1] == "solve_continuous_are" {
                    l::solve_continuous_are(
                        &convert_matrix(&a()),
                        &convert_matrix(&matrix(c[3])),
                        &convert_matrix(&matrix(c[4])),
                        &convert_matrix(&matrix(c[5])),
                    )
                } else {
                    l::solve_discrete_are(
                        &convert_matrix(&a()),
                        &convert_matrix(&matrix(c[3])),
                        &convert_matrix(&matrix(c[4])),
                        &convert_matrix(&matrix(c[5])),
                    )
                }
                .unwrap();
                mm(f);
            }
            "matrix_balance" => {
                let (b, t) =
                    l::matrix_balance(&convert_matrix(&a()), boolean(c[3]), boolean(c[4])).unwrap();
                mm(mul(&mul(&t, &b), &l::inv(&convert_matrix(&t)).unwrap()));
                eigenvalues(l::eigvals(&convert_matrix(&b)).unwrap(), e[2], e[3]);
            }
            "clarkson_woodruff_transform" => {
                let buckets: Vec<_> = vector(c[4]).iter().map(|&i| i as usize).collect();
                mm(l::clarkson_woodruff_transform(
                    &convert_matrix(&a()),
                    c[3].parse().unwrap(),
                    &buckets,
                    &convert_vector(&vector(c[5])),
                )
                .unwrap());
            }
            "cossin" => {
                let (p, q) = (
                    c[3].parse::<usize>().unwrap(),
                    c[4].parse::<usize>().unwrap(),
                );
                let f = l::cossin(&convert_matrix(&a()), p, q).unwrap();
                vm(f.theta);
                // Each central block is a permuted diagonal of cosines,
                // sines, or identities, rather than an arbitrary factor.
                for (start, end) in [(0, p), (p, f.cs.len())] {
                    for (left, right) in [(0, q), (q, f.cs.len())] {
                        for row in &f.cs[start..end] {
                            assert!(
                                row[left..right].iter().filter(|v| v.abs() > 1e-10).count() <= 1
                            );
                        }
                        for j in left..right {
                            assert!(
                                (start..end).filter(|&i| f.cs[i][j].abs() > 1e-10).count() <= 1
                            );
                        }
                    }
                }
                check_matrix(&mul(&mul(&f.u, &f.cs), &f.vh), &a());
                check_matrix(&mul(&trans(&f.u), &f.u), &identity(f.u.len()));
                check_matrix(&mul(&f.vh, &trans(&f.vh)), &identity(f.vh.len()));
                for i in 0..f.u.len() {
                    for j in 0..f.u.len() {
                        if (i < p) != (j < p) {
                            finite_close(f.u[i][j], 0., 3e-11, 3e-11);
                        }
                        if (i < q) != (j < q) {
                            finite_close(f.vh[i][j], 0., 3e-11, 3e-11);
                        }
                    }
                }
            }
            "find_best_blas_type" => {
                let arrays: Vec<_> = if c[2].is_empty() {
                    vec![]
                } else {
                    c[2].split('|').map(matrix).collect()
                };
                let arrays: Vec<_> = arrays.iter().map(|a| convert_matrix(a)).collect();
                let t = l::find_best_blas_type(&arrays).unwrap();
                assert_eq!(t.prefix.to_string(), e[1]);
                assert_eq!(t.dtype, "f64");
                assert_eq!(t.prefer_fortran, boolean(e[3]));
            }
            "get_blas_funcs" => match l::get_blas_funcs(&[c[2]]).unwrap()[0] {
                l::BlasFunction::Dot(f) => finite_close(
                    f(&vector(c[3]), &vector(c[4])).unwrap(),
                    e[1].parse().unwrap(),
                    3e-11,
                    3e-11,
                ),
                l::BlasFunction::Nrm2(f) => finite_close(
                    f(&vector(c[3])).unwrap(),
                    e[1].parse().unwrap(),
                    3e-11,
                    3e-11,
                ),
                l::BlasFunction::Gemv(f) => vm(f(&matrix(c[3]), &vector(c[4])).unwrap()),
                l::BlasFunction::Gemm(f) => mm(f(&matrix(c[3]), &matrix(c[4])).unwrap()),
            },
            "get_lapack_funcs" => match l::get_lapack_funcs(&[c[2]]).unwrap()[0] {
                l::LapackFunction::Gesv(f) => vm(f(&matrix(c[3]), &vector(c[4])).unwrap()),
                l::LapackFunction::Getrf(f) => mm(f(&matrix(c[3])).unwrap().packed),
                _ => panic!("unsupported fixture"),
            },
            "interpolative.estimate_rank" => assert_eq!(
                l::interpolative::estimate_rank(&convert_matrix(&a()), c[3].parse().unwrap())
                    .unwrap(),
                e[1].parse::<usize>().unwrap()
            ),
            "interpolative.estimate_spectral_norm" => finite_close(
                l::interpolative::estimate_spectral_norm(&convert_matrix(&a())).unwrap(),
                e[1].parse().unwrap(),
                3e-11,
                3e-11,
            ),
            "interpolative.estimate_spectral_norm_diff" => finite_close(
                l::interpolative::estimate_spectral_norm_diff(
                    &convert_matrix(&a()),
                    &convert_matrix(&matrix(c[3])),
                )
                .unwrap(),
                e[1].parse().unwrap(),
                3e-11,
                3e-11,
            ),
            "interpolative.interp_decomp" => {
                let f = l::interpolative::interp_decomp(
                    &convert_matrix(&a()),
                    parsed_call!(l::interpolative::Rank::Fixed; c[3]),
                )
                .unwrap();
                let b = l::interpolative::reconstruct_skel_matrix(
                    &convert_matrix(&a()),
                    f.rank,
                    &f.indices,
                )
                .unwrap();
                let actual = l::interpolative::reconstruct_matrix_from_id(
                    &convert_matrix(&b),
                    &f.indices,
                    &convert_matrix(&f.projection),
                )
                .unwrap();
                // IDs have nonunique pivot choices; compare spectral residual to
                // SciPy ID's residual rather than requiring the same skeleton.
                let expected = matrix(e[1]);
                let residual = l::interpolative::estimate_spectral_norm_diff(
                    &convert_matrix(&a()),
                    &convert_matrix(&actual),
                )
                .unwrap();
                let reference = l::interpolative::estimate_spectral_norm_diff(
                    &convert_matrix(&a()),
                    &convert_matrix(&expected),
                )
                .unwrap();
                assert!(
                    residual <= 1.01 * reference + 3e-11,
                    "ID residual {residual} > {reference}"
                );
                assert_eq!(f.rank, e[2].parse().unwrap());
            }
            "interpolative.svd" => {
                let f = l::interpolative::svd(
                    &convert_matrix(&a()),
                    parsed_call!(l::interpolative::Rank::Fixed; c[3]),
                )
                .unwrap();
                if f.singular_values.is_empty() {
                    mm(vec![vec![0.; a()[0].len()]; a().len()]);
                } else {
                    let d = l::diagsvd(
                        &convert_vector(&f.singular_values),
                        f.singular_values.len(),
                        f.singular_values.len(),
                    )
                    .unwrap();
                    mm(mul(&mul(&f.u, &d), &f.vh));
                }
                check_vector(&f.singular_values, &vector(e[2]));
            }
            "interpolative.reconstruct_interp_matrix" => {
                let indices: Vec<_> = vector(c[2]).iter().map(|&x| x as usize).collect();
                mm(l::interpolative::reconstruct_interp_matrix(
                    &indices,
                    &convert_matrix(&matrix(c[3])),
                )
                .unwrap());
            }
            "interpolative.reconstruct_skel_matrix" => {
                let indices: Vec<_> = vector(c[4]).iter().map(|&x| x as usize).collect();
                mm(l::interpolative::reconstruct_skel_matrix(
                    &convert_matrix(&a()),
                    c[3].parse().unwrap(),
                    &indices,
                )
                .unwrap());
            }
            "interpolative.reconstruct_matrix_from_id" | "interpolative.id_to_svd" => {
                let indices: Vec<_> = vector(c[3]).iter().map(|&x| x as usize).collect();
                let projection = matrix(c[4]);
                if c[1] == "interpolative.id_to_svd" {
                    let f = l::interpolative::id_to_svd(
                        &convert_matrix(&a()),
                        &indices,
                        &convert_matrix(&projection),
                    )
                    .unwrap();
                    let d = l::diagsvd(
                        &convert_vector(&f.singular_values),
                        f.singular_values.len(),
                        f.singular_values.len(),
                    )
                    .unwrap();
                    mm(mul(&mul(&f.u, &d), &f.vh));
                    check_vector(&f.singular_values, &vector(e[2]));
                } else {
                    mm(l::interpolative::reconstruct_matrix_from_id(
                        &convert_matrix(&a()),
                        &indices,
                        &convert_matrix(&projection),
                    )
                    .unwrap());
                }
            }
            "eigvals" => {
                let mut w = l::eigvals(&convert_matrix(&a())).unwrap();
                w.sort_by(|x, y| x.re.total_cmp(&y.re).then(x.im.total_cmp(&y.im)));
                vm(w.iter().map(|z| z.re).collect());
                check_vector(&w.iter().map(|z| z.im).collect::<Vec<_>>(), &vector(e[2]));
            }
            "norm" => {
                let order = match c[3] {
                    "fro" => l::MatrixNorm::Frobenius,
                    "1" => l::MatrixNorm::One,
                    "-1" => l::MatrixNorm::NegativeOne,
                    "inf" => l::MatrixNorm::Infinity,
                    "-inf" => l::MatrixNorm::NegativeInfinity,
                    "2" => l::MatrixNorm::Spectral,
                    "-2" => l::MatrixNorm::SmallestSingular,
                    "nuc" => l::MatrixNorm::Nuclear,
                    _ => panic!(),
                };
                finite_close(
                    l::norm(&convert_matrix(&a()), order).unwrap(),
                    e[1].parse().unwrap(),
                    3e-11,
                    3e-11,
                );
            }
            "solve_banded" => vm(l::solve_banded(
                c[2].parse().unwrap(),
                c[3].parse().unwrap(),
                &convert_matrix(&matrix(c[4])),
                &convert_vector(&vector(c[5])),
            )
            .unwrap()),
            "solveh_banded" => {
                vm(
                    l::solveh_banded(&convert_matrix(&a()), &convert_vector(&b()), boolean(c[4]))
                        .unwrap(),
                )
            }
            "cholesky_banded" => {
                mm(l::cholesky_banded(&convert_matrix(&a()), boolean(c[3])).unwrap())
            }
            "cho_solve_banded" => vm(l::cho_solve_banded(
                &convert_matrix(&l::cholesky_banded(&convert_matrix(&a()), boolean(c[4])).unwrap()),
                &convert_vector(&b()),
                boolean(c[4]),
            )
            .unwrap()),
            "eig_banded" => eigen(
                l::eig_banded(&convert_matrix(&a()), boolean(c[3])).unwrap(),
                None,
                &e,
            ),
            "eigvals_banded" => {
                vm(l::eigvals_banded(&convert_matrix(&a()), boolean(c[3])).unwrap())
            }
            "subspace_angles" => vm(l::subspace_angles(
                &convert_matrix(&a()),
                &convert_matrix(&matrix(c[3])),
            )
            .unwrap()),
            "expm" => mm(l::expm(&convert_matrix(&a())).unwrap()),
            "sinm" => mm(l::sinm(&convert_matrix(&a())).unwrap()),
            "cosm" => mm(l::cosm(&convert_matrix(&a())).unwrap()),
            "tanm" => mm(l::tanm(&convert_matrix(&a())).unwrap()),
            "sinhm" => mm(l::sinhm(&convert_matrix(&a())).unwrap()),
            "coshm" => mm(l::coshm(&convert_matrix(&a())).unwrap()),
            "tanhm" => mm(l::tanhm(&convert_matrix(&a())).unwrap()),
            "expm_frechet" => {
                let (exp, derivative) =
                    l::expm_frechet(&convert_matrix(&a()), &convert_matrix(&matrix(c[3]))).unwrap();
                mm(exp);
                check_matrix(&derivative, &matrix(e[2]));
            }
            "expm_cond" => finite_close(
                l::expm_cond(&convert_matrix(&a())).unwrap(),
                e[1].parse().unwrap(),
                3e-11,
                3e-11,
            ),
            "solve_sylvester" => mm(l::solve_sylvester(
                &convert_matrix(&a()),
                &convert_matrix(&matrix(c[3])),
                &convert_matrix(&matrix(c[4])),
            )
            .unwrap()),
            "solve_continuous_lyapunov" => mm(l::solve_continuous_lyapunov(
                &convert_matrix(&a()),
                &convert_matrix(&matrix(c[3])),
            )
            .unwrap()),
            "solve_lyapunov" => mm(l::solve_lyapunov(&a(), &matrix(c[3])).unwrap()),
            "solve_discrete_lyapunov" => mm(l::solve_discrete_lyapunov(
                &convert_matrix(&a()),
                &convert_matrix(&matrix(c[3])),
            )
            .unwrap()),
            "bandwidth" => {
                let (lo, hi) = l::bandwidth(&convert_matrix(&a())).unwrap();
                assert_eq!((lo, hi), (e[1].parse().unwrap(), e[2].parse().unwrap()));
            }
            "issymmetric" => assert_eq!(
                l::issymmetric(&convert_matrix(&a()), optional(c[3]), optional(c[4])).unwrap(),
                boolean(e[1])
            ),
            "ishermitian" => assert_eq!(
                l::ishermitian(&a(), optional(c[3]), optional(c[4])).unwrap(),
                boolean(e[1])
            ),
            _ => panic!("missing dispatcher {}", c[1]),
        }
    }
    assert_eq!(covered.len(), 105);
}

#[test]
fn invalid_shapes_domains_and_work_limits_are_explicit() {
    let singular = vec![vec![1., 2.], vec![2., 4.]];
    let positive = vec![vec![4., 1.], vec![1., 3.]];
    let zeros = vec![vec![0., 0.], vec![0., 0.]];
    assert!(matches!(
        l::solve(&singular, &[1., 2.]),
        Err(Error::NumericalFailure(_))
    ));
    assert!(l::inv(&singular).is_err());
    assert_eq!(l::det(&singular).unwrap(), 0.);
    assert!(l::solve_triangular(&zeros, &[1., 2.], true, false, false).is_err());
    assert!(l::solve_triangular(&positive, &[1.], true, false, false).is_err());
    assert!(l::solve_toeplitz(&[1., 1.], None, &[1., 2.]).is_err());
    assert!(l::solve_toeplitz(&[1., 2.], Some(&[1.]), &[1., 2.]).is_err());
    assert!(l::matmul_toeplitz(&[1., 2.], None, &[vec![1.]]).is_err());
    assert!(l::solve_circulant(&[1., 1.], &[1., 2.]).is_err());
    assert!(l::cholesky(&singular, true).is_err());
    assert!(l::cho_factor(&[vec![-1.]], false).is_err());
    assert!(
        l::cho_solve(
            &l::CholeskyFactor {
                factor: vec![vec![-1.]],
                lower: true
            },
            &[1.]
        )
        .is_err()
    );
    let factor = l::lu_factor(&singular).unwrap();
    assert!(l::lu_solve(&factor, &[1., 2.], false).is_err());
    let bad = l::LuFactor {
        packed: positive.clone(),
        row_permutation: vec![0, 0],
    };
    assert!(l::lu_solve(&bad, &[1., 2.], true).is_err());
    let bad = l::LuFactor {
        packed: positive.clone(),
        row_permutation: vec![0, usize::MAX],
    };
    assert!(l::lu_solve(&bad, &[1., 2.], false).is_err());
    assert!(l::lu_factor(&[vec![1., 2.]]).is_err());
    for invalid in [-1., f64::NAN, f64::INFINITY] {
        assert!(l::pinv(&positive, invalid, None).is_err());
        assert!(l::pinv(&positive, 0., Some(invalid)).is_err());
        assert!(l::pinvh(&positive, invalid, None, true).is_err());
        assert!(l::pinvh(&positive, 0., Some(invalid), true).is_err());
        assert!(l::orth(&positive, Some(invalid)).is_err());
        assert!(l::null_space(&positive, Some(invalid)).is_err());
        assert!(l::lstsq(&positive, &[1., 2.], Some(invalid)).is_err());
        assert!(l::issymmetric(&positive, Some(invalid), None).is_err());
        assert!(l::ishermitian(&positive, None, Some(invalid)).is_err());
    }
    assert!(l::orthogonal_procrustes(&positive, &[vec![1.]]).is_err());
    assert!(l::subspace_angles(&positive, &[vec![1.]]).is_err());
    assert!(l::eigh_tridiagonal(&[1., 2.], &[]).is_err());
    assert!(l::eigvalsh_tridiagonal(&[], &[]).is_err());
    assert!(l::eigh(&[vec![1., 2.]], true).is_err());
    assert!(l::eigvalsh(&[vec![1., 2.]], false).is_err());
    assert!(l::solve_banded(1, 1, &[vec![1., 2.]], &[1., 2.]).is_err());
    assert!(l::solve_banded(usize::MAX, 0, &[vec![1.]], &[1.]).is_err());
    assert!(l::solveh_banded(&[vec![-1.]], &[1.], true).is_err());
    assert!(l::cholesky_banded(&[vec![1.], vec![2.]], false).is_err());
    assert!(l::cho_solve_banded(&[vec![0.]], &[1.], true).is_err());
    assert!(l::eig_banded(&[vec![1.], vec![2.]], true).is_err());
    assert!(l::eigvals_banded(&[] as &[Vec<f64>], false).is_err());
    assert!(l::solve_sylvester(&[vec![1.]], &[vec![-1.]], &[vec![2.]]).is_err());
    assert!(l::solve_sylvester(&positive, &positive, &[vec![1.]]).is_err());
    assert!(l::solve_continuous_lyapunov(&zeros, &positive).is_err());
    assert!(l::solve_lyapunov(&zeros, &positive).is_err());
    assert!(l::solve_discrete_lyapunov(&identity(2), &positive).is_err());
    assert!(l::solve_discrete_lyapunov(&positive, &[vec![1.]]).is_err());
    assert!(l::solve_discrete_lyapunov(&identity(17), &identity(17)).is_err());
    assert!(l::solve_sylvester(&identity(17), &identity(16), &vec![vec![1.; 16]; 17]).is_err());
    assert!(l::solve_lyapunov(&identity(17), &identity(17)).is_err());
    assert!(l::solve(&[vec![1.]], &[f64::INFINITY]).is_err());
    assert!(l::lstsq(&[vec![1.]], &[], None).is_err());
    for a in [
        vec![],
        vec![vec![]],
        vec![vec![1.], vec![1., 2.]],
        vec![vec![f64::NAN]],
        vec![vec![f64::INFINITY]],
        vec![vec![1.; 257]],
    ] {
        assert!(l::svd(&a).is_err());
        assert!(l::svdvals(&a).is_err());
        assert!(l::qr(&a).is_err());
        assert!(l::lu(&a).is_err());
        assert!(l::lu_factor(&a).is_err());
        assert!(l::solve(&a, &[1.]).is_err());
        assert!(l::inv(&a).is_err());
        assert!(l::det(&a).is_err());
        assert!(l::eigh(&a, true).is_err());
        assert!(l::eigvalsh(&a, true).is_err());
        assert!(l::eigvals(&a).is_err());
        assert!(l::hessenberg(&a).is_err());
        assert!(l::schur(&a).is_err());
        assert!(l::polar(&a, true).is_err());
        assert!(l::pinv(&a, 0., None).is_err());
        assert!(l::pinvh(&a, 0., None, true).is_err());
        assert!(l::orth(&a, None).is_err());
        assert!(l::null_space(&a, None).is_err());
        assert!(l::norm(&a, l::MatrixNorm::Frobenius).is_err());
    }
    assert!(l::toeplitz(&[f64::NAN], None).is_err());
    assert!(l::hankel(&[1.], Some(&[f64::NAN])).is_err());
    assert!(l::circulant(&[f64::INFINITY]).is_err());
    assert!(l::fiedler(&[f64::NAN]).is_err());
    assert!(l::companion(&[0., 1.]).is_err());
    assert!(l::companion(&[1.]).is_err());
    assert!(l::fiedler_companion(&[0., 1.]).is_err());
    assert!(l::fiedler_companion(&[1., f64::NAN]).is_err());
    assert!(l::leslie(&[1.], &[]).is_err());
    assert!(l::leslie(&[1., 2.], &[f64::NAN]).is_err());
    assert!(l::block_diag(&[vec![vec![1.], vec![1., 2.]]]).is_err());
    assert!(l::block_diag(&[vec![vec![f64::NAN]]]).is_err());
    assert!(l::convolution_matrix(&[], 1, Mode::Full).is_err());
    assert!(l::convolution_matrix(&[1.], 0, Mode::Full).is_err());
    assert!(l::convolution_matrix(&[1.], usize::MAX, Mode::Same).is_err());
    assert!(l::hadamard(0).is_err());
    assert!(l::hadamard(3).is_err());
    assert!(l::helmert(0, false).is_err());
    assert!(l::dft(0, l::DftScale::None).is_err());
    assert!(l::diagsvd(&[1.], 2, 2).is_err());
    assert!(l::diagsvd(&[f64::NAN], 1, 1).is_err());
    assert!(l::invhilbert(129).is_err());
    assert!(l::pascal(257, l::PascalKind::Lower).is_err());
    assert!(l::invpascal(257, l::PascalKind::Upper).is_err());
    assert!(l::bandwidth(&[vec![f64::NAN]]).is_err());
    assert!(l::issymmetric(&[vec![1., 2.]], None, None).is_err());
    for n in [1001, usize::MAX] {
        assert!(l::hilbert(n).is_err());
        assert!(l::hadamard(n).is_err());
        assert!(l::helmert(n, true).is_err());
        assert!(l::dft(n, l::DftScale::None).is_err());
        assert!(l::diagsvd(&[], n, n).is_err());
    }
    assert!(l::toeplitz(&vec![1.; 1001], Some(&vec![1.; 1000])).is_err());
    assert!(l::hankel(&vec![1.; 1001], Some(&vec![1.; 1000])).is_err());
    assert!(l::circulant(&vec![1.; 1001]).is_err());
    assert!(l::fiedler(&vec![1.; 1001]).is_err());
}

#[test]
fn ieee_values_overflow_and_stable_scaling() {
    let nan = f64::NAN;
    let inf = f64::INFINITY;
    type Predicate = fn(&[Vec<f64>], Option<f64>, Option<f64>) -> sci::Result<bool>;
    for predicate in [l::issymmetric as Predicate, l::ishermitian as Predicate] {
        assert!(predicate(&[vec![nan, inf], vec![inf, nan]], None, None).unwrap());
        assert!(!predicate(&[vec![nan, inf], vec![inf, nan]], Some(0.), None).unwrap());
        assert!(!predicate(&[vec![1., nan], vec![nan, 2.]], None, None).unwrap());
        assert!(!predicate(&[vec![1., inf], vec![-inf, 2.]], Some(1.), Some(1.)).unwrap());
        assert!(predicate(&[vec![inf]], Some(0.), Some(0.)).unwrap());
    }
    let t = l::toeplitz(&[-0.], None).unwrap();
    assert!(t[0][0].is_sign_negative());
    let h = l::hankel(&[-0.], None).unwrap();
    assert!(h[0][0].is_sign_negative());
    let s = l::diagsvd(&[-0.], 1, 1).unwrap();
    assert!(s[0][0].is_sign_negative());
    assert!(l::fiedler(&[f64::MAX, -f64::MAX]).is_err());
    assert!(l::companion(&[1e-300, f64::MAX]).is_err());
    assert!(l::fiedler_companion(&[1e-300, f64::MAX]).is_err());
    assert!(l::inv(&[vec![1e-320]]).is_err());
    assert!(l::solve(&[vec![1e-320]], &[1.]).is_err());
    assert!(l::det(&[vec![1e200, 0.], vec![0., 1e200]]).is_err());
    for scale in [1e-200, 1e200] {
        let a = vec![vec![2. * scale, 0.], vec![0., scale]];
        let s = l::svdvals(&a).unwrap();
        finite_close(s[0] / scale, 2., 3e-11, 3e-11);
        finite_close(s[1] / scale, 1., 3e-11, 3e-11);
        let e = l::eigvalsh(&a, true).unwrap();
        finite_close(e[0] / scale, 1., 3e-11, 3e-11);
        finite_close(e[1] / scale, 2., 3e-11, 3e-11);
        let e = l::eigvals(&a).unwrap();
        assert!(e.iter().all(|z| z.im == 0.));
        finite_close(
            l::norm(&a, l::MatrixNorm::Frobenius).unwrap() / scale,
            5.0_f64.sqrt(),
            3e-11,
            3e-11,
        );
    }
    assert!(l::pinv(&[vec![1.]], f64::MAX, Some(f64::MAX)).is_err());
    assert!(l::pinvh(&[vec![1.]], f64::MAX, Some(f64::MAX), true).is_err());
}

#[test]
fn inverse_identities_and_cross_module_interoperability() {
    let a = vec![vec![1., 2., 0.], vec![2., 4., 1.]];
    let (p, rank) = l::pinv(&a, 0., None).unwrap();
    assert_eq!(rank, 2);
    check_matrix(&mul(&mul(&a, &p), &a), &a);
    check_matrix(&mul(&mul(&p, &a), &p), &p);
    let ap = mul(&a, &p);
    let pa = mul(&p, &a);
    check_matrix(&ap, &trans(&ap));
    check_matrix(&pa, &trans(&pa));
    for n in 1..=6 {
        for k in [
            l::PascalKind::Lower,
            l::PascalKind::Upper,
            l::PascalKind::Symmetric,
        ] {
            check_matrix(
                &mul(&l::pascal(n, k).unwrap(), &l::invpascal(n, k).unwrap()),
                &identity(n),
            );
        }
        let h = l::helmert(n, true).unwrap();
        check_matrix(&mul(&h, &trans(&h)), &identity(n));
    }
    let x = vec![1., -2., 3.];
    let kernel = vec![1., 0.5];
    for mode in [Mode::Full, Mode::Same, Mode::Valid] {
        let c = l::convolution_matrix(&kernel, x.len(), mode).unwrap();
        check_vector(
            &l::matvec(&c, &x).unwrap(),
            &sci::signal::convolve(&x, &kernel, mode).unwrap(),
        );
    }
    let n = x.len();
    let f = l::dft(n, l::DftScale::None).unwrap();
    let spectrum = sci::fft::rfft(&x);
    for (k, z) in spectrum.iter().enumerate() {
        finite_close(
            f[k].iter().zip(&x).map(|(z, x)| z.re * x).sum(),
            z.re,
            3e-11,
            3e-11,
        );
        finite_close(
            f[k].iter().zip(&x).map(|(z, x)| z.im * x).sum(),
            z.im,
            3e-11,
            3e-11,
        );
    }
    let jac = sci::differentiate::jacobian(
        |x| vec![2. * x[0] + x[1], x[0] + 3. * x[1]],
        &[1., 2.],
        1e-4,
    )
    .unwrap();
    let result = l::solve(&jac, &[4., 7.]).unwrap();
    check_vector(&result, &[1., 2.]);
    let a = vec![vec![-1., 0.2], vec![0., -2.]];
    let q = identity(2);
    let x = l::solve_lyapunov(&a, &q).unwrap();
    let ax = mul(&a, &x);
    let xat = mul(&x, &trans(&a));
    let sum: Matrix = ax
        .iter()
        .zip(xat)
        .map(|(r, s)| r.iter().zip(s).map(|(a, b)| a + b).collect())
        .collect();
    check_matrix(&sum, &q);
    let a = vec![vec![0.2, 0.1], vec![-0.1, 0.3]];
    let x = l::solve_discrete_lyapunov(&a, &q).unwrap();
    let axat = mul(&mul(&a, &x), &trans(&a));
    let residual: Matrix = x
        .iter()
        .zip(axat)
        .map(|(r, s)| r.iter().zip(s).map(|(a, b)| a - b).collect())
        .collect();
    check_matrix(&residual, &q);
}

#[test]
fn matrix_functions_validate_work_and_satisfy_identities() {
    let a = vec![vec![0.2, 0.3], vec![0., -0.4]];
    let exp = l::expm(&a).unwrap();
    let minus: Matrix = a.iter().map(|r| r.iter().map(|x| -x).collect()).collect();
    check_matrix(&mul(&exp, &l::expm(&minus).unwrap()), &identity(2));
    let twice: Matrix = a
        .iter()
        .map(|r| r.iter().map(|x| 2. * x).collect())
        .collect();
    check_matrix(&mul(&exp, &exp), &l::expm(&twice).unwrap());
    let sin = l::sinm(&a).unwrap();
    let cos = l::cosm(&a).unwrap();
    let ss = mul(&sin, &sin);
    let cc = mul(&cos, &cos);
    let sum: Matrix = ss
        .iter()
        .zip(cc)
        .map(|(r, s)| r.iter().zip(s).map(|(a, b)| a + b).collect())
        .collect();
    check_matrix(&sum, &identity(2));
    let (_, d) = l::expm_frechet(&a, &identity(2)).unwrap();
    check_matrix(&d, &exp);
    let (_, zero) = l::expm_frechet(&a, &vec![vec![0.; 2]; 2]).unwrap();
    check_matrix(&zero, &vec![vec![0.; 2]; 2]);
    let n = l::expm_cond(&[vec![-700.]]).unwrap();
    finite_close(n, 700., 3e-11, 3e-11);
    let direction = vec![vec![1., 2.], vec![-1., 0.5]];
    let (_, d) = l::expm_frechet(&a, &direction).unwrap();
    let h = 1e-5;
    let plus: Matrix = a
        .iter()
        .zip(&direction)
        .map(|(r, s)| r.iter().zip(s).map(|(x, y)| x + h * y).collect())
        .collect();
    let minus: Matrix = a
        .iter()
        .zip(&direction)
        .map(|(r, s)| r.iter().zip(s).map(|(x, y)| x - h * y).collect())
        .collect();
    let plus = l::expm(&plus).unwrap();
    let minus = l::expm(&minus).unwrap();
    for (i, row) in d.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            assert!((v - (plus[i][j] - minus[i][j]) / (2. * h)).abs() < 1e-8);
        }
    }
    for invalid in [
        vec![],
        vec![vec![1., 2.]],
        vec![vec![f64::NAN]],
        vec![vec![f64::INFINITY]],
        vec![vec![1e200]],
    ] {
        assert!(l::expm(&invalid).is_err());
        assert!(l::cosm(&invalid).is_err());
        assert!(l::sinm(&invalid).is_err());
        assert!(l::tanm(&invalid).is_err());
        assert!(l::coshm(&invalid).is_err());
        assert!(l::sinhm(&invalid).is_err());
        assert!(l::tanhm(&invalid).is_err());
        assert!(l::expm_cond(&invalid).is_err());
        assert!(l::expm_frechet(&invalid, &invalid).is_err());
    }
    assert!(l::expm(&[vec![1000.]]).is_err());
    assert!(l::coshm(&[vec![1000.]]).is_err());
    assert!(l::sinhm(&[vec![1000.]]).is_err());
    assert_eq!(l::tanhm(&[vec![1000.]]).unwrap(), vec![vec![1.]]);
    assert!(l::expm_cond(&[vec![-1000.]]).is_err());
    assert!(l::expm_cond(&identity(9)).is_err());
    assert!(l::expm_frechet(&identity(129), &identity(129)).is_err());
    assert!(l::expm_frechet(&identity(2), &identity(1)).is_err());
    assert_eq!(l::expm(&[vec![-1000.]]).unwrap(), vec![vec![0.]]);
}

#[test]
fn qr_adapters_validate_shapes_and_ranges() {
    let q = identity(2);
    let r = vec![vec![2., 1.], vec![0., 3.]];
    assert!(l::khatri_rao(&r, &[vec![1.]]).is_err());
    assert!(l::khatri_rao(&[vec![f64::MAX]], &[vec![2.]]).is_err());
    for bad in [vec![], vec![vec![f64::NAN]], vec![vec![1.], vec![1., 2.]]] {
        assert!(l::khatri_rao(&bad, &r).is_err());
        assert!(l::rq(&bad).is_err());
        assert!(l::qr_multiply(&bad, &r, true).is_err());
        assert!(l::qr_update(&bad, &r, &[1., 2.], &[1., 2.]).is_err());
        assert!(l::qr_insert(&q, &r, &bad, 0, true).is_err());
        assert!(l::qr_delete(&bad, &r, 0, 1, true).is_err());
    }
    assert!(l::qr_multiply(&r, &[vec![1.]], true).is_err());
    assert!(l::qr_multiply(&r, &[vec![1.]], false).is_err());
    assert!(l::qr_update(&q, &r, &[1.], &[1., 2.]).is_err());
    assert!(l::qr_update(&q, &r, &[1., f64::NAN], &[1., 2.]).is_err());
    assert!(l::qr_update(&q, &r, &[f64::MAX; 2], &[2.; 2]).is_err());
    assert!(l::qr_insert(&q, &r, &[vec![1., 2.]], usize::MAX, true).is_err());
    assert!(l::qr_insert(&q, &r, &[vec![1.]], 1, true).is_err());
    assert!(l::qr_insert(&q, &r, &[vec![1., 2.]], 0, false).is_err());
    assert!(l::qr_insert(&q, &r, &vec![vec![1., 2.]; 255], 2, true).is_err());
    for (index, count) in [(0, 0), (0, 2), (2, 1), (usize::MAX, 1), (0, usize::MAX)] {
        assert!(l::qr_delete(&q, &r, index, count, true).is_err());
        assert!(l::qr_delete(&q, &r, index, count, false).is_err());
    }
    let c = vec![vec![1., 2.], vec![3., 5.]];
    let (product, r) = l::qr_multiply(&r, &c, true).unwrap();
    let recovered = mul(&product, &l::inv(&c).unwrap());
    check_matrix(&mul(&recovered, &r), &[vec![2., 1.], vec![0., 3.]]);
    assert!(l::qr_delete(&q, &vec![vec![f64::MAX; 2]; 2], 0, 1, true).is_ok());
}

#[test]
fn advanced_domains_and_matrix_identities() {
    use sci::complex::Complex;
    let a = [vec![1., 2.], vec![0., 3.]];
    let id = identity(2);
    let zero = [vec![0., 0.], vec![0., 0.]];
    let jordan = [vec![1., 1.], vec![0., 1.]];
    let nilpotent = [vec![0., 1.], vec![0., 0.]];
    for invalid in [
        vec![],
        vec![vec![1., 2.]],
        vec![vec![f64::NAN]],
        vec![vec![f64::INFINITY]],
        identity(257),
    ] {
        assert!(l::eig(&invalid).is_err());
        assert!(l::ldl(&invalid, true).is_err());
        assert!(l::sqrtm(&invalid).is_err());
        assert!(l::logm(&invalid).is_err());
        assert!(l::signm(&invalid).is_err());
        assert!(l::fractional_matrix_power(&invalid, 0.5).is_err());
        assert!(l::funm(&invalid, |z| z).is_err());
        assert!(l::qz(&invalid, &invalid).is_err());
        assert!(l::matrix_balance(&invalid, true, true).is_err());
    }
    assert!(l::logm(&zero).is_err());
    assert!(l::sqrtm(&nilpotent).is_err());
    assert!(l::signm(&zero).is_err());
    assert!(l::signm(&[vec![0., -1.], vec![1., 0.]]).is_err());
    assert!(l::funm(&jordan, sci::complex::exp).is_err());
    assert!(
        l::funm(&a, |_| Complex {
            re: f64::NAN,
            im: 0.
        })
        .is_err()
    );
    for exponent in [f64::NAN, f64::INFINITY, 4_294_967_297.] {
        assert!(l::fractional_matrix_power(&a, exponent).is_err());
    }
    assert!(l::fractional_matrix_power(&zero, -1.).is_err());
    ccheck(
        &l::fractional_matrix_power(&nilpotent, 0.).unwrap(),
        &cfrom(&id),
    );
    ccheck(
        &l::fractional_matrix_power(&nilpotent, 2.).unwrap(),
        &cfrom(&zero),
    );
    let e = l::eig(&[vec![0., -1.], vec![1., 0.]]).unwrap();
    let (d, v) = l::cdf2rdf(&e.eigenvalues, &e.eigenvectors).unwrap();
    check_matrix(&mul(&[vec![0., -1.], vec![1., 0.]], &v), &mul(&v, &d));
    let log = l::logm(&jordan).unwrap();
    ccheck(&log, &cfrom(&nilpotent));
    for scale in [1e-200, 1e200] {
        let root = l::sqrtm(&[vec![4. * scale]]).unwrap();
        finite_close(root[0][0].re / (2. * scale.sqrt()), 1., 3e-11, 3e-11);
        let logarithm = l::logm(&[vec![scale]]).unwrap();
        finite_close(logarithm[0][0].re, scale.ln(), 3e-11, 3e-11);
        let power = l::fractional_matrix_power(&[vec![scale]], 0.5).unwrap();
        finite_close(power[0][0].re / scale.sqrt(), 1., 3e-11, 3e-11);
        let eig = l::eig(&[vec![scale]]).unwrap();
        finite_close(eig.eigenvalues[0].re / scale, 1., 3e-11, 3e-11);
    }
    assert!(l::rsf2csf(&a, &zero).is_err());
    assert!(
        l::rsf2csf(
            &[vec![1., 0., 0.], vec![1., 1., 0.], vec![0., 1., 1.]],
            &identity(3)
        )
        .is_err()
    );
    assert!(l::qz(&a, &[vec![1.]]).is_err());
    let f = l::qz(&[vec![0.]], &[vec![0.]]).unwrap();
    assert_eq!(f.alpha[0], Complex::default());
    assert_eq!(f.beta[0], Complex::default());
    let f = l::qz(&[vec![2.]], &[vec![0.]]).unwrap();
    assert!(f.alpha[0].re.abs() > 0.);
    assert_eq!(f.beta[0], Complex::default());
    let w = [Complex { re: 0., im: 1. }];
    assert!(l::cdf2rdf(&w, &[vec![Complex { re: 1., im: 0. }]]).is_err());
    assert!(
        l::cdf2rdf(
            &[Complex { re: 1., im: 0. }],
            &[vec![Complex { re: 1., im: 1. }]]
        )
        .is_err()
    );
    assert!(l::clarkson_woodruff_transform(&a, 1, &[0], &[1.]).is_err());
    assert!(l::clarkson_woodruff_transform(&a, 1, &[0, 1], &[1., -1.]).is_err());
    assert!(l::clarkson_woodruff_transform(&a, 1, &[0, 0], &[0., 1.]).is_err());
    assert!(l::clarkson_woodruff_transform(&a, 0, &[0, 0], &[1., 1.]).is_err());
    assert!(l::cossin(&a, 1, 1).is_err());
    assert!(l::cossin(&id, 0, 1).is_err());
    assert!(l::cossin(&id, 1, 2).is_err());
    let (balanced, t) = l::matrix_balance(&[vec![1., 1e8], vec![1e-8, 2.]], false, true).unwrap();
    assert!(balanced[0][1].abs() + balanced[1][0].abs() < 3.);
    check_matrix(
        &mul(&mul(&t, &balanced), &l::inv(&t).unwrap()),
        &[vec![1., 1e8], vec![1e-8, 2.]],
    );
    let (balanced, t) = l::matrix_balance(&a, false, false).unwrap();
    check_matrix(&balanced, &a);
    check_matrix(&t, &id);
    for discrete in [false, true] {
        type Are = fn(&[Vec<f64>], &[Vec<f64>], &[Vec<f64>], &[Vec<f64>]) -> sci::Result<Matrix>;
        let are: Are = if discrete {
            l::solve_discrete_are
        } else {
            l::solve_continuous_are
        };
        assert!(are(&a, &[vec![1.]], &id, &[vec![1.]]).is_err());
        assert!(are(&a, &id, &id, &zero).is_err());
        assert!(are(&a, &id, &a, &id).is_err());
        assert!(are(&[vec![2.]], &[vec![0.]], &[vec![1.]], &[vec![1.]]).is_err());
        assert!(
            are(
                &identity(129),
                &identity(129),
                &identity(129),
                &identity(129)
            )
            .is_err()
        );
    }
}

#[test]
fn interpolative_validation_and_round_trips() {
    use l::interpolative::{self as ip, Rank};
    let a = [vec![1., 2., 3.], vec![2., 4., 6.]];
    for bad in [
        Rank::Fixed(3),
        Rank::Relative(-1.),
        Rank::Relative(1.),
        Rank::Relative(f64::NAN),
    ] {
        assert!(ip::interp_decomp(&a, bad).is_err());
        assert!(ip::svd(&a, bad).is_err());
    }
    assert!(ip::estimate_rank(&[] as &[Vec<f64>], 1e-10).is_err());
    assert!(ip::estimate_spectral_norm(&[vec![f64::NAN]]).is_err());
    assert!(ip::estimate_spectral_norm_diff(&a, &[vec![1.]]).is_err());
    for indices in [vec![], vec![0, 0], vec![2, 0], (0..257).collect()] {
        assert!(ip::reconstruct_interp_matrix(&indices, &[vec![1.]]).is_err());
    }
    assert!(ip::reconstruct_interp_matrix(&[0, 1], &[vec![1., 2.]]).is_err());
    assert!(ip::reconstruct_matrix_from_id(&[vec![f64::NAN]], &[0, 1], &[vec![1.]]).is_err());
    assert!(ip::reconstruct_skel_matrix(&a, 1, &[0, 1]).is_err());
    assert!(ip::reconstruct_skel_matrix(&a, 4, &[0, 1, 2]).is_err());
    let f = ip::interp_decomp(&a, Rank::Relative(1e-12)).unwrap();
    assert_eq!(f.rank, 1);
    let b = ip::reconstruct_skel_matrix(&a, f.rank, &f.indices).unwrap();
    check_matrix(
        &ip::reconstruct_matrix_from_id(&b, &f.indices, &f.projection).unwrap(),
        &a,
    );
    let f = ip::svd(&a, Rank::Relative(1e-12)).unwrap();
    assert_eq!(f.singular_values.len(), 1);
    let zero = [vec![0., 0.], vec![0., 0.]];
    let f = ip::interp_decomp(&zero, Rank::Relative(1e-12)).unwrap();
    assert_eq!(f.rank, 0);
    let b = ip::reconstruct_skel_matrix(&zero, 0, &f.indices).unwrap();
    assert_eq!(b, vec![vec![], vec![]]);
    check_matrix(
        &ip::reconstruct_matrix_from_id(&b, &f.indices, &f.projection).unwrap(),
        &zero,
    );
    let converted = ip::id_to_svd(&b, &f.indices, &f.projection).unwrap();
    assert!(converted.singular_values.is_empty());
    assert_eq!(converted.u.len(), 2);
    assert!(converted.vh.is_empty());
    let f = ip::interp_decomp(&a, Rank::Fixed(2)).unwrap();
    assert_eq!(f.projection[0].len(), 1);
    let b = ip::reconstruct_skel_matrix(&a, 2, &f.indices).unwrap();
    check_matrix(
        &ip::reconstruct_matrix_from_id(&b, &f.indices, &f.projection).unwrap(),
        &a,
    );
}

#[test]
fn typed_dispatch_calls_shared_apis_and_rejects_unknown_names() {
    use l::{BlasFunction as B, LapackFunction as L};
    assert!(l::get_blas_funcs(&["sdot"]).is_err());
    assert!(l::get_lapack_funcs(&["zgeev"]).is_err());
    assert!(l::get_blas_funcs(&["unknown"]).is_err());
    assert!(l::get_lapack_funcs(&["unknown"]).is_err());
    assert!(l::find_best_blas_type(&[vec![vec![1.], vec![1., 2.]]]).is_err());
    assert!(l::get_blas_funcs(&[]).unwrap().is_empty());
    assert!(l::get_lapack_funcs(&[]).unwrap().is_empty());
    if let B::Dot(f) = l::get_blas_funcs(&["ddot"]).unwrap()[0] {
        assert!(f(&[1.], &[]).is_err());
        assert!(f(&[f64::NAN], &[1.]).is_err());
    }
    if let B::Nrm2(f) = l::get_blas_funcs(&["dnrm2"]).unwrap()[0] {
        finite_close(f(&[3e200, 4e200]).unwrap() / 1e200, 5., 3e-11, 3e-11);
        assert!(f(&[f64::INFINITY]).is_err());
    }
    if let B::Gemv(f) = l::get_blas_funcs(&["gemv"]).unwrap()[0] {
        assert_eq!(f(&[vec![f64::MAX]], &[2.]), Err(Error::NonFiniteEvaluation));
    }
    let a = [vec![2., 0.], vec![0., 3.]];
    let id = identity(2);
    let handles = l::get_lapack_funcs(&[
        "getrf", "getrs", "potrf", "potrs", "geev", "syev", "syevd", "gesvd", "gesdd", "gges",
    ])
    .unwrap();
    for f in handles {
        match f {
            L::Getrf(f) => assert_eq!(f(&a).unwrap().row_permutation.len(), 2),
            L::Getrs(f) => check_vector(
                &f(&l::lu_factor(&a).unwrap(), &[4., 9.], false).unwrap(),
                &[2., 3.],
            ),
            L::Potrf(f) => check_matrix(
                &mul(&f(&a, true).unwrap(), &trans(&f(&a, true).unwrap())),
                &a,
            ),
            L::Potrs(f) => check_vector(
                &f(&l::cho_factor(&a, true).unwrap(), &[4., 9.]).unwrap(),
                &[2., 3.],
            ),
            L::Geev(f) => assert_eq!(f(&a).unwrap().eigenvalues.len(), 2),
            L::Syev(f) => check_vector(&f(&a, true).unwrap().eigenvalues, &[2., 3.]),
            L::Gesvd(f) => check_vector(&f(&a).unwrap().singular_values, &[3., 2.]),
            L::Gges(f) => {
                let f = f(&a, &id).unwrap();
                ccheck(&cmul(&cmul(&f.q, &f.s), &cadjoint(&f.z)), &cfrom(&a));
            }
            L::Gesv(_) => unreachable!(),
        }
    }
}

#[test]
fn public_apis_match_scipy_references() {
    check_scipy_references(<[Vec<f64>]>::to_vec, <[f64]>::to_vec);
}

#[test]
#[cfg(feature = "linalg-ndarray")]
fn ndarray_inputs_match_scipy_references() {
    use l::{MatrixInput, VectorInput};
    check_scipy_references(
        |a| a.to_ndarray().unwrap(),
        |x| x.to_ndarray_vector().unwrap(),
    );
}

#[test]
#[cfg(feature = "linalg-nalgebra")]
fn nalgebra_inputs_match_scipy_references() {
    use l::{MatrixInput, VectorInput};
    check_scipy_references(
        |a| a.to_nalgebra().unwrap(),
        |x| x.to_nalgebra_vector().unwrap(),
    );
}

#[test]
#[cfg(feature = "linalg-faer")]
fn faer_inputs_match_scipy_references() {
    use l::{MatrixInput, VectorInput};
    check_scipy_references(|a| a.to_faer().unwrap(), |x| x.to_faer_col().unwrap());
}

#[test]
fn hyperbolic_extreme_scales_match_independent_references() {
    let inputs = common::lines(include_str!("fixtures/hyperbolic_inputs.csv"));
    let references = common::lines(include_str!("fixtures/hyperbolic_expected.csv"));
    assert_eq!(inputs.clone().count(), references.clone().count());
    for (input, reference) in inputs.zip(references) {
        let row: Vec<_> = common::fields(input);
        let expected: Vec<_> = common::fields(reference);
        assert_eq!(row[0], expected[0]);
        let a = matrix(row[2]);
        let actual = match row[1] {
            "sinhm" => l::sinhm(&a),
            "coshm" => l::coshm(&a),
            "tanhm" => l::tanhm(&a),
            _ => unreachable!(),
        }
        .unwrap();
        let expected = matrix(expected[1]);
        assert_eq!(actual.len(), expected.len());
        let general_dense = actual.len() == 3;
        let scale = expected
            .iter()
            .flatten()
            .fold(0.0_f64, |s, x| s.max(x.abs()));
        for (a, b) in actual.iter().zip(&expected) {
            assert_eq!(a.len(), b.len());
            for (&a, &b) in a.iter().zip(b) {
                // General dense matrices and zero entries incur product/solve
                // roundoff at the matrix scale. The small spectral cases use
                // strict relative comparisons for every nonzero component.
                let tolerance = if b == 0.0 || general_dense {
                    32.0 * f64::EPSILON * scale + 5e-12 * b.abs()
                } else {
                    (5e-12 * b.abs()).max(f64::from_bits(1))
                };
                if b != 0.0 && !general_dense {
                    assert_ne!(
                        a, 0.0,
                        "{input}: nonzero spectral reference underflowed to zero"
                    );
                }
                assert!(
                    a.is_finite() && approx::abs_diff_eq!(a, b, epsilon = tolerance),
                    "{input}: {a} != {b}"
                );
            }
        }
    }
}
