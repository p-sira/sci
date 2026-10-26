mod common;
use common::close;
use common::{matrix, vector};

use sci::{Error, Spline, SurfaceSpline, interpolate as p};

fn spline(s: &str) -> Spline {
    let f: Vec<_> = s.split('|').collect();
    Spline {
        knots: vector(f[0]),
        coefficients: vector(f[1]),
        degree: f[2].parse().unwrap(),
    }
}
fn surface(s: &str) -> SurfaceSpline {
    let f: Vec<_> = s.split('|').collect();
    SurfaceSpline {
        x_knots: vector(f[0]),
        y_knots: vector(f[1]),
        coefficients: vector(f[2]),
        x_degree: f[3].parse().unwrap(),
        y_degree: f[4].parse().unwrap(),
    }
}
fn optional(s: &str) -> Option<Vec<f64>> {
    (!s.is_empty()).then(|| vector(s))
}
fn check(actual: &[f64], expected: &str) {
    let expected = vector(expected);
    assert_eq!(actual.len(), expected.len());
    for (&a, e) in actual.iter().zip(expected) {
        close(a, e, 2e-10, 2e-10);
    }
}
fn check_matrix(actual: &[Vec<f64>], expected: &str) {
    let rows: Vec<_> = expected.split(':').collect();
    assert_eq!(actual.len(), rows.len());
    for (a, e) in actual.iter().zip(rows) {
        check(a, e);
    }
}
fn extension(i: &str) -> p::SplineExtension {
    match i {
        "0" => p::SplineExtension::Extrapolate,
        "1" => p::SplineExtension::Zero,
        "2" => p::SplineExtension::Error,
        "3" => p::SplineExtension::Clamp,
        _ => panic!(),
    }
}
fn check_fit(s: &Spline, query: &[f64], expected: &[&str]) {
    check(
        &p::splev(query, s, 0, p::SplineExtension::Extrapolate).unwrap(),
        expected[1],
    );
    check(&s.knots, expected[2]);
    assert_eq!(s.degree, expected[3].parse::<usize>().unwrap());
}
fn check_transform(s: &Spline, query: &[f64], expected: &[&str]) {
    check(
        &p::splev(query, s, 0, p::SplineExtension::Extrapolate).unwrap(),
        expected[1],
    );
    check(&s.knots, expected[2]);
    check(&s.coefficients, expected[3]);
}

#[test]
fn every_added_inventory_name_matches_scipy() {
    let mut covered = std::collections::BTreeSet::new();
    let mut failures = Vec::new();
    let input =
        common::lines(include_str!("fixtures/interpolation_inputs.csv")).collect::<Vec<_>>();
    let expected =
        common::lines(include_str!("fixtures/interpolation_expected.csv")).collect::<Vec<_>>();
    assert_eq!(input.len(), expected.len());
    for (input, expected) in input.into_iter().zip(expected) {
        let c: Vec<_> = common::fields(input);
        let e: Vec<_> = common::fields(expected);
        assert_eq!(c[0], e[0]);
        covered.insert(c[1]);
        let a = &c[2..];
        let result = std::panic::catch_unwind(|| match c[1] {
            "make_interp_spline" => check_fit(
                &p::make_interp_spline(&vector(a[0]), &vector(a[1]), a[2].parse().unwrap())
                    .unwrap(),
                &vector(a[3]),
                &e,
            ),
            "make_lsq_spline" => check_fit(
                &p::make_lsq_spline(
                    &vector(a[0]),
                    &vector(a[1]),
                    &vector(a[2]),
                    a[3].parse().unwrap(),
                    optional(a[4]).as_deref(),
                )
                .unwrap(),
                &vector(a[5]),
                &e,
            ),
            "make_splrep" | "splrep" => {
                let f = p::make_splrep;
                check_fit(
                    &f(
                        &vector(a[0]),
                        &vector(a[1]),
                        a[2].parse().unwrap(),
                        a[3].parse().unwrap(),
                        optional(a[4]).as_deref(),
                    )
                    .unwrap(),
                    &vector(a[5]),
                    &e,
                );
            }
            "make_smoothing_spline" => check_fit(
                &p::make_smoothing_spline(
                    &vector(a[0]),
                    &vector(a[1]),
                    optional(a[2]).as_deref(),
                    a[3].parse().unwrap(),
                )
                .unwrap(),
                &vector(a[4]),
                &e,
            ),
            "make_splprep" | "splprep" => {
                let f = p::make_splprep;
                let s = f(
                    &matrix(a[0]),
                    optional(a[1]).as_deref(),
                    a[2].parse().unwrap(),
                    a[3].parse().unwrap(),
                    optional(a[4]).as_deref(),
                )
                .unwrap();
                let query = vector(a[5]);
                let mut out = vec![Vec::new(); query.len()];
                for component in &s.components {
                    let v =
                        p::splev(&query, component, 0, p::SplineExtension::Extrapolate).unwrap();
                    for (row, value) in out.iter_mut().zip(v) {
                        row.push(value);
                    }
                }
                check_matrix(&out, e[1]);
                check(&s.parameters, e[2]);
                check(&s.components[0].knots, e[3]);
                assert!(
                    s.components
                        .iter()
                        .all(|c| c.knots == s.components[0].knots)
                );
            }
            "approximate_taylor_polynomial" => {
                let f: fn(f64) -> f64 = match a[0] {
                    "square" => |x| x * x,
                    "exp" => f64::exp,
                    "sin" => f64::sin,
                    "constant" => |_| 1.,
                    _ => panic!(),
                };
                check(
                    &p::approximate_taylor_polynomial(
                        f,
                        a[1].parse().unwrap(),
                        a[2].parse().unwrap(),
                        a[3].parse().unwrap(),
                        (!a[4].is_empty()).then(|| a[4].parse().unwrap()),
                    )
                    .unwrap(),
                    e[1],
                );
            }
            "generate_knots" => check_matrix(
                &p::generate_knots(
                    &vector(a[0]),
                    &vector(a[1]),
                    a[2].parse().unwrap(),
                    a[3].parse().unwrap(),
                    optional(a[4]).as_deref(),
                )
                .unwrap(),
                e[1],
            ),
            "splev" => check(
                &p::splev(
                    &vector(a[0]),
                    &spline(a[1]),
                    a[2].parse().unwrap(),
                    extension(a[3]),
                )
                .unwrap(),
                e[1],
            ),
            "splder" => check_transform(
                &p::splder(&spline(a[0]), a[1].parse().unwrap()).unwrap(),
                &vector(a[2]),
                &e,
            ),
            "splantider" => check_transform(
                &p::splantider(&spline(a[0]), a[1].parse().unwrap()).unwrap(),
                &vector(a[2]),
                &e,
            ),
            "spalde" => check_matrix(&p::spalde(&vector(a[0]), &spline(a[1])).unwrap(), e[1]),
            "splint" => close(
                p::splint(a[0].parse().unwrap(), a[1].parse().unwrap(), &spline(a[2])).unwrap(),
                e[1].parse().unwrap(),
                2e-10,
                2e-10,
            ),
            "insert" => check_transform(
                &p::insert(a[0].parse().unwrap(), &spline(a[1]), a[2].parse().unwrap()).unwrap(),
                &vector(a[3]),
                &e,
            ),
            "sproot" => check(&p::sproot(&spline(a[0])).unwrap(), e[1]),
            "interpn" => check(
                &p::interpn(
                    &matrix(a[0]),
                    &vector(a[1]),
                    &matrix(a[2]),
                    if a[3] == "linear" {
                        p::GridMethod::Linear
                    } else {
                        p::GridMethod::Nearest
                    },
                    a[4].parse().unwrap(),
                    (!a[5].is_empty()).then(|| a[5].parse().unwrap()),
                )
                .unwrap(),
                e[1],
            ),
            "griddata" => check(
                &p::griddata(
                    &matrix(a[0]),
                    &vector(a[1]),
                    &matrix(a[2]),
                    match a[3] {
                        "linear" => p::ScatteredMethod::Linear,
                        "nearest" => p::ScatteredMethod::Nearest,
                        "cubic" => p::ScatteredMethod::Cubic,
                        _ => panic!(),
                    },
                    a[4].parse().unwrap(),
                    a[5].parse().unwrap(),
                )
                .unwrap(),
                e[1],
            ),
            "bisplrep" => {
                let s = p::bisplrep(
                    &vector(a[0]),
                    &vector(a[1]),
                    &vector(a[2]),
                    &vector(a[3]),
                    &vector(a[4]),
                    a[5].parse().unwrap(),
                    a[6].parse().unwrap(),
                    optional(a[7]).as_deref(),
                )
                .unwrap();
                check(&s.x_knots, e[1]);
                check(&s.y_knots, e[2]);
                check(&s.coefficients, e[3]);
            }
            "bisplev" => check_matrix(
                &p::bisplev(
                    &vector(a[0]),
                    &vector(a[1]),
                    &surface(a[2]),
                    a[3].parse().unwrap(),
                    a[4].parse().unwrap(),
                )
                .unwrap(),
                e[1],
            ),
            _ => panic!("unknown API {}", c[1]),
        });
        if result.is_err() {
            failures.push((c[0], c[1]));
        }
    }
    assert!(failures.is_empty(), "failed cases: {failures:?}");
    assert_eq!(covered.len(), 20);
}

#[test]
fn spline_calculus_and_refinement_preserve_identities() {
    let boundary_x: Vec<_> = (0..128).map(|i| i as f64 / 127.0).collect();
    let boundary_y: Vec<_> = boundary_x.iter().map(|x| 2.0 * x + 1.0).collect();
    let boundary_fit = p::make_interp_spline(&boundary_x, &boundary_y, 1).unwrap();
    for (&x, value) in boundary_x
        .iter()
        .zip(p::splev(&boundary_x, &boundary_fit, 0, p::SplineExtension::Error).unwrap())
    {
        close(value, 2.0 * x + 1.0, 2e-10, 2e-10);
    }
    let maximal_degree = Spline {
        knots: [vec![0.0; 33], vec![1.0; 33]].concat(),
        coefficients: vec![1.0; 33],
        degree: 32,
    };
    for value in p::splev(
        &[0.0, 0.2, 0.5, 1.0],
        &maximal_degree,
        0,
        p::SplineExtension::Error,
    )
    .unwrap()
    {
        close(value, 1.0, 2e-10, 2e-10);
    }
    assert!(p::splantider(&maximal_degree, 1).is_err());
    let x = [0., 0.3, 0.7, 1., 1.5, 2., 2.4, 3.];
    let y = x.map(|x| x * x * x - 2. * x + 1.);
    let s = p::make_interp_spline(&x, &y, 3).unwrap();
    let query: Vec<_> = (0..=60).map(|i| i as f64 / 20.).collect();
    let values = p::splev(&query, &s, 0, p::SplineExtension::Error).unwrap();
    for (&q, v) in query.iter().zip(&values) {
        close(*v, q * q * q - 2. * q + 1., 2e-10, 2e-10);
    }
    let d = p::splder(&s, 1).unwrap();
    for (&q, v) in query
        .iter()
        .zip(p::splev(&query, &d, 0, p::SplineExtension::Error).unwrap())
    {
        close(v, 3. * q * q - 2., 2e-10, 2e-10);
    }
    let round = p::splder(&p::splantider(&s, 2).unwrap(), 2).unwrap();
    for (&v, r) in values
        .iter()
        .zip(p::splev(&query, &round, 0, p::SplineExtension::Error).unwrap())
    {
        close(v, r, 2e-10, 2e-10);
    }
    let refined = p::insert(0.8, &s, 2).unwrap();
    for (&v, r) in values
        .iter()
        .zip(p::splev(&query, &refined, 0, p::SplineExtension::Error).unwrap())
    {
        close(v, r, 2e-10, 2e-10);
    }
    close(p::splint(0., 3., &s).unwrap(), 14.25, 2e-10, 2e-10);
    close(p::splint(3., 0., &s).unwrap(), -14.25, 2e-10, 2e-10);
    let roots = p::sproot(&s).unwrap();
    for r in roots {
        close(
            p::splev(&[r], &s, 0, p::SplineExtension::Error).unwrap()[0],
            0.,
            2e-10,
            2e-10,
        );
    }
    let noisy = [0., 0.4, -0.1, 0.8, 0.2, 1., 0.7, 1.5];
    let sequence = p::generate_knots(&x, &noisy, 3, 0.01, None).unwrap();
    assert!(
        sequence
            .windows(2)
            .all(|pair| pair[1].len() > pair[0].len())
    );
    let final_fit = p::make_lsq_spline(&x, &noisy, sequence.last().unwrap(), 3, None).unwrap();
    let v = p::splev(&x, &final_fit, 0, p::SplineExtension::Error).unwrap();
    assert!(
        v.iter()
            .zip(noisy)
            .map(|(v, y)| (v - y).powi(2))
            .sum::<f64>()
            <= 0.01001
    );
    let smooth = p::make_smoothing_spline(&x, &noisy, None, 1.).unwrap();
    for q in [x[0], x[x.len() - 1]] {
        close(
            p::splev(&[q], &smooth, 2, p::SplineExtension::Error).unwrap()[0],
            0.,
            2e-10,
            2e-10,
        );
    }
}

#[test]
fn invalid_shapes_domains_and_work_limits_are_errors() {
    let x = [0., 1., 2., 3., 4.];
    let y = [0., 1., 2., 3., 4.];
    let s = p::make_interp_spline(&x, &y, 3).unwrap();
    assert!(p::make_interp_spline(&x, &y, 0).is_err());
    assert!(p::make_interp_spline(&[0., 0.], &[1., 2.], 1).is_err());
    assert!(p::make_interp_spline(&x, &y[..4], 3).is_err());
    assert!(p::make_interp_spline(&[0.; 129], &[0.; 129], 1).is_err());
    assert!(p::make_lsq_spline(&x, &y, &[0., 0., 4., 4.], 1, Some(&[0.; 5])).is_err());
    assert!(p::make_lsq_spline(&x, &y, &[0., 0., 4., 4.], 1, Some(&[1.; 4])).is_err());
    assert!(p::make_lsq_spline(&x, &y, &[0., 0., 4., 4., 4.], 1, None).is_err());
    assert!(p::make_splrep(&x, &y, 3, -1., None).is_err());
    assert!(p::make_splrep(&x, &y, 2, 1., None).is_err());
    assert!(p::make_splrep(&x, &y, 1, 1., None).is_err());
    assert!(p::make_smoothing_spline(&x[..4], &y[..4], None, 1.).is_err());
    assert!(p::make_smoothing_spline(&x, &y, None, -1.).is_err());
    assert!(p::generate_knots(&x, &y, 3, -1., None).is_err());
    assert!(p::generate_knots(&[0.; 65], &[0.; 65], 1, 1., None).is_err());
    assert!(p::make_splprep(&[], None, 1, 0., None).is_err());
    assert!(p::make_splprep(&[vec![0., 0.], vec![0., 0.]], None, 1, 0., None).is_err());
    assert!(p::make_splprep(&[vec![0., 0.], vec![1.]], None, 1, 0., None).is_err());
    assert!(p::make_splprep(&[vec![0., 0.], vec![1., 1.]], Some(&[0., 0.]), 1, 0., None).is_err());
    assert!(p::splev(&[-1.], &s, 0, p::SplineExtension::Error).is_err());
    assert!(p::spalde(&[-1.], &s).is_err());
    assert!(p::splder(&s, 4).is_err());
    assert!(p::splantider(&s, usize::MAX).is_err());
    assert!(p::insert(0., &s, 1).is_err());
    assert!(p::insert(1., &s, 0).is_err());
    assert!(p::insert(1., &s, usize::MAX).is_err());
    assert!(p::sproot(&p::splder(&s, 1).unwrap()).is_err());
    let bad = Spline {
        degree: usize::MAX,
        ..s.clone()
    };
    assert!(p::splev(&[], &bad, 0, p::SplineExtension::Zero).is_err());
    assert!(p::splder(&bad, 0).is_err());
    assert!(p::splantider(&bad, 0).is_err());
    assert!(p::spalde(&[], &bad).is_err());
    assert!(p::splint(0., 1., &bad).is_err());
    assert!(p::insert(1., &bad, 1).is_err());
    assert!(p::sproot(&bad).is_err());
    let empty = Spline {
        knots: vec![],
        coefficients: vec![],
        degree: 0,
    };
    assert!(p::splev(&[], &empty, 0, p::SplineExtension::Clamp).is_err());
    let steps = p::insert(1., &s, 4).unwrap();
    assert!(p::splder(&steps, 1).is_err());
    assert!(p::splev(&vec![0.; 1_000_000], &s, 0, p::SplineExtension::Clamp).is_err());
    assert_eq!(
        p::splev(&[], &s, 0, p::SplineExtension::Error).unwrap(),
        vec![]
    );
    assert!(
        p::approximate_taylor_polynomial(
            |_| panic!("invalid controls must precede callbacks"),
            0.,
            32,
            1.,
            None
        )
        .is_err()
    );
    assert!(p::approximate_taylor_polynomial(|_| f64::NAN, 0., 2, 1., None).is_err());
    assert!(
        p::approximate_taylor_polynomial(
            |_| panic!("indistinguishable nodes must precede callbacks"),
            1e308,
            2,
            1.0,
            None,
        )
        .is_err()
    );
    for bad in [f64::NAN, f64::INFINITY] {
        assert!(p::splev(&[bad], &s, 0, p::SplineExtension::Clamp).is_err());
        assert!(p::spalde(&[bad], &s).is_err());
        assert!(p::splint(bad, 1., &s).is_err());
        assert!(p::insert(bad, &s, 1).is_err());
        assert!(p::make_splrep(&x, &y, 3, bad, None).is_err());
        assert!(p::make_smoothing_spline(&x, &y, None, bad).is_err());
        assert!(p::generate_knots(&x, &y, 3, bad, None).is_err());
    }
    assert!(
        p::interpn(
            &[vec![0., 0.]],
            &[1., 2.],
            &[],
            p::GridMethod::Linear,
            true,
            None
        )
        .is_err()
    );
    assert!(
        p::interpn(
            &[vec![0., 1.]],
            &[1.],
            &[],
            p::GridMethod::Linear,
            true,
            None
        )
        .is_err()
    );
    assert!(
        p::interpn(
            &[vec![0., 1.]],
            &[1., 2.],
            &[vec![2.]],
            p::GridMethod::Linear,
            true,
            None
        )
        .is_err()
    );
    assert!(
        p::griddata(
            &[vec![0., 0.], vec![1., 0.]],
            &[1., 2.],
            &[],
            p::ScatteredMethod::Linear,
            f64::NAN,
            false
        )
        .is_err()
    );
    assert!(
        p::griddata(
            &[vec![0., 0.], vec![1., 1.]],
            &[1., 2.],
            &[],
            p::ScatteredMethod::Cubic,
            f64::NAN,
            false
        )
        .is_err()
    );
    assert!(p::bisplrep(&x, &y, &y, &[0., 0., 4., 4.], &[0., 0., 4., 4.], 1, 1, None).is_err());
    let surface = SurfaceSpline {
        x_knots: vec![0., 0., 1., 1.],
        y_knots: vec![0., 0., 1., 1.],
        coefficients: vec![1.; 4],
        x_degree: 1,
        y_degree: 1,
    };
    assert!(p::bisplev(&[1., 0.], &[0.], &surface, 0, 0).is_err());
    assert!(p::bisplev(&[0.], &[0.], &surface, 2, 0).is_err());
    assert!(
        p::bisplev(
            &[],
            &[],
            &SurfaceSpline {
                x_degree: usize::MAX,
                ..surface.clone()
            },
            0,
            0
        )
        .is_err()
    );
    let invalid = SurfaceSpline {
        coefficients: vec![],
        ..surface
    };
    assert_eq!(
        p::bisplev(&[0.], &[0.], &invalid, 0, 0),
        Err(Error::DimensionMismatch)
    );
}
