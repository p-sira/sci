mod common;
use common::{matrix, vector};

use g::{DistanceForm, DistanceOutput, Metric};
use sci::{linalg, spatial as g, spatial};

fn metric<'a>(name: &str, extra: &str, v: &'a [f64], m: &'a [Vec<f64>]) -> Metric<'a> {
    match name {
        "euclidean" => Metric::Euclidean,
        "sqeuclidean" => Metric::SqEuclidean,
        "cityblock" => Metric::Cityblock,
        "chebyshev" => Metric::Chebyshev,
        "canberra" => Metric::Canberra,
        "braycurtis" => Metric::Braycurtis,
        "cosine" => Metric::Cosine,
        "correlation" => Metric::Correlation,
        "hamming" => Metric::Hamming,
        "jaccard" => Metric::Jaccard,
        "dice" => Metric::Dice,
        "rogerstanimoto" => Metric::Rogerstanimoto,
        "russellrao" => Metric::Russellrao,
        "sokalsneath" => Metric::Sokalsneath,
        "yule" => Metric::Yule,
        "minkowski" => Metric::Minkowski(extra.parse().unwrap()),
        "seuclidean" => Metric::Seuclidean(v),
        "mahalanobis" => Metric::Mahalanobis(m),
        "jensenshannon" => Metric::JensenShannon(None),
        _ => panic!("unknown metric"),
    }
}
fn close(a: f64, b: f64) {
    if b.is_nan() {
        assert!(a.is_nan(), "{a} != NaN");
    } else if b.is_infinite() {
        assert_eq!(a, b);
    } else {
        assert!(
            a.is_finite() && (a - b).abs() <= 3e-12 + 3e-12 * b.abs(),
            "{a} != {b}"
        );
    }
}

#[test]
fn all_numerical_inventory_names_match_scipy() {
    let inputs: Vec<_> = common::lines(include_str!("fixtures/geometry_inputs.csv")).collect();
    let expected: Vec<_> = common::lines(include_str!("fixtures/geometry_expected.csv")).collect();
    assert_eq!(inputs.len(), expected.len());
    let mut covered = std::collections::BTreeSet::new();
    for (input, expected) in inputs.into_iter().zip(expected) {
        let c: Vec<_> = common::fields(input);
        let e: Vec<_> = common::fields(expected);
        assert_eq!(c[0], e[0]);
        covered.insert(c[1]);
        let (u, v, w) = (
            vector(c[2].split(':').next().unwrap()),
            vector(c[3].split(':').next().unwrap()),
            vector(c[4]),
        );
        let weights = (!w.is_empty()).then_some(w.as_slice());
        let scalar = match c[1] {
            "euclidean" => Some(g::euclidean(&u, &v, weights)),
            "sqeuclidean" => Some(g::sqeuclidean(&u, &v, weights)),
            "cityblock" => Some(g::cityblock(&u, &v, weights)),
            "chebyshev" => Some(g::chebyshev(&u, &v, weights)),
            "canberra" => Some(g::canberra(&u, &v, weights)),
            "braycurtis" => Some(g::braycurtis(&u, &v, weights)),
            "cosine" => Some(g::cosine(&u, &v, weights)),
            "correlation" => Some(g::correlation(&u, &v, weights, c[6] == "true")),
            "hamming" => Some(g::hamming(&u, &v, weights)),
            "jaccard" => Some(g::jaccard(&u, &v, weights)),
            "dice" => Some(g::dice(&u, &v, weights)),
            "rogerstanimoto" => Some(g::rogerstanimoto(&u, &v, weights)),
            "russellrao" => Some(g::russellrao(&u, &v, weights)),
            "sokalsneath" => Some(g::sokalsneath(&u, &v, weights)),
            "yule" => Some(g::yule(&u, &v, weights)),
            "minkowski" => Some(g::minkowski(&u, &v, c[5].parse().unwrap(), weights)),
            "minkowski_distance" => Some(g::minkowski_distance(&u, &v, c[5].parse().unwrap())),
            "minkowski_distance_p" => Some(g::minkowski_distance_p(&u, &v, c[5].parse().unwrap())),
            "seuclidean" => Some(g::seuclidean(&u, &v, &vector(c[6]))),
            "mahalanobis" => Some(g::mahalanobis(&u, &v, &matrix(c[6]))),
            "jensenshannon" => Some(g::jensenshannon(
                &u,
                &v,
                if c[5].is_empty() {
                    None
                } else {
                    Some(c[5].parse().unwrap())
                },
            )),
            "is_valid_y" => Some(Ok(f64::from(g::is_valid_y(&u)))),
            "is_valid_dm" => Some(Ok(f64::from(g::is_valid_dm(
                &matrix(c[2]),
                c[5].parse().unwrap(),
            )))),
            "num_obs_y" => Some(g::num_obs_y(&u).map(|x| x as f64)),
            "num_obs_dm" => Some(g::num_obs_dm(&matrix(c[2])).map(|x| x as f64)),
            _ => None,
        };
        let actual = if let Some(s) = scalar {
            vec![vec![s.unwrap_or_else(|err| panic!("{input}: {err}"))]]
        } else {
            let (x, y) = (matrix(c[2]), matrix(c[3]));
            let extra_vector = vector(c[6].split(':').next().unwrap());
            let extra_matrix = matrix(c[6]);
            match c[1] {
                "cdist" => {
                    g::cdist(&x, &y, metric(c[5], c[6], &extra_vector, &extra_matrix)).unwrap()
                }
                "pdist" => {
                    vec![g::pdist(&x, metric(c[5], c[6], &extra_vector, &extra_matrix)).unwrap()]
                }
                "distance_matrix" => g::distance_matrix(&x, &y, c[5].parse().unwrap()).unwrap(),
                "geometric_slerp" => g::geometric_slerp(&u, &v, &w, c[5].parse().unwrap()).unwrap(),
                "squareform" => match g::squareform(if c[5] == "condensed" {
                    DistanceForm::Condensed(&u)
                } else {
                    DistanceForm::Square(&x)
                })
                .unwrap()
                {
                    DistanceOutput::Square(m) => m,
                    DistanceOutput::Condensed(v) => vec![v],
                },
                "procrustes" => {
                    let r = g::procrustes(&x, &y).unwrap();
                    assert_eq!(r.reference.len(), x.len());
                    assert_eq!(r.transformed.len(), x.len());
                    assert!(
                        r.reference
                            .iter()
                            .chain(&r.transformed)
                            .all(|r| r.len() == x[0].len())
                    );
                    vec![
                        r.reference
                            .into_iter()
                            .flatten()
                            .chain(r.transformed.into_iter().flatten())
                            .chain([r.disparity])
                            .collect(),
                    ]
                }
                "directed_hausdorff" => {
                    let r = g::directed_hausdorff(&x, &y).unwrap();
                    close(
                        g::euclidean(&x[r.index_u], &y[r.index_v], None).unwrap(),
                        r.distance,
                    );
                    let nearest = y
                        .iter()
                        .map(|b| g::euclidean(&x[r.index_u], b, None).unwrap())
                        .fold(f64::INFINITY, f64::min);
                    close(nearest, r.distance);
                    vec![vec![r.distance, r.index_u as f64, r.index_v as f64]]
                }
                "tsearch" => {
                    let cells: Vec<_> = c[4]
                        .split(':')
                        .map(|r| r.split(';').map(|x| x.parse().unwrap()).collect())
                        .collect();
                    vec![
                        g::tsearch(&x, &cells, &y, c[5].parse().unwrap())
                            .unwrap()
                            .into_iter()
                            .map(|x| x.map_or(-1.0, |x| x as f64))
                            .collect(),
                    ]
                }
                _ => panic!("unknown fixture {input}"),
            }
        };
        let (rows, cols): (usize, usize) = (e[1].parse().unwrap(), e[2].parse().unwrap());
        assert_eq!(actual.len(), rows, "{input}");
        assert!(actual.iter().all(|r| r.len() == cols), "{input}");
        let reference = vector(e[3]);
        assert_eq!(reference.len(), rows * cols);
        for (i, (a, b)) in actual.into_iter().flatten().zip(reference).enumerate() {
            // SciPy shuffles, so tied witness indices may differ. Shapes and
            // the returned witnesses are checked above; compare the distance.
            if c[1] != "directed_hausdorff" || i == 0 {
                close(a, b);
            }
        }
    }
    assert_eq!(covered.len(), 33);
}

#[test]
fn domains_special_values_and_work_limits() {
    type Weighted = fn(&[f64], &[f64], Option<&[f64]>) -> sci::Result<f64>;
    let functions: [Weighted; 14] = [
        g::euclidean,
        g::sqeuclidean,
        g::cityblock,
        g::chebyshev,
        g::canberra,
        g::braycurtis,
        g::cosine,
        g::hamming,
        g::jaccard,
        g::dice,
        g::rogerstanimoto,
        g::russellrao,
        g::sokalsneath,
        g::yule,
    ];
    for f in functions {
        assert!(f(&[1.0], &[], None).is_err());
        for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(f(&[x], &[1.0], None).is_err());
        }
        for w in [&[-1.0][..], &[0.0], &[f64::NAN], &[f64::INFINITY], &[]] {
            assert!(f(&[1.0], &[1.0], Some(w)).is_err());
        }
        assert!(f(&vec![1.0; 1_000_001], &vec![1.0; 1_000_001], None).is_err());
    }
    for p in [0.0, -1.0, f64::NAN, f64::NEG_INFINITY] {
        assert!(g::minkowski(&[0.0], &[1.0], p, None).is_err());
        assert!(g::minkowski_distance(&[0.0], &[1.0], p).is_err());
        assert!(g::minkowski_distance_p(&[0.0], &[1.0], p).is_err());
        assert!(g::distance_matrix(&[], &[], p).is_err());
    }
    for f in [
        g::dice,
        g::rogerstanimoto,
        g::russellrao,
        g::sokalsneath,
        g::yule,
    ] {
        assert!(f(&[2.0], &[1.0], None).is_err());
    }
    assert!(g::sokalsneath(&[0.0], &[0.0], None).is_err());
    assert!(g::cosine(&[0.0], &[0.0], None).unwrap().is_nan());
    assert!(
        g::correlation(&[1.0, 1.0], &[2.0, 2.0], None, true)
            .unwrap()
            .is_nan()
    );
    assert!(g::correlation(&[1.0], &[], None, true).is_err());
    for variance in [vec![0.0], vec![-1.0], vec![f64::NAN], vec![]] {
        assert!(g::seuclidean(&[0.0], &[1.0], &variance).is_err());
    }
    for vi in [
        vec![vec![-1.0]],
        vec![vec![0.0]],
        vec![vec![f64::NAN]],
        vec![],
    ] {
        assert!(g::mahalanobis(&[0.0], &[1.0], &vi).is_err());
    }
    assert!(g::mahalanobis(&[0.0, 0.0], &[1.0, 1.0], &[vec![1.0, 1.0], vec![0.0, 1.0]]).is_err());
    for (p, q, b) in [
        (vec![0.0], vec![1.0], None),
        (vec![-1.0], vec![1.0], None),
        (vec![1.0], vec![1.0], Some(1.0)),
        (vec![1.0], vec![1.0], Some(f64::INFINITY)),
        (vec![f64::NAN], vec![1.0], None),
    ] {
        assert!(g::jensenshannon(&p, &q, b).is_err());
    }
    assert!(!g::is_valid_dm(&[vec![f64::NAN]], 0.0));
    assert!(!g::is_valid_dm(&[vec![0.0]], -1.0));
    assert!(!g::is_valid_dm(&[vec![0.0]], f64::NAN));
    assert!(!g::is_valid_dm(&[vec![-1.0]], 0.01));
    assert!(g::num_obs_dm(&[vec![0.0, 1.0]]).is_err());
    assert!(g::num_obs_y(&[]).is_err());
    assert!(g::num_obs_y(&[1.0, 2.0]).is_err());
    assert!(g::squareform(DistanceForm::Condensed(&[1.0, 2.0])).is_err());
    assert!(g::squareform(DistanceForm::Condensed(&[f64::NAN])).is_err());
    assert!(g::squareform(DistanceForm::Square(&[vec![1.0]])).is_err());
    let ragged = vec![vec![1.0], vec![1.0, 2.0]];
    assert!(g::cdist(&ragged, &ragged, Metric::Euclidean).is_err());
    assert!(g::pdist(&ragged, Metric::Euclidean).is_err());
    assert!(g::cdist(&[vec![1.0]], &[vec![1.0, 2.0]], Metric::Euclidean).is_err());
    assert!(g::pdist(&vec![vec![0.0]; 1500], Metric::Euclidean).is_err());
    assert!(
        g::cdist(
            &vec![vec![0.0; 20]; 800],
            &vec![vec![0.0; 20]; 800],
            Metric::Euclidean
        )
        .is_err()
    );
    assert!(g::cdist(&[], &[], Metric::Minkowski(0.0)).is_err());
    assert!(g::pdist(&[], Metric::JensenShannon(Some(0.5))).is_err());
    assert!(g::directed_hausdorff(&[], &[]).is_err());
    assert!(g::directed_hausdorff(&ragged, &ragged).is_err());
    assert!(g::directed_hausdorff(&vec![vec![0.0; 20]; 800], &vec![vec![0.0; 20]; 800]).is_err());
    assert!(g::euclidean(&[f64::MAX], &[-f64::MAX], None).is_err());
    assert!(g::sqeuclidean(&[1e200], &[0.0], None).is_err());
    assert!(g::minkowski_distance_p(&[1e200], &[0.0], 2.0).is_err());
    assert_eq!(
        g::euclidean(&[-0.0], &[0.0], None).unwrap().to_bits(),
        0.0_f64.to_bits()
    );
}

#[test]
fn spatial_domains_and_degenerate_shapes() {
    for (a, b) in [
        (&[2.0, 0.0][..], &[2.0, 0.0][..]),
        (&[1.0, 0.0], &[-1.0, 0.0]),
        (&[1.0], &[1.0]),
        (&[f64::NAN, 0.0], &[1.0, 0.0]),
        (&[1.0, 0.0], &[1.0]),
    ] {
        assert!(g::geometric_slerp(a, b, &[0.5], 1e-7).is_err());
    }
    for tol in [-1.0, 1.0, f64::NAN, f64::INFINITY] {
        assert!(g::geometric_slerp(&[1.0, 0.0], &[0.0, 1.0], &[], tol).is_err());
    }
    assert!(g::geometric_slerp(&[1.0, 0.0], &[0.0, 1.0], &[f64::NAN], 0.0).is_err());
    assert!(g::geometric_slerp(&[1.0, 0.0], &[0.0, 1.0], &vec![0.5; 500_001], 0.0).is_err());
    assert!(
        g::geometric_slerp(&[1.0, 0.0], &[0.0, 1.0], &[], 0.0)
            .unwrap()
            .is_empty()
    );
    for x in [
        vec![],
        vec![vec![1.0]],
        vec![vec![1.0], vec![1.0]],
        vec![vec![f64::NAN], vec![1.0]],
        vec![vec![]],
    ] {
        assert!(g::procrustes(&x, &x).is_err());
    }
    assert!(g::procrustes(&[vec![0.0], vec![1.0]], &[vec![0.0]]).is_err());
    assert!(g::procrustes(&vec![vec![0.0; 65]; 2], &vec![vec![1.0; 65]; 2]).is_err());
    let v = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
    for cells in [vec![vec![0, 1]], vec![vec![0, 1, 3]], vec![vec![0, 1, 1]]] {
        assert!(g::tsearch(&v, &cells, &[], 1e-12).is_err());
    }
    assert!(g::tsearch(&v, &[vec![0, 1, 2]], &[vec![1.0]], 0.0).is_err());
    assert!(g::tsearch(&v, &[], &[], -1.0).is_err());
    assert_eq!(
        g::tsearch(&v, &[], &[vec![0.0, 0.0]], 0.0).unwrap(),
        vec![None]
    );
    assert!(
        g::pdist(&[vec![0.0]], Metric::Euclidean)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        g::cdist(&[vec![0.0]], &[], Metric::Euclidean).unwrap(),
        vec![Vec::<f64>::new()]
    );
}

#[test]
fn independent_properties_and_interoperability() {
    let x = vec![
        vec![1.0, 2.0, 3.0],
        vec![3.0, -1.0, 2.0],
        vec![0.0, 2.0, -1.0],
    ];
    let y = g::pdist(&x, Metric::Euclidean).unwrap();
    let DistanceOutput::Square(dm) = g::squareform(DistanceForm::Condensed(&y)).unwrap() else {
        panic!()
    };
    assert_eq!(
        g::squareform(DistanceForm::Square(&dm)).unwrap(),
        DistanceOutput::Condensed(y)
    );
    assert_eq!(g::distance_matrix(&x, &x, 2.0).unwrap(), dm);
    assert_eq!(
        spatial::distance::euclidean(&x[0], &x[1], None).unwrap(),
        dm[0][1]
    );
    close(linalg::matvec(&dm, &[1.0, 0.0, 0.0]).unwrap()[1], dm[1][0]);
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                assert!(dm[i][k] <= dm[i][j] + dm[j][k] + 1e-14);
            }
        }
    }
    for row in g::geometric_slerp(
        &[1.0, 0.0, 0.0],
        &[0.0, 1.0, 0.0],
        &[-2.0, 0.1, 0.5, 1.7],
        0.0,
    )
    .unwrap()
    {
        close(linalg::dot(&row, &row).unwrap(), 1.0);
    }
    let shifted: Vec<_> = x
        .iter()
        .map(|r| vec![3.0 + 2.0 * r[1], 1.0 - 2.0 * r[0], 4.0 + 2.0 * r[2]])
        .collect();
    let fit = g::procrustes(&x, &shifted).unwrap();
    close(fit.disparity, 0.0);
    close(fit.reference.iter().flatten().map(|x| x * x).sum(), 1.0);
    let reverse = g::procrustes(&shifted, &x).unwrap();
    close(fit.disparity, reverse.disparity);
    let vertices = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    assert_eq!(
        g::tsearch(
            &vertices,
            &[vec![0, 1, 2], vec![1, 3, 2]],
            &[vec![0.5, 0.5], vec![0.8, 0.8], vec![2.0, 2.0]],
            1e-12
        )
        .unwrap(),
        vec![Some(0), Some(1), None]
    );
    close(
        g::euclidean(&[1e300, 1e300], &[0.0, 0.0], None).unwrap() / 1e300,
        2.0_f64.sqrt(),
    );
    close(g::canberra(&[1e308], &[-1e308], None).unwrap(), 1.0);
    close(
        g::hamming(&[1.0, 0.0], &[0.0, 0.0], Some(&[1e308, 1e308])).unwrap(),
        0.5,
    );
    close(
        g::dice(&[1.0, 0.0], &[1.0, 1.0], Some(&[1e308, 1e308])).unwrap(),
        1.0 / 3.0,
    );
    close(
        g::braycurtis(&[1.0, 2.0], &[2.0, 1.0], Some(&[1e308, 1e308])).unwrap(),
        1.0 / 3.0,
    );
    close(
        g::minkowski_distance_p(&[1e300, 1e300], &[0.0, 0.0], 0.1).unwrap() / 1e30,
        2.0,
    );
    let tie = g::directed_hausdorff(&[vec![0.0], vec![2.0]], &[vec![1.0]]).unwrap();
    assert_eq!((tie.index_u, tie.index_v), (0, 0));
    close(
        g::cosine(&[1e300, 1e300], &[1e300, 1e300], None).unwrap(),
        0.0,
    );
    close(
        g::jensenshannon(&[1e308, 1e308], &[1.0, 1.0], None).unwrap(),
        0.0,
    );
    assert_eq!(
        g::chebyshev(&[f64::MAX, 0.0], &[-f64::MAX, 2.0], Some(&[0.0, 1.0])).unwrap(),
        2.0
    );
}
