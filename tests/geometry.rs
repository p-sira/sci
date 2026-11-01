mod common;
use common::close;
use common::{matrix, vector};

use g::distance::{DistanceForm, DistanceOutput, Metric};
use sci::{linalg, spatial as g, spatial};

#[test]
fn distance_weight_normalization_preserves_caller_storage() {
    let u = [0., 1., 0.];
    let v = [1., 1., 1.];
    let weights = [0., f64::MAX, f64::MAX];
    let original = weights;
    // The two active coordinates have one mismatch. Large weights require
    // normalization for Hamming, while Euclidean uses their absolute scale.
    assert_eq!(g::distance::hamming(&u, &v, Some(&weights)).unwrap(), 0.5);
    assert_eq!(
        g::distance::euclidean(&u, &v, Some(&weights)).unwrap(),
        f64::MAX.sqrt()
    );
    assert_eq!(weights, original);
}

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
            "euclidean" => Some(g::distance::euclidean(&u, &v, weights)),
            "sqeuclidean" => Some(g::distance::sqeuclidean(&u, &v, weights)),
            "cityblock" => Some(g::distance::cityblock(&u, &v, weights)),
            "chebyshev" => Some(g::distance::chebyshev(&u, &v, weights)),
            "canberra" => Some(g::distance::canberra(&u, &v, weights)),
            "braycurtis" => Some(g::distance::braycurtis(&u, &v, weights)),
            "cosine" => Some(g::distance::cosine(&u, &v, weights)),
            "correlation" => Some(g::distance::correlation(&u, &v, weights, c[6] == "true")),
            "hamming" => Some(g::distance::hamming(&u, &v, weights)),
            "jaccard" => Some(g::distance::jaccard(&u, &v, weights)),
            "dice" => Some(g::distance::dice(&u, &v, weights)),
            "rogerstanimoto" => Some(g::distance::rogerstanimoto(&u, &v, weights)),
            "russellrao" => Some(g::distance::russellrao(&u, &v, weights)),
            "sokalsneath" => Some(g::distance::sokalsneath(&u, &v, weights)),
            "yule" => Some(g::distance::yule(&u, &v, weights)),
            "minkowski" => Some(g::distance::minkowski(
                &u,
                &v,
                c[5].parse().unwrap(),
                weights,
            )),
            "minkowski_distance" => Some(g::distance::minkowski_distance(
                &u,
                &v,
                c[5].parse().unwrap(),
            )),
            "minkowski_distance_p" => Some(g::distance::minkowski_distance_p(
                &u,
                &v,
                c[5].parse().unwrap(),
            )),
            "seuclidean" => Some(g::distance::seuclidean(&u, &v, &vector(c[6]))),
            "mahalanobis" => Some(g::distance::mahalanobis(&u, &v, &matrix(c[6]))),
            "jensenshannon" => Some(g::distance::jensenshannon(
                &u,
                &v,
                if c[5].is_empty() {
                    None
                } else {
                    Some(c[5].parse().unwrap())
                },
            )),
            "is_valid_y" => Some(Ok(f64::from(g::distance::is_valid_y(&u)))),
            "is_valid_dm" => Some(Ok(f64::from(g::distance::is_valid_dm(
                &matrix(c[2]),
                c[5].parse().unwrap(),
            )))),
            "num_obs_y" => Some(g::distance::num_obs_y(&u).map(|x| x as f64)),
            "num_obs_dm" => Some(g::distance::num_obs_dm(&matrix(c[2])).map(|x| x as f64)),
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
                    g::distance::cdist(&x, &y, metric(c[5], c[6], &extra_vector, &extra_matrix))
                        .unwrap()
                }
                "pdist" => {
                    vec![
                        g::distance::pdist(&x, metric(c[5], c[6], &extra_vector, &extra_matrix))
                            .unwrap(),
                    ]
                }
                "distance_matrix" => {
                    g::distance::distance_matrix(&x, &y, c[5].parse().unwrap()).unwrap()
                }
                "geometric_slerp" => {
                    g::transform::geometric_slerp(&u, &v, &w, c[5].parse().unwrap()).unwrap()
                }
                "squareform" => match g::distance::squareform(if c[5] == "condensed" {
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
                    let r = g::transform::procrustes(&x, &y).unwrap();
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
                    let r = g::distance::directed_hausdorff(&x, &y).unwrap();
                    close(
                        g::distance::euclidean(&x[r.index_u], &y[r.index_v], None).unwrap(),
                        r.distance,
                        3e-12,
                        3e-12,
                    );
                    let nearest = y
                        .iter()
                        .map(|b| g::distance::euclidean(&x[r.index_u], b, None).unwrap())
                        .fold(f64::INFINITY, f64::min);
                    close(nearest, r.distance, 3e-12, 3e-12);
                    vec![vec![r.distance, r.index_u as f64, r.index_v as f64]]
                }
                "tsearch" => {
                    let cells: Vec<_> = c[4]
                        .split(':')
                        .map(|r| r.split(';').map(|x| x.parse().unwrap()).collect())
                        .collect();
                    vec![
                        g::transform::tsearch(&x, &cells, &y, c[5].parse().unwrap())
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
                close(a, b, 3e-12, 3e-12);
            }
        }
    }
    assert_eq!(covered.len(), 33);
}

#[test]
fn domains_special_values_and_work_limits() {
    type Weighted = fn(&[f64], &[f64], Option<&[f64]>) -> sci::Result<f64>;
    let functions: [Weighted; 14] = [
        g::distance::euclidean,
        g::distance::sqeuclidean,
        g::distance::cityblock,
        g::distance::chebyshev,
        g::distance::canberra,
        g::distance::braycurtis,
        g::distance::cosine,
        g::distance::hamming,
        g::distance::jaccard,
        g::distance::dice,
        g::distance::rogerstanimoto,
        g::distance::russellrao,
        g::distance::sokalsneath,
        g::distance::yule,
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
        assert!(g::distance::minkowski(&[0.0], &[1.0], p, None).is_err());
        assert!(g::distance::minkowski_distance(&[0.0], &[1.0], p).is_err());
        assert!(g::distance::minkowski_distance_p(&[0.0], &[1.0], p).is_err());
        assert!(g::distance::distance_matrix(&[], &[], p).is_err());
    }
    for f in [
        g::distance::dice,
        g::distance::rogerstanimoto,
        g::distance::russellrao,
        g::distance::sokalsneath,
        g::distance::yule,
    ] {
        assert!(f(&[2.0], &[1.0], None).is_err());
    }
    assert!(g::distance::sokalsneath(&[0.0], &[0.0], None).is_err());
    assert!(g::distance::cosine(&[0.0], &[0.0], None).unwrap().is_nan());
    assert!(
        g::distance::correlation(&[1.0, 1.0], &[2.0, 2.0], None, true)
            .unwrap()
            .is_nan()
    );
    assert!(g::distance::correlation(&[1.0], &[], None, true).is_err());
    for variance in [vec![0.0], vec![-1.0], vec![f64::NAN], vec![]] {
        assert!(g::distance::seuclidean(&[0.0], &[1.0], &variance).is_err());
    }
    for vi in [
        vec![vec![-1.0]],
        vec![vec![0.0]],
        vec![vec![f64::NAN]],
        vec![],
    ] {
        assert!(g::distance::mahalanobis(&[0.0], &[1.0], &vi).is_err());
    }
    assert!(
        g::distance::mahalanobis(&[0.0, 0.0], &[1.0, 1.0], &[vec![1.0, 1.0], vec![0.0, 1.0]])
            .is_err()
    );
    for (p, q, b) in [
        (vec![0.0], vec![1.0], None),
        (vec![-1.0], vec![1.0], None),
        (vec![1.0], vec![1.0], Some(1.0)),
        (vec![1.0], vec![1.0], Some(f64::INFINITY)),
        (vec![f64::NAN], vec![1.0], None),
    ] {
        assert!(g::distance::jensenshannon(&p, &q, b).is_err());
    }
    assert!(!g::distance::is_valid_dm(&[vec![f64::NAN]], 0.0));
    assert!(!g::distance::is_valid_dm(&[vec![0.0]], -1.0));
    assert!(!g::distance::is_valid_dm(&[vec![0.0]], f64::NAN));
    assert!(!g::distance::is_valid_dm(&[vec![-1.0]], 0.01));
    assert!(g::distance::num_obs_dm(&[vec![0.0, 1.0]]).is_err());
    assert!(g::distance::num_obs_y(&[]).is_err());
    assert!(g::distance::num_obs_y(&[1.0, 2.0]).is_err());
    assert!(g::distance::squareform(DistanceForm::Condensed(&[1.0, 2.0])).is_err());
    assert!(g::distance::squareform(DistanceForm::Condensed(&[f64::NAN])).is_err());
    assert!(g::distance::squareform(DistanceForm::Square(&[vec![1.0]])).is_err());
    let ragged = vec![vec![1.0], vec![1.0, 2.0]];
    assert!(g::distance::cdist(&ragged, &ragged, Metric::Euclidean).is_err());
    assert!(g::distance::pdist(&ragged, Metric::Euclidean).is_err());
    assert!(g::distance::cdist(&[vec![1.0]], &[vec![1.0, 2.0]], Metric::Euclidean).is_err());
    assert!(g::distance::pdist(&vec![vec![0.0]; 1500], Metric::Euclidean).is_err());
    assert!(
        g::distance::cdist(
            &vec![vec![0.0; 20]; 800],
            &vec![vec![0.0; 20]; 800],
            Metric::Euclidean
        )
        .is_err()
    );
    assert!(g::distance::cdist(&[], &[], Metric::Minkowski(0.0)).is_err());
    assert!(g::distance::pdist(&[], Metric::JensenShannon(Some(0.5))).is_err());
    assert!(g::distance::directed_hausdorff(&[], &[]).is_err());
    assert!(g::distance::directed_hausdorff(&ragged, &ragged).is_err());
    assert!(
        g::distance::directed_hausdorff(&vec![vec![0.0; 20]; 800], &vec![vec![0.0; 20]; 800])
            .is_err()
    );
    assert!(g::distance::euclidean(&[f64::MAX], &[-f64::MAX], None).is_err());
    assert!(g::distance::sqeuclidean(&[1e200], &[0.0], None).is_err());
    assert!(g::distance::minkowski_distance_p(&[1e200], &[0.0], 2.0).is_err());
    assert_eq!(
        g::distance::euclidean(&[-0.0], &[0.0], None)
            .unwrap()
            .to_bits(),
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
        assert!(g::transform::geometric_slerp(a, b, &[0.5], 1e-7).is_err());
    }
    for tol in [-1.0, 1.0, f64::NAN, f64::INFINITY] {
        assert!(g::transform::geometric_slerp(&[1.0, 0.0], &[0.0, 1.0], &[], tol).is_err());
    }
    assert!(g::transform::geometric_slerp(&[1.0, 0.0], &[0.0, 1.0], &[f64::NAN], 0.0).is_err());
    assert!(
        g::transform::geometric_slerp(&[1.0, 0.0], &[0.0, 1.0], &vec![0.5; 500_001], 0.0).is_err()
    );
    assert!(
        g::transform::geometric_slerp(&[1.0, 0.0], &[0.0, 1.0], &[], 0.0)
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
        assert!(g::transform::procrustes(&x, &x).is_err());
    }
    assert!(g::transform::procrustes(&[vec![0.0], vec![1.0]], &[vec![0.0]]).is_err());
    assert!(g::transform::procrustes(&vec![vec![0.0; 65]; 2], &vec![vec![1.0; 65]; 2]).is_err());
    let v = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
    for cells in [vec![vec![0, 1]], vec![vec![0, 1, 3]], vec![vec![0, 1, 1]]] {
        assert!(g::transform::tsearch(&v, &cells, &[], 1e-12).is_err());
    }
    assert!(g::transform::tsearch(&v, &[vec![0, 1, 2]], &[vec![1.0]], 0.0).is_err());
    assert!(g::transform::tsearch(&v, &[], &[], -1.0).is_err());
    assert_eq!(
        g::transform::tsearch(&v, &[], &[vec![0.0, 0.0]], 0.0).unwrap(),
        vec![None]
    );
    assert!(
        g::distance::pdist(&[vec![0.0]], Metric::Euclidean)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        g::distance::cdist(&[vec![0.0]], &[], Metric::Euclidean).unwrap(),
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
    let y = g::distance::pdist(&x, Metric::Euclidean).unwrap();
    let DistanceOutput::Square(dm) = g::distance::squareform(DistanceForm::Condensed(&y)).unwrap()
    else {
        panic!()
    };
    assert_eq!(
        g::distance::squareform(DistanceForm::Square(&dm)).unwrap(),
        DistanceOutput::Condensed(y)
    );
    assert_eq!(g::distance::distance_matrix(&x, &x, 2.0).unwrap(), dm);
    assert_eq!(
        spatial::distance::euclidean(&x[0], &x[1], None).unwrap(),
        dm[0][1]
    );
    close(
        linalg::basic::matvec(&dm, &[1.0, 0.0, 0.0]).unwrap()[1],
        dm[1][0],
        3e-12,
        3e-12,
    );
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                assert!(dm[i][k] <= dm[i][j] + dm[j][k] + 1e-14);
            }
        }
    }
    for row in g::transform::geometric_slerp(
        &[1.0, 0.0, 0.0],
        &[0.0, 1.0, 0.0],
        &[-2.0, 0.1, 0.5, 1.7],
        0.0,
    )
    .unwrap()
    {
        close(linalg::basic::dot(&row, &row).unwrap(), 1.0, 3e-12, 3e-12);
    }
    let shifted: Vec<_> = x
        .iter()
        .map(|r| vec![3.0 + 2.0 * r[1], 1.0 - 2.0 * r[0], 4.0 + 2.0 * r[2]])
        .collect();
    let fit = g::transform::procrustes(&x, &shifted).unwrap();
    close(fit.disparity, 0.0, 3e-12, 3e-12);
    close(
        fit.reference.iter().flatten().map(|x| x * x).sum(),
        1.0,
        3e-12,
        3e-12,
    );
    let reverse = g::transform::procrustes(&shifted, &x).unwrap();
    close(fit.disparity, reverse.disparity, 3e-12, 3e-12);
    let vertices = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    assert_eq!(
        g::transform::tsearch(
            &vertices,
            &[vec![0, 1, 2], vec![1, 3, 2]],
            &[vec![0.5, 0.5], vec![0.8, 0.8], vec![2.0, 2.0]],
            1e-12
        )
        .unwrap(),
        vec![Some(0), Some(1), None]
    );
    close(
        g::distance::euclidean(&[1e300, 1e300], &[0.0, 0.0], None).unwrap() / 1e300,
        2.0_f64.sqrt(),
        3e-12,
        3e-12,
    );
    close(
        g::distance::canberra(&[1e308], &[-1e308], None).unwrap(),
        1.0,
        3e-12,
        3e-12,
    );
    close(
        g::distance::hamming(&[1.0, 0.0], &[0.0, 0.0], Some(&[1e308, 1e308])).unwrap(),
        0.5,
        3e-12,
        3e-12,
    );
    close(
        g::distance::dice(&[1.0, 0.0], &[1.0, 1.0], Some(&[1e308, 1e308])).unwrap(),
        1.0 / 3.0,
        3e-12,
        3e-12,
    );
    close(
        g::distance::braycurtis(&[1.0, 2.0], &[2.0, 1.0], Some(&[1e308, 1e308])).unwrap(),
        1.0 / 3.0,
        3e-12,
        3e-12,
    );
    close(
        g::distance::minkowski_distance_p(&[1e300, 1e300], &[0.0, 0.0], 0.1).unwrap() / 1e30,
        2.0,
        3e-12,
        3e-12,
    );
    let tie = g::distance::directed_hausdorff(&[vec![0.0], vec![2.0]], &[vec![1.0]]).unwrap();
    assert_eq!((tie.index_u, tie.index_v), (0, 0));
    close(
        g::distance::cosine(&[1e300, 1e300], &[1e300, 1e300], None).unwrap(),
        0.0,
        3e-12,
        3e-12,
    );
    close(
        g::distance::jensenshannon(&[1e308, 1e308], &[1.0, 1.0], None).unwrap(),
        0.0,
        3e-12,
        3e-12,
    );
    assert_eq!(
        g::distance::chebyshev(&[f64::MAX, 0.0], &[-f64::MAX, 2.0], Some(&[0.0, 1.0])).unwrap(),
        2.0
    );
}
