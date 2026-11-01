mod common;
use common::{matrix, vector};

use sci::cluster::hierarchy::{self as h, Criterion, Cut, Method};
fn integers(s: &str) -> Vec<usize> {
    vector(s).into_iter().map(|v| v as usize).collect()
}
fn criterion<'a>(s: &str, depth: usize, stats: &'a [f64]) -> Criterion<'a> {
    match s {
        "distance" => Criterion::Distance,
        "inconsistent" => Criterion::Inconsistent(depth),
        "maxclust" => Criterion::MaxClust,
        "monocrit" => Criterion::Monocrit(stats),
        "maxclust_monocrit" => Criterion::MaxClustMonocrit(stats),
        _ => panic!("unknown criterion"),
    }
}
#[test]
fn public_hierarchy_matches_scipy() {
    let inputs: Vec<_> = common::lines(include_str!("fixtures/cluster_inputs.csv")).collect();
    let expected: Vec<_> = common::lines(include_str!("fixtures/cluster_expected.csv")).collect();
    assert_eq!(inputs.len(), expected.len());
    let mut covered = std::collections::BTreeSet::new();
    for (input, expected) in inputs.iter().zip(expected) {
        let c: Vec<_> = common::fields(input);
        let e: Vec<_> = common::fields(expected);
        assert_eq!(c[0], e[0]);
        covered.insert(c[1]);
        let z = matrix(c[2]);
        let y = vector(c[3]);
        let depth = c[4].parse().unwrap();
        let boolean = |v: bool| vec![vec![if v { 1.0 } else { 0.0 }]];
        let labels = |v: Vec<usize>| vec![v.into_iter().map(|v| v as f64).collect::<Vec<_>>()];
        let got = match c[1] {
            "linkage" => h::linkage(&y, Method::Average).unwrap(),
            "single" => h::single(&y).unwrap(),
            "complete" => h::complete(&y).unwrap(),
            "average" => h::average(&y).unwrap(),
            "weighted" => h::weighted(&y).unwrap(),
            "centroid" => h::centroid(&y).unwrap(),
            "median" => h::median(&y).unwrap(),
            "ward" => h::ward(&y).unwrap(),
            "num_obs_linkage" => vec![vec![h::num_obs_linkage(&z).unwrap() as f64]],
            "is_valid_linkage" => boolean(h::is_valid_linkage(&z)),
            "is_monotonic" => boolean(h::is_monotonic(&z).unwrap()),
            "leaves_list" => labels(h::leaves_list(&z).unwrap()),
            "maxdists" => vec![h::maxdists(&z).unwrap()],
            "to_mlab_linkage" => h::to_mlab_linkage(&z).unwrap(),
            "from_mlab_linkage" => h::from_mlab_linkage(&z).unwrap(),
            "to_tree" => h::to_tree(&z)
                .unwrap()
                .into_iter()
                .map(|v| {
                    vec![
                        v.id as f64,
                        v.left.map_or(-1.0, |v| v as f64),
                        v.right.map_or(-1.0, |v| v as f64),
                        v.distance,
                        v.count as f64,
                    ]
                })
                .collect(),
            "correspond" => boolean(h::correspond(&z, &y).unwrap()),
            "cophenet" => {
                let (corr, d) = h::cophenet(&z, Some(&y)).unwrap();
                vec![std::iter::once(corr.unwrap()).chain(d).collect()]
            }
            "inconsistent" => h::inconsistent(&z, depth).unwrap(),
            "is_valid_im" => boolean(h::is_valid_im(&z)),
            "maxRstat" => vec![
                h::max_rstat(
                    &z,
                    &h::inconsistent(&z, depth).unwrap(),
                    c[5].parse().unwrap(),
                )
                .unwrap(),
            ],
            "maxinconsts" => {
                vec![h::maxinconsts(&z, &h::inconsistent(&z, depth).unwrap()).unwrap()]
            }
            "is_isomorphic" => boolean(h::is_isomorphic(&integers(c[8]), &integers(c[9])).unwrap()),
            "fcluster" => {
                let stats = h::maxdists(&z).unwrap();
                labels(
                    h::fcluster(&z, c[6].parse().unwrap(), criterion(c[7], depth, &stats)).unwrap(),
                )
            }
            "fclusterdata" => labels(
                h::fclusterdata(
                    &matrix(c[12]),
                    c[6].parse().unwrap(),
                    Method::Single,
                    Criterion::Distance,
                )
                .unwrap(),
            ),
            "leaders" => {
                let (nodes, ids) = h::leaders(&z, &integers(c[8])).unwrap();
                nodes
                    .into_iter()
                    .zip(ids)
                    .map(|(a, b)| vec![a as f64, b as f64])
                    .collect()
            }
            "cut_tree" => {
                let counts = integers(c[10]);
                let heights = vector(c[11]);
                let cut = if !counts.is_empty() {
                    Cut::Counts(&counts)
                } else if !heights.is_empty() {
                    Cut::Heights(&heights)
                } else {
                    Cut::All
                };
                h::cut_tree(&z, cut)
                    .unwrap()
                    .into_iter()
                    .map(|r| r.into_iter().map(|v| v as f64).collect())
                    .collect()
            }
            _ => panic!("unknown API"),
        };
        let want = vector(e[3]);
        assert_eq!(got.len(), e[1].parse::<usize>().unwrap(), "{} rows", c[0]);
        assert!(
            got.iter()
                .all(|r| r.len() == e[2].parse::<usize>().unwrap()),
            "{} columns",
            c[0]
        );
        let got: Vec<_> = got.into_iter().flatten().collect();
        if c[1] == "fcluster" || c[1] == "fclusterdata" {
            assert!(
                h::is_isomorphic(
                    &got.iter().map(|v| *v as usize).collect::<Vec<_>>(),
                    &want.iter().map(|v| *v as usize).collect::<Vec<_>>()
                )
                .unwrap(),
                "{}: {got:?} != {want:?}",
                c[0]
            );
        } else {
            for (a, b) in got.into_iter().zip(want) {
                assert!(
                    if b.is_nan() {
                        a.is_nan()
                    } else {
                        a.is_finite()
                            && approx::abs_diff_eq!(a, b, epsilon = 3e-12 + 3e-12 * b.abs())
                    },
                    "{}: {a} != {b}",
                    c[0]
                );
            }
        }
    }
    assert_eq!(covered.len(), 27);
}

#[test]
fn validation_and_tree_properties() {
    let z = h::average(&[1.0, 4.0, 6.0, 3.0, 5.0, 2.0]).unwrap();
    assert_eq!(
        h::from_mlab_linkage(&h::to_mlab_linkage(&z).unwrap()).unwrap(),
        z
    );
    let leaves = h::leaves_list(&z).unwrap();
    let mut sorted = leaves.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, vec![0, 1, 2, 3]);
    let nodes = h::to_tree(&z).unwrap();
    assert_eq!(nodes.last().unwrap().count, 4);
    let (_, d) = h::cophenet(&z, None).unwrap();
    assert_eq!(h::correspond(&z, &d), Ok(true));
    let cuts = h::cut_tree(&z, Cut::All).unwrap();
    for stage in 0..4 {
        assert_eq!(
            cuts.iter()
                .map(|r| r[stage])
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            4 - stage
        );
    }
    let duplicate_cuts = h::cut_tree(&z, Cut::Counts(&[4, 4, 2, 2])).unwrap();
    for r in duplicate_cuts {
        assert_eq!(r[0], r[1]);
        assert_eq!(r[2], r[3]);
    }
    let codes = sci::cluster::vq::whiten(&[vec![0.0], vec![2.0], vec![4.0]]).unwrap();
    assert_eq!(
        h::fclusterdata(&codes, 10.0, Method::Ward, Criterion::Distance).unwrap(),
        vec![1, 1, 1]
    );
    for invalid in [
        vec![],
        vec![vec![0.0, 0.0, 1.0, 2.0]],
        vec![vec![0.0, 1.0, f64::NAN, 2.0]],
        vec![vec![0.0, 1.0, 1.0, 3.0]],
        vec![vec![0.0, 3.0, 1.0, 2.0]],
        vec![vec![0.0, 1.0]],
        vec![vec![0.5, 1.0, 1.0, 2.0]],
    ] {
        assert!(!h::is_valid_linkage(&invalid));
        assert!(h::leaves_list(&invalid).is_err());
        assert!(h::cophenet(&invalid, None).is_err());
        assert!(h::to_tree(&invalid).is_err());
        assert!(h::cut_tree(&invalid, Cut::All).is_err());
        assert!(h::maxdists(&invalid).is_err());
        assert!(h::inconsistent(&invalid, 2).is_err());
    }
    for y in [
        vec![],
        vec![1.0, 2.0],
        vec![-1.0],
        vec![f64::NAN],
        vec![f64::INFINITY],
        vec![1.0; 513 * 512 / 2],
    ] {
        assert!(h::single(&y).is_err());
    }
    assert!(h::ward(&[f64::MAX]).is_err());
    assert!(h::from_mlab_linkage(&[vec![0.0, 2.0, 1.0]]).is_err());
    assert!(h::from_mlab_linkage(&[vec![1.0, 1.0, 1.0]]).is_err());
    assert!(h::from_mlab_linkage(&[vec![1.0, 2.0, f64::INFINITY]]).is_err());
    assert!(h::is_isomorphic(&[1], &[]).is_err());
    assert!(h::fcluster(&z, f64::NAN, Criterion::Distance).is_err());
    assert!(h::fcluster(&z, 0.0, Criterion::MaxClust).is_err());
    assert!(h::fcluster(&z, 1.5, Criterion::MaxClust).is_err());
    assert!(h::fcluster(&z, 1.0, Criterion::Monocrit(&[1.0])).is_err());
    assert!(h::fcluster(&z, 1.0, Criterion::Monocrit(&[3.0, 2.0, 1.0])).is_err());
    assert!(h::leaders(&z, &[0, 1, 0, 1]).is_err());
    assert!(h::leaders(&z, &[0]).is_err());
    assert!(h::max_rstat(&z, &h::inconsistent(&z, 2).unwrap(), 4).is_err());
    assert!(h::max_rstat(&z, &[], 0).is_err());
    assert!(h::cophenet(&z, Some(&[1.0])).is_err());
    assert!(h::cophenet(&z, Some(&[f64::NAN; 6])).is_err());
    assert!(h::cut_tree(&z, Cut::Counts(&[0])).is_err());
    assert!(h::cut_tree(&z, Cut::Heights(&[f64::INFINITY])).is_err());
    let inverted = vec![vec![0.0, 1.0, 3.0, 2.0], vec![2.0, 3.0, 1.0, 3.0]];
    assert!(!h::is_monotonic(&inverted).unwrap());
    assert_eq!(h::maxdists(&inverted).unwrap(), vec![3.0, 3.0]);
    assert!(h::cut_tree(&inverted, Cut::Heights(&[2.0])).is_err());
    assert!(!h::is_valid_im(&[vec![0.0, 0.0, 1.0, f64::NAN]]));
}

#[test]
fn whiten_matches_shared_population_std_at_extreme_scales() {
    use sci::cluster::vq;

    let cases = [
        vec![vec![1.0], vec![3.0]],
        vec![vec![1.0, -0.0], vec![1.0, 0.0]],
        vec![vec![1e-300], vec![-1e-300], vec![0.0]],
        vec![vec![f64::MAX], vec![-f64::MAX]],
        vec![vec![1e12 + 0.125], vec![1e12 + 0.25], vec![1e12 + 0.5]],
        vec![
            vec![1e200, -1e200, 7.0],
            vec![1e200 + 1e185, -1e200 + 1e185, 7.0],
            vec![1e200 + 2e185, -1e200 + 2e185, 7.0],
        ],
    ];
    for obs in cases {
        let columns = obs[0].len();
        let scales: Vec<_> = (0..columns)
            .map(|j| {
                let column: Vec<_> = obs.iter().map(|row| row[j]).collect();
                sci::stats::std(&column, 0)
            })
            .collect::<Result<_, _>>()
            .unwrap();
        let actual = vq::whiten(&obs).unwrap();
        for (row, original) in actual.iter().zip(&obs) {
            for (j, &value) in row.iter().enumerate() {
                let denominator = if scales[j] == 0.0 { 1.0 } else { scales[j] };
                let expected = original[j] / denominator;
                assert_eq!(
                    value.to_bits(),
                    expected.to_bits(),
                    "whiten changed f64 rounding at column {j}, input {}",
                    original[j]
                );
            }
        }
    }
}

#[cfg(feature = "spatial")]
#[test]
fn hierarchy_euclidean_matches_spatial_condensed_order_and_errors() {
    use sci::spatial::{Metric, pdist};

    let observations = [
        vec![vec![0.0], vec![2.0], vec![5.0], vec![9.0]],
        vec![
            vec![0.0, 0.0],
            vec![3.0, 4.0],
            vec![3.0, -4.0],
            vec![0.0, 0.0],
        ],
        vec![
            vec![1e150, -1e150],
            vec![1e150 + 1e135, -1e150 + 1e135],
            vec![1e150 + 2e135, -1e150 + 2e135],
        ],
    ];
    for x in observations {
        let distances = pdist(&x, Metric::Euclidean).unwrap();
        for method in [
            Method::Single,
            Method::Complete,
            Method::Average,
            Method::Weighted,
            Method::Centroid,
            Method::Median,
            Method::Ward,
        ] {
            for threshold in [0.0, 3.0, 10.0] {
                let expected = h::fcluster(
                    &h::linkage(&distances, method).unwrap(),
                    threshold,
                    Criterion::Distance,
                )
                .unwrap();
                assert_eq!(
                    h::fclusterdata(&x, threshold, method, Criterion::Distance).unwrap(),
                    expected,
                    "{method:?} threshold {threshold}"
                );
            }
        }
    }

    let z = vec![vec![0.0, 1.0, 2.0, 2.0]];
    for length in [0, 1, 2, 3, 4, 6, 7, 10] {
        let y = vec![1.0; length];
        let expected = sci::spatial::num_obs_y(&y).map(|n| n == 2);
        assert_eq!(h::correspond(&z, &y), expected, "length {length}");
    }
    assert_eq!(
        h::fclusterdata(
            &[vec![0.0], vec![f64::NAN]],
            1.0,
            Method::Single,
            Criterion::Distance,
        ),
        Err(sci::Error::InvalidInput("coordinates must be finite")),
    );
    assert_eq!(
        h::fclusterdata(
            &[vec![f64::MAX], vec![-f64::MAX]],
            1.0,
            Method::Single,
            Criterion::Distance,
        ),
        Err(sci::Error::NonFiniteEvaluation),
    );
    assert_eq!(
        h::fclusterdata(
            &[vec![0.0], vec![0.0, 1.0]],
            1.0,
            Method::Single,
            Criterion::Distance,
        ),
        Err(sci::Error::DimensionMismatch),
    );
    let excessive_work = vec![vec![0.0; 100]; 512];
    assert_eq!(
        h::fclusterdata(&excessive_work, 1.0, Method::Single, Criterion::Distance),
        Err(sci::Error::InvalidInput("geometry work limit exceeded")),
    );
}
