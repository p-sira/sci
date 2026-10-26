//! Benchmarks covering every public function of `sci::cluster`.
//!
//! Inputs are deterministic so that results are reproducible across runs.

mod common;

use common::points;
use divan::{Bencher, black_box};
use sci::cluster::{hierarchy, vq};
use sci::spatial;

fn main() {
    divan::main();
}

/// Number of observations used by the hierarchy benchmarks.
const LEAVES: usize = 100;

/// Three loose blobs of 3-D observations.
fn blobs(n: usize) -> Vec<Vec<f64>> {
    let centers = [[0.0, 0.0, 0.0], [4.0, 4.0, 0.0], [0.0, 4.0, 4.0]];
    points(n, 3, 41)
        .into_iter()
        .enumerate()
        .map(|(i, p)| {
            let c = centers[i % centers.len()];
            p.iter().zip(c).map(|(v, c)| v + c).collect()
        })
        .collect()
}

/// Condensed Euclidean distances between the observations.
fn condensed() -> Vec<f64> {
    spatial::pdist(&blobs(LEAVES), spatial::Metric::Euclidean).unwrap()
}

/// Average-linkage tree of the observations.
fn tree() -> Vec<Vec<f64>> {
    hierarchy::linkage(&condensed(), hierarchy::Method::Average).unwrap()
}

mod vq_benches {
    use super::*;

    fn codes() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 1.0, 1.0],
            vec![3.0, 3.0, 1.0],
            vec![1.0, 3.0, 3.0],
            vec![2.0, 2.0, 2.0],
        ]
    }

    #[divan::bench]
    fn vq(bencher: Bencher) {
        let obs = blobs(300);
        let book = codes();
        bencher.bench(|| vq::vq(black_box(&obs), black_box(&book)).unwrap());
    }

    #[divan::bench]
    fn whiten(bencher: Bencher) {
        let obs = blobs(300);
        bencher.bench(|| vq::whiten(black_box(&obs)).unwrap());
    }

    #[divan::bench]
    fn kmeans2(bencher: Bencher) {
        let obs = blobs(300);
        let initial = codes();
        bencher.bench(|| vq::kmeans2(black_box(&obs), black_box(&initial), 5).unwrap());
    }

    #[divan::bench]
    fn kmeans(bencher: Bencher) {
        let obs = blobs(300);
        let initial = codes();
        bencher.bench(|| {
            vq::kmeans(black_box(&obs), black_box(&initial), black_box(1e-8), 50).unwrap()
        });
    }
}

mod linkage_benches {
    use super::*;

    #[divan::bench]
    fn linkage(bencher: Bencher) {
        let y = condensed();
        bencher.bench(|| hierarchy::linkage(black_box(&y), hierarchy::Method::Average).unwrap());
    }

    macro_rules! method {
        ($($name:ident),* $(,)?) => {$(
            #[divan::bench]
            fn $name(bencher: Bencher) {
                let y = condensed();
                bencher.bench(|| hierarchy::$name(black_box(&y)).unwrap());
            }
        )*};
    }

    method!(single, complete, average, weighted, centroid, median, ward);

    #[divan::bench]
    fn fclusterdata(bencher: Bencher) {
        let x = blobs(50);
        bencher.bench(|| {
            hierarchy::fclusterdata(
                black_box(&x),
                black_box(3.0),
                hierarchy::Method::Average,
                hierarchy::Criterion::MaxClust,
            )
            .unwrap()
        });
    }
}

mod tree_benches {
    use super::*;

    #[divan::bench]
    fn num_obs_linkage(bencher: Bencher) {
        let z = tree();
        bencher.bench(|| hierarchy::num_obs_linkage(black_box(&z)).unwrap());
    }

    #[divan::bench]
    fn is_valid_linkage(bencher: Bencher) {
        let z = tree();
        bencher.bench(|| hierarchy::is_valid_linkage(black_box(&z)));
    }

    #[divan::bench]
    fn is_monotonic(bencher: Bencher) {
        let z = tree();
        bencher.bench(|| hierarchy::is_monotonic(black_box(&z)).unwrap());
    }

    #[divan::bench]
    fn leaves_list(bencher: Bencher) {
        let z = tree();
        bencher.bench(|| hierarchy::leaves_list(black_box(&z)).unwrap());
    }

    #[divan::bench]
    fn maxdists(bencher: Bencher) {
        let z = tree();
        bencher.bench(|| hierarchy::maxdists(black_box(&z)).unwrap());
    }

    #[divan::bench]
    fn to_mlab_linkage(bencher: Bencher) {
        let z = tree();
        bencher.bench(|| hierarchy::to_mlab_linkage(black_box(&z)).unwrap());
    }

    #[divan::bench]
    fn from_mlab_linkage(bencher: Bencher) {
        let m = hierarchy::to_mlab_linkage(&tree()).unwrap();
        bencher.bench(|| hierarchy::from_mlab_linkage(black_box(&m)).unwrap());
    }

    #[divan::bench]
    fn correspond(bencher: Bencher) {
        let y = condensed();
        let z = hierarchy::linkage(&y, hierarchy::Method::Average).unwrap();
        bencher.bench(|| hierarchy::correspond(black_box(&z), black_box(&y)).unwrap());
    }

    #[divan::bench]
    fn cophenet(bencher: Bencher) {
        let y = condensed();
        let z = hierarchy::linkage(&y, hierarchy::Method::Average).unwrap();
        bencher.bench(|| hierarchy::cophenet(black_box(&z), Some(black_box(&y))).unwrap());
    }

    #[divan::bench]
    fn to_tree(bencher: Bencher) {
        let z = tree();
        bencher.bench(|| hierarchy::to_tree(black_box(&z)).unwrap());
    }
}

mod inconsistency_benches {
    use super::*;

    #[divan::bench]
    fn inconsistent(bencher: Bencher) {
        let z = tree();
        bencher.bench(|| hierarchy::inconsistent(black_box(&z), black_box(2)).unwrap());
    }

    #[divan::bench]
    fn is_valid_im(bencher: Bencher) {
        let r = hierarchy::inconsistent(&tree(), 2).unwrap();
        bencher.bench(|| hierarchy::is_valid_im(black_box(&r)));
    }

    #[divan::bench]
    fn max_rstat(bencher: Bencher) {
        let z = tree();
        let r = hierarchy::inconsistent(&z, 2).unwrap();
        bencher.bench(|| hierarchy::max_rstat(black_box(&z), black_box(&r), 3).unwrap());
    }

    #[divan::bench]
    fn maxinconsts(bencher: Bencher) {
        let z = tree();
        let r = hierarchy::inconsistent(&z, 2).unwrap();
        bencher.bench(|| hierarchy::maxinconsts(black_box(&z), black_box(&r)).unwrap());
    }
}

mod flat_benches {
    use super::*;

    #[divan::bench]
    fn fcluster(bencher: Bencher) {
        let z = tree();
        bencher.bench(|| {
            hierarchy::fcluster(
                black_box(&z),
                black_box(3.0),
                hierarchy::Criterion::MaxClust,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn leaders(bencher: Bencher) {
        let z = tree();
        let labels = hierarchy::fcluster(&z, 3.0, hierarchy::Criterion::MaxClust).unwrap();
        bencher.bench(|| hierarchy::leaders(black_box(&z), black_box(&labels)).unwrap());
    }

    #[divan::bench]
    fn is_isomorphic(bencher: Bencher) {
        let y = condensed();
        let a = hierarchy::fcluster(
            &hierarchy::linkage(&y, hierarchy::Method::Average).unwrap(),
            3.0,
            hierarchy::Criterion::MaxClust,
        )
        .unwrap();
        let b = hierarchy::fcluster(
            &hierarchy::linkage(&y, hierarchy::Method::Complete).unwrap(),
            3.0,
            hierarchy::Criterion::MaxClust,
        )
        .unwrap();
        bencher.bench(|| hierarchy::is_isomorphic(black_box(&a), black_box(&b)).unwrap());
    }

    #[divan::bench]
    fn cut_tree(bencher: Bencher) {
        let z = tree();
        let counts = [2, 5, 10, 20];
        bencher.bench(|| {
            hierarchy::cut_tree(black_box(&z), hierarchy::Cut::Counts(black_box(&counts))).unwrap()
        });
    }
}
