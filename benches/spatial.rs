//! Benchmarks covering every public function of `sci::spatial`.
//!
//! Inputs are deterministic so that results are reproducible across runs.

mod common;

use common::{linspace, points, positive_values, spd_matrix, values, warm};
use divan::{Bencher, black_box};
use sci::spatial;

fn main() {
    divan::main();
}

/// Length of the vectors used by the single-pair distance benchmarks.
const N: usize = 1000;

/// Deterministic 0/1 vector for the boolean dissimilarities.
fn boolean(n: usize, seed: u64) -> Vec<f64> {
    values(n, seed)
        .into_iter()
        .map(|v| if v > 0.0 { 1.0 } else { 0.0 })
        .collect()
}

mod vector_distance_benches {
    use super::*;

    macro_rules! real_metric {
        ($($name:ident),* $(,)?) => {$(
            #[divan::bench]
            fn $name(bencher: Bencher) {
                let u = values(N, 1);
                let v = values(N, 2);
                bencher.bench(|| spatial::$name(black_box(&u), black_box(&v), None).unwrap());
            }
        )*};
    }

    macro_rules! boolean_metric {
        ($($name:ident),* $(,)?) => {$(
            #[divan::bench]
            fn $name(bencher: Bencher) {
                let u = boolean(N, 1);
                let v = boolean(N, 2);
                bencher.bench(|| spatial::$name(black_box(&u), black_box(&v), None).unwrap());
            }
        )*};
    }

    real_metric!(
        euclidean,
        sqeuclidean,
        cityblock,
        chebyshev,
        canberra,
        braycurtis,
        cosine,
        hamming,
    );

    boolean_metric!(jaccard, dice, rogerstanimoto, russellrao, sokalsneath, yule,);

    #[divan::bench]
    fn minkowski(bencher: Bencher) {
        let u = values(N, 1);
        let v = values(N, 2);
        bencher.bench(|| {
            spatial::minkowski(black_box(&u), black_box(&v), black_box(3.0), None).unwrap()
        });
    }

    #[divan::bench]
    fn correlation(bencher: Bencher) {
        let u = values(N, 1);
        let v = values(N, 2);
        bencher.bench(|| {
            spatial::correlation(black_box(&u), black_box(&v), None, black_box(true)).unwrap()
        });
    }

    #[divan::bench]
    fn seuclidean(bencher: Bencher) {
        let u = values(N, 1);
        let v = values(N, 2);
        let variance = positive_values(N, 3);
        bencher.bench(|| {
            spatial::seuclidean(black_box(&u), black_box(&v), black_box(&variance)).unwrap()
        });
    }

    #[divan::bench]
    fn mahalanobis(bencher: Bencher) {
        let u = values(16, 1);
        let v = values(16, 2);
        let vi = spd_matrix(16);
        warm(bencher, || {
            spatial::mahalanobis(black_box(&u), black_box(&v), black_box(&vi)).unwrap()
        });
    }

    #[divan::bench]
    fn jensenshannon(bencher: Bencher) {
        let p = positive_values(N, 1);
        let q = positive_values(N, 2);
        bencher.bench(|| {
            spatial::jensenshannon(black_box(&p), black_box(&q), black_box(Some(2.0))).unwrap()
        });
    }

    #[divan::bench]
    fn minkowski_distance(bencher: Bencher) {
        let u = values(N, 1);
        let v = values(N, 2);
        bencher.bench(|| {
            spatial::minkowski_distance(black_box(&u), black_box(&v), black_box(3.0)).unwrap()
        });
    }

    #[divan::bench]
    fn minkowski_distance_p(bencher: Bencher) {
        let u = values(N, 1);
        let v = values(N, 2);
        bencher.bench(|| {
            spatial::minkowski_distance_p(black_box(&u), black_box(&v), black_box(3.0)).unwrap()
        });
    }
}

mod pairwise_benches {
    use super::*;

    #[divan::bench]
    fn pdist(bencher: Bencher) {
        let x = points(60, 3, 1);
        bencher.bench(|| spatial::pdist(black_box(&x), spatial::Metric::Euclidean).unwrap());
    }

    #[divan::bench]
    fn cdist(bencher: Bencher) {
        let x = points(40, 3, 1);
        let y = points(40, 3, 2);
        bencher.bench(|| {
            spatial::cdist(black_box(&x), black_box(&y), spatial::Metric::Euclidean).unwrap()
        });
    }

    #[divan::bench]
    fn distance_matrix(bencher: Bencher) {
        let x = points(40, 3, 1);
        let y = points(40, 3, 2);
        bencher.bench(|| {
            spatial::distance_matrix(black_box(&x), black_box(&y), black_box(2.0)).unwrap()
        });
    }

    #[divan::bench]
    fn directed_hausdorff(bencher: Bencher) {
        let u = points(50, 3, 1);
        let v = points(50, 3, 2);
        bencher.bench(|| spatial::directed_hausdorff(black_box(&u), black_box(&v)).unwrap());
    }
}

mod distance_form_benches {
    use super::*;

    fn condensed() -> Vec<f64> {
        spatial::pdist(&points(100, 3, 1), spatial::Metric::Euclidean).unwrap()
    }

    fn square() -> Vec<Vec<f64>> {
        let x = points(100, 3, 1);
        spatial::cdist(&x, &x, spatial::Metric::Euclidean).unwrap()
    }

    #[divan::bench]
    fn squareform_to_square(bencher: Bencher) {
        let y = condensed();
        bencher.bench(|| {
            spatial::squareform(spatial::DistanceForm::Condensed(black_box(&y))).unwrap()
        });
    }

    #[divan::bench]
    fn squareform_to_condensed(bencher: Bencher) {
        let d = square();
        bencher
            .bench(|| spatial::squareform(spatial::DistanceForm::Square(black_box(&d))).unwrap());
    }

    #[divan::bench]
    fn is_valid_dm(bencher: Bencher) {
        let d = square();
        bencher.bench(|| spatial::is_valid_dm(black_box(&d), black_box(1e-12)));
    }

    #[divan::bench]
    fn is_valid_y(bencher: Bencher) {
        let y = condensed();
        bencher.bench(|| spatial::is_valid_y(black_box(&y)));
    }

    #[divan::bench]
    fn num_obs_dm(bencher: Bencher) {
        let d = square();
        bencher.bench(|| spatial::num_obs_dm(black_box(&d)).unwrap());
    }

    #[divan::bench]
    fn num_obs_y(bencher: Bencher) {
        let y = condensed();
        bencher.bench(|| spatial::num_obs_y(black_box(&y)).unwrap());
    }
}

mod geometry_benches {
    use super::*;

    #[divan::bench]
    fn geometric_slerp(bencher: Bencher) {
        let start = [1.0, 0.0, 0.0];
        let s = 0.5_f64.sqrt();
        let end = [0.0, s, s];
        let t = linspace(0.0, 1.0, 200);
        bencher.bench(|| {
            spatial::geometric_slerp(
                black_box(&start),
                black_box(&end),
                black_box(&t),
                black_box(1e-7),
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn procrustes(bencher: Bencher) {
        let a = points(200, 3, 1);
        // A rotated, scaled, shifted, and perturbed copy of `a`.
        let noise = values(600, 2);
        let b: Vec<Vec<f64>> = a
            .iter()
            .enumerate()
            .map(|(i, p)| {
                vec![
                    2.0 * p[1] + 1.0 + 0.01 * noise[3 * i],
                    -2.0 * p[0] - 3.0 + 0.01 * noise[3 * i + 1],
                    2.0 * p[2] + 0.5 + 0.01 * noise[3 * i + 2],
                ]
            })
            .collect();
        warm(bencher, || {
            spatial::procrustes(black_box(&a), black_box(&b)).unwrap()
        });
    }

    #[divan::bench]
    fn tsearch(bencher: Bencher) {
        // Regular triangulation of a 10 x 10 vertex grid on [0, 1]^2.
        let m = 10;
        let grid = linspace(0.0, 1.0, m);
        let vertices: Vec<Vec<f64>> = grid
            .iter()
            .flat_map(|y| grid.iter().map(move |x| vec![*x, *y]))
            .collect();
        let mut simplices = Vec::new();
        for j in 0..m - 1 {
            for i in 0..m - 1 {
                let k = j * m + i;
                simplices.push(vec![k, k + 1, k + m]);
                simplices.push(vec![k + 1, k + m + 1, k + m]);
            }
        }
        let queries: Vec<Vec<f64>> = points(50, 2, 3)
            .into_iter()
            .map(|p| p.iter().map(|v| 0.5 * (v + 1.0)).collect())
            .collect();
        warm(bencher, || {
            spatial::tsearch(
                black_box(&vertices),
                black_box(&simplices),
                black_box(&queries),
                black_box(1e-12),
            )
            .unwrap()
        });
    }
}
