//! Benchmarks covering every public function of `sci::integrate`.
//!
//! Inputs are deterministic so that results are reproducible across runs.

mod common;

use common::{linspace, sine_signal};
use divan::{Bencher, black_box};

fn main() {
    divan::main();
}

/// Number of samples for the sampled rules.
const N: usize = 4096;
/// Sample spacing matching `sine_signal`.
const DX: f64 = 0.001;

/// Strictly increasing, unevenly spaced coordinates on `[0, 4]`.
fn uneven_coordinates(n: usize) -> Vec<f64> {
    linspace(0.0, 2.0, n).into_iter().map(|t| t * t).collect()
}

mod sampled {
    use super::*;

    #[divan::bench]
    fn trapezoid(bencher: Bencher) {
        let y = sine_signal(N);
        let x = uneven_coordinates(N);
        bencher.bench(|| {
            sci::integrate::sampled::trapezoid(black_box(&y), Some(black_box(&x)), DX).unwrap()
        });
    }

    #[divan::bench]
    fn cumulative_trapezoid(bencher: Bencher) {
        let y = sine_signal(N);
        bencher.bench(|| {
            sci::integrate::sampled::cumulative_trapezoid(black_box(&y), None, DX, Some(0.0))
                .unwrap()
        });
    }

    #[divan::bench]
    fn simpson(bencher: Bencher) {
        let y = sine_signal(N);
        let x = uneven_coordinates(N);
        bencher.bench(|| {
            sci::integrate::sampled::simpson(black_box(&y), Some(black_box(&x)), DX).unwrap()
        });
    }

    #[divan::bench]
    fn cumulative_simpson(bencher: Bencher) {
        let y = sine_signal(N);
        bencher.bench(|| {
            sci::integrate::sampled::cumulative_simpson(black_box(&y), None, DX, None).unwrap()
        });
    }

    #[divan::bench]
    fn romb(bencher: Bencher) {
        let y = sine_signal(N + 1);
        bencher.bench(|| sci::integrate::sampled::romb(black_box(&y), DX).unwrap());
    }
}

mod quadrature {
    use super::*;

    /// 16-point Gauss–Legendre rule (includes computing the nodes and weights).
    #[divan::bench]
    fn fixed_quad(bencher: Bencher) {
        bencher.bench(|| {
            sci::integrate::quadrature::fixed_quad(
                |x: f64| (-x * x).exp() * x.cos(),
                black_box(-3.0),
                black_box(3.0),
                16,
            )
            .unwrap()
        });
    }

    /// Weights of the highest supported uniform rule (13 nodes).
    #[divan::bench]
    fn newton_cotes(bencher: Bencher) {
        let nodes: Vec<f64> = (0..=12).map(f64::from).collect();
        bencher.bench(|| sci::integrate::quadrature::newton_cotes(black_box(&nodes)).unwrap());
    }
}
