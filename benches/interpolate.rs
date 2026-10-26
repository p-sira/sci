//! Benchmarks covering every public function of `sci::interpolate`.
//!
//! Inputs are deterministic so that results are reproducible across runs.

mod common;

use common::{linspace, points, values, warm};
use divan::{Bencher, black_box};
use sci::interpolate;

fn main() {
    divan::main();
}

/// Smooth samples `sin(x) + 0.1 cos(3x)` on an evenly spaced grid.
fn smooth(x: &[f64]) -> Vec<f64> {
    x.iter().map(|v| v.sin() + 0.1 * (3.0 * v).cos()).collect()
}

/// Cubic interpolating spline through `n` smooth samples on `[0, 10]`.
fn cubic_spline(n: usize) -> sci::Spline {
    let x = linspace(0.0, 10.0, n);
    let y = smooth(&x);
    interpolate::make_interp_spline(&x, &y, 3).unwrap()
}

mod polynomial_benches {
    use super::*;

    #[divan::bench]
    fn barycentric_interpolate(bencher: Bencher) {
        let xi = linspace(-1.0, 1.0, 32);
        let yi = smooth(&xi);
        let x = linspace(-0.99, 0.99, 200);
        bencher.bench(|| {
            interpolate::barycentric_interpolate(black_box(&xi), black_box(&yi), black_box(&x))
                .unwrap()
        });
    }

    #[divan::bench]
    fn krogh_interpolate(bencher: Bencher) {
        let xi = linspace(-1.0, 1.0, 16);
        let yi = smooth(&xi);
        let x = linspace(-0.99, 0.99, 200);
        bencher.bench(|| {
            interpolate::krogh_interpolate(black_box(&xi), black_box(&yi), black_box(&x)).unwrap()
        });
    }

    #[divan::bench]
    fn lagrange(bencher: Bencher) {
        let x = linspace(-1.0, 1.0, 16);
        let y = smooth(&x);
        bencher.bench(|| interpolate::lagrange(black_box(&x), black_box(&y)).unwrap());
    }

    #[divan::bench]
    fn pchip_interpolate(bencher: Bencher) {
        let xi = linspace(0.0, 10.0, 200);
        let yi = smooth(&xi);
        let x = linspace(0.01, 9.99, 500);
        bencher.bench(|| {
            interpolate::pchip_interpolate(black_box(&xi), black_box(&yi), black_box(&x)).unwrap()
        });
    }

    #[divan::bench]
    fn pade(bencher: Bencher) {
        // Taylor coefficients of exp(x).
        let mut an = vec![1.0; 12];
        for k in 1..an.len() {
            an[k] = an[k - 1] / k as f64;
        }
        warm(bencher, || {
            interpolate::pade(black_box(&an), black_box(5), black_box(None)).unwrap()
        });
    }

    #[divan::bench]
    fn approximate_taylor_polynomial(bencher: Bencher) {
        bencher.bench(|| {
            interpolate::approximate_taylor_polynomial(
                |x| x.sin() * x.exp(),
                black_box(0.5),
                black_box(6),
                black_box(0.5),
                black_box(Some(10)),
            )
            .unwrap()
        });
    }
}

mod spline_eval_benches {
    use super::*;

    #[divan::bench]
    fn splev(bencher: Bencher) {
        let spline = cubic_spline(64);
        let x = linspace(0.0, 10.0, 500);
        bencher.bench(|| {
            interpolate::splev(
                black_box(&x),
                black_box(&spline),
                0,
                interpolate::SplineExtension::Extrapolate,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn splder(bencher: Bencher) {
        let spline = cubic_spline(64);
        bencher.bench(|| interpolate::splder(black_box(&spline), black_box(1)).unwrap());
    }

    #[divan::bench]
    fn splantider(bencher: Bencher) {
        let spline = cubic_spline(64);
        bencher.bench(|| interpolate::splantider(black_box(&spline), black_box(1)).unwrap());
    }

    #[divan::bench]
    fn spalde(bencher: Bencher) {
        let spline = cubic_spline(64);
        let x = linspace(0.05, 9.95, 100);
        bencher.bench(|| interpolate::spalde(black_box(&x), black_box(&spline)).unwrap());
    }

    #[divan::bench]
    fn splint(bencher: Bencher) {
        let spline = cubic_spline(64);
        bencher.bench(|| {
            interpolate::splint(black_box(1.0), black_box(9.0), black_box(&spline)).unwrap()
        });
    }

    #[divan::bench]
    fn insert(bencher: Bencher) {
        let spline = cubic_spline(64);
        bencher.bench(|| {
            interpolate::insert(black_box(5.05), black_box(&spline), black_box(2)).unwrap()
        });
    }

    #[divan::bench]
    fn sproot(bencher: Bencher) {
        let spline = cubic_spline(32);
        warm(bencher, || interpolate::sproot(black_box(&spline)).unwrap());
    }
}

mod spline_fit_benches {
    use super::*;

    #[divan::bench]
    fn make_interp_spline(bencher: Bencher) {
        let x = linspace(0.0, 10.0, 32);
        let y = smooth(&x);
        warm(bencher, || {
            interpolate::make_interp_spline(black_box(&x), black_box(&y), 3).unwrap()
        });
    }

    #[divan::bench]
    fn make_lsq_spline(bencher: Bencher) {
        let x = linspace(0.0, 10.0, 128);
        let noise = values(128, 3);
        let y: Vec<f64> = smooth(&x)
            .iter()
            .zip(&noise)
            .map(|(v, e)| v + 0.05 * e)
            .collect();
        let mut knots = vec![0.0; 3];
        knots.extend(linspace(0.0, 10.0, 12));
        knots.extend([10.0; 3]);
        warm(bencher, || {
            interpolate::make_lsq_spline(black_box(&x), black_box(&y), black_box(&knots), 3, None)
                .unwrap()
        });
    }

    #[divan::bench]
    fn make_splrep(bencher: Bencher) {
        let x = linspace(0.0, 10.0, 64);
        let noise = values(64, 5);
        let y: Vec<f64> = smooth(&x)
            .iter()
            .zip(&noise)
            .map(|(v, e)| v + 0.05 * e)
            .collect();
        bencher.bench(|| {
            interpolate::make_splrep(black_box(&x), black_box(&y), 3, black_box(0.2), None).unwrap()
        });
    }

    #[divan::bench]
    fn make_splprep(bencher: Bencher) {
        let t = linspace(0.0, 6.0, 64);
        let noise = values(128, 9);
        let pts: Vec<Vec<f64>> = t
            .iter()
            .enumerate()
            .map(|(i, v)| {
                vec![
                    v.cos() * (1.0 + 0.1 * v) + 0.01 * noise[2 * i],
                    v.sin() * (1.0 + 0.1 * v) + 0.01 * noise[2 * i + 1],
                ]
            })
            .collect();
        bencher.bench(|| {
            interpolate::make_splprep(black_box(&pts), None, 3, black_box(0.05), None).unwrap()
        });
    }

    #[divan::bench]
    fn generate_knots(bencher: Bencher) {
        let x = linspace(0.0, 10.0, 48);
        let noise = values(48, 13);
        let y: Vec<f64> = smooth(&x)
            .iter()
            .zip(&noise)
            .map(|(v, e)| v + 0.05 * e)
            .collect();
        warm(bencher, || {
            interpolate::generate_knots(black_box(&x), black_box(&y), 3, black_box(0.5), None)
                .unwrap()
        });
    }

    #[divan::bench]
    fn make_smoothing_spline(bencher: Bencher) {
        let x = linspace(0.0, 10.0, 32);
        let noise = values(32, 17);
        let y: Vec<f64> = smooth(&x)
            .iter()
            .zip(&noise)
            .map(|(v, e)| v + 0.05 * e)
            .collect();
        warm(bencher, || {
            interpolate::make_smoothing_spline(black_box(&x), black_box(&y), None, black_box(0.1))
                .unwrap()
        });
    }
}

mod surface_benches {
    use super::*;

    /// Scattered samples of a smooth surface on `[0, 1]^2` and cubic knots.
    fn surface_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let pts = points(128, 2, 21);
        let x: Vec<f64> = pts.iter().map(|p| 0.5 * (p[0] + 1.0)).collect();
        let y: Vec<f64> = pts.iter().map(|p| 0.5 * (p[1] + 1.0)).collect();
        let z: Vec<f64> = x
            .iter()
            .zip(&y)
            .map(|(a, b)| (3.0 * a).sin() * (2.0 * b).cos())
            .collect();
        let knots = vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        (x, y, z, knots)
    }

    #[divan::bench]
    fn bisplrep(bencher: Bencher) {
        let (x, y, z, knots) = surface_data();
        warm(bencher, || {
            interpolate::bisplrep(
                black_box(&x),
                black_box(&y),
                black_box(&z),
                black_box(&knots),
                black_box(&knots),
                3,
                3,
                None,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn bisplev(bencher: Bencher) {
        let (x, y, z, knots) = surface_data();
        let spline = interpolate::bisplrep(&x, &y, &z, &knots, &knots, 3, 3, None).unwrap();
        let qx = linspace(0.0, 1.0, 20);
        let qy = linspace(0.0, 1.0, 20);
        bencher.bench(|| {
            interpolate::bisplev(black_box(&qx), black_box(&qy), black_box(&spline), 0, 0).unwrap()
        });
    }
}

mod multidimensional_benches {
    use super::*;

    #[divan::bench]
    fn interpn(bencher: Bencher) {
        let axes = vec![linspace(0.0, 1.0, 20), linspace(0.0, 1.0, 20)];
        let grid: Vec<f64> = axes[0]
            .iter()
            .flat_map(|a| axes[1].iter().map(move |b| (3.0 * a).sin() + b * b))
            .collect();
        let queries: Vec<Vec<f64>> = points(200, 2, 23)
            .into_iter()
            .map(|p| p.iter().map(|v| 0.5 * (v + 1.0)).collect())
            .collect();
        bencher.bench(|| {
            interpolate::interpn(
                black_box(&axes),
                black_box(&grid),
                black_box(&queries),
                interpolate::GridMethod::Linear,
                true,
                None,
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn griddata(bencher: Bencher) {
        let pts = points(100, 2, 29);
        let vals: Vec<f64> = pts.iter().map(|p| p[0] * p[0] - p[1]).collect();
        let queries: Vec<Vec<f64>> = points(100, 2, 31)
            .into_iter()
            .map(|p| p.iter().map(|v| 0.8 * v).collect())
            .collect();
        bencher.bench(|| {
            interpolate::griddata(
                black_box(&pts),
                black_box(&vals),
                black_box(&queries),
                interpolate::ScatteredMethod::Linear,
                f64::NAN,
                false,
            )
            .unwrap()
        });
    }
}
