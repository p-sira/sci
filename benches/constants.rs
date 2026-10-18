//! Benchmarks covering every public function of `sci::constants`.
//!
//! Lookups and conversions are evaluated over fixed batches of deterministic
//! inputs so that a single iteration is long enough to measure.

mod common;

use common::positive_values;
use divan::{Bencher, black_box};
use sci::constants::{PHYSICAL_CONSTANTS, TemperatureScale};

fn main() {
    divan::main();
}

/// Number of scalar evaluations per iteration.
const BATCH: usize = 128;

/// Evenly spread selection of valid lookup keys from the CODATA table.
fn keys() -> Vec<&'static str> {
    let stride = PHYSICAL_CONSTANTS.len() / BATCH;
    PHYSICAL_CONSTANTS
        .iter()
        .step_by(stride.max(1))
        .take(BATCH)
        .map(|entry| entry.name)
        .collect()
}

mod lookup {
    use super::*;

    #[divan::bench]
    fn value(bencher: Bencher) {
        let keys = keys();
        bencher.bench(|| {
            black_box(&keys)
                .iter()
                .map(|key| sci::constants::value(key).unwrap())
                .sum::<f64>()
        });
    }

    #[divan::bench]
    fn unit(bencher: Bencher) {
        let keys = keys();
        bencher.bench(|| {
            black_box(&keys)
                .iter()
                .map(|key| sci::constants::unit(key).unwrap().len())
                .sum::<usize>()
        });
    }

    #[divan::bench]
    fn precision(bencher: Bencher) {
        let keys = keys();
        bencher.bench(|| {
            black_box(&keys)
                .iter()
                .map(|key| sci::constants::precision(key).unwrap())
                .sum::<f64>()
        });
    }

    /// Case-insensitive substring search over all current names.
    #[divan::bench]
    fn find(bencher: Bencher) {
        bencher.bench(|| sci::constants::find(black_box(Some("electron"))));
    }
}

mod conversions {
    use super::*;

    /// Converts a batch of temperatures through every pair of distinct scales.
    #[divan::bench]
    fn convert_temperature(bencher: Bencher) {
        use TemperatureScale::*;
        const SCALES: [TemperatureScale; 4] = [Celsius, Kelvin, Fahrenheit, Rankine];
        let temperatures = positive_values(BATCH, 1);
        bencher.bench(|| {
            let mut total = 0.0;
            for &t in black_box(&temperatures) {
                for from in SCALES {
                    for to in SCALES {
                        total += sci::constants::convert_temperature(100.0 * t, from, to);
                    }
                }
            }
            total
        });
    }

    #[divan::bench]
    fn lambda2nu(bencher: Bencher) {
        let wavelengths: Vec<f64> = positive_values(BATCH, 2).iter().map(|v| v * 1e-7).collect();
        bencher.bench(|| {
            black_box(&wavelengths)
                .iter()
                .map(|&w| sci::constants::lambda2nu(w))
                .sum::<f64>()
        });
    }

    #[divan::bench]
    fn nu2lambda(bencher: Bencher) {
        let frequencies: Vec<f64> = positive_values(BATCH, 3).iter().map(|v| v * 1e14).collect();
        bencher.bench(|| {
            black_box(&frequencies)
                .iter()
                .map(|&f| sci::constants::nu2lambda(f))
                .sum::<f64>()
        });
    }
}
