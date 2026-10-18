//! Benchmarks covering every public function of `sci::complex`.
//!
//! Each scalar function is evaluated over a fixed batch of deterministic
//! complex points so that a single iteration is long enough to measure.

mod common;

use common::{positive_values, values};
use divan::{Bencher, black_box};
use sci::complex::{self, Complex};

fn main() {
    divan::main();
}

/// Number of scalar evaluations per iteration.
const BATCH: usize = 256;

/// Deterministic complex points with components in `[-1, 1)`.
fn points(seed: u64) -> Vec<Complex> {
    values(2 * BATCH, seed)
        .chunks(2)
        .map(|c| Complex { re: c[0], im: c[1] })
        .collect()
}

fn sum(values: impl Iterator<Item = Complex>) -> Complex {
    values.fold(Complex::default(), complex::add)
}

/// Defines a benchmark applying a unary complex-valued function to the batch.
macro_rules! unary {
    ($($name:ident),* $(,)?) => {
        $(
            #[divan::bench]
            fn $name(bencher: Bencher) {
                let z = points(1);
                bencher.bench(|| sum(black_box(&z).iter().map(|&z| complex::$name(z))));
            }
        )*
    };
}

/// Defines a benchmark applying a binary complex-valued function to the batch.
macro_rules! binary {
    ($($name:ident),* $(,)?) => {
        $(
            #[divan::bench]
            fn $name(bencher: Bencher) {
                let a = points(1);
                let b = points(2);
                bencher.bench(|| {
                    sum(black_box(&a)
                        .iter()
                        .zip(black_box(&b))
                        .map(|(&a, &b)| complex::$name(a, b)))
                });
            }
        )*
    };
}

/// Defines a benchmark applying a real-valued function to the batch.
macro_rules! real_valued {
    ($($name:ident),* $(,)?) => {
        $(
            #[divan::bench]
            fn $name(bencher: Bencher) {
                let z = points(3);
                bencher.bench(|| black_box(&z).iter().map(|&z| complex::$name(z)).sum::<f64>());
            }
        )*
    };
}

mod arithmetic {
    use super::*;

    binary!(add, subtract, multiply, divide);
    unary!(negate, conjugate, reciprocal);
}

mod components {
    use super::*;

    real_valued!(real, imag, abs, norm_sqr, arg);

    #[divan::bench]
    fn to_polar(bencher: Bencher) {
        let z = points(4);
        bencher.bench(|| {
            black_box(&z)
                .iter()
                .map(|&z| {
                    let (r, theta) = complex::to_polar(z);
                    r + theta
                })
                .sum::<f64>()
        });
    }

    #[divan::bench]
    fn from_polar(bencher: Bencher) {
        let radii = positive_values(BATCH, 5);
        let phases = values(BATCH, 6);
        bencher.bench(|| {
            sum(black_box(&radii)
                .iter()
                .zip(black_box(&phases))
                .map(|(&r, &theta)| complex::from_polar(r, 3.0 * theta).unwrap()))
        });
    }
}

mod exponential {
    use super::*;

    unary!(exp, ln, sqrt, cbrt);
}

mod trigonometric {
    use super::*;

    unary!(sin, cos, tan, asin, acos, atan);
}

mod hyperbolic {
    use super::*;

    unary!(sinh, cosh, tanh, asinh, acosh, atanh);
}

mod power {
    use super::*;

    binary!(powc);

    #[divan::bench]
    fn powf(bencher: Bencher) {
        let z = points(7);
        bencher.bench(|| sum(black_box(&z).iter().map(|&z| complex::powf(z, 2.5))));
    }

    #[divan::bench]
    fn powi(bencher: Bencher) {
        let z = points(8);
        bencher.bench(|| sum(black_box(&z).iter().map(|&z| complex::powi(z, -7))));
    }

    #[divan::bench]
    fn powu(bencher: Bencher) {
        let z = points(9);
        bencher.bench(|| sum(black_box(&z).iter().map(|&z| complex::powu(z, 13))));
    }
}
