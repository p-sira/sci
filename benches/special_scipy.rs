//! Benchmarks covering the SciPy-compatible scalar functions of `sci::special`.
//!
//! Every scalar function is evaluated over a small deterministic vector of
//! inputs drawn from its valid domain, and the results are summed.

mod common;

use common::{unit_values, values};
use divan::{Bencher, black_box};
use sci::special;

fn main() {
    divan::main();
}

/// Number of evaluation points for cheap scalar functions.
const N: usize = 256;
/// Number of evaluation points for moderately expensive scalar functions.
const HALF: usize = 128;
/// Number of evaluation points for expensive scalar functions.
const SMALL: usize = 64;
/// Number of evaluation points for more expensive scalar functions.
const HALF_SMALL: usize = 32;
/// Number of evaluation points for very expensive scalar functions.
const TINY: usize = 16;

/// Deterministic values uniformly spread over `(lo, hi)`.
fn range(n: usize, seed: u64, lo: f64, hi: f64) -> Vec<f64> {
    unit_values(n, seed)
        .into_iter()
        .map(|u| lo + (hi - lo) * u)
        .collect()
}

/// Deterministic integers in `[lo, hi]`, stored as `f64`.
fn ints(n: usize, seed: u64, lo: i32, hi: i32) -> Vec<f64> {
    unit_values(n, seed)
        .into_iter()
        .map(|u| (lo as f64 + ((hi - lo + 1) as f64 * u).floor()).min(hi as f64))
        .collect()
}

/// Sums `f(x)` over `xs`.
fn sum1(bencher: Bencher, xs: Vec<f64>, f: impl Fn(f64) -> f64 + Sync) {
    bencher.bench(|| black_box(&xs).iter().map(|&x| f(x)).sum::<f64>());
}

/// Sums `f(x, y)` over paired inputs.
fn sum2(bencher: Bencher, xs: Vec<f64>, ys: Vec<f64>, f: impl Fn(f64, f64) -> f64 + Sync) {
    bencher.bench(|| {
        black_box(&xs)
            .iter()
            .zip(black_box(&ys))
            .map(|(&x, &y)| f(x, y))
            .sum::<f64>()
    });
}

/// Sums `f(x, y, z)` over triplets of inputs.
fn sum3(
    bencher: Bencher,
    xs: Vec<f64>,
    ys: Vec<f64>,
    zs: Vec<f64>,
    f: impl Fn(f64, f64, f64) -> f64 + Sync,
) {
    bencher.bench(|| {
        black_box(&xs)
            .iter()
            .zip(black_box(&ys))
            .zip(black_box(&zs))
            .map(|((&x, &y), &z)| f(x, y, z))
            .sum::<f64>()
    });
}

/// Sums `f(a, b, c, x)` over quadruplets of inputs.
fn sum4(bencher: Bencher, inputs: [Vec<f64>; 4], f: impl Fn(f64, f64, f64, f64) -> f64 + Sync) {
    bencher.bench(|| {
        let [a, b, c, x] = black_box(&inputs);
        (0..x.len()).map(|i| f(a[i], b[i], c[i], x[i])).sum::<f64>()
    });
}

fn pair((a, b): (f64, f64)) -> f64 {
    a + b
}

fn quad(v: [f64; 4]) -> f64 {
    v.iter().sum()
}

mod airy_elliptic_benches {
    use super::*;

    #[divan::bench]
    fn agm(bencher: Bencher) {
        sum2(
            bencher,
            range(N, 1, 0.1, 10.0),
            range(N, 2, 0.1, 10.0),
            special::agm,
        );
    }

    #[divan::bench]
    fn airy(bencher: Bencher) {
        sum1(bencher, range(N, 3, -10.0, 10.0), |x| {
            quad(special::airy(x))
        });
    }

    #[divan::bench]
    fn airye(bencher: Bencher) {
        sum1(bencher, range(SMALL, 4, 0.0, 20.0), |x| {
            quad(special::airye(x))
        });
    }

    #[divan::bench]
    fn itairy(bencher: Bencher) {
        sum1(bencher, range(N, 5, -10.0, 10.0), |x| {
            quad(special::itairy(x))
        });
    }

    #[divan::bench]
    fn ellipj(bencher: Bencher) {
        sum2(
            bencher,
            range(N, 6, -5.0, 5.0),
            unit_values(N, 7),
            |u, m| {
                let (sn, cn, dn, ph) = special::ellipj(u, m);
                sn + cn + dn + ph
            },
        );
    }
}

mod gamma_error_benches {
    use super::*;

    #[divan::bench]
    fn binom(bencher: Bencher) {
        let n = range(N, 8, 0.0, 30.0);
        let k = n
            .iter()
            .zip(unit_values(N, 9))
            .map(|(n, u)| n * u)
            .collect();
        sum2(bencher, n, k, special::binom);
    }

    #[divan::bench]
    fn erfcx(bencher: Bencher) {
        sum1(bencher, range(N, 10, -5.0, 20.0), special::erfcx);
    }

    #[divan::bench]
    fn erfi(bencher: Bencher) {
        sum1(bencher, range(N, 11, -5.0, 5.0), special::erfi);
    }

    #[divan::bench]
    fn erfcinv(bencher: Bencher) {
        sum1(bencher, range(N, 12, 0.0, 2.0), special::erfcinv);
    }

    #[divan::bench]
    fn dawsn(bencher: Bencher) {
        sum1(bencher, range(N, 13, -10.0, 10.0), special::dawsn);
    }

    #[divan::bench]
    fn gammasgn(bencher: Bencher) {
        sum1(bencher, range(N, 14, -10.0, 10.0), special::gammasgn);
    }

    #[divan::bench]
    fn gammaincc(bencher: Bencher) {
        sum2(
            bencher,
            range(N, 15, 0.1, 20.0),
            range(N, 16, 0.0, 20.0),
            special::gammaincc,
        );
    }

    #[divan::bench]
    fn gammaincinv(bencher: Bencher) {
        sum2(
            bencher,
            range(N, 17, 0.1, 20.0),
            unit_values(N, 18),
            special::gammaincinv,
        );
    }

    #[divan::bench]
    fn gammainccinv(bencher: Bencher) {
        sum2(
            bencher,
            range(N, 19, 0.1, 20.0),
            unit_values(N, 20),
            special::gammainccinv,
        );
    }

    #[divan::bench]
    fn loggamma(bencher: Bencher) {
        sum1(bencher, range(N, 21, 0.01, 50.0), special::loggamma);
    }

    #[divan::bench]
    fn rgamma(bencher: Bencher) {
        sum1(bencher, range(N, 22, -10.0, 10.0), special::rgamma);
    }

    #[divan::bench]
    fn polygamma(bencher: Bencher) {
        sum2(
            bencher,
            ints(N, 24, 0, 4),
            range(N, 25, 0.1, 20.0),
            |n, x| special::polygamma(n as u32, x),
        );
    }

    #[divan::bench]
    fn multigammaln(bencher: Bencher) {
        sum1(bencher, range(N, 26, 1.5, 30.0), |a| {
            special::multigammaln(a, 3)
        });
    }
}

mod bessel_benches {
    use super::*;

    fn orders() -> Vec<f64> {
        range(N, 30, 0.0, 5.0)
    }

    fn args() -> Vec<f64> {
        range(N, 31, 0.1, 20.0)
    }

    fn half(mut v: Vec<f64>) -> Vec<f64> {
        v.truncate(HALF);
        v
    }

    #[divan::bench]
    fn jv(bencher: Bencher) {
        sum2(bencher, orders(), args(), special::jv);
    }

    #[divan::bench]
    fn jve(bencher: Bencher) {
        sum2(bencher, orders(), args(), special::jve);
    }

    #[divan::bench]
    fn yv(bencher: Bencher) {
        sum2(bencher, half(orders()), half(args()), special::yv);
    }

    #[divan::bench]
    fn yve(bencher: Bencher) {
        sum2(bencher, half(orders()), half(args()), special::yve);
    }

    #[divan::bench]
    fn iv(bencher: Bencher) {
        sum2(bencher, orders(), args(), special::iv);
    }

    #[divan::bench]
    fn ive(bencher: Bencher) {
        sum2(bencher, orders(), args(), special::ive);
    }

    #[divan::bench]
    fn kv(bencher: Bencher) {
        sum2(bencher, orders(), args(), special::kv);
    }

    #[divan::bench]
    fn kve(bencher: Bencher) {
        sum2(bencher, orders(), args(), special::kve);
    }

    #[divan::bench]
    fn jn(bencher: Bencher) {
        sum2(bencher, ints(N, 32, 0, 5), args(), |n, x| {
            special::jn(n as i32, x)
        });
    }

    #[divan::bench]
    fn yn(bencher: Bencher) {
        sum2(bencher, ints(HALF, 33, 0, 5), half(args()), |n, x| {
            special::yn(n as i32, x)
        });
    }

    #[divan::bench]
    fn kn(bencher: Bencher) {
        sum2(bencher, ints(N, 34, 0, 5), args(), |n, x| {
            special::kn(n as i32, x)
        });
    }

    #[divan::bench]
    fn j0(bencher: Bencher) {
        sum1(bencher, args(), special::j0);
    }

    #[divan::bench]
    fn j1(bencher: Bencher) {
        sum1(bencher, args(), special::j1);
    }

    #[divan::bench]
    fn y0(bencher: Bencher) {
        sum1(bencher, args(), special::y0);
    }

    #[divan::bench]
    fn y1(bencher: Bencher) {
        sum1(bencher, args(), special::y1);
    }

    #[divan::bench]
    fn i0(bencher: Bencher) {
        sum1(bencher, args(), special::i0);
    }

    #[divan::bench]
    fn i1(bencher: Bencher) {
        sum1(bencher, args(), special::i1);
    }

    #[divan::bench]
    fn i0e(bencher: Bencher) {
        sum1(bencher, args(), special::i0e);
    }

    #[divan::bench]
    fn i1e(bencher: Bencher) {
        sum1(bencher, args(), special::i1e);
    }

    #[divan::bench]
    fn k0(bencher: Bencher) {
        sum1(bencher, args(), special::k0);
    }

    #[divan::bench]
    fn k1(bencher: Bencher) {
        sum1(bencher, args(), special::k1);
    }

    #[divan::bench]
    fn k0e(bencher: Bencher) {
        sum1(bencher, args(), special::k0e);
    }

    #[divan::bench]
    fn k1e(bencher: Bencher) {
        sum1(bencher, args(), special::k1e);
    }

    #[divan::bench]
    fn besselpoly(bencher: Bencher) {
        sum3(
            bencher,
            range(SMALL, 35, 0.0, 2.0),
            range(SMALL, 36, 0.0, 2.0),
            range(SMALL, 37, 0.0, 3.0),
            special::besselpoly,
        );
    }

    #[divan::bench]
    fn itj0y0(bencher: Bencher) {
        sum1(bencher, args(), |x| pair(special::itj0y0(x)));
    }

    #[divan::bench]
    fn it2j0y0(bencher: Bencher) {
        sum1(bencher, args(), |x| pair(special::it2j0y0(x)));
    }

    #[divan::bench]
    fn iti0k0(bencher: Bencher) {
        sum1(bencher, args(), |x| pair(special::iti0k0(x)));
    }

    #[divan::bench]
    fn it2i0k0(bencher: Bencher) {
        sum1(bencher, args(), |x| pair(special::it2i0k0(x)));
    }
}

mod struve_benches {
    use super::*;

    #[divan::bench]
    fn struve(bencher: Bencher) {
        sum2(
            bencher,
            range(HALF_SMALL, 40, 0.0, 3.0),
            range(HALF_SMALL, 41, 0.1, 20.0),
            special::struve,
        );
    }

    #[divan::bench]
    fn modstruve(bencher: Bencher) {
        sum2(
            bencher,
            range(SMALL, 42, 0.0, 3.0),
            range(SMALL, 43, 0.1, 20.0),
            special::modstruve,
        );
    }

    #[divan::bench]
    fn itstruve0(bencher: Bencher) {
        sum1(bencher, range(N, 44, 0.1, 20.0), special::itstruve0);
    }

    #[divan::bench]
    fn it2struve0(bencher: Bencher) {
        sum1(bencher, range(N, 45, 0.1, 20.0), special::it2struve0);
    }

    #[divan::bench]
    fn itmodstruve0(bencher: Bencher) {
        sum1(bencher, range(N, 46, 0.1, 20.0), special::itmodstruve0);
    }
}

mod hypergeometric_benches {
    use super::*;

    #[divan::bench]
    fn hyp0f1(bencher: Bencher) {
        sum2(
            bencher,
            range(N, 50, 0.5, 5.0),
            range(N, 51, -10.0, 10.0),
            special::hyp0f1,
        );
    }

    #[divan::bench]
    fn hyp1f1(bencher: Bencher) {
        sum3(
            bencher,
            range(N, 52, -3.0, 3.0),
            range(N, 53, 0.5, 5.0),
            range(N, 54, -5.0, 5.0),
            special::hyp1f1,
        );
    }

    #[divan::bench]
    fn hyp2f1(bencher: Bencher) {
        sum4(
            bencher,
            [
                range(N, 55, -2.0, 2.0),
                range(N, 56, -2.0, 2.0),
                range(N, 57, 0.5, 5.0),
                range(N, 58, -0.9, 0.9),
            ],
            special::hyp2f1,
        );
    }

    #[divan::bench]
    fn hyperu(bencher: Bencher) {
        sum3(
            bencher,
            range(SMALL, 59, 0.5, 2.0),
            range(SMALL, 60, 0.5, 3.0),
            range(SMALL, 61, 0.5, 10.0),
            special::hyperu,
        );
    }
}

mod integral_benches {
    use super::*;

    #[divan::bench]
    fn exp1(bencher: Bencher) {
        sum1(bencher, range(N, 70, 0.01, 20.0), special::exp1);
    }

    #[divan::bench]
    fn expi(bencher: Bencher) {
        sum1(bencher, range(N, 71, 0.01, 20.0), special::expi);
    }

    #[divan::bench]
    fn expn(bencher: Bencher) {
        sum2(
            bencher,
            ints(N, 72, 0, 5),
            range(N, 73, 0.01, 20.0),
            |n, x| special::expn(n as u32, x),
        );
    }

    #[divan::bench]
    fn sici(bencher: Bencher) {
        sum1(bencher, range(N, 74, -20.0, 20.0), |x| {
            pair(special::sici(x))
        });
    }

    #[divan::bench]
    fn shichi(bencher: Bencher) {
        sum1(bencher, range(N, 75, -20.0, 20.0), |x| {
            pair(special::shichi(x))
        });
    }

    #[divan::bench]
    fn fresnel(bencher: Bencher) {
        sum1(bencher, range(N, 76, -10.0, 10.0), |x| {
            pair(special::fresnel(x))
        });
    }

    #[divan::bench]
    fn spence(bencher: Bencher) {
        sum1(bencher, range(N, 77, 0.0, 10.0), special::spence);
    }

    #[divan::bench]
    fn zeta(bencher: Bencher) {
        sum2(
            bencher,
            range(N, 78, 1.1, 11.0),
            range(N, 79, 0.1, 10.0),
            special::zeta,
        );
    }

    #[divan::bench]
    fn zetac(bencher: Bencher) {
        sum1(bencher, range(N, 80, 1.1, 30.0), special::zetac);
    }

    #[divan::bench]
    fn wrightomega(bencher: Bencher) {
        sum1(bencher, range(N, 81, -10.0, 10.0), special::wrightomega);
    }

    #[divan::bench]
    fn wright_bessel(bencher: Bencher) {
        sum3(
            bencher,
            range(HALF_SMALL, 82, 0.0, 1.0),
            range(HALF_SMALL, 83, 0.0, 2.0),
            range(HALF_SMALL, 84, 0.0, 10.0),
            special::wright_bessel,
        );
    }

    #[divan::bench]
    fn log_wright_bessel(bencher: Bencher) {
        sum3(
            bencher,
            range(HALF_SMALL, 85, 0.0, 1.0),
            range(HALF_SMALL, 86, 0.0, 2.0),
            range(HALF_SMALL, 87, 0.0, 10.0),
            special::log_wright_bessel,
        );
    }

    #[divan::bench]
    fn voigt_profile(bencher: Bencher) {
        sum3(
            bencher,
            range(N, 88, -5.0, 5.0),
            range(N, 89, 0.1, 2.0),
            range(N, 90, 0.1, 2.0),
            special::voigt_profile,
        );
    }
}

mod distribution_benches {
    use super::*;

    #[divan::bench]
    fn bdtr(bencher: Bencher) {
        sum2(bencher, ints(N, 100, 0, 19), unit_values(N, 101), |k, p| {
            special::bdtr(k, 20, p)
        });
    }

    #[divan::bench]
    fn bdtrc(bencher: Bencher) {
        sum2(bencher, ints(N, 102, 0, 19), unit_values(N, 103), |k, p| {
            special::bdtrc(k, 20, p)
        });
    }

    #[divan::bench]
    fn bdtri(bencher: Bencher) {
        sum2(bencher, ints(N, 104, 0, 19), unit_values(N, 105), |k, y| {
            special::bdtri(k, 20, y)
        });
    }

    #[divan::bench]
    fn chdtr(bencher: Bencher) {
        sum2(
            bencher,
            range(N, 106, 0.5, 20.0),
            range(N, 107, 0.0, 30.0),
            special::chdtr,
        );
    }

    #[divan::bench]
    fn chdtrc(bencher: Bencher) {
        sum2(
            bencher,
            range(N, 108, 0.5, 20.0),
            range(N, 109, 0.0, 30.0),
            special::chdtrc,
        );
    }

    #[divan::bench]
    fn chdtri(bencher: Bencher) {
        sum2(
            bencher,
            range(N, 110, 0.5, 20.0),
            unit_values(N, 111),
            special::chdtri,
        );
    }

    #[divan::bench]
    fn chdtriv(bencher: Bencher) {
        sum2(
            bencher,
            range(SMALL, 112, 0.05, 0.95),
            range(SMALL, 113, 1.0, 20.0),
            special::chdtriv,
        );
    }

    #[divan::bench]
    fn fdtr(bencher: Bencher) {
        sum3(
            bencher,
            range(HALF, 114, 1.0, 20.0),
            range(HALF, 115, 1.0, 20.0),
            range(N, 116, 0.0, 5.0),
            special::fdtr,
        );
    }

    #[divan::bench]
    fn fdtrc(bencher: Bencher) {
        sum3(
            bencher,
            range(N, 117, 1.0, 20.0),
            range(N, 118, 1.0, 20.0),
            range(N, 119, 0.0, 5.0),
            special::fdtrc,
        );
    }

    #[divan::bench]
    fn fdtri(bencher: Bencher) {
        sum3(
            bencher,
            range(HALF, 120, 1.0, 20.0),
            range(HALF, 121, 1.0, 20.0),
            unit_values(HALF, 122),
            special::fdtri,
        );
    }

    #[divan::bench]
    fn gdtr(bencher: Bencher) {
        sum3(
            bencher,
            range(N, 123, 0.1, 5.0),
            range(N, 124, 0.5, 10.0),
            range(N, 125, 0.0, 10.0),
            special::gdtr,
        );
    }

    #[divan::bench]
    fn gdtrc(bencher: Bencher) {
        sum3(
            bencher,
            range(N, 126, 0.1, 5.0),
            range(N, 127, 0.5, 10.0),
            range(N, 128, 0.0, 10.0),
            special::gdtrc,
        );
    }

    #[divan::bench]
    fn gdtrib(bencher: Bencher) {
        sum3(
            bencher,
            range(SMALL, 129, 0.1, 5.0),
            range(SMALL, 130, 0.05, 0.95),
            range(SMALL, 131, 0.1, 10.0),
            special::gdtrib,
        );
    }

    #[divan::bench]
    fn gdtria(bencher: Bencher) {
        sum3(
            bencher,
            range(SMALL, 132, 0.05, 0.95),
            range(SMALL, 133, 0.5, 10.0),
            range(SMALL, 134, 0.1, 10.0),
            special::gdtria,
        );
    }

    #[divan::bench]
    fn gdtrix(bencher: Bencher) {
        sum3(
            bencher,
            range(SMALL, 135, 0.1, 5.0),
            range(SMALL, 136, 0.5, 10.0),
            range(SMALL, 137, 0.05, 0.95),
            special::gdtrix,
        );
    }

    #[divan::bench]
    fn nbdtr(bencher: Bencher) {
        sum2(bencher, ints(N, 138, 0, 30), unit_values(N, 139), |k, p| {
            special::nbdtr(k as i32, 10, p)
        });
    }

    #[divan::bench]
    fn nbdtrc(bencher: Bencher) {
        sum2(bencher, ints(N, 140, 0, 30), unit_values(N, 141), |k, p| {
            special::nbdtrc(k as i32, 10, p)
        });
    }

    #[divan::bench]
    fn nbdtri(bencher: Bencher) {
        sum2(bencher, ints(N, 142, 0, 30), unit_values(N, 143), |k, y| {
            special::nbdtri(k as i32, 10, y)
        });
    }

    #[divan::bench]
    fn ndtr(bencher: Bencher) {
        sum1(bencher, range(N, 144, -10.0, 10.0), special::ndtr);
    }

    #[divan::bench]
    fn ndtri(bencher: Bencher) {
        sum1(bencher, unit_values(N, 145), special::ndtri);
    }

    #[divan::bench]
    fn log_ndtr(bencher: Bencher) {
        sum1(bencher, range(N, 146, -30.0, 10.0), special::log_ndtr);
    }

    #[divan::bench]
    fn ndtri_exp(bencher: Bencher) {
        sum1(bencher, range(N, 147, -30.0, -0.01), special::ndtri_exp);
    }

    #[divan::bench]
    fn nrdtrimn(bencher: Bencher) {
        sum3(
            bencher,
            unit_values(N, 148),
            range(N, 149, 0.5, 3.0),
            range(N, 150, -5.0, 5.0),
            special::nrdtrimn,
        );
    }

    #[divan::bench]
    fn nrdtrisd(bencher: Bencher) {
        sum3(
            bencher,
            range(N, 151, -5.0, -0.1),
            range(N, 152, 0.55, 0.99),
            range(N, 153, 0.1, 5.0),
            special::nrdtrisd,
        );
    }

    #[divan::bench]
    fn pdtr(bencher: Bencher) {
        sum2(
            bencher,
            ints(N, 154, 0, 20),
            range(N, 155, 0.1, 15.0),
            special::pdtr,
        );
    }

    #[divan::bench]
    fn pdtrc(bencher: Bencher) {
        sum2(
            bencher,
            ints(N, 156, 0, 20),
            range(N, 157, 0.1, 15.0),
            special::pdtrc,
        );
    }

    #[divan::bench]
    fn pdtri(bencher: Bencher) {
        sum2(bencher, ints(N, 158, 0, 20), unit_values(N, 159), |k, p| {
            special::pdtri(k as i32, p)
        });
    }

    #[divan::bench]
    fn stdtr(bencher: Bencher) {
        sum2(
            bencher,
            range(N, 160, 1.0, 30.0),
            range(N, 161, -5.0, 5.0),
            special::stdtr,
        );
    }

    #[divan::bench]
    fn stdtrit(bencher: Bencher) {
        sum2(
            bencher,
            range(SMALL, 162, 1.0, 30.0),
            unit_values(SMALL, 163),
            special::stdtrit,
        );
    }

    #[divan::bench]
    fn kolmogorov(bencher: Bencher) {
        sum1(bencher, range(N, 164, 0.1, 3.0), special::kolmogorov);
    }

    #[divan::bench]
    fn kolmogi(bencher: Bencher) {
        sum1(bencher, unit_values(N, 165), special::kolmogi);
    }

    #[divan::bench]
    fn smirnov(bencher: Bencher) {
        sum1(bencher, unit_values(SMALL, 166), |x| {
            special::smirnov(10, x)
        });
    }

    #[divan::bench]
    fn smirnovi(bencher: Bencher) {
        sum1(bencher, unit_values(TINY, 167), |q| {
            special::smirnovi(10, q)
        });
    }

    #[divan::bench]
    fn owens_t(bencher: Bencher) {
        sum2(
            bencher,
            range(N, 168, -5.0, 5.0),
            range(N, 169, -5.0, 5.0),
            special::owens_t,
        );
    }

    #[divan::bench]
    fn tklmbda(bencher: Bencher) {
        sum2(
            bencher,
            range(HALF, 170, -3.0, 3.0),
            range(HALF, 171, -0.5, 2.0),
            special::tklmbda,
        );
    }

    #[divan::bench]
    fn betaincc(bencher: Bencher) {
        sum3(
            bencher,
            range(N, 172, 0.5, 10.0),
            range(N, 173, 0.5, 10.0),
            unit_values(N, 174),
            special::betaincc,
        );
    }

    #[divan::bench]
    fn betainccinv(bencher: Bencher) {
        sum3(
            bencher,
            range(HALF, 175, 0.5, 10.0),
            range(HALF, 176, 0.5, 10.0),
            unit_values(HALF, 177),
            special::betainccinv,
        );
    }
}

mod parabolic_mathieu_benches {
    use super::*;

    #[divan::bench]
    fn pbdv(bencher: Bencher) {
        sum2(
            bencher,
            range(HALF_SMALL, 180, 0.0, 5.0),
            range(HALF_SMALL, 181, -5.0, 5.0),
            |v, x| pair(special::pbdv(v, x)),
        );
    }

    #[divan::bench]
    fn pbvv(bencher: Bencher) {
        sum2(
            bencher,
            range(HALF_SMALL, 182, 0.0, 5.0),
            range(HALF_SMALL, 183, -5.0, 5.0),
            |v, x| pair(special::pbvv(v, x)),
        );
    }

    #[divan::bench]
    fn pbwa(bencher: Bencher) {
        sum2(
            bencher,
            range(SMALL, 184, -4.9, 4.9),
            range(SMALL, 185, -4.9, 4.9),
            |a, x| pair(special::pbwa(a, x)),
        );
    }

    #[divan::bench]
    fn mathieu_a(bencher: Bencher) {
        sum2(
            bencher,
            ints(SMALL, 186, 0, 5),
            range(SMALL, 187, 0.0, 10.0),
            |m, q| special::mathieu_a(m as u32, q),
        );
    }

    #[divan::bench]
    fn mathieu_b(bencher: Bencher) {
        sum2(
            bencher,
            ints(SMALL, 188, 1, 5),
            range(SMALL, 189, 0.0, 10.0),
            |m, q| special::mathieu_b(m as u32, q),
        );
    }

    #[divan::bench]
    fn mathieu_cem(bencher: Bencher) {
        sum3(
            bencher,
            ints(SMALL, 190, 0, 5),
            range(SMALL, 191, 0.0, 10.0),
            range(SMALL, 192, 0.0, 180.0),
            |m, q, x| pair(special::mathieu_cem(m as u32, q, x)),
        );
    }

    #[divan::bench]
    fn mathieu_sem(bencher: Bencher) {
        sum3(
            bencher,
            ints(SMALL, 193, 1, 5),
            range(SMALL, 194, 0.0, 10.0),
            range(SMALL, 195, 0.0, 180.0),
            |m, q, x| pair(special::mathieu_sem(m as u32, q, x)),
        );
    }

    #[divan::bench]
    fn mathieu_modcem1(bencher: Bencher) {
        sum3(
            bencher,
            ints(SMALL, 196, 0, 5),
            range(SMALL, 197, 0.5, 10.0),
            range(SMALL, 198, 0.1, 2.0),
            |m, q, x| pair(special::mathieu_modcem1(m as u32, q, x)),
        );
    }

    #[divan::bench]
    fn mathieu_modcem2(bencher: Bencher) {
        sum3(
            bencher,
            ints(SMALL, 199, 0, 5),
            range(SMALL, 200, 0.5, 10.0),
            range(SMALL, 201, 0.1, 2.0),
            |m, q, x| pair(special::mathieu_modcem2(m as u32, q, x)),
        );
    }

    #[divan::bench]
    fn mathieu_modsem1(bencher: Bencher) {
        sum3(
            bencher,
            ints(SMALL, 202, 1, 5),
            range(SMALL, 203, 0.5, 10.0),
            range(SMALL, 204, 0.1, 2.0),
            |m, q, x| pair(special::mathieu_modsem1(m as u32, q, x)),
        );
    }

    #[divan::bench]
    fn mathieu_modsem2(bencher: Bencher) {
        sum3(
            bencher,
            ints(SMALL, 205, 1, 5),
            range(SMALL, 206, 0.5, 10.0),
            range(SMALL, 207, 0.1, 2.0),
            |m, q, x| pair(special::mathieu_modsem2(m as u32, q, x)),
        );
    }
}

mod kelvin_benches {
    use super::*;

    /// Number of zeros requested from the Kelvin zero finders.
    const ZEROS: usize = 16;

    fn args(seed: u64) -> Vec<f64> {
        range(N, seed, 0.1, 20.0)
    }

    #[divan::bench]
    fn ber(bencher: Bencher) {
        sum1(bencher, args(210), special::ber);
    }

    #[divan::bench]
    fn bei(bencher: Bencher) {
        sum1(bencher, args(211), special::bei);
    }

    #[divan::bench]
    fn ker(bencher: Bencher) {
        sum1(bencher, args(212), special::ker);
    }

    #[divan::bench]
    fn kei(bencher: Bencher) {
        sum1(bencher, args(213), special::kei);
    }

    #[divan::bench]
    fn berp(bencher: Bencher) {
        sum1(bencher, args(214), special::berp);
    }

    #[divan::bench]
    fn beip(bencher: Bencher) {
        sum1(bencher, args(215), special::beip);
    }

    #[divan::bench]
    fn kerp(bencher: Bencher) {
        sum1(bencher, args(216), special::kerp);
    }

    #[divan::bench]
    fn keip(bencher: Bencher) {
        sum1(bencher, args(217), special::keip);
    }

    #[divan::bench]
    fn ber_zeros(bencher: Bencher) {
        bencher.bench(|| special::ber_zeros(black_box(ZEROS)).unwrap());
    }

    #[divan::bench]
    fn bei_zeros(bencher: Bencher) {
        bencher.bench(|| special::bei_zeros(black_box(ZEROS)).unwrap());
    }

    #[divan::bench]
    fn ker_zeros(bencher: Bencher) {
        bencher.bench(|| special::ker_zeros(black_box(ZEROS)).unwrap());
    }

    #[divan::bench]
    fn kei_zeros(bencher: Bencher) {
        bencher.bench(|| special::kei_zeros(black_box(ZEROS)).unwrap());
    }

    #[divan::bench]
    fn berp_zeros(bencher: Bencher) {
        bencher.bench(|| special::berp_zeros(black_box(ZEROS)).unwrap());
    }

    #[divan::bench]
    fn beip_zeros(bencher: Bencher) {
        bencher.bench(|| special::beip_zeros(black_box(ZEROS)).unwrap());
    }

    #[divan::bench]
    fn kerp_zeros(bencher: Bencher) {
        bencher.bench(|| special::kerp_zeros(black_box(ZEROS)).unwrap());
    }

    #[divan::bench]
    fn keip_zeros(bencher: Bencher) {
        bencher.bench(|| special::keip_zeros(black_box(ZEROS)).unwrap());
    }

    #[divan::bench]
    fn kelvin_zeros(bencher: Bencher) {
        bencher.bench(|| special::kelvin_zeros(black_box(ZEROS)).unwrap());
    }
}

mod legendre_benches {
    use super::*;

    #[divan::bench]
    fn lpmv(bencher: Bencher) {
        sum3(
            bencher,
            ints(N, 220, 0, 3),
            range(N, 221, 3.0, 8.0),
            range(N, 222, -0.99, 0.99),
            |m, v, x| special::lpmv(m as i32, v, x),
        );
    }

    #[divan::bench]
    fn legendre_p(bencher: Bencher) {
        sum2(
            bencher,
            ints(N, 223, 0, 20),
            range(N, 224, -1.0, 1.0),
            |n, x| special::legendre_p(n as i32, x),
        );
    }

    #[divan::bench]
    fn assoc_legendre_p(bencher: Bencher) {
        sum3(
            bencher,
            ints(N, 225, 4, 20),
            ints(N, 226, 0, 4),
            range(N, 227, -0.99, 0.99),
            |n, m, x| special::assoc_legendre_p(n as i32, m as i32, x),
        );
    }

    #[divan::bench]
    fn sph_legendre_p(bencher: Bencher) {
        sum3(
            bencher,
            ints(N, 228, 4, 20),
            ints(N, 229, 0, 4),
            range(N, 230, 0.0, std::f64::consts::PI),
            |n, m, theta| special::sph_legendre_p(n as i32, m as i32, theta),
        );
    }

    #[divan::bench]
    fn lqn(bencher: Bencher) {
        let xs = range(16, 231, -0.9, 0.9);
        bencher.bench(|| {
            black_box(&xs)
                .iter()
                .map(|&x| special::lqn(32, x).unwrap())
                .collect::<Vec<_>>()
        });
    }

    #[divan::bench]
    fn lqmn(bencher: Bencher) {
        let xs = range(4, 232, -0.9, 0.9);
        bencher.bench(|| {
            black_box(&xs)
                .iter()
                .map(|&x| special::lqmn(8, 8, x).unwrap())
                .collect::<Vec<_>>()
        });
    }
}

mod elementary_benches {
    use super::*;

    #[divan::bench]
    fn expit(bencher: Bencher) {
        sum1(bencher, range(N, 245, -20.0, 20.0), special::expit);
    }

    #[divan::bench]
    fn logit(bencher: Bencher) {
        sum1(bencher, unit_values(N, 246), special::logit);
    }

    #[divan::bench]
    fn softplus(bencher: Bencher) {
        sum1(bencher, range(N, 247, -20.0, 20.0), special::softplus);
    }

    #[divan::bench]
    fn log_expit(bencher: Bencher) {
        sum1(bencher, range(N, 248, -20.0, 20.0), special::log_expit);
    }

    #[divan::bench]
    fn exprel(bencher: Bencher) {
        sum1(bencher, range(N, 249, -5.0, 5.0), special::exprel);
    }

    #[divan::bench]
    fn sinc(bencher: Bencher) {
        sum1(bencher, range(N, 250, -10.0, 10.0), special::sinc);
    }

    #[divan::bench]
    fn cosm1(bencher: Bencher) {
        sum1(bencher, values(N, 251), special::cosm1);
    }

    #[divan::bench]
    fn sindg(bencher: Bencher) {
        sum1(bencher, range(N, 252, -360.0, 360.0), special::sindg);
    }

    #[divan::bench]
    fn cosdg(bencher: Bencher) {
        sum1(bencher, range(N, 253, -360.0, 360.0), special::cosdg);
    }

    #[divan::bench]
    fn tandg(bencher: Bencher) {
        sum1(bencher, range(N, 254, -80.0, 80.0), special::tandg);
    }

    #[divan::bench]
    fn cotdg(bencher: Bencher) {
        sum1(bencher, range(N, 255, 10.0, 170.0), special::cotdg);
    }

    #[divan::bench]
    fn radian(bencher: Bencher) {
        sum3(
            bencher,
            range(N, 256, 0.0, 360.0),
            range(N, 257, 0.0, 60.0),
            range(N, 258, 0.0, 60.0),
            special::radian,
        );
    }
}

mod information_benches {
    use super::*;

    #[divan::bench]
    fn entr(bencher: Bencher) {
        sum1(bencher, unit_values(N, 260), special::entr);
    }

    #[divan::bench]
    fn rel_entr(bencher: Bencher) {
        sum2(
            bencher,
            unit_values(N, 261),
            unit_values(N, 262),
            special::rel_entr,
        );
    }

    #[divan::bench]
    fn kl_div(bencher: Bencher) {
        sum2(
            bencher,
            unit_values(N, 263),
            unit_values(N, 264),
            special::kl_div,
        );
    }

    #[divan::bench]
    fn huber(bencher: Bencher) {
        sum2(
            bencher,
            range(N, 265, 0.1, 2.0),
            range(N, 266, -5.0, 5.0),
            special::huber,
        );
    }

    #[divan::bench]
    fn pseudo_huber(bencher: Bencher) {
        sum2(
            bencher,
            range(N, 267, 0.1, 2.0),
            range(N, 268, -5.0, 5.0),
            special::pseudo_huber,
        );
    }

    #[divan::bench]
    fn xlogy(bencher: Bencher) {
        sum2(
            bencher,
            range(N, 269, 0.0, 5.0),
            range(N, 270, 0.01, 10.0),
            special::xlogy,
        );
    }

    #[divan::bench]
    fn xlog1py(bencher: Bencher) {
        sum2(
            bencher,
            range(N, 271, 0.0, 5.0),
            range(N, 272, -0.9, 10.0),
            special::xlog1py,
        );
    }

    #[divan::bench]
    fn powm1(bencher: Bencher) {
        sum2(
            bencher,
            range(N, 273, 0.1, 5.0),
            range(N, 274, -3.0, 3.0),
            special::powm1,
        );
    }

    #[divan::bench]
    fn boxcox(bencher: Bencher) {
        sum2(
            bencher,
            range(N, 275, 0.1, 10.0),
            range(N, 276, -2.0, 2.0),
            special::boxcox,
        );
    }

    #[divan::bench]
    fn boxcox1p(bencher: Bencher) {
        sum2(
            bencher,
            range(N, 277, -0.9, 10.0),
            range(N, 278, -2.0, 2.0),
            special::boxcox1p,
        );
    }

    #[divan::bench]
    fn inv_boxcox(bencher: Bencher) {
        sum2(
            bencher,
            range(N, 279, 0.0, 2.0),
            range(N, 280, -0.4, 2.0),
            special::inv_boxcox,
        );
    }

    #[divan::bench]
    fn inv_boxcox1p(bencher: Bencher) {
        sum2(
            bencher,
            range(N, 281, 0.0, 2.0),
            range(N, 282, -0.4, 2.0),
            special::inv_boxcox1p,
        );
    }
}
