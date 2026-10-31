//! Benchmarks covering the public functions of `sci::special`, except the
//! SciPy-compatibility functions defined in `scipy.rs`, which have their own
//! benchmark target.
//!
//! Scalar functions are evaluated over a small deterministic set of valid
//! inputs and the results are summed, so each iteration stays short.

mod common;

use common::{linspace, positive_values, unit_values, values};
use divan::{Bencher, black_box};
use sci::complex::Complex;
use sci::special;

fn main() {
    divan::main();
}

/// Number of points used for cheap scalar functions.
const N: usize = 128;
/// Number of points used for scalar functions solved by iteration.
const N_SLOW: usize = 16;

/// Sums `f` over every input.
fn sum1(xs: &[f64], f: impl Fn(f64) -> f64) -> f64 {
    xs.iter().map(|&x| f(x)).sum()
}

/// Sums `f` over paired inputs.
fn sum2(xs: &[f64], ys: &[f64], f: impl Fn(f64, f64) -> f64) -> f64 {
    xs.iter().zip(ys).map(|(&x, &y)| f(x, y)).sum()
}

/// Sums the real and imaginary parts of a complex value.
fn csum(z: Complex) -> f64 {
    z.re + z.im
}

/// Integer degrees cycling through `0..max`.
fn degrees(n: usize, max: i32) -> Vec<i32> {
    (0..n).map(|i| i as i32 % max).collect()
}

mod gamma_benches {
    use super::*;

    #[divan::bench]
    fn gamma(bencher: Bencher) {
        let x = linspace(0.1, 20.0, N);
        bencher.bench(|| sum1(black_box(&x), special::gamma::gamma));
    }

    #[divan::bench]
    fn gammaln(bencher: Bencher) {
        let x = linspace(0.1, 200.0, N);
        bencher.bench(|| sum1(black_box(&x), special::gamma::gammaln));
    }

    #[divan::bench]
    fn digamma(bencher: Bencher) {
        let x = linspace(0.1, 50.0, N);
        bencher.bench(|| sum1(black_box(&x), special::gamma::digamma));
    }

    #[divan::bench]
    fn trigamma(bencher: Bencher) {
        let x = linspace(0.1, 50.0, N);
        bencher.bench(|| sum1(black_box(&x), special::gamma::trigamma));
    }

    #[divan::bench]
    fn beta(bencher: Bencher) {
        let a = positive_values(N, 1);
        let b = positive_values(N, 2);
        bencher.bench(|| sum2(black_box(&a), black_box(&b), special::beta::beta));
    }

    #[divan::bench]
    fn betaln(bencher: Bencher) {
        let a = positive_values(N, 1);
        let b = positive_values(N, 2);
        bencher.bench(|| sum2(black_box(&a), black_box(&b), special::beta::betaln));
    }

    #[divan::bench]
    fn betainc(bencher: Bencher) {
        let a = positive_values(N, 1);
        let b = positive_values(N, 2);
        let x = unit_values(N, 3);
        bencher.bench(|| {
            let (a, b, x) = black_box((&a, &b, &x));
            (0..N)
                .map(|i| special::beta::betainc(a[i], b[i], x[i]))
                .sum::<f64>()
        });
    }

    #[divan::bench]
    fn betaincinv(bencher: Bencher) {
        let a = positive_values(N, 1);
        let b = positive_values(N, 2);
        let p = unit_values(N, 3);
        bencher.bench(|| {
            let (a, b, p) = black_box((&a, &b, &p));
            (0..N)
                .map(|i| special::beta::betaincinv(a[i], b[i], p[i]))
                .sum::<f64>()
        });
    }

    #[divan::bench]
    fn gammainc(bencher: Bencher) {
        let a = positive_values(N, 1);
        let x = positive_values(N, 2);
        bencher.bench(|| sum2(black_box(&a), black_box(&x), special::gamma::gammainc));
    }

    #[divan::bench]
    fn poch(bencher: Bencher) {
        let a = positive_values(N, 1);
        let m = linspace(0.0, 5.0, N);
        bencher.bench(|| sum2(black_box(&a), black_box(&m), special::combinatorics::poch));
    }
}

mod error_function_benches {
    use super::*;

    #[divan::bench]
    fn erf(bencher: Bencher) {
        let x = linspace(-4.0, 4.0, N);
        bencher.bench(|| sum1(black_box(&x), special::elementary::erf));
    }

    #[divan::bench]
    fn erfc(bencher: Bencher) {
        let x = linspace(-4.0, 4.0, N);
        bencher.bench(|| sum1(black_box(&x), special::elementary::erfc));
    }

    #[divan::bench]
    fn erfinv(bencher: Bencher) {
        let x = values(N, 4);
        bencher.bench(|| sum1(black_box(&x), special::elementary::erfinv));
    }

    #[divan::bench]
    fn wofz(bencher: Bencher) {
        let re = linspace(-3.0, 3.0, N);
        let im = linspace(0.01, 2.0, N);
        bencher.bench(|| {
            sum2(black_box(&re), black_box(&im), |re, im| {
                csum(special::elementary::wofz(Complex { re, im }))
            })
        });
    }

    #[divan::bench]
    fn erf_zeros(bencher: Bencher) {
        bencher.bench(|| special::elementary::erf_zeros(black_box(16)).unwrap());
    }
}

mod lambertw_benches {
    use super::*;

    #[divan::bench]
    fn lambertw(bencher: Bencher) {
        let x = linspace(-0.3, 20.0, N);
        bencher.bench(|| sum1(black_box(&x), special::elementary::lambertw));
    }

    #[divan::bench]
    fn lambertwm1(bencher: Bencher) {
        let x = linspace(-0.36, -0.001, N);
        bencher.bench(|| sum1(black_box(&x), special::elementary::lambertwm1));
    }
}

mod elementary_benches {
    use super::*;

    #[divan::bench]
    fn logsumexp(bencher: Bencher) {
        let x = values(1024, 5);
        bencher.bench(|| special::elementary::logsumexp(black_box(&x)));
    }

    #[divan::bench]
    fn softmax(bencher: Bencher) {
        let x = values(1024, 5);
        bencher.bench(|| special::elementary::softmax(black_box(&x)));
    }

    #[divan::bench]
    fn log_softmax(bencher: Bencher) {
        let x = values(1024, 5);
        bencher.bench(|| special::elementary::log_softmax(black_box(&x)));
    }

    #[divan::bench]
    fn diric(bencher: Bencher) {
        let x = linspace(-6.0, 6.0, N);
        bencher.bench(|| sum1(black_box(&x), |x| special::elementary::diric(x, 7)));
    }
}

mod bessel_derivative_benches {
    use super::*;

    /// Derivatives combine several Bessel evaluations, so use fewer points.
    const N: usize = 32;

    #[divan::bench]
    fn jvp(bencher: Bencher) {
        let x = linspace(0.5, 20.0, N);
        bencher.bench(|| sum1(black_box(&x), |x| special::bessel::jvp(1.5, x, 2)));
    }

    #[divan::bench]
    fn yvp(bencher: Bencher) {
        let x = linspace(0.5, 20.0, N);
        bencher.bench(|| sum1(black_box(&x), |x| special::bessel::yvp(1.5, x, 2)));
    }

    #[divan::bench]
    fn ivp(bencher: Bencher) {
        let x = linspace(0.5, 20.0, N);
        bencher.bench(|| sum1(black_box(&x), |x| special::bessel::ivp(1.5, x, 2)));
    }

    #[divan::bench]
    fn kvp(bencher: Bencher) {
        let x = linspace(0.5, 20.0, N);
        bencher.bench(|| sum1(black_box(&x), |x| special::bessel::kvp(1.5, x, 2)));
    }

    #[divan::bench]
    fn h1vp(bencher: Bencher) {
        let x = linspace(0.5, 20.0, N);
        bencher.bench(|| sum1(black_box(&x), |x| csum(special::bessel::h1vp(1.5, x, 2))));
    }

    #[divan::bench]
    fn h2vp(bencher: Bencher) {
        let x = linspace(0.5, 20.0, N);
        bencher.bench(|| sum1(black_box(&x), |x| csum(special::bessel::h2vp(1.5, x, 2))));
    }
}

mod spherical_bessel_benches {
    use super::*;

    #[divan::bench]
    fn spherical_jn(bencher: Bencher) {
        let x = linspace(0.5, 20.0, N);
        bencher.bench(|| {
            sum1(black_box(&x), |x| {
                special::bessel::spherical_jn(3, x, false)
                    + special::bessel::spherical_jn(3, x, true)
            })
        });
    }

    #[divan::bench]
    fn spherical_yn(bencher: Bencher) {
        let x = linspace(0.5, 20.0, N);
        bencher.bench(|| {
            sum1(black_box(&x), |x| {
                special::bessel::spherical_yn(3, x, false)
                    + special::bessel::spherical_yn(3, x, true)
            })
        });
    }

    #[divan::bench]
    fn spherical_in(bencher: Bencher) {
        let x = linspace(0.5, 20.0, N);
        bencher.bench(|| {
            sum1(black_box(&x), |x| {
                special::bessel::spherical_in(3, x, false)
                    + special::bessel::spherical_in(3, x, true)
            })
        });
    }

    #[divan::bench]
    fn spherical_kn(bencher: Bencher) {
        let x = linspace(0.5, 20.0, N);
        bencher.bench(|| {
            sum1(black_box(&x), |x| {
                special::bessel::spherical_kn(3, x, false)
                    + special::bessel::spherical_kn(3, x, true)
            })
        });
    }

    #[divan::bench]
    fn riccati_jn(bencher: Bencher) {
        bencher.bench(|| special::bessel::riccati_jn(black_box(32), black_box(7.5)).unwrap());
    }

    #[divan::bench]
    fn riccati_yn(bencher: Bencher) {
        bencher.bench(|| special::bessel::riccati_yn(black_box(32), black_box(7.5)).unwrap());
    }

    #[divan::bench]
    fn lmbda(bencher: Bencher) {
        bencher.bench(|| special::wave::lmbda(black_box(16.5), black_box(7.5)).unwrap());
    }
}

mod hankel_benches {
    use super::*;

    #[divan::bench]
    fn hankel1(bencher: Bencher) {
        let x = linspace(0.5, 20.0, N);
        bencher.bench(|| sum1(black_box(&x), |x| csum(special::bessel::hankel1(1.5, x))));
    }

    #[divan::bench]
    fn hankel1e(bencher: Bencher) {
        let x = linspace(0.5, 20.0, N);
        bencher.bench(|| sum1(black_box(&x), |x| csum(special::bessel::hankel1e(1.5, x))));
    }

    #[divan::bench]
    fn hankel2(bencher: Bencher) {
        let x = linspace(0.5, 20.0, N);
        bencher.bench(|| sum1(black_box(&x), |x| csum(special::bessel::hankel2(1.5, x))));
    }

    #[divan::bench]
    fn hankel2e(bencher: Bencher) {
        let x = linspace(0.5, 20.0, N);
        bencher.bench(|| sum1(black_box(&x), |x| csum(special::bessel::hankel2e(1.5, x))));
    }

    #[divan::bench]
    fn kelvin(bencher: Bencher) {
        let x = linspace(0.1, 10.0, N);
        bencher.bench(|| {
            sum1(black_box(&x), |x| {
                special::bessel::kelvin(x)
                    .into_iter()
                    .map(csum)
                    .sum::<f64>()
            })
        });
    }
}

mod fresnel_benches {
    use super::*;

    #[divan::bench]
    fn modfresnelp(bencher: Bencher) {
        let x = linspace(-5.0, 5.0, N);
        bencher.bench(|| {
            sum1(black_box(&x), |x| {
                let (a, b) = special::integrals::modfresnelp(x);
                csum(a) + csum(b)
            })
        });
    }

    #[divan::bench]
    fn modfresnelm(bencher: Bencher) {
        let x = linspace(-5.0, 5.0, N);
        bencher.bench(|| {
            sum1(black_box(&x), |x| {
                let (a, b) = special::integrals::modfresnelm(x);
                csum(a) + csum(b)
            })
        });
    }

    #[divan::bench]
    fn fresnel_zeros(bencher: Bencher) {
        bencher.bench(|| special::integrals::fresnel_zeros(black_box(8)).unwrap());
    }

    #[divan::bench]
    fn fresnels_zeros(bencher: Bencher) {
        bencher.bench(|| special::integrals::fresnels_zeros(black_box(8)).unwrap());
    }

    #[divan::bench]
    fn fresnelc_zeros(bencher: Bencher) {
        bencher.bench(|| special::integrals::fresnelc_zeros(black_box(8)).unwrap());
    }
}

mod parabolic_cylinder_benches {
    use super::*;

    #[divan::bench]
    fn pbdv_seq(bencher: Bencher) {
        bencher.bench(|| special::wave::pbdv_seq(black_box(16.5), black_box(1.5)).unwrap());
    }

    #[divan::bench]
    fn pbvv_seq(bencher: Bencher) {
        bencher.bench(|| special::wave::pbvv_seq(black_box(16.5), black_box(1.5)).unwrap());
    }

    #[divan::bench]
    fn pbdn_seq(bencher: Bencher) {
        bencher.bench(|| special::wave::pbdn_seq(black_box(16), black_box(1.5)).unwrap());
    }
}

mod combinatorics_benches {
    use super::*;

    #[divan::bench]
    fn factorial(bencher: Bencher) {
        let x = linspace(0.0, 100.0, N);
        bencher.bench(|| sum1(black_box(&x), special::combinatorics::factorial));
    }

    #[divan::bench]
    fn factorial2(bencher: Bencher) {
        let n = degrees(N, 120);
        bencher.bench(|| {
            black_box(&n)
                .iter()
                .map(|&n| special::combinatorics::factorial2(n))
                .sum::<f64>()
        });
    }

    #[divan::bench]
    fn factorialk(bencher: Bencher) {
        let n = degrees(N, 120);
        bencher.bench(|| {
            black_box(&n)
                .iter()
                .map(|&n| special::combinatorics::factorialk(n, 3))
                .sum::<f64>()
        });
    }

    #[divan::bench]
    fn comb(bencher: Bencher) {
        let k = linspace(0.0, 60.0, N);
        bencher.bench(|| sum1(black_box(&k), |k| special::combinatorics::comb(60.0, k)));
    }

    #[divan::bench]
    fn perm(bencher: Bencher) {
        bencher.bench(|| {
            (0..N as u32)
                .map(|k| special::combinatorics::perm(black_box(150.0), k % 60))
                .sum::<f64>()
        });
    }

    #[divan::bench]
    fn stirling2(bencher: Bencher) {
        bencher.bench(|| special::combinatorics::stirling2(black_box(64), black_box(20)));
    }

    #[divan::bench]
    fn bernoulli(bencher: Bencher) {
        bencher.bench(|| special::combinatorics::bernoulli(black_box(64)).unwrap());
    }

    #[divan::bench]
    fn euler(bencher: Bencher) {
        bencher.bench(|| special::combinatorics::euler(black_box(64)).unwrap());
    }

    #[divan::bench]
    fn harmonic(bencher: Bencher) {
        bencher.bench(|| {
            (0..N as u64)
                .map(|n| special::combinatorics::harmonic(black_box(n * 1000)))
                .sum::<f64>()
        });
    }

    #[divan::bench]
    fn gen_harmonic(bencher: Bencher) {
        bencher.bench(|| {
            special::combinatorics::gen_harmonic(black_box(1000), black_box(1.5)).unwrap()
        });
    }

    #[divan::bench]
    fn ln_factorial(bencher: Bencher) {
        bencher.bench(|| {
            (0..N as u64)
                .map(|n| special::combinatorics::ln_factorial(black_box(n * 1000)))
                .sum::<f64>()
        });
    }

    #[divan::bench]
    fn ln_binomial(bencher: Bencher) {
        bencher.bench(|| {
            special::combinatorics::ln_binomial(black_box(10_000), black_box(500)).unwrap()
        });
    }

    #[divan::bench]
    fn ln_multinomial(bencher: Bencher) {
        let counts: Vec<u64> = (1..=32).collect();
        let n = counts.iter().sum();
        bencher.bench(|| {
            special::combinatorics::ln_multinomial(black_box(n), black_box(&counts)).unwrap()
        });
    }

    #[divan::bench]
    fn multinomial(bencher: Bencher) {
        let counts: Vec<u64> = vec![3, 4, 5, 6];
        let n = counts.iter().sum();
        bencher.bench(|| {
            special::combinatorics::multinomial(black_box(n), black_box(&counts)).unwrap()
        });
    }
}

mod orthogonal_eval_benches {
    use super::*;

    #[divan::bench]
    fn eval_jacobi(bencher: Bencher) {
        let x = values(N, 6);
        bencher.bench(|| {
            sum1(black_box(&x), |x| {
                special::orthogonal::eval_jacobi(10, 0.5, 0.3, x)
            })
        });
    }

    #[divan::bench]
    fn eval_legendre(bencher: Bencher) {
        let x = values(N, 6);
        bencher.bench(|| sum1(black_box(&x), |x| special::orthogonal::eval_legendre(10, x)));
    }

    #[divan::bench]
    fn eval_gegenbauer(bencher: Bencher) {
        let x = values(N, 6);
        bencher.bench(|| {
            sum1(black_box(&x), |x| {
                special::orthogonal::eval_gegenbauer(10, 0.5, x)
            })
        });
    }

    #[divan::bench]
    fn eval_genlaguerre(bencher: Bencher) {
        let x = positive_values(N, 6);
        bencher.bench(|| {
            sum1(black_box(&x), |x| {
                special::orthogonal::eval_genlaguerre(10, 0.5, x)
            })
        });
    }

    #[divan::bench]
    fn eval_laguerre(bencher: Bencher) {
        let x = positive_values(N, 6);
        bencher.bench(|| sum1(black_box(&x), |x| special::orthogonal::eval_laguerre(10, x)));
    }

    #[divan::bench]
    fn eval_chebyt(bencher: Bencher) {
        let x = values(N, 6);
        bencher.bench(|| sum1(black_box(&x), |x| special::orthogonal::eval_chebyt(10, x)));
    }

    #[divan::bench]
    fn eval_chebyu(bencher: Bencher) {
        let x = values(N, 6);
        bencher.bench(|| sum1(black_box(&x), |x| special::orthogonal::eval_chebyu(10, x)));
    }

    #[divan::bench]
    fn eval_chebyc(bencher: Bencher) {
        let x = linspace(-2.0, 2.0, N);
        bencher.bench(|| sum1(black_box(&x), |x| special::orthogonal::eval_chebyc(10, x)));
    }

    #[divan::bench]
    fn eval_chebys(bencher: Bencher) {
        let x = linspace(-2.0, 2.0, N);
        bencher.bench(|| sum1(black_box(&x), |x| special::orthogonal::eval_chebys(10, x)));
    }

    #[divan::bench]
    fn eval_hermite(bencher: Bencher) {
        let x = linspace(-3.0, 3.0, N);
        bencher.bench(|| sum1(black_box(&x), |x| special::orthogonal::eval_hermite(10, x)));
    }

    #[divan::bench]
    fn eval_hermitenorm(bencher: Bencher) {
        let x = linspace(-3.0, 3.0, N);
        bencher.bench(|| {
            sum1(black_box(&x), |x| {
                special::orthogonal::eval_hermitenorm(10, x)
            })
        });
    }

    #[divan::bench]
    fn eval_sh_chebyt(bencher: Bencher) {
        let x = unit_values(N, 6);
        bencher.bench(|| {
            sum1(black_box(&x), |x| {
                special::orthogonal::eval_sh_chebyt(10, x)
            })
        });
    }

    #[divan::bench]
    fn eval_sh_chebyu(bencher: Bencher) {
        let x = unit_values(N, 6);
        bencher.bench(|| {
            sum1(black_box(&x), |x| {
                special::orthogonal::eval_sh_chebyu(10, x)
            })
        });
    }

    #[divan::bench]
    fn eval_sh_legendre(bencher: Bencher) {
        let x = unit_values(N, 6);
        bencher.bench(|| {
            sum1(black_box(&x), |x| {
                special::orthogonal::eval_sh_legendre(10, x)
            })
        });
    }

    #[divan::bench]
    fn eval_sh_jacobi(bencher: Bencher) {
        let x = unit_values(N, 6);
        bencher.bench(|| {
            sum1(black_box(&x), |x| {
                special::orthogonal::eval_sh_jacobi(10, 2.5, 1.5, x)
            })
        });
    }
}

mod orthogonal_polynomial_benches {
    use super::*;

    /// Polynomial degree / number of quadrature nodes.
    const DEGREE: usize = 16;

    #[divan::bench]
    fn jacobi(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::jacobi(black_box(DEGREE), 0.5, 0.3, false).unwrap());
    }

    #[divan::bench]
    fn roots_jacobi(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::roots_jacobi(black_box(DEGREE), 0.5, 0.3).unwrap());
    }

    #[divan::bench]
    fn legendre(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::legendre(black_box(DEGREE), false).unwrap());
    }

    #[divan::bench]
    fn roots_legendre(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::roots_legendre(black_box(DEGREE)).unwrap());
    }

    #[divan::bench]
    fn gegenbauer(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::gegenbauer(black_box(DEGREE), 0.5, false).unwrap());
    }

    #[divan::bench]
    fn roots_gegenbauer(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::roots_gegenbauer(black_box(DEGREE), 0.5).unwrap());
    }

    #[divan::bench]
    fn genlaguerre(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::genlaguerre(black_box(DEGREE), 0.5, false).unwrap());
    }

    #[divan::bench]
    fn roots_genlaguerre(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::roots_genlaguerre(black_box(DEGREE), 0.5).unwrap());
    }

    #[divan::bench]
    fn laguerre(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::laguerre(black_box(DEGREE), false).unwrap());
    }

    #[divan::bench]
    fn roots_laguerre(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::roots_laguerre(black_box(DEGREE)).unwrap());
    }

    #[divan::bench]
    fn hermite(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::hermite(black_box(DEGREE), false).unwrap());
    }

    #[divan::bench]
    fn roots_hermite(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::roots_hermite(black_box(DEGREE)).unwrap());
    }

    #[divan::bench]
    fn hermitenorm(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::hermitenorm(black_box(DEGREE), false).unwrap());
    }

    #[divan::bench]
    fn roots_hermitenorm(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::roots_hermitenorm(black_box(DEGREE)).unwrap());
    }

    #[divan::bench]
    fn chebyt(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::chebyt(black_box(DEGREE), false).unwrap());
    }

    #[divan::bench]
    fn roots_chebyt(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::roots_chebyt(black_box(DEGREE)).unwrap());
    }

    #[divan::bench]
    fn chebyu(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::chebyu(black_box(DEGREE), false).unwrap());
    }

    #[divan::bench]
    fn roots_chebyu(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::roots_chebyu(black_box(DEGREE)).unwrap());
    }

    #[divan::bench]
    fn chebyc(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::chebyc(black_box(DEGREE), false).unwrap());
    }

    #[divan::bench]
    fn roots_chebyc(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::roots_chebyc(black_box(DEGREE)).unwrap());
    }

    #[divan::bench]
    fn chebys(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::chebys(black_box(DEGREE), false).unwrap());
    }

    #[divan::bench]
    fn roots_chebys(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::roots_chebys(black_box(DEGREE)).unwrap());
    }

    #[divan::bench]
    fn sh_chebyt(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::sh_chebyt(black_box(DEGREE), false).unwrap());
    }

    #[divan::bench]
    fn roots_sh_chebyt(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::roots_sh_chebyt(black_box(DEGREE)).unwrap());
    }

    #[divan::bench]
    fn sh_chebyu(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::sh_chebyu(black_box(DEGREE), false).unwrap());
    }

    #[divan::bench]
    fn roots_sh_chebyu(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::roots_sh_chebyu(black_box(DEGREE)).unwrap());
    }

    #[divan::bench]
    fn sh_legendre(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::sh_legendre(black_box(DEGREE), false).unwrap());
    }

    #[divan::bench]
    fn roots_sh_legendre(bencher: Bencher) {
        bencher.bench(|| special::orthogonal::roots_sh_legendre(black_box(DEGREE)).unwrap());
    }

    #[divan::bench]
    fn sh_jacobi(bencher: Bencher) {
        bencher
            .bench(|| special::orthogonal::sh_jacobi(black_box(DEGREE), 2.5, 1.5, false).unwrap());
    }

    #[divan::bench]
    fn roots_sh_jacobi(bencher: Bencher) {
        bencher
            .bench(|| special::orthogonal::roots_sh_jacobi(black_box(DEGREE), 2.5, 1.5).unwrap());
    }
}

mod legendre_function_benches {
    use super::*;

    #[divan::bench]
    fn sph_harm_y(bencher: Bencher) {
        let theta = linspace(0.1, 3.0, N);
        let phi = linspace(0.0, 6.0, N);
        bencher.bench(|| {
            sum2(black_box(&theta), black_box(&phi), |t, p| {
                csum(special::wave::sph_harm_y(8, 3, t, p))
            })
        });
    }

    #[divan::bench]
    fn sph_harm_y_all(bencher: Bencher) {
        bencher.bench(|| {
            special::wave::sph_harm_y_all(
                black_box(16),
                black_box(8),
                black_box(0.5),
                black_box(0.3),
            )
            .unwrap()
        });
    }

    #[divan::bench]
    fn legendre_p_all(bencher: Bencher) {
        bencher
            .bench(|| special::orthogonal::legendre_p_all(black_box(64), black_box(0.3)).unwrap());
    }

    #[divan::bench]
    fn assoc_legendre_p_all(bencher: Bencher) {
        bencher.bench(|| {
            special::orthogonal::assoc_legendre_p_all(black_box(16), black_box(8), black_box(0.3))
                .unwrap()
        });
    }

    #[divan::bench]
    fn sph_legendre_p_all(bencher: Bencher) {
        bencher.bench(|| {
            special::orthogonal::sph_legendre_p_all(black_box(16), black_box(8), black_box(0.3))
                .unwrap()
        });
    }
}

mod mathieu_benches {
    use super::*;

    #[divan::bench]
    fn mathieu_even_coef(bencher: Bencher) {
        bencher.bench(|| special::wave::mathieu_even_coef(black_box(4), black_box(5.0)).unwrap());
    }

    #[divan::bench]
    fn mathieu_odd_coef(bencher: Bencher) {
        bencher.bench(|| special::wave::mathieu_odd_coef(black_box(4), black_box(5.0)).unwrap());
    }
}

mod spheroidal_benches {
    use super::*;

    /// Spheroidal functions solve an eigenproblem per call.
    const N_SLOW: usize = 8;

    /// Angular coordinates in (-1, 1).
    fn angles() -> Vec<f64> {
        linspace(-0.9, 0.9, N_SLOW)
    }

    /// Radial coordinates greater than one.
    fn radii() -> Vec<f64> {
        linspace(1.1, 5.0, N_SLOW)
    }

    #[divan::bench]
    fn pro_cv(bencher: Bencher) {
        let c = linspace(0.1, 4.0, N_SLOW);
        bencher.bench(|| sum1(black_box(&c), |c| special::wave::pro_cv(1, 3, c)));
    }

    #[divan::bench]
    fn pro_cv_seq(bencher: Bencher) {
        bencher.bench(|| {
            special::wave::pro_cv_seq(black_box(1), black_box(6), black_box(2.0)).unwrap()
        });
    }

    #[divan::bench]
    fn pro_ang1(bencher: Bencher) {
        let x = angles();
        bencher.bench(|| sum1(black_box(&x), |x| special::wave::pro_ang1(1, 3, 2.0, x).0));
    }

    #[divan::bench]
    fn pro_ang1_cv(bencher: Bencher) {
        let x = angles();
        let cv = special::wave::pro_cv(1, 3, 2.0);
        bencher.bench(|| {
            sum1(black_box(&x), |x| {
                special::wave::pro_ang1_cv(1, 3, 2.0, cv, x).0
            })
        });
    }

    #[divan::bench]
    fn pro_rad1(bencher: Bencher) {
        let x = radii();
        bencher.bench(|| sum1(black_box(&x), |x| special::wave::pro_rad1(1, 3, 2.0, x).0));
    }

    #[divan::bench]
    fn pro_rad1_cv(bencher: Bencher) {
        let x = radii();
        let cv = special::wave::pro_cv(1, 3, 2.0);
        bencher.bench(|| {
            sum1(black_box(&x), |x| {
                special::wave::pro_rad1_cv(1, 3, 2.0, cv, x).0
            })
        });
    }

    #[divan::bench]
    fn pro_rad2(bencher: Bencher) {
        let x = radii();
        bencher.bench(|| sum1(black_box(&x), |x| special::wave::pro_rad2(1, 3, 2.0, x).0));
    }

    #[divan::bench]
    fn pro_rad2_cv(bencher: Bencher) {
        let x = radii();
        let cv = special::wave::pro_cv(1, 3, 2.0);
        bencher.bench(|| {
            sum1(black_box(&x), |x| {
                special::wave::pro_rad2_cv(1, 3, 2.0, cv, x).0
            })
        });
    }

    #[divan::bench]
    fn obl_cv(bencher: Bencher) {
        let c = linspace(0.1, 4.0, N_SLOW);
        bencher.bench(|| sum1(black_box(&c), |c| special::wave::obl_cv(1, 3, c)));
    }

    #[divan::bench]
    fn obl_cv_seq(bencher: Bencher) {
        bencher.bench(|| {
            special::wave::obl_cv_seq(black_box(1), black_box(6), black_box(2.0)).unwrap()
        });
    }

    #[divan::bench]
    fn obl_ang1(bencher: Bencher) {
        let x = angles();
        bencher.bench(|| sum1(black_box(&x), |x| special::wave::obl_ang1(1, 3, 2.0, x).0));
    }

    #[divan::bench]
    fn obl_ang1_cv(bencher: Bencher) {
        let x = angles();
        let cv = special::wave::obl_cv(1, 3, 2.0);
        bencher.bench(|| {
            sum1(black_box(&x), |x| {
                special::wave::obl_ang1_cv(1, 3, 2.0, cv, x).0
            })
        });
    }

    #[divan::bench]
    fn obl_rad1(bencher: Bencher) {
        let x = radii();
        bencher.bench(|| sum1(black_box(&x), |x| special::wave::obl_rad1(1, 3, 2.0, x).0));
    }

    #[divan::bench]
    fn obl_rad1_cv(bencher: Bencher) {
        let x = radii();
        let cv = special::wave::obl_cv(1, 3, 2.0);
        bencher.bench(|| {
            sum1(black_box(&x), |x| {
                special::wave::obl_rad1_cv(1, 3, 2.0, cv, x).0
            })
        });
    }

    #[divan::bench]
    fn obl_rad2(bencher: Bencher) {
        let x = radii();
        bencher.bench(|| sum1(black_box(&x), |x| special::wave::obl_rad2(1, 3, 2.0, x).0));
    }

    #[divan::bench]
    fn obl_rad2_cv(bencher: Bencher) {
        let x = radii();
        let cv = special::wave::obl_cv(1, 3, 2.0);
        bencher.bench(|| {
            sum1(black_box(&x), |x| {
                special::wave::obl_rad2_cv(1, 3, 2.0, cv, x).0
            })
        });
    }
}

mod ellipsoidal_benches {
    use super::*;

    #[divan::bench]
    fn ellip_harm(bencher: Bencher) {
        let s = linspace(1.5, 5.0, N_SLOW);
        bencher.bench(|| {
            sum1(black_box(&s), |s| {
                special::wave::ellip_harm(0.5, 1.5, 3, 2, s, 1.0, 1.0).unwrap()
            })
        });
    }

    #[divan::bench]
    fn ellip_harm_2(bencher: Bencher) {
        bencher.bench(|| {
            special::wave::ellip_harm_2(black_box(0.5), black_box(1.5), 3, 2, black_box(2.0))
                .unwrap()
        });
    }

    #[divan::bench]
    fn ellip_normal(bencher: Bencher) {
        bencher
            .bench(|| special::wave::ellip_normal(black_box(0.5), black_box(1.5), 3, 2).unwrap());
    }
}

mod complete_elliptic_benches {
    use super::*;

    /// Parameters `m` in (-1, 1).
    fn parameters() -> Vec<f64> {
        linspace(-0.99, 0.99, N)
    }

    #[divan::bench]
    fn ellipk(bencher: Bencher) {
        let m = parameters();
        bencher.bench(|| sum1(black_box(&m), |m| special::elliptic::ellipk(m).unwrap()));
    }

    #[divan::bench]
    fn ellipkm1(bencher: Bencher) {
        let p = linspace(1e-6, 1.0, N);
        bencher.bench(|| sum1(black_box(&p), |p| special::elliptic::ellipkm1(p).unwrap()));
    }

    #[divan::bench]
    fn ellipe(bencher: Bencher) {
        let m = parameters();
        bencher.bench(|| sum1(black_box(&m), |m| special::elliptic::ellipe(m).unwrap()));
    }

    #[divan::bench]
    fn ellipke(bencher: Bencher) {
        let m = parameters();
        bencher.bench(|| {
            sum1(black_box(&m), |m| {
                let (k, e) = special::elliptic::ellipke(m).unwrap();
                k + e
            })
        });
    }

    #[divan::bench]
    fn ellippi(bencher: Bencher) {
        let n = linspace(-0.9, 0.9, N);
        let m = parameters();
        bencher.bench(|| {
            sum2(black_box(&n), black_box(&m), |n, m| {
                special::elliptic::ellippi(n, m).unwrap()
            })
        });
    }

    #[divan::bench]
    fn ellipd(bencher: Bencher) {
        let m = parameters();
        bencher.bench(|| sum1(black_box(&m), |m| special::elliptic::ellipd(m).unwrap()));
    }
}

mod incomplete_elliptic_benches {
    use super::*;

    /// Amplitudes in radians.
    fn amplitudes() -> Vec<f64> {
        linspace(-1.5, 1.5, N)
    }

    /// Parameters `m` in [0, 1).
    fn parameters() -> Vec<f64> {
        linspace(0.0, 0.95, N)
    }

    #[divan::bench]
    fn ellipkinc(bencher: Bencher) {
        let (phi, m) = (amplitudes(), parameters());
        bencher.bench(|| {
            sum2(black_box(&phi), black_box(&m), |p, m| {
                special::elliptic::ellipkinc(p, m).unwrap()
            })
        });
    }

    #[divan::bench]
    fn ellipeinc(bencher: Bencher) {
        let (phi, m) = (amplitudes(), parameters());
        bencher.bench(|| {
            sum2(black_box(&phi), black_box(&m), |p, m| {
                special::elliptic::ellipeinc(p, m).unwrap()
            })
        });
    }

    #[divan::bench]
    fn ellipdinc(bencher: Bencher) {
        let (phi, m) = (amplitudes(), parameters());
        bencher.bench(|| {
            sum2(black_box(&phi), black_box(&m), |p, m| {
                special::elliptic::ellipdinc(p, m).unwrap()
            })
        });
    }

    #[divan::bench]
    fn ellippiinc(bencher: Bencher) {
        let (phi, m) = (amplitudes(), parameters());
        bencher.bench(|| {
            sum2(black_box(&phi), black_box(&m), |p, m| {
                special::elliptic::ellippiinc(p, 0.4, m).unwrap()
            })
        });
    }

    #[divan::bench]
    fn ellippiinc_bulirsch(bencher: Bencher) {
        let (phi, m) = (amplitudes(), parameters());
        bencher.bench(|| {
            sum2(black_box(&phi), black_box(&m), |p, m| {
                special::elliptic::ellippiinc_bulirsch(p, 0.4, m).unwrap()
            })
        });
    }

    #[divan::bench]
    fn jacobi_zeta(bencher: Bencher) {
        let (phi, m) = (amplitudes(), parameters());
        bencher.bench(|| {
            sum2(black_box(&phi), black_box(&m), |p, m| {
                special::elliptic::jacobi_zeta(p, m).unwrap()
            })
        });
    }

    #[divan::bench]
    fn heuman_lambda(bencher: Bencher) {
        let (phi, m) = (amplitudes(), parameters());
        bencher.bench(|| {
            sum2(black_box(&phi), black_box(&m), |p, m| {
                special::elliptic::heuman_lambda(p, m).unwrap()
            })
        });
    }
}

mod carlson_benches {
    use super::*;

    #[divan::bench]
    fn elliprf(bencher: Bencher) {
        let (x, y, z) = (
            positive_values(N, 1),
            positive_values(N, 2),
            positive_values(N, 3),
        );
        bencher.bench(|| {
            let (x, y, z) = black_box((&x, &y, &z));
            (0..N)
                .map(|i| special::elliptic::elliprf(x[i], y[i], z[i]).unwrap())
                .sum::<f64>()
        });
    }

    #[divan::bench]
    fn elliprd(bencher: Bencher) {
        let (x, y, z) = (
            positive_values(N, 1),
            positive_values(N, 2),
            positive_values(N, 3),
        );
        bencher.bench(|| {
            let (x, y, z) = black_box((&x, &y, &z));
            (0..N)
                .map(|i| special::elliptic::elliprd(x[i], y[i], z[i]).unwrap())
                .sum::<f64>()
        });
    }

    #[divan::bench]
    fn elliprg(bencher: Bencher) {
        let (x, y, z) = (
            positive_values(N, 1),
            positive_values(N, 2),
            positive_values(N, 3),
        );
        bencher.bench(|| {
            let (x, y, z) = black_box((&x, &y, &z));
            (0..N)
                .map(|i| special::elliptic::elliprg(x[i], y[i], z[i]).unwrap())
                .sum::<f64>()
        });
    }

    #[divan::bench]
    fn elliprj(bencher: Bencher) {
        let (x, y, z) = (
            positive_values(N, 1),
            positive_values(N, 2),
            positive_values(N, 3),
        );
        let p = positive_values(N, 4);
        bencher.bench(|| {
            let (x, y, z, p) = black_box((&x, &y, &z, &p));
            (0..N)
                .map(|i| special::elliptic::elliprj(x[i], y[i], z[i], p[i]).unwrap())
                .sum::<f64>()
        });
    }

    #[divan::bench]
    fn elliprc(bencher: Bencher) {
        let (x, y) = (positive_values(N, 1), positive_values(N, 2));
        bencher.bench(|| {
            sum2(black_box(&x), black_box(&y), |x, y| {
                special::elliptic::elliprc(x, y).unwrap()
            })
        });
    }
}

mod bulirsch_benches {
    use super::*;

    /// Complementary moduli `kc` in (0, 1].
    fn moduli() -> Vec<f64> {
        linspace(0.05, 1.0, N)
    }

    #[divan::bench]
    fn cel(bencher: Bencher) {
        let kc = moduli();
        bencher.bench(|| {
            sum1(black_box(&kc), |kc| {
                special::elliptic::cel(kc, 0.7, 1.0, 0.5).unwrap()
            })
        });
    }

    #[divan::bench]
    fn cel1(bencher: Bencher) {
        let kc = moduli();
        bencher.bench(|| sum1(black_box(&kc), |kc| special::elliptic::cel1(kc).unwrap()));
    }

    #[divan::bench]
    fn cel2(bencher: Bencher) {
        let kc = moduli();
        bencher.bench(|| {
            sum1(black_box(&kc), |kc| {
                special::elliptic::cel2(kc, 1.0, 0.5).unwrap()
            })
        });
    }

    #[divan::bench]
    fn cel3(bencher: Bencher) {
        let kc = moduli();
        bencher.bench(|| {
            sum1(black_box(&kc), |kc| {
                special::elliptic::cel3(kc, 0.7).unwrap()
            })
        });
    }

    #[divan::bench]
    fn el1(bencher: Bencher) {
        let x = linspace(-5.0, 5.0, N);
        let kc = moduli();
        bencher.bench(|| {
            sum2(black_box(&x), black_box(&kc), |x, kc| {
                special::elliptic::el1(x, kc).unwrap()
            })
        });
    }

    #[divan::bench]
    fn el2(bencher: Bencher) {
        let x = linspace(-5.0, 5.0, N);
        let kc = moduli();
        bencher.bench(|| {
            sum2(black_box(&x), black_box(&kc), |x, kc| {
                special::elliptic::el2(x, kc, 1.0, 0.5).unwrap()
            })
        });
    }

    #[divan::bench]
    fn el3(bencher: Bencher) {
        let x = linspace(-5.0, 5.0, N);
        let kc = moduli();
        bencher.bench(|| {
            sum2(black_box(&x), black_box(&kc), |x, kc| {
                special::elliptic::el3(x, kc, 0.7).unwrap()
            })
        });
    }
}

mod distribution_inverse_benches {
    use super::*;

    /// Probabilities in (0, 1).
    fn probabilities(n: usize) -> Vec<f64> {
        linspace(0.05, 0.95, n)
    }

    #[divan::bench]
    fn btdtria(bencher: Bencher) {
        let p = probabilities(8);
        bencher.bench(|| {
            sum1(black_box(&p), |p| {
                special::probability::btdtria(p, 2.5, 0.5)
            })
        });
    }

    #[divan::bench]
    fn btdtrib(bencher: Bencher) {
        let p = probabilities(8);
        bencher.bench(|| {
            sum1(black_box(&p), |p| {
                special::probability::btdtrib(2.5, p, 0.5)
            })
        });
    }

    #[divan::bench]
    fn bdtrik(bencher: Bencher) {
        let y = probabilities(16);
        bencher.bench(|| {
            sum1(black_box(&y), |y| {
                special::probability::bdtrik(y, 20.0, 0.5)
            })
        });
    }

    #[divan::bench]
    fn bdtrin(bencher: Bencher) {
        let y = probabilities(8);
        bencher.bench(|| sum1(black_box(&y), |y| special::probability::bdtrin(2.0, y, 0.5)));
    }

    #[divan::bench]
    fn nbdtrik(bencher: Bencher) {
        let y = probabilities(8);
        bencher.bench(|| {
            sum1(black_box(&y), |y| {
                special::probability::nbdtrik(y, 5.0, 0.5)
            })
        });
    }

    #[divan::bench]
    fn nbdtrin(bencher: Bencher) {
        let y = probabilities(8);
        bencher.bench(|| {
            sum1(black_box(&y), |y| {
                special::probability::nbdtrin(2.0, y, 0.5)
            })
        });
    }

    #[divan::bench]
    fn pdtrik(bencher: Bencher) {
        let p = probabilities(8);
        bencher.bench(|| sum1(black_box(&p), |p| special::probability::pdtrik(p, 5.0)));
    }

    #[divan::bench]
    fn stdtridf(bencher: Bencher) {
        let p = linspace(0.55, 0.68, 8);
        bencher.bench(|| sum1(black_box(&p), |p| special::probability::stdtridf(p, 0.5)));
    }

    #[divan::bench]
    fn fdtridfd(bencher: Bencher) {
        bencher.bench(|| {
            special::probability::fdtridfd(
                black_box(3.0),
                black_box(0.30154736269507604),
                black_box(0.5),
            )
        });
    }
}

mod noncentral_benches {
    use super::*;

    #[divan::bench]
    fn chndtr(bencher: Bencher) {
        let x = positive_values(16, 7);
        bencher.bench(|| sum1(black_box(&x), |x| special::probability::chndtr(x, 3.0, 0.5)));
    }

    #[divan::bench]
    fn chndtrix(bencher: Bencher) {
        let p = linspace(0.05, 0.95, 2);
        bencher.bench(|| {
            sum1(black_box(&p), |p| {
                special::probability::chndtrix(p, 3.0, 0.5)
            })
        });
    }

    #[divan::bench]
    fn chndtridf(bencher: Bencher) {
        bencher.bench(|| {
            special::probability::chndtridf(black_box(2.5), black_box(0.3), black_box(0.5))
        });
    }

    #[divan::bench]
    fn chndtrinc(bencher: Bencher) {
        bencher.bench(|| {
            special::probability::chndtrinc(black_box(2.5), black_box(3.0), black_box(0.3))
        });
    }

    #[divan::bench]
    fn ncfdtr(bencher: Bencher) {
        let x = positive_values(16, 7);
        bencher.bench(|| {
            sum1(black_box(&x), |x| {
                special::probability::ncfdtr(3.0, 5.0, 0.5, x)
            })
        });
    }

    #[divan::bench]
    fn ncfdtri(bencher: Bencher) {
        bencher.bench(|| {
            special::probability::ncfdtri(
                black_box(3.0),
                black_box(5.0),
                black_box(0.5),
                black_box(0.3),
            )
        });
    }

    #[divan::bench]
    fn ncfdtridfd(bencher: Bencher) {
        bencher.bench(|| {
            special::probability::ncfdtridfd(
                black_box(3.0),
                black_box(0.25637506749643246),
                black_box(0.5),
                black_box(0.5),
            )
        });
    }

    #[divan::bench]
    fn ncfdtridfn(bencher: Bencher) {
        bencher.bench(|| {
            special::probability::ncfdtridfn(
                black_box(0.25637506749643246),
                black_box(5.0),
                black_box(0.5),
                black_box(0.5),
            )
        });
    }

    #[divan::bench]
    fn ncfdtrinc(bencher: Bencher) {
        bencher.bench(|| {
            special::probability::ncfdtrinc(
                black_box(3.0),
                black_box(5.0),
                black_box(0.3),
                black_box(2.5),
            )
        });
    }

    #[divan::bench]
    fn nctdtr(bencher: Bencher) {
        let t = linspace(-3.0, 3.0, 16);
        bencher.bench(|| sum1(black_box(&t), |t| special::probability::nctdtr(3.0, 0.5, t)));
    }

    #[divan::bench]
    fn nctdtrit(bencher: Bencher) {
        bencher.bench(|| {
            special::probability::nctdtrit(black_box(3.0), black_box(0.5), black_box(0.3))
        });
    }

    #[divan::bench]
    fn nctdtrinc(bencher: Bencher) {
        bencher.bench(|| {
            special::probability::nctdtrinc(black_box(3.0), black_box(0.3), black_box(0.5))
        });
    }

    #[divan::bench]
    fn nctdtridf(bencher: Bencher) {
        bencher.bench(|| {
            special::probability::nctdtridf(
                black_box(0.4120716320853016),
                black_box(0.5),
                black_box(0.3),
            )
        });
    }
}

mod zeros_benches {
    use super::*;

    /// Number of zeros requested.
    const NT: usize = 2;

    #[divan::bench]
    fn jn_zeros(bencher: Bencher) {
        bencher.bench(|| special::bessel::jn_zeros(black_box(2), black_box(NT)).unwrap());
    }

    #[divan::bench]
    fn jnp_zeros(bencher: Bencher) {
        bencher.bench(|| special::bessel::jnp_zeros(black_box(2), black_box(NT)).unwrap());
    }

    #[divan::bench]
    fn yn_zeros(bencher: Bencher) {
        bencher.bench(|| special::bessel::yn_zeros(black_box(2), black_box(NT)).unwrap());
    }

    #[divan::bench]
    fn ynp_zeros(bencher: Bencher) {
        bencher.bench(|| special::bessel::ynp_zeros(black_box(2), black_box(NT)).unwrap());
    }

    #[divan::bench]
    fn jnyn_zeros(bencher: Bencher) {
        bencher.bench(|| special::bessel::jnyn_zeros(black_box(2), black_box(1)).unwrap());
    }

    #[divan::bench]
    fn jnjnp_zeros(bencher: Bencher) {
        bencher.bench(|| special::bessel::jnjnp_zeros(black_box(1)).unwrap());
    }

    #[divan::bench]
    fn y0_zeros(bencher: Bencher) {
        bencher.bench(|| special::bessel::y0_zeros(black_box(NT)).unwrap());
    }

    #[divan::bench]
    fn y1_zeros(bencher: Bencher) {
        bencher.bench(|| special::bessel::y1_zeros(black_box(NT)).unwrap());
    }

    #[divan::bench]
    fn y1p_zeros(bencher: Bencher) {
        bencher.bench(|| special::bessel::y1p_zeros(black_box(NT)).unwrap());
    }

    #[divan::bench]
    fn ai_zeros(bencher: Bencher) {
        bencher.bench(|| special::wave::ai_zeros(black_box(NT)).unwrap());
    }

    #[divan::bench]
    fn bi_zeros(bencher: Bencher) {
        bencher.bench(|| special::wave::bi_zeros(black_box(NT)).unwrap());
    }
}
