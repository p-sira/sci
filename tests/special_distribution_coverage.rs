//! Numerical and domain coverage for the Kolmogorov, Smirnov, and discrete
//! distribution kernels, exercised through the public `sci::special` API.
use sci::special as s;

fn close(label: &str, actual: f64, expected: f64, atol: f64, rtol: f64) {
    assert!(actual.is_finite(), "{label}: nonfinite result {actual}");
    assert!(
        (actual - expected).abs() <= atol + rtol * expected.abs(),
        "{label}: actual={actual:.17e}, expected={expected:.17e}, atol={atol}, rtol={rtol}"
    );
}

fn probability(label: &str, p: f64) {
    assert!(p.is_finite() && (0.0..=1.0).contains(&p), "{label}: {p}");
}

// Independently evaluate the Kolmogorov survival function from its classical
// theta-series (small x) or alternating exponential series (large x), rather
// than reusing the backend's scaled/geometric term recurrence.
fn kolmogorov_series_reference(x: f64) -> f64 {
    if x <= 0.82 {
        let terms = (0..8)
            .map(|k| {
                let odd = (2 * k + 1) as f64;
                (-(odd * odd) * std::f64::consts::PI.powi(2) / (8.0 * x * x)).exp()
            })
            .sum::<f64>();
        1.0 - (2.0 * std::f64::consts::PI).sqrt() * terms / x
    } else {
        2.0 * (1..=12)
            .map(|k| {
                let term = (-2.0 * (k * k) as f64 * x * x).exp();
                if k % 2 == 0 { -term } else { term }
            })
            .sum::<f64>()
    }
}

// Binomial coefficients are small enough in the tests below to evaluate by
// their exact finite product in binary64, without calling beta/gamma kernels.
fn choose(n: i32, k: i32) -> f64 {
    (1..=k).fold(1.0, |value, j| value * (n - j + 1) as f64 / j as f64)
}

#[test]
fn kolmogorov_survival_monotonicity_and_asymptotic_tails() {
    assert_eq!(s::kolmogorov(f64::NEG_INFINITY), 1.0);
    assert_eq!(s::kolmogorov(-1.0), 1.0);
    assert_eq!(s::kolmogorov(0.0), 1.0);
    assert_eq!(s::kolmogorov(f64::INFINITY), 0.0);
    assert_eq!(s::kolmogorov(f64::MAX), 0.0);
    assert!(s::kolmogorov(f64::NAN).is_nan());

    let mut previous = 1.0;
    for x in [
        1e-300, 0.001, 0.01, 0.05, 0.2, 0.4, 0.6, 0.819, 0.82, 0.821, 1.0, 1.5, 2.0, 3.0, 5.0,
        10.0, 18.0, 19.0, 20.0, 1e200,
    ] {
        let q = s::kolmogorov(x);
        probability("Kolmogorov SF", q);
        assert!(
            q <= previous,
            "SF must decrease: x={x}, previous={previous}, q={q}"
        );
        previous = q;
    }
    // Anchors within both analytic representations, including the x=0.82
    // switch. Monotonicity alone would not catch an incorrect flat SF.
    for x in [
        0.2, 0.4, 0.6, 0.819, 0.82, 0.821, 1.0, 1.5, 2.0, 3.0, 5.0, 19.0,
    ] {
        let expected = kolmogorov_series_reference(x);
        close(
            "Kolmogorov independent series",
            s::kolmogorov(x),
            expected,
            if expected < 1e-8 { 0.0 } else { 2e-14 },
            if x >= 18.0 { 2e-9 } else { 2e-12 },
        );
    }
    // The second term of the large-x alternating series is exponentially
    // negligible by x=3, so the leading survival term is 2*exp(-2*x^2).
    close(
        "Kolmogorov leading Gaussian tail",
        s::kolmogorov(3.0),
        2.0 * (-18.0_f64).exp(),
        0.0,
        2e-14,
    );
    // x=19 lies in the subnormal survival regime and exercises the split
    // exponential branch; x=20 underflows to an actual zero.
    close(
        "Kolmogorov subnormal asymptotic",
        s::kolmogorov(19.0),
        2.0 * (-722.0_f64).exp(),
        0.0,
        2e-9,
    );
    assert_eq!(s::kolmogorov(20.0), 0.0);

    let small: f32 = s::kolmogorov(0.4_f32);
    close(
        "Kolmogorov f32 promotion",
        small as f64,
        s::kolmogorov(0.4_f64),
        0.0,
        2e-6,
    );
}

#[test]
fn kolmogi_inverse_spans_both_series_and_probability_tails() {
    assert_eq!(s::kolmogi(1.0_f64), 0.0);
    assert_eq!(s::kolmogi(0.0_f64), f64::INFINITY);
    for invalid in [-0.1, 1.1, f64::NEG_INFINITY, f64::INFINITY, f64::NAN] {
        assert!(s::kolmogi(invalid).is_nan(), "inverse domain q={invalid}");
    }

    let mut previous_x = f64::INFINITY;
    for q in [
        1e-300, 1e-100, 1e-20, 1e-6, 0.001, 0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95, 0.99, 0.999999,
    ] {
        let x: f64 = s::kolmogi(q);
        assert!(x.is_finite() && x > 0.0, "inverse q={q}: x={x}");
        assert!(
            x <= previous_x,
            "inverse SF must decrease as q grows: q={q}, x={x}"
        );
        previous_x = x;
        let reconstructed = s::kolmogorov(x);
        // An absolute tolerance suitable for ordinary probabilities would
        // mask a complete underflow at q=1e-100 or 1e-300.
        let atol = if q < 1e-10 { 0.0 } else { 2e-14 };
        close("Kolmogorov SF/inverse", reconstructed, q, atol, 2e-8);
    }

    // Starting from the statistic instead of the probability checks the
    // switch at x=0.82 from the theta-series CDF to the survival series.
    for x in [0.25, 0.5, 0.8, 0.819, 0.821, 1.2, 2.5, 5.0, 10.0] {
        let q: f64 = s::kolmogorov(x);
        let recovered: f64 = s::kolmogi(q);
        close("Kolmogorov inverse/SF", recovered, x, 2e-10, 2e-9);
    }
}

#[test]
fn smirnov_one_sample_and_finite_n_exact_boundary_formulas() {
    // With one observation the one-sided KS statistic is uniform on [0,1].
    for x in [0.0, 0.01, 0.2, 0.5, 0.8, 1.0] {
        close("Smirnov n=1", s::smirnov(1, x), 1.0 - x, 2e-15, 2e-14);
    }
    // When n*x<=1, CDF=x*(1+x)^(n-1); near x=1 the positive finite
    // Smirnov sum has only its j=0 term, giving SF=(1-x)^n.
    for (n, x) in [(2, 0.2), (5, 0.1), (20, 0.04), (1_000_000, 1e-8)] {
        let expected = 1.0_f64 - x * (1.0_f64 + x).powi(n - 1);
        close("Smirnov small nx", s::smirnov(n, x), expected, 3e-14, 2e-12);
    }
    for (n, x) in [(2, 0.75), (5, 0.9), (50, 0.99)] {
        close(
            "Smirnov upper statistic tail",
            s::smirnov(n, x),
            (1.0_f64 - x).powi(n),
            0.0,
            8e-12,
        );
    }
    for n in [1, 5, 20, 1000] {
        let mut previous = 1.0;
        for x in [0.0, 0.002, 0.03, 0.1, 0.2, 0.4, 0.8, 1.0] {
            let q = s::smirnov(n, x);
            probability("finite-sample Smirnov SF", q);
            assert!(
                q <= previous,
                "Smirnov n={n}, x={x}, q={q}, previous={previous}"
            );
            previous = q;
        }
    }
    // Also pin an interior probability for the largest sample size with a
    // survival/inverse identity; the nearby end branches have exact formulas.
    let x = 0.03;
    let q = s::smirnov(1000, x);
    close(
        "Smirnov n=1000 inverse",
        s::smirnovi(1000, q),
        x,
        2e-12,
        2e-9,
    );
}

#[test]
fn smirnov_inverse_survival_roundtrips_and_rejects_invalid_domains() {
    for n in [1, 2, 5, 20, 100] {
        assert_eq!(s::smirnov(n, 0.0), 1.0);
        assert_eq!(s::smirnov(n, 1.0), 0.0);
        assert_eq!(s::smirnovi(n, 0.0), 1.0);
        assert_eq!(s::smirnovi(n, 1.0), 0.0);
        for q in [1e-6, 0.01, 0.2, 0.5, 0.8, 0.99] {
            let x = s::smirnovi(n, q);
            assert!(
                (0.0..=1.0).contains(&x),
                "Smirnov inverse n={n}, q={q}: x={x}"
            );
            close("Smirnov inverse/SF", s::smirnov(n, x), q, 2e-13, 1e-8);
            if n == 1 {
                close("Smirnov uniform inverse", x, 1.0 - q, 2e-12, 2e-12);
            }
        }
    }
    for n in [-1, 0, 1_000_001] {
        assert!(s::smirnov(n, 0.3).is_nan(), "invalid sample size n={n}");
        assert!(s::smirnovi(n, 0.3).is_nan(), "invalid inverse size n={n}");
    }
    for bad in [-0.01, 1.01, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(s::smirnov(5, bad).is_nan(), "Smirnov x={bad}");
        assert!(s::smirnovi(5, bad).is_nan(), "Smirnov inverse q={bad}");
    }
}

#[test]
fn binomial_cdf_survival_floor_and_inverse_special_cases() {
    // k=0 uses log1p/expm1 rather than an incomplete-beta call.
    for p in [0.001, 0.3, 0.9, 1.0] {
        let expected = (1.0_f64 - p).powi(12);
        close(
            "binomial k=0 CDF",
            s::bdtr(0.0, 12, p),
            expected,
            if expected < 1e-6 { 0.0 } else { 5e-15 },
            2e-13,
        );
        close(
            "binomial k=0 SF",
            s::bdtrc(0.0, 12, p),
            1.0 - expected,
            5e-15,
            2e-13,
        );
    }
    for (n, k, p) in [(5, 2.0, 0.3), (20, 7.0, 0.45), (200, 40.0, 0.3)] {
        let cdf = s::bdtr(k, n, p);
        let sf = s::bdtrc(k, n, p);
        probability("binomial CDF", cdf);
        probability("binomial SF", sf);
        close("binomial complement", cdf + sf, 1.0, 2e-12, 2e-12);
        // A nonintegral count means the CDF includes outcomes through floor(k).
        close("binomial floor(k)", s::bdtr(k + 0.75, n, p), cdf, 0.0, 0.0);
        close(
            "binomial floor(k) SF",
            s::bdtrc(k + 0.75, n, p),
            sf,
            0.0,
            0.0,
        );
        // The discrete CDF jump must equal the independently computed PMF.
        let mass = choose(n, k as i32) * p.powf(k) * (1.0 - p).powf(n as f64 - k);
        close(
            "binomial CDF increment / PMF",
            cdf - s::bdtr(k - 1.0, n, p),
            mass,
            2e-14,
            2e-10,
        );
    }
    assert_eq!(s::bdtr(5.0, 5, 0.3_f64), 1.0);
    assert_eq!(s::bdtrc(5.0, 5, 0.3_f64), 0.0);
    assert_eq!(s::bdtrc(-0.1, 5, 0.3_f64), 1.0);
    assert!(s::bdtr(-0.1, 5, 0.3_f64).is_nan());
    for (k, n, p) in [
        (6.0, 5, 0.3_f64),
        (2.0, -1, 0.3),
        (2.0, 5, -0.1),
        (2.0, 5, 1.1),
    ] {
        assert!(s::bdtr(k, n, p).is_nan());
        assert!(s::bdtrc(k, n, p).is_nan());
    }

    // The zero-success-count inverse has an exact closed form, while k>0
    // uses two complementary inverse-beta branches depending on the CDF.
    for y in [0.001, 0.3, 0.9] {
        let p = s::bdtri(0.0_f64, 12, y);
        close("binomial inverse k=0", s::bdtr(0.0, 12, p), y, 2e-14, 2e-12);
        close(
            "binomial zero-count formula",
            p,
            1.0 - y.powf(1.0 / 12.0),
            2e-14,
            2e-12,
        );
    }
    for p in [0.1, 0.3, 0.6] {
        let target = s::bdtr(3.0_f64, 20, p);
        let recovered = s::bdtri(3.0, 20, target);
        close("binomial inverse-beta", recovered, p, 2e-12, 2e-10);
    }
    assert!(s::bdtri(3.0_f64, 3, 0.5).is_nan());
    assert!(s::bdtri(-1.0_f64, 10, 0.5).is_nan());
    assert!(s::bdtri(2.0_f64, 10, 1.1).is_nan());
}

#[test]
fn negative_binomial_and_other_distribution_complements() {
    // P(no failures before n successes)=p^n; the survival path uses expm1.
    for p in [0.01, 0.3, 0.9, 1.0] {
        let lower = s::nbdtr(0, 7, p);
        let upper = s::nbdtrc(0, 7, p);
        close(
            "negative binomial zero failures",
            lower,
            p.powi(7),
            0.0,
            2e-12,
        );
        close(
            "negative binomial zero complement",
            lower + upper,
            1.0,
            3e-15,
            2e-12,
        );
    }
    for (k, n, p) in [(2, 3, 0.3), (7, 10, 0.7), (50, 25, 0.4)] {
        let cdf = s::nbdtr(k, n, p);
        let sf = s::nbdtrc(k, n, p);
        probability("negative binomial CDF", cdf);
        probability("negative binomial SF", sf);
        close("negative binomial complement", cdf + sf, 1.0, 2e-12, 2e-12);
        close(
            "negative binomial inverse",
            s::nbdtri(k, n, cdf),
            p,
            2e-11,
            2e-9,
        );
        let mass = choose(n + k - 1, k) * (1.0 - p).powi(k) * p.powi(n);
        close(
            "negative binomial CDF increment / PMF",
            cdf - s::nbdtr(k - 1, n, p),
            mass,
            2e-13,
            2e-10,
        );
    }
    for (k, n, p) in [
        (-1, 5, 0.3_f64),
        (2, 0, 0.3),
        (2, -1, 0.3),
        (2, 5, -0.1),
        (2, 5, 1.1),
    ] {
        assert!(s::nbdtr(k, n, p).is_nan());
        assert!(s::nbdtrc(k, n, p).is_nan());
        assert!(s::nbdtri(k, n, p).is_nan());
    }

    for (k, mean) in [(0.0, 0.01), (3.0, 2.0), (20.0, 30.0)] {
        close(
            "Poisson complement",
            s::pdtr(k, mean) + s::pdtrc(k, mean),
            1.0,
            3e-12,
            3e-12,
        );
    }
    // Independent degree-zero / two-degree closed forms rule out a pair of
    // mutually complementary but jointly incorrect CDF and SF kernels.
    for mean in [0.01, 2.0, 30.0] {
        close(
            "Poisson P(X=0)",
            s::pdtr(0.0, mean),
            (-mean).exp(),
            0.0,
            3e-13,
        );
        close(
            "Poisson P(X>0)",
            s::pdtrc(0.0, mean),
            -(-mean).exp_m1(),
            2e-14,
            3e-13,
        );
    }
    for (df, x) in [(1.0, 0.2), (5.0, 3.0), (40.0, 50.0)] {
        close(
            "chi-square complement",
            s::chdtr(df, x) + s::chdtrc(df, x),
            1.0,
            3e-12,
            3e-12,
        );
    }
    for x in [0.1, 2.0, 20.0] {
        close(
            "chi-square df=2 CDF",
            s::chdtr(2.0, x),
            -(-x / 2.0).exp_m1(),
            2e-14,
            3e-13,
        );
        close(
            "chi-square df=2 SF",
            s::chdtrc(2.0, x),
            (-x / 2.0).exp(),
            0.0,
            3e-13,
        );
    }
    for (dfn, dfd, x) in [(1.0, 2.0, 0.1), (3.0, 8.0, 1.0), (20.0, 40.0, 3.0)] {
        close(
            "F complement",
            s::fdtr(dfn, dfd, x) + s::fdtrc(dfn, dfd, x),
            1.0,
            3e-12,
            3e-12,
        );
    }
    for x in [0.1, 1.0, 10.0] {
        close(
            "F(2,2) CDF",
            s::fdtr(2.0, 2.0, x),
            x / (1.0 + x),
            2e-14,
            3e-13,
        );
        close(
            "F(2,2) SF",
            s::fdtrc(2.0, 2.0, x),
            1.0 / (1.0 + x),
            2e-14,
            3e-13,
        );
    }
    for (a, b, x) in [(1.0, 0.5, 0.1), (2.0, 3.0, 0.5), (10.0, 5.0, 10.0)] {
        close(
            "gamma complement",
            s::gdtr(a, b, x) + s::gdtrc(a, b, x),
            1.0,
            3e-12,
            3e-12,
        );
    }
    for x in [0.1, 1.0, 5.0] {
        close(
            "gamma shape=1 CDF",
            s::gdtr(2.0, 1.0, x),
            -(-2.0 * x).exp_m1(),
            2e-14,
            3e-13,
        );
        close(
            "gamma shape=1 SF",
            s::gdtrc(2.0, 1.0, x),
            (-2.0 * x).exp(),
            0.0,
            3e-13,
        );
    }
}
