mod common;
use common::{number, rows};

use sci::{complex::Complex, differentiate, fft, optimize, special, stats};

#[track_caller]
fn assert_close(actual: f64, expected: f64, relative_tolerance: f64, absolute_tolerance: f64) {
    let difference = (actual - expected).abs();
    let tolerance = absolute_tolerance.max(relative_tolerance * expected.abs());
    assert!(
        approx::abs_diff_eq!(actual, expected, epsilon = tolerance),
        "actual {actual:.17e} differs from SciPy {expected:.17e} by {difference:.3e} (tolerance {tolerance:.3e})"
    );
}

#[test]
fn differentiation_matches_scipy() {
    for row in rows(include_str!("fixtures/differentiation.csv")) {
        let x = number(row[1]);
        let step = number(row[2]);
        let expected = number(row[3]);
        let actual = match row[0] {
            "sin" => differentiate::derivative(f64::sin, x, step),
            "exp" => differentiate::derivative(f64::exp, x, step),
            "cubic" => differentiate::derivative(|value| value.powi(3) - 2.0 * value, x, step),
            function => panic!("unknown fixture function {function}"),
        }
        .unwrap();
        assert_close(actual, expected, 1e-10, 1e-10);
    }
}

#[test]
fn special_functions_match_scipy() {
    for row in rows(include_str!("fixtures/special.csv")) {
        let x: f64 = number(row[1]);
        let y: f64 = number(row[2]);
        let expected = number(row[3]);
        let actual = match row[0] {
            "gamma" => special::gamma::gamma(x),
            "digamma" => special::gamma::digamma(x),
            "erf" => special::elementary::erf(x),
            "gammaln" => special::gamma::gammaln(x),
            "trigamma" => special::gamma::trigamma(x),
            "erfc" => special::elementary::erfc(x),
            "erfinv" => special::elementary::erfinv(x),
            "betaln" => special::beta::betaln(x, y),
            "lambertw" => special::elementary::lambertw(x),
            "lambertwm1" => special::elementary::lambertwm1(x),
            "gammainc" => special::gamma::gammainc(x, y),
            "beta" => special::beta::beta(x, y),
            "ellipk" => special::elliptic::ellipk(x).unwrap(),
            "ellipe" => special::elliptic::ellipe(x).unwrap(),
            function => panic!("unknown fixture function {function}"),
        };
        assert_close(actual, expected, 2e-12, 2e-14);
    }
}

#[test]
fn incomplete_beta_matches_scipy() {
    for row in rows(include_str!("fixtures/incomplete_beta.csv")) {
        let a = number(row[0]);
        let b = number(row[1]);
        let x = number(row[2]);
        let expected = number(row[3]);
        assert_close(special::beta::betainc(a, b, x), expected, 2e-12, 2e-14);
        assert_close(
            special::beta::betaincinv(a, b, expected),
            number(row[4]),
            2e-12,
            2e-14,
        );
    }
}

#[test]
fn fft_matches_scipy() {
    let data = rows(include_str!("fixtures/fft.csv"))
        .map(|row| {
            (
                Complex {
                    re: number(row[1]),
                    im: number(row[2]),
                },
                Complex {
                    re: number(row[3]),
                    im: number(row[4]),
                },
            )
        })
        .collect::<Vec<_>>();
    let inputs = data.iter().map(|(input, _)| *input).collect::<Vec<_>>();
    let actual = fft::fourier::fft(&inputs);

    for (value, (_, expected)) in actual.iter().zip(&data) {
        assert_close(value.re, expected.re, 1e-13, 1e-13);
        assert_close(value.im, expected.im, 1e-13, 1e-13);
    }
}

#[test]
fn statistics_match_scipy() {
    for row in rows(include_str!("fixtures/statistics.csv")) {
        let mean = number(row[1]);
        let std_dev = number(row[2]);
        let x = number(row[3]);
        let expected = number(row[4]);
        let actual = match row[0] {
            "normal_pdf" => stats::distributions::normal_pdf(x, mean, std_dev).unwrap(),
            "normal_cdf" => stats::distributions::normal_cdf(x, mean, std_dev).unwrap(),
            "normal_ppf" => stats::distributions::normal_ppf(x, mean, std_dev).unwrap(),
            function => panic!("unknown fixture function {function}"),
        };
        // Statrs uses an approximate erf for its normal CDF.
        let tolerance = if row[0] == "normal_cdf" { 1e-10 } else { 2e-12 };
        assert_close(actual, expected, tolerance, 2e-15);
    }
}

#[test]
fn jacobian_and_linear_algebra_match_scipy() {
    let data = rows(include_str!("fixtures/jacobian.csv")).collect::<Vec<_>>();
    assert_eq!(data.len(), 2);
    let point = [number(data[0][1]), number(data[0][2])];
    let jacobian = differentiate::jacobian(
        |x| vec![x[0] * x[1], x[0] + x[1]],
        &point,
        number(data[0][3]),
    )
    .unwrap();
    let direction = [number(data[0][6]), number(data[0][7])];
    let product = sci::linalg::basic::matvec(&jacobian, &direction).unwrap();
    for (index, row) in data.iter().enumerate() {
        assert_close(jacobian[index][0], number(row[4]), 1e-10, 1e-10);
        assert_close(jacobian[index][1], number(row[5]), 1e-10, 1e-10);
        assert_close(product[index], number(row[8]), 1e-10, 1e-10);
    }
    let data = rows(include_str!("fixtures/linalg.csv")).collect::<Vec<_>>();
    let matrix = data
        .iter()
        .map(|row| vec![number(row[1]), number(row[2])])
        .collect::<Vec<_>>();
    let transpose = sci::linalg::basic::transpose(&matrix).unwrap();
    let gram = sci::linalg::basic::matmul(&matrix, &transpose).unwrap();
    for (index, row) in data.iter().enumerate() {
        assert_close(gram[index][0], number(row[3]), 1e-14, 1e-14);
        assert_close(gram[index][1], number(row[4]), 1e-14, 1e-14);
    }
}

#[test]
fn fft_complex_and_statistics_interoperate() {
    let expected = rows(include_str!("fixtures/spectrum_statistics.csv")).collect::<Vec<_>>();
    let spectrum = fft::real::rfft(&[1.0, 2.0, 3.0, 4.0]);
    let amplitudes = spectrum
        .into_iter()
        .map(sci::complex::abs)
        .collect::<Vec<_>>();
    assert_eq!(amplitudes.len(), expected.len());
    for (actual, row) in amplitudes.iter().zip(&expected) {
        assert_close(*actual, number(row[1]), 1e-13, 1e-13);
    }
    assert_close(
        stats::descriptive::mean(&amplitudes).unwrap(),
        number(expected[0][2]),
        1e-13,
        1e-13,
    );
    assert_close(
        stats::descriptive::variance(&amplitudes, 1).unwrap(),
        number(expected[0][3]),
        1e-13,
        1e-13,
    );
}

#[test]
fn optimization_matches_scipy() {
    for row in rows(include_str!("fixtures/optimization.csv")) {
        assert_eq!(row[0], "sphere");
        let start = vec![number(row[1]), number(row[2])];
        let expected = [number(row[3]), number(row[4])];
        let expected_cost = number(row[5]);
        let result = optimize::minimize::minimize(
            |x| sci::linalg::basic::dot(x, x).unwrap(),
            &start,
            optimize::minimize::Options {
                tolerance: 1e-10,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(result.status, optimize::minimize::Status::Converged);

        for (actual, expected) in result.point.iter().zip(expected) {
            assert_close(*actual, expected, 0.0, 1e-6);
        }
        assert_close(result.cost, expected_cost, 0.0, 1e-12);
    }
}
