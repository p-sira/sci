mod common;
use common::close;
use common::parsed_call;
use common::{csv_row, matrix, vector};

use sci::{cluster::vq, constants as c, integrate as i, interpolate as p};

fn check_vector(actual: &[f64], expected: &str) {
    let expected = vector(expected);
    assert_eq!(actual.len(), expected.len());
    for (&a, b) in actual.iter().zip(expected) {
        close(a, b, 3e-12, 3e-12);
    }
}
fn check_matrix(actual: &[Vec<f64>], expected: &str) {
    let expected = matrix(expected);
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.len(), b.len());
        for (&a, b) in a.iter().zip(b) {
            close(a, b, 3e-12, 3e-12);
        }
    }
}

#[test]
fn public_numerical_apis_match_scipy() {
    let inputs: Vec<_> = common::lines(include_str!("fixtures/numerics_inputs.csv")).collect();
    let expected: Vec<_> = common::lines(include_str!("fixtures/numerics_expected.csv")).collect();
    assert_eq!(inputs.len(), expected.len());
    let mut covered = std::collections::BTreeSet::new();
    for (input, expected) in inputs.into_iter().zip(expected) {
        let c: Vec<_> = common::fields(input);
        let e: Vec<_> = common::fields(expected);
        assert_eq!(c[0], e[0]);
        covered.insert((c[1], c[2]));
        let a = vector(if c[2] == "fixed_quad" {
            ""
        } else {
            c[3].split(':').next().unwrap()
        });
        match (c[1], c[2]) {
            ("integrate", api) => {
                let x = vector(c[4]);
                let x = (!c[4].is_empty()).then_some(x.as_slice());
                let dx: f64 = c[5].parse().unwrap_or(1.0);
                let initial = (!c[6].is_empty()).then(|| c[6].parse().unwrap());
                let check_integral = |actual: &[f64]| {
                    let expected = vector(e[1]);
                    assert_eq!(actual.len(), expected.len(), "{input}");
                    for (&actual, expected) in actual.iter().zip(expected) {
                        if c[0].starts_with("integration_scaled_") {
                            // Absolute error near cancellation scales with the
                            // integrated data, rather than the small residual.
                            let width_scale = x.map_or(dx.abs(), |x| {
                                x.windows(2)
                                    .fold(0_f64, |m, pair| m.max((pair[1] - pair[0]).abs()))
                            });
                            let data_scale =
                                a.iter().fold(0_f64, |m, y| m.max(y.abs())) * width_scale;
                            let constant = a.iter().all(|value| *value == a[0]);
                            let tolerance = if constant {
                                (3e-12 * expected.abs()).max(f64::from_bits(1))
                            } else {
                                ((32.0 * f64::EPSILON * data_scale) * a.len() as f64
                                    + 3e-12 * expected.abs())
                                .max(f64::from_bits(1))
                            };
                            if expected != 0.0 && (constant || expected.abs() < 1e-100) {
                                assert_ne!(actual, 0.0, "{input}: lost a nonzero integral");
                            }
                            assert!(
                                actual.is_finite()
                                    && approx::abs_diff_eq!(actual, expected, epsilon = tolerance),
                                "{input}: {actual} != {expected}"
                            );
                        } else {
                            close(actual, expected, 3e-12, 3e-12);
                        }
                    }
                };
                match api {
                    "trapezoid" => check_integral(&[i::trapezoid(&a, x, dx).unwrap()]),
                    "simpson" => check_integral(&[i::simpson(&a, x, dx).unwrap()]),
                    "cumulative_trapezoid" => {
                        check_integral(&i::cumulative_trapezoid(&a, x, dx, initial).unwrap())
                    }
                    "cumulative_simpson" => {
                        check_integral(&i::cumulative_simpson(&a, x, dx, initial).unwrap())
                    }
                    "romb" => {
                        let actual = i::romb(&a, c[4].parse().unwrap()).unwrap();
                        let expected: f64 = e[1].parse().unwrap();
                        if c[0].starts_with("romb_") && expected != 0.0 {
                            assert_ne!(actual, 0.0, "{input}");
                            assert!(
                                (actual - expected).abs()
                                    <= (3e-12 * expected.abs()).max(f64::from_bits(1)),
                                "{input}: {actual} != {expected}"
                            );
                        } else {
                            close(actual, expected, 3e-12, 3e-12);
                        }
                    }
                    "newton_cotes" => {
                        let (w, b) = i::newton_cotes(&a).unwrap();
                        check_vector(&w, e[1]);
                        close(b, e[2].parse().unwrap(), 3e-12, 3e-12);
                    }
                    "fixed_quad" => {
                        let f: fn(f64) -> f64 = match c[3] {
                            "sin" => f64::sin,
                            "exp" => f64::exp,
                            "square" => |x| x * x,
                            "cubic" => |x| x * x * x,
                            "constant" => |_| 1.0,
                            "huge_constant" => |_| 1e308,
                            "tiny_constant" => |_| 1e-308,
                            "subnormal_constant" => |_| f64::from_bits(1),
                            "scaled_square" => |x| (x / 1e308).powi(2) * 1e-308,
                            _ => panic!(),
                        };
                        let actual = i::fixed_quad(
                            f,
                            c[4].parse().unwrap(),
                            c[5].parse().unwrap(),
                            c[6].parse().unwrap(),
                        )
                        .unwrap();
                        let expected: f64 = e[1].parse().unwrap();
                        if c[0].starts_with("gauss_") {
                            if expected != 0.0 {
                                assert_ne!(actual, 0.0, "{input}");
                            }
                            assert!(
                                actual.is_finite()
                                    && (actual - expected).abs()
                                        <= (3e-12 * expected.abs()).max(f64::from_bits(1)),
                                "{input}: {actual} != {expected}"
                            );
                        } else {
                            close(actual, expected, 3e-12, 3e-12);
                        }
                    }
                    _ => panic!("{api}"),
                }
            }
            ("interpolate", api) => {
                if api == "pade" {
                    let (num, den) = p::pade(
                        &a,
                        c[4].parse().unwrap(),
                        (!c[5].is_empty()).then(|| c[5].parse().unwrap()),
                    )
                    .unwrap();
                    check_vector(&num, e[1]);
                    check_vector(&den, e[2]);
                } else {
                    let b = vector(c[4]);
                    let query = vector(c[5]);
                    let result = match api {
                        "barycentric_interpolate" => p::barycentric_interpolate(&a, &b, &query),
                        "krogh_interpolate" => p::krogh_interpolate(&a, &b, &query),
                        "pchip_interpolate" => p::pchip_interpolate(&a, &b, &query),
                        "lagrange" => p::lagrange(&a, &b),
                        _ => panic!("{api}"),
                    };
                    let result = result.unwrap_or_else(|error| {
                        panic!("interpolation case {} ({api}) failed: {error}", c[0])
                    });
                    check_vector(&result, e[1]);
                }
            }
            ("cluster", api) => {
                let obs = matrix(c[3]);
                let codes = matrix(c[4]);
                match api {
                    "whiten" => check_matrix(&vq::whiten(&obs).unwrap(), e[1]),
                    "vq" | "py_vq" => {
                        let (labels, distances) = if api == "vq" {
                            vq::vq(&obs, &codes)
                        } else {
                            vq::py_vq(&obs, &codes)
                        }
                        .unwrap();
                        check_vector(&labels.iter().map(|v| *v as f64).collect::<Vec<_>>(), e[1]);
                        check_vector(&distances, e[2]);
                    }
                    "kmeans2" => {
                        let (centroids, labels) =
                            vq::kmeans2(&obs, &codes, c[5].parse().unwrap()).unwrap();
                        check_matrix(&centroids, e[1]);
                        check_vector(&labels.iter().map(|v| *v as f64).collect::<Vec<_>>(), e[2]);
                    }
                    "kmeans" => {
                        let (centroids, distortion) =
                            vq::kmeans(&obs, &codes, c[5].parse().unwrap(), c[6].parse().unwrap())
                                .unwrap();
                        check_matrix(&centroids, e[1]);
                        close(distortion, e[2].parse().unwrap(), 3e-12, 3e-12);
                    }
                    _ => panic!("{api}"),
                }
            }
            _ => panic!(),
        }
    }
    assert_eq!(covered.len(), 17);
}

fn scale(s: &str) -> c::TemperatureScale {
    match s {
        "Celsius" => c::TemperatureScale::Celsius,
        "Kelvin" => c::TemperatureScale::Kelvin,
        "Fahrenheit" => c::TemperatureScale::Fahrenheit,
        "Rankine" => c::TemperatureScale::Rankine,
        _ => panic!(),
    }
}
#[test]
fn constants_match_scipy_and_full_codata_snapshot() {
    let inputs: Vec<_> = common::lines(include_str!("fixtures/constants_inputs.csv")).collect();
    let expected: Vec<_> = common::lines(include_str!("fixtures/constants_expected.csv")).collect();
    assert_eq!(inputs.len(), expected.len());
    let mut covered = std::collections::BTreeSet::new();
    for (input, expected) in inputs.into_iter().zip(expected) {
        let row = csv_row(input);
        let e = csv_row(expected);
        assert_eq!(row[0], e[0]);
        covered.insert(row[0].clone());
        let value = match row[0].as_str() {
            "value" => c::value(&row[1]).unwrap(),
            "precision" => c::precision(&row[1]).unwrap(),
            "unit" => {
                assert_eq!(c::unit(&row[1]).unwrap(), e[1]);
                continue;
            }
            "find" => {
                assert_eq!(c::find(Some(&row[1])).join(";"), e[1]);
                continue;
            }
            "convert_temperature" => {
                c::convert_temperature(row[1].parse().unwrap(), scale(&row[2]), scale(&row[3]))
            }
            "lambda2nu" => parsed_call!(c::lambda2nu; row[1]),
            "nu2lambda" => parsed_call!(c::nu2lambda; row[1]),
            _ => panic!(),
        };
        close(value, e[1].parse().unwrap(), 3e-12, 3e-12);
    }
    assert_eq!(covered.len(), 7);
    let table: Vec<_> = common::lines(include_str!("fixtures/constants_table.csv")).collect();
    assert_eq!(table.len(), c::PHYSICAL_CONSTANTS.len());
    for (line, entry) in table.into_iter().zip(c::PHYSICAL_CONSTANTS) {
        let fields = csv_row(line);
        let value: f64 = fields[1].parse().unwrap();
        let uncertainty: f64 = fields[3].parse().unwrap();
        assert_eq!(entry.name, fields[0]);
        assert_eq!(entry.value, value);
        assert_eq!(entry.unit, fields[2]);
        assert_eq!(entry.uncertainty, uncertainty);
        assert_eq!(c::value(&fields[0]).unwrap(), value);
        assert_eq!(c::unit(&fields[0]).unwrap(), fields[2]);
        assert_eq!(c::precision(&fields[0]).unwrap(), uncertainty / value);
    }
}

#[test]
fn integration_domains_boundaries_and_properties() {
    use sci::Error;
    assert_eq!(
        i::trapezoid(&[], None, 1.0),
        Err(Error::InvalidInput("sample count must be in 1..=1000001"))
    );
    assert_eq!(
        i::simpson(&[1.0, 2.0], Some(&[0.0]), 1.0),
        Err(Error::DimensionMismatch)
    );
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(i::trapezoid(&[bad], None, 1.0).is_err());
        assert!(i::simpson(&[1.0], None, bad).is_err());
        assert!(i::fixed_quad(|_| bad, 0.0, 1.0, 2).is_err());
        assert!(i::fixed_quad(|x| x, bad, 1.0, 2).is_err());
    }
    assert!(i::simpson(&[1.0, 2.0, 3.0], Some(&[0.0, 0.0, 1.0]), 1.0).is_err());
    assert!(i::simpson(&[1.0, 2.0, 3.0], Some(&[0.0, 1.0, 0.0]), 1.0).is_err());
    assert!(i::cumulative_simpson(&[1.0, 2.0], Some(&[1.0, 0.0]), 1.0, None).is_err());
    assert!(i::cumulative_simpson(&[1.0, 2.0, 3.0], None, -1.0, None).is_err());
    assert!(i::cumulative_simpson(&[1.0], None, 1.0, Some(f64::NAN)).is_err());
    assert!(i::cumulative_trapezoid(&[1.0], None, 1.0, Some(1.0)).is_err());
    assert!(i::romb(&[0.0; 4], 1.0).is_err());
    assert!(i::romb(&[1.0], 1.0).is_err());
    for n in [0, 257, usize::MAX] {
        assert!(i::fixed_quad(|x| x, 0.0, 1.0, n).is_err());
    }
    assert_eq!(
        i::fixed_quad(|_| panic!("zero-width callback"), 2.0, 2.0, 1).unwrap(),
        0.0
    );
    for nodes in [
        &[0.0][..],
        &[0.0, 0.0, 2.0],
        &[1.0, 2.0],
        &[0.0, f64::NAN, 2.0],
    ] {
        assert!(i::newton_cotes(nodes).is_err());
    }
    assert!(i::newton_cotes(&(0..=13).map(|x| x as f64).collect::<Vec<_>>()).is_err());
    let x = [0.0, 0.3, 1.0, 1.4, 2.0];
    let y: Vec<_> = x.iter().map(|v| v * v).collect();
    close(
        i::simpson(&y, Some(&x), 1.0).unwrap(),
        8.0 / 3.0,
        3e-12,
        3e-12,
    );
    let integral = i::cumulative_simpson(&y, Some(&x), 1.0, Some(0.0)).unwrap();
    for (&x, value) in x.iter().zip(integral) {
        close(value, x * x * x / 3.0, 3e-12, 3e-12);
    }
    for n in 1..=8 {
        let (nodes, weights) = sci::special::roots_legendre(n).unwrap();
        let samples: Vec<_> = nodes.iter().map(|v| v.powi((2 * n - 2) as i32)).collect();
        close(
            i::fixed_quad(|v| v.powi((2 * n - 2) as i32), -1.0, 1.0, n).unwrap(),
            sci::linalg::dot(&weights, &samples).unwrap(),
            3e-12,
            3e-12,
        );
    }
    for n in 1..=8 {
        let nodes: Vec<_> = (0..=n).map(|v| v as f64).collect();
        let (w, _) = i::newton_cotes(&nodes).unwrap();
        for k in 0..=n {
            let y: Vec<_> = nodes.iter().map(|v| v.powi(k)).collect();
            close(
                sci::linalg::dot(&w, &y).unwrap(),
                (n as f64).powi(k + 1) / (k + 1) as f64,
                3e-12,
                3e-12,
            );
        }
    }
}

#[test]
fn interpolation_preserves_finite_results_at_extreme_scales() {
    let tiny = f64::from_bits(1);
    let values =
        p::barycentric_interpolate(&[0.0, 1.0], &[2.0, 3.0], &[tiny, -tiny, 1e-310]).unwrap();
    assert_eq!(values, vec![2.0; 3]);

    for scale in [1e-310, -1e-310, 1.0, -1.0, 1e308, -1e308] {
        let query = [0.25, 0.5, 0.75, 1.5];
        let values =
            p::barycentric_interpolate(&[0.0, 1.0, 2.0], &[scale, scale, scale], &query).unwrap();
        assert_eq!(values, vec![scale; query.len()]);
        let values =
            p::pchip_interpolate(&[0.0, 0.5, 1.0], &[0.0, 0.5 * scale, scale], &query[..3])
                .unwrap();
        for (&t, value) in query.iter().zip(values) {
            // Normalize before comparing so subnormal results receive a useful tolerance.
            close(value / scale, t, 3e-12, 3e-12);
        }
    }

    for zero in [-0.0, 0.0] {
        let values = p::pchip_interpolate(&[-0.0, 1.0], &[-0.0, 1.0], &[zero]).unwrap();
        assert_eq!(values[0].to_bits(), (-0.0_f64).to_bits());
        let values = p::barycentric_interpolate(&[-0.0, 1.0], &[-0.0, 1.0], &[zero]).unwrap();
        assert_eq!(values[0].to_bits(), (-0.0_f64).to_bits());
    }
    let values =
        p::barycentric_interpolate(&[-1.0, 0.0, 1.0], &[f64::MAX; 3], &[-0.5, 0.5]).unwrap();
    assert_eq!(values, vec![f64::MAX; 2]);
    assert!(p::barycentric_interpolate(&[0.0, 1.0], &[0.0, 1e308], &[2.0]).is_err());
    assert!(p::pchip_interpolate(&[0.0, 1.0], &[0.0, 1e308], &[2.0]).is_err());
    // The endpoint derivative is 2e308, beyond f64's representable range.
    assert!(p::pchip_interpolate(&[0.0, 0.5, 1.0], &[0.0, 5e307, 0.0], &[0.25]).is_err());
}

#[test]
fn interpolation_domains_and_independent_identities() {
    assert!(p::barycentric_interpolate(&[0.0, 0.0], &[1.0, 2.0], &[]).is_err());
    assert!(p::krogh_interpolate(&[0.0, 1.0, 0.0], &[1.0, 2.0, 3.0], &[]).is_err());
    assert!(p::lagrange(&[0.0, 0.0], &[1.0, 2.0]).is_err());
    assert!(p::pchip_interpolate(&[1.0, 0.0], &[1.0, 2.0], &[]).is_err());
    assert!(p::pchip_interpolate(&[0.0], &[1.0], &[]).is_err());
    assert!(p::pade(&[1.0, 1.0], 2, None).is_err());
    assert!(p::pade(&[1.0, 1.0], 1, Some(1)).is_err());
    assert!(p::pade(&[0.0, 0.0, 1.0], 1, None).is_err());
    for bad in [f64::NAN, f64::INFINITY] {
        assert!(p::barycentric_interpolate(&[0.0], &[bad], &[]).is_err());
        assert!(p::krogh_interpolate(&[bad], &[0.0], &[]).is_err());
        assert!(p::pchip_interpolate(&[0.0, 1.0], &[0.0, 1.0], &[bad]).is_err());
        assert!(p::pade(&[bad], 0, None).is_err());
    }
    let nodes = [-1.0, 0.0, 1.0, 2.0];
    let data: Vec<_> = nodes.iter().map(|x| x * x * x - 2.0 * x + 1.0).collect();
    let query = [-2.0, -0.3, 0.2, 1.2, 3.0];
    let bary = p::barycentric_interpolate(&nodes, &data, &query).unwrap();
    let krogh = p::krogh_interpolate(&nodes, &data, &query).unwrap();
    let coefficients = p::lagrange(&nodes, &data).unwrap();
    for ((&x, b), k) in query.iter().zip(bary).zip(krogh) {
        let expected = x * x * x - 2.0 * x + 1.0;
        close(b, expected, 3e-12, 3e-12);
        close(k, expected, 3e-12, 3e-12);
        close(
            coefficients.iter().fold(0.0, |value, c| value * x + c),
            expected,
            3e-12,
            3e-12,
        );
    }
    let hermite =
        p::krogh_interpolate(&[0.0, 0.0, 0.0, 0.0], &[1.0, -2.0, 0.0, 6.0], &query).unwrap();
    for (&x, value) in query.iter().zip(hermite) {
        close(value, x * x * x - 2.0 * x + 1.0, 3e-12, 3e-12);
    }
    // Reciprocal slopes would overflow here. Linear reproduction remains valid.
    let tiny = p::pchip_interpolate(&[0.0, 1.0, 2.0], &[0.0, 1e-310, 2e-310], &[1.5]).unwrap();
    close(tiny[0] / 1e-310, 1.5, 3e-12, 3e-12);
    assert!(p::barycentric_interpolate(&[0.0; 257], &[0.0; 257], &[]).is_err());
    assert!(p::krogh_interpolate(&[0.0; 33], &[0.0; 33], &[]).is_err());
    assert!(p::lagrange(&[0.0; 33], &[0.0; 33]).is_err());
    assert!(p::pade(&[1.0; 66], 0, None).is_err());
    let query: Vec<_> = (0..=200).map(|v| v as f64 / 100.0).collect();
    let values =
        p::pchip_interpolate(&[0.0, 0.2, 1.0, 2.0], &[0.0, 1.0, 1.1, 3.0], &query).unwrap();
    assert!(values.windows(2).all(|v| v[0] <= v[1]));
    assert_eq!(values[0], 0.0);
    assert_eq!(values[200], 3.0);
    for (m, n) in [(1, 1), (2, 2), (0, 4)] {
        let t = [1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0];
        let (mut num, mut den) = p::pade(&t, m, Some(n)).unwrap();
        num.reverse();
        den.reverse();
        for k in 0..=m + n {
            let product: f64 = (0..=k.min(m)).map(|j| den[j] * t[k - j]).sum();
            close(product, num.get(k).copied().unwrap_or(0.0), 3e-12, 3e-12);
        }
    }
}

#[test]
fn clustering_validation_ties_and_interoperability() {
    assert!(vq::whiten(&[]).is_err());
    assert!(vq::vq(&[vec![1.0]], &[]).is_err());
    assert!(vq::vq(&[vec![1.0]], &[vec![1.0, 2.0]]).is_err());
    assert!(vq::whiten(&[vec![1.0], vec![]]).is_err());
    assert!(vq::whiten(&[vec![f64::NAN]]).is_err());
    assert!(vq::kmeans2(&[vec![1.0]], &[vec![0.0]], 0).is_err());
    assert!(vq::kmeans(&[vec![1.0]], &[vec![0.0]], -1.0, 10).is_err());
    assert!(vq::kmeans(&[vec![1.0]], &[vec![0.0]], 0.0, usize::MAX).is_err());
    assert!(vq::kmeans2(&[vec![1.0]], &[vec![0.0]], 10_000_001).is_err());
    let (labels, d) = vq::vq(&[vec![2.0, 0.0]], &[vec![0.0, 0.0], vec![4.0, 0.0]]).unwrap();
    assert_eq!(labels, vec![0]);
    assert_eq!(d, vec![2.0]);
    let (_, d) = vq::vq(&[vec![1e200, 0.0]], &[vec![0.0, 0.0]]).unwrap();
    assert_eq!(d, vec![1e200]);
    let obs = vec![
        vec![0.0, 0.0],
        vec![2.0, 0.0],
        vec![8.0, 1.0],
        vec![10.0, 1.0],
    ];
    let whitened = vq::whiten(&obs).unwrap();
    close(
        sci::stats::std(&whitened.iter().map(|r| r[0]).collect::<Vec<_>>(), 0).unwrap(),
        1.0,
        3e-12,
        3e-12,
    );
    let (codes, distortion) =
        vq::kmeans(&obs, &[vec![0.0, 0.0], vec![10.0, 1.0]], 1e-10, 20).unwrap();
    let (labels, distances) = vq::vq(&obs, &codes).unwrap();
    assert_eq!(labels, vec![0, 0, 1, 1]);
    close(
        distortion,
        sci::stats::mean(&distances).unwrap(),
        3e-12,
        3e-12,
    );
    for (row, &label) in obs.iter().zip(&labels) {
        close(
            sci::spatial::distance::euclidean(row, &codes[label], None).unwrap(),
            1.0,
            3e-12,
            3e-12,
        );
    }
}

#[test]
fn constants_special_values_round_trips_and_aliases() {
    assert!(c::value("missing-key").is_err());
    assert!(c::unit("missing-key").is_err());
    assert!(c::precision("missing-key").is_err());
    assert_eq!(c::lambda2nu(-0.0), f64::NEG_INFINITY);
    assert_eq!(c::nu2lambda(-0.0), f64::NEG_INFINITY);
    assert!(c::lambda2nu(f64::INFINITY).is_sign_positive());
    assert!(c::nu2lambda(f64::NEG_INFINITY).is_sign_negative());
    assert!(c::lambda2nu(f64::NAN).is_nan());
    assert!(c::nu2lambda(f64::NAN).is_nan());
    for wavelength in [1e-9, 1.0, 1e6] {
        close(
            c::nu2lambda(c::lambda2nu(wavelength)),
            wavelength,
            3e-12,
            3e-12,
        );
    }
    use c::TemperatureScale::*;
    for from in [Celsius, Kelvin, Fahrenheit, Rankine] {
        for to in [Celsius, Kelvin, Fahrenheit, Rankine] {
            for x in [-273.15, 0.0, 100.0, 1e6] {
                close(
                    c::convert_temperature(c::convert_temperature(x, from, to), to, from),
                    x,
                    3e-12,
                    3e-12,
                );
            }
            assert!(c::convert_temperature(f64::NAN, from, to).is_nan());
            assert_eq!(
                c::convert_temperature(f64::INFINITY, from, to),
                f64::INFINITY
            );
        }
    }
    assert_eq!(c::c, c::speed_of_light);
    assert_eq!(c::h, c::Planck);
    assert_eq!(c::k, c::Boltzmann);
    assert_eq!(c::r#dyn, c::dyne);
    assert_eq!(c::Avogadro, c::N_A);
    assert_eq!(c::find(None), c::find(Some("")));
}

#[test]
fn integration_scaling_and_compensation_preserve_independent_identities() {
    let tiny = f64::from_bits(1);
    assert_eq!(i::trapezoid(&[tiny, tiny], None, 1.0).unwrap(), tiny);
    assert_eq!(i::trapezoid(&[tiny, 0.0], None, 2.0).unwrap(), tiny);
    assert_eq!(i::simpson(&[1.0; 3], None, tiny).unwrap(), 2.0 * tiny);
    close(
        i::simpson(&[1e308; 3], None, 1e-300).unwrap(),
        2e8,
        3e-12,
        3e-12,
    );
    let y = [1e16, 1e16, 1.0, 1.0, -1e16, -1e16];
    let x = [0.0, 1.0, 1.0, 2.0, 2.0, 3.0];
    assert_eq!(i::trapezoid(&y, Some(&x), 1.0).unwrap(), 1.0);
    assert_eq!(
        *i::cumulative_trapezoid(&y, Some(&x), 1.0, Some(0.0))
            .unwrap()
            .last()
            .unwrap(),
        1.0
    );
    for amplitude in [1e-300, 1.0, 1e300] {
        let y = [amplitude; 5];
        let forward = i::simpson(&y, None, 1.0 / amplitude).unwrap();
        let backward = i::simpson(&y, None, -1.0 / amplitude).unwrap();
        close(forward, 4.0, 3e-12, 3e-12);
        assert_eq!(backward, -forward);
        assert_eq!(
            i::cumulative_simpson(&y, None, 1.0 / amplitude, Some(0.0))
                .unwrap()
                .last(),
            Some(&forward)
        );
    }
    let prefix =
        i::cumulative_simpson(&[1e308; 3], Some(&[0.0, tiny, 1.0]), 1.0, Some(-0.0)).unwrap();
    assert!(prefix[0].is_sign_negative());
    assert!(prefix[1] > 0.0);
    assert!((prefix[1] - tiny * 1e308).abs() <= 3e-14 * prefix[1]);
    assert!(i::simpson(&[f64::MAX; 3], None, 1.0).is_err());
    assert!(i::trapezoid(&[f64::MAX; 2], None, 2.0).is_err());
}

#[test]
fn quadrature_extreme_scales_obey_moments_and_callback_contracts() {
    let tiny = f64::from_bits(1);
    for order in [1, 2, 5] {
        let calls = std::cell::Cell::new(0);
        assert_eq!(
            i::fixed_quad(
                |x| {
                    assert!((0.0..=tiny).contains(&x));
                    calls.set(calls.get() + 1);
                    1.0
                },
                0.0,
                tiny,
                order
            )
            .unwrap(),
            tiny
        );
        assert_eq!(calls.get(), order);
        let forward = i::fixed_quad(
            |x| {
                assert!(x.is_finite() && (-1.7e308..=1.7e308).contains(&x));
                1e-308
            },
            -1.7e308,
            1.7e308,
            order,
        )
        .unwrap();
        close(forward, 3.4, 3e-12, 3e-12);
        let backward = i::fixed_quad(|_| 1e-308, 1.7e308, -1.7e308, order).unwrap();
        close(backward, -forward, 3e-12, 3e-12);
    }
    for n in [2, 3, 5, 9] {
        assert_eq!(i::romb(&vec![1.0; n], tiny).unwrap(), (n - 1) as f64 * tiny);
        close(
            i::romb(&vec![1e-308; n], 1e308).unwrap(),
            (n - 1) as f64,
            3e-12,
            3e-12,
        );
    }
    close(
        i::romb(&[-5.5e307, 1.55e308, -5.5e307], 1.0).unwrap(),
        1.7e308,
        3e-12,
        3e-12,
    );
    assert!(
        i::fixed_quad(|_| 0.0, 1.0, 0.0, 3)
            .unwrap()
            .is_sign_negative()
    );
    assert!(i::fixed_quad(|_| f64::MAX, 0.0, 2.0, 3).is_err());
    assert!(i::fixed_quad(|_| f64::NAN, 0.0, 1.0, 3).is_err());
    assert!(i::romb(&[f64::MAX; 3], 1.0).is_err());
    assert_eq!(
        i::fixed_quad(|_| panic!("zero-width interval evaluated"), 1.0, 1.0, 3).unwrap(),
        0.0
    );
}
