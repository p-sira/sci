//! Generic precision coverage against independent checked-in references.
use s::elliptic as e;
use sci::{
    Float, complex as c, constants, differentiate as d, fft, integrate as q, interpolate as i,
    special as s, stats,
};

fn cast<T: Float>(value: f64) -> T {
    T::from(value).unwrap()
}
fn real<T: Float>(values: impl IntoIterator<Item = T>) -> Vec<f64> {
    values.into_iter().map(|x| x.to_f64().unwrap()).collect()
}
fn complex<T: Float>(values: impl IntoIterator<Item = c::Complex<T>>) -> Vec<f64> {
    values
        .into_iter()
        .flat_map(|z| [z.re.to_f64().unwrap(), z.im.to_f64().unwrap()])
        .collect()
}

fn evaluate<T: fft::FftFloat + e::BulirschFloat>(
    group: &str,
    name: &str,
    x: &[T],
    p: T,
) -> Vec<f64> {
    let zero = T::zero();
    let one = T::one();
    match group {
        "complex" => {
            let z = c::Complex { re: x[0], im: x[1] };
            let b = c::Complex { re: x[2], im: x[3] };
            match name {
                "add" => complex([c::add(z, b)]),
                "subtract" => complex([c::subtract(z, b)]),
                "multiply" => complex([c::multiply(z, b)]),
                "divide" => complex([c::divide(z, b)]),
                "powc" => complex([c::powc(z, b)]),
                "negate" => complex([c::negate(z)]),
                "conjugate" => complex([c::conjugate(z)]),
                "reciprocal" => complex([c::reciprocal(z)]),
                "exp" => complex([c::exp(z)]),
                "ln" => complex([c::ln(z)]),
                "sqrt" => complex([c::sqrt(z)]),
                "sin" => complex([c::sin(z)]),
                "cos" => complex([c::cos(z)]),
                "sinh" => complex([c::sinh(z)]),
                "cosh" => complex([c::cosh(z)]),
                "tan" => complex([c::tan(z)]),
                "tanh" => complex([c::tanh(z)]),
                "asin" => complex([c::asin(z)]),
                "acos" => complex([c::acos(z)]),
                "atan" => complex([c::atan(z)]),
                "asinh" => complex([c::asinh(z)]),
                "acosh" => complex([c::acosh(z)]),
                "atanh" => complex([c::atanh(z)]),
                "cbrt" => complex([c::cbrt(z)]),
                "real" => real([c::real(z)]),
                "imag" => real([c::imag(z)]),
                "abs" => real([c::abs(z)]),
                "norm_sqr" => real([c::norm_sqr(z)]),
                "arg" => real([c::arg(z)]),
                "to_polar" => {
                    let (a, b) = c::to_polar(z);
                    real([a, b])
                }
                "from_polar" => complex([c::from_polar(x[0], x[1]).unwrap()]),
                "powf" => complex([c::powf(z, b.re)]),
                "powi" => complex([c::powi(z, 3)]),
                "powu" => complex([c::powu(z, 3)]),
                _ => panic!("unknown complex function {name}"),
            }
        }
        "special" => match name {
            "ellipk" => real([e::ellipk(x[0]).unwrap()]),
            "ellipe" => real([e::ellipe(x[0]).unwrap()]),
            "ellippi" => real([e::ellippi(x[0], x[1]).unwrap()]),
            "ellipd" => real([e::ellipd(x[0]).unwrap()]),
            "ellipf" => real([e::ellipf(x[0], x[1]).unwrap()]),
            "ellipeinc" => real([e::ellipeinc(x[0], x[1]).unwrap()]),
            "ellippiinc" => real([e::ellippiinc(x[0], x[1], x[2]).unwrap()]),
            "ellipdinc" => real([e::ellipdinc(x[0], x[1]).unwrap()]),
            "elliprf" => real([e::elliprf(x[0], x[1], x[2]).unwrap()]),
            "elliprd" => real([e::elliprd(x[0], x[1], x[2]).unwrap()]),
            "elliprg" => real([e::elliprg(x[0], x[1], x[2]).unwrap()]),
            "elliprj" => real([e::elliprj(x[0], x[1], x[2], x[3]).unwrap()]),
            "elliprc" => real([e::elliprc(x[0], x[1]).unwrap()]),
            "ellipke" => {
                let (k, e) = e::ellipke(x[0]).unwrap();
                real([k, e])
            }
            "cel" => real([e::cel(x[0], x[1], x[2], x[3]).unwrap()]),
            "cel1" => real([e::cel1(x[0]).unwrap()]),
            "cel2" => real([e::cel2(x[0], x[1], x[2]).unwrap()]),
            "cel3" => real([e::cel3(x[0], x[1]).unwrap()]),
            "el1" => real([e::el1(x[0], x[1]).unwrap()]),
            "el2" => real([e::el2(x[0], x[1], x[2], x[3]).unwrap()]),
            "el3" => real([e::el3(x[0], x[1], x[2]).unwrap()]),
            "jacobi_zeta" => real([e::jacobi_zeta(x[0], x[1]).unwrap()]),
            "heuman_lambda" => real([e::heuman_lambda(x[0], x[1]).unwrap()]),
            "ellippiinc_bulirsch" => real([e::ellippiinc_bulirsch(x[0], x[1], x[2]).unwrap()]),
            "ellipkm1" => real([e::ellipkm1(x[0]).unwrap()]),
            "logsumexp" => real([s::logsumexp(x)]),
            "softmax" => real(s::softmax(x)),
            "log_softmax" => real(s::log_softmax(x)),
            _ => panic!("unknown special function {name}"),
        },
        "differentiate" => match name {
            "derivative" => real([d::derivative(|x| x * x * x, x[0], p).unwrap()]),
            "gradient" => {
                real(d::gradient(|x| x[0] * x[0] + x[0] * x[1] + x[1] * x[1], x, p).unwrap())
            }
            "jacobian" => real(
                d::jacobian(|x| vec![x[0] * x[1], x[0] + x[1]], x, p)
                    .unwrap()
                    .into_iter()
                    .flatten(),
            ),
            "hessian" => real(
                d::hessian(|x| x[0] * x[0] + x[0] * x[1] + x[1] * x[1], x, p)
                    .unwrap()
                    .into_iter()
                    .flatten(),
            ),
            _ => panic!("unknown differentiation function {name}"),
        },
        "integrate" => match name {
            "trapezoid" => real([q::trapezoid(x, None, p).unwrap()]),
            "simpson" => real([q::simpson(x, None, p).unwrap()]),
            "cumulative_trapezoid" => {
                real(q::cumulative_trapezoid(x, None, p, Some(zero)).unwrap())
            }
            "cumulative_simpson" => real(q::cumulative_simpson(x, None, p, Some(zero)).unwrap()),
            "romb" => real([q::romb(x, p).unwrap()]),
            "fixed_quad" => real([q::fixed_quad(|x| x * x, zero, one, 4).unwrap()]),
            _ => panic!("unknown integration function {name}"),
        },
        "interpolate" => {
            let nodes = [zero, one, cast(2.), cast(3.)];
            let query = [cast(-0.5), zero, cast(0.5), cast(1.5), cast(3.), cast(3.5)];
            match name {
                "barycentric_interpolate" => {
                    real(i::barycentric_interpolate(&nodes, x, &query).unwrap())
                }
                "krogh_interpolate" => real(i::krogh_interpolate(&nodes, x, &query).unwrap()),
                "pchip_interpolate" => real(i::pchip_interpolate(&nodes, x, &query).unwrap()),
                "lagrange" => real(i::lagrange(&nodes, x).unwrap()),
                _ => panic!("unknown interpolation function {name}"),
            }
        }
        "stats" => match name {
            "mean" => real([stats::mean(x).unwrap()]),
            "median" => real([stats::median(x).unwrap()]),
            "iqr" => real([stats::iqr(x).unwrap()]),
            "gmean" => real([stats::gmean(x).unwrap()]),
            "hmean" => real([stats::hmean(x).unwrap()]),
            "variance" => real([stats::variance(x, 1).unwrap()]),
            "std" => real([stats::std(x, 1).unwrap()]),
            "sem" => real([stats::sem(x, 1).unwrap()]),
            "variation" => real([stats::variation(x, 1).unwrap()]),
            "gstd" => real([stats::gstd(x, 1).unwrap()]),
            "zscore" => real(stats::zscore(x, 1).unwrap()),
            "gzscore" => real(stats::gzscore(x, 1).unwrap()),
            "quantile" => real([stats::quantile(x, cast(0.3)).unwrap()]),
            "scoreatpercentile" => real([stats::scoreatpercentile(x, cast(30.)).unwrap()]),
            "median_abs_deviation" => real([stats::median_abs_deviation(x, cast(2.)).unwrap()]),
            "moment" => real([stats::moment(x, 3).unwrap()]),
            "skew" => real([stats::skew(x, false).unwrap()]),
            "kurtosis" => real([stats::kurtosis(x, true, false).unwrap()]),
            "zmap" => {
                real(stats::zmap(&x.iter().map(|&v| v + one).collect::<Vec<_>>(), x, 1).unwrap())
            }
            "trimboth" => real(stats::trimboth(x, cast(0.125)).unwrap()),
            "trim_mean" => real([stats::trim_mean(x, cast(0.125)).unwrap()]),
            "mode" => {
                let (v, n) = stats::mode(x).unwrap();
                vec![v.to_f64().unwrap(), n as f64]
            }
            "describe" => {
                let a = stats::describe(x, 1, false).unwrap();
                real([
                    cast(a.count as f64),
                    a.min,
                    a.max,
                    a.mean,
                    a.variance,
                    a.skewness,
                    a.kurtosis,
                ])
            }
            _ => panic!("unknown statistics function {name}"),
        },
        "constants" => match name {
            "convert_temperature" => real([constants::convert_temperature(
                x[0],
                constants::TemperatureScale::Celsius,
                constants::TemperatureScale::Fahrenheit,
            )]),
            "lambda2nu" => real([constants::lambda2nu(x[0])]),
            "nu2lambda" => real([constants::nu2lambda(x[0])]),
            _ => panic!("unknown constants function {name}"),
        },
        "fft" => {
            let z: Vec<_> = x
                .iter()
                .zip(x.iter().rev())
                .map(|(&re, &im)| c::Complex {
                    re,
                    im: cast::<T>(0.25) * im,
                })
                .collect();
            let matrix: Vec<_> = z.chunks(2).map(|r| r.to_vec()).collect();
            let rows: Vec<_> = x.chunks(2).map(|r| r.to_vec()).collect();
            let shape = [2, 2];
            let axes = [0, 1];
            let kind = match p.to_usize().unwrap() {
                1 => fft::RealTransformType::I,
                3 => fft::RealTransformType::III,
                4 => fft::RealTransformType::IV,
                _ => fft::RealTransformType::II,
            };
            let period = cast(std::f64::consts::TAU);
            match name {
                "fft" => complex(fft::fft(&z)),
                "ifft" => complex(fft::ifft(&z)),
                "fftshift" => complex(fft::fftshift(&z).unwrap()),
                "ifftshift" => complex(fft::ifftshift(&z).unwrap()),
                "rfft" => complex(fft::rfft(x)),
                "ihfft" => complex(fft::ihfft(x, None).unwrap()),
                "irfft" => real(fft::irfft(&z[..3], Some(4)).unwrap()),
                "hfft" => real(fft::hfft(&z[..3], Some(4)).unwrap()),
                "fftn" => complex(fft::fftn(&z, &shape, &axes).unwrap()),
                "ifftn" => complex(fft::ifftn(&z, &shape, &axes).unwrap()),
                "rfftn" => {
                    let (v, s) = fft::rfftn(x, &shape, &axes).unwrap();
                    assert_eq!(s, vec![2, 2]);
                    complex(v)
                }
                "ihfftn" => {
                    let (v, s) = fft::ihfftn(x, &shape, &axes).unwrap();
                    assert_eq!(s, vec![2, 2]);
                    complex(v)
                }
                "irfftn" => real(fft::irfftn(&z, &shape, &axes).unwrap()),
                "hfftn" => real(fft::hfftn(&z, &shape, &axes).unwrap()),
                "fft2" => complex(fft::fft2(&matrix).unwrap().into_iter().flatten()),
                "ifft2" => complex(fft::ifft2(&matrix).unwrap().into_iter().flatten()),
                "rfft2" => complex(fft::rfft2(&rows).unwrap().into_iter().flatten()),
                "ihfft2" => complex(fft::ihfft2(&rows).unwrap().into_iter().flatten()),
                "irfft2" => real(fft::irfft2(&matrix, 2).unwrap().into_iter().flatten()),
                "hfft2" => real(fft::hfft2(&matrix, 2).unwrap().into_iter().flatten()),
                "fftfreq" => real(fft::fftfreq(4, cast::<T>(0.5)).unwrap()),
                "rfftfreq" => real(fft::rfftfreq(4, cast::<T>(0.5)).unwrap()),
                "dct" => real(fft::dct(x, kind).unwrap()),
                "idct" => real(fft::idct(x, kind).unwrap()),
                "dst" => real(fft::dst(x, kind).unwrap()),
                "idst" => real(fft::idst(x, kind).unwrap()),
                "dctn" => real(fft::dctn(x, &shape, &axes, kind).unwrap()),
                "idctn" => real(fft::idctn(x, &shape, &axes, kind).unwrap()),
                "dstn" => real(fft::dstn(x, &shape, &axes, kind).unwrap()),
                "idstn" => real(fft::idstn(x, &shape, &axes, kind).unwrap()),
                "hilbert" => real(fft::hilbert(x).unwrap()),
                "ihilbert" => real(fft::ihilbert(x).unwrap()),
                "diff" => real(fft::diff(x, 1, period).unwrap()),
                "shift" => real(fft::shift(x, cast(0.25), period).unwrap()),
                "tilbert" => real(fft::tilbert(x, cast(0.5), period).unwrap()),
                "itilbert" => real(fft::itilbert(x, cast(0.5), period).unwrap()),
                "cc_diff" => real(fft::cc_diff(x, cast(0.25), cast(0.5), period).unwrap()),
                "cs_diff" => real(fft::cs_diff(x, cast(0.25), cast(0.5), period).unwrap()),
                "sc_diff" => real(fft::sc_diff(x, cast(0.25), cast(0.5), period).unwrap()),
                "ss_diff" => real(fft::ss_diff(x, cast(0.25), cast(0.5), period).unwrap()),
                _ => panic!("unknown FFT function {name}"),
            }
        }
        _ => panic!("unknown fixture group {group}"),
    }
}

fn fixtures<T: fft::FftFloat + e::BulirschFloat>(precision: &str, tolerance: f64) {
    let inputs: Vec<_> = include_str!("fixtures/generic_float_inputs.csv")
        .lines()
        .skip(1)
        .collect();
    for line in include_str!("fixtures/generic_float_expected.csv")
        .lines()
        .skip(1)
    {
        let expected: Vec<_> = line.split(',').collect();
        if expected[1] != precision {
            continue;
        }
        let case: usize = expected[0].parse().unwrap();
        let input: Vec<_> = inputs[case].split(',').collect();
        let values: Vec<T> = input[2]
            .split(';')
            .map(|v| cast(v.parse().unwrap()))
            .collect();
        let p = cast(input[3].parse().unwrap_or(0.));
        let actual = evaluate(input[0], input[1], &values, p);
        let expected: Vec<f64> = expected[2].split(';').map(|v| v.parse().unwrap()).collect();
        assert_eq!(actual.len(), expected.len(), "{case}: {}", input[1]);
        let tolerance = if precision == "32" && input[0] != "differentiate" {
            5e-6
        } else {
            tolerance
        };
        let atol = tolerance
            + if input[1] == "derivative" {
                p.to_f64().unwrap().powi(2) * 1.01
            } else {
                0.
            };
        for (a, e) in actual.into_iter().zip(expected) {
            if e.is_nan() {
                assert!(a.is_nan());
            } else if e.is_infinite() {
                assert_eq!(a, e);
            } else {
                assert!(
                    (a - e).abs() <= atol + tolerance * e.abs(),
                    "{precision} case {case} {}: {a} != {e}",
                    input[1]
                );
            }
        }
    }
}

#[test]
fn binary32_public_apis_match_independent_references() {
    fixtures::<f32>("32", 3e-4);
}
#[test]
fn binary64_public_apis_match_independent_references() {
    fixtures::<f64>("64", 5e-12);
}

fn boundary_properties<T: fft::FftFloat + e::BulirschFloat>() {
    let zero = T::zero();
    let one = T::one();
    let two = one + one;
    let nan = T::nan();
    let inf = T::infinity();
    let max = T::max_value();
    let tiny = T::min_positive_value() * T::epsilon();
    assert_eq!(sci_core::is_finite_slice(&[-zero, tiny]), Ok(()));
    assert!(sci_core::is_finite_slice(&[nan]).is_err());
    assert!(sci_core::is_finite_slice(&[inf]).is_err());
    assert!(d::derivative(|x| x, one, T::epsilon() / cast(4.)).is_err());
    assert!(d::gradient(|x| x[0], &[one], zero).is_err());
    assert!(d::jacobian(|_| vec![nan], &[one], one).is_err());
    assert!(d::hessian(|_| inf, &[one], one).is_err());
    assert_eq!(d::derivative(|x| x, zero, max).unwrap(), one);
    assert_eq!(d::derivative(|x| x, zero, tiny).unwrap(), one);
    assert_eq!(stats::mean(&[max, max]).unwrap(), max);
    assert_eq!(stats::mean(&[-max, max]).unwrap(), zero);
    assert_eq!(stats::std(&[-max, max], 0).unwrap(), max);
    assert_eq!(stats::variance(&[max, max], 0).unwrap(), zero);
    assert_eq!(q::trapezoid(&[tiny, tiny], None, one).unwrap(), tiny);
    assert_eq!(
        q::simpson(&[tiny, tiny, tiny], None, one).unwrap(),
        two * tiny
    );
    assert_eq!(q::romb(&[tiny, tiny, tiny], one).unwrap(), two * tiny);
    assert_eq!(q::fixed_quad(|_| tiny, zero, one, 3).unwrap(), tiny);
    assert!(q::fixed_quad(|_| nan, zero, one, 3).is_err());
    assert!(q::fixed_quad(|x| x, zero, one, 257).is_err());
    assert!(q::trapezoid::<T>(&[], None, one).is_err());
    assert!(q::simpson(&[one, nan], None, one).is_err());
    assert!(q::cumulative_trapezoid(&[one], None, one, Some(inf)).is_err());
    assert!(q::cumulative_simpson(&[one, one], Some(&[one, zero]), one, None).is_err());
    assert!(q::romb(&[one, one, one, one], one).is_err());
    assert!(fft::fftn::<T>(&[], &[2], &[0]).is_err());
    assert!(fft::rfftn(&[nan], &[1], &[0]).is_err());
    assert!(fft::dct(&[one], fft::RealTransformType::I).is_err());
    assert!(fft::diff(&[one], 1, zero).is_err());
    assert!(fft::fftfreq(4, zero).is_err());
    assert!(fft::rfftfreq(0, one).is_err());
    assert!(fft::rfft::<T>(&[]).is_empty());
    for n in [1, 2, 3, 4, 7, 16, 3] {
        let input: Vec<_> = (0..n)
            .map(|i| c::Complex {
                re: cast(i as f64 / 8.),
                im: -one,
            })
            .collect();
        let restored = fft::ifft(&fft::fft(&input));
        for (a, b) in restored.iter().zip(&input) {
            assert!((a.re - b.re).abs() <= cast::<T>(100.) * T::epsilon());
            assert!((a.im - b.im).abs() <= cast::<T>(100.) * T::epsilon());
        }
    }
    // Interoperate across precision-generic public modules.
    let gradient = d::gradient(|x| x[0] * x[0] + x[1] * x[1], &[one, two], cast(0.125)).unwrap();
    assert_eq!(stats::mean(&gradient).unwrap(), cast(3.));
    let spectrum = fft::rfft(&gradient);
    assert_eq!(c::real(spectrum[0]), cast(6.));
    let product = c::multiply(
        c::Complex { re: max, im: max },
        c::Complex {
            re: cast(0.5),
            im: cast(0.5),
        },
    );
    assert_eq!(product.re, zero);
    assert_eq!(product.im, max);
    assert_eq!(
        c::multiply(
            c::Complex { re: tiny, im: zero },
            c::Complex { re: one, im: zero }
        )
        .re,
        tiny
    );
    let branch = c::sqrt(c::Complex {
        re: -one,
        im: -zero,
    });
    assert_eq!(branch.im, -one);
    assert!(
        c::ln(c::Complex {
            re: -one,
            im: -zero
        })
        .im
        .is_sign_negative()
    );
    assert!(c::from_polar(-one, zero).is_err());
    assert_eq!(s::logsumexp::<T>(&[]), -inf);
    assert!(s::logsumexp(&[nan]).is_nan());
    assert_eq!(s::logsumexp(&[inf, one]), inf);
    assert_eq!(constants::lambda2nu(-zero), -inf);
    assert!(constants::nu2lambda(-inf).is_sign_negative());
    assert!(
        constants::convert_temperature(
            nan,
            constants::TemperatureScale::Celsius,
            constants::TemperatureScale::Kelvin
        )
        .is_nan()
    );
    assert_eq!(e::ellipe(one).unwrap(), one);
    assert!(e::ellipk(one).is_err());
    assert!(e::ellipkm1(zero).is_err());
    for bad in [nan, inf, -inf] {
        assert!(e::ellipk(bad).is_err());
        assert!(e::ellipe(bad).is_err());
        assert!(e::ellippi(bad, zero).is_err());
        assert!(e::ellipd(bad).is_err());
        assert!(e::ellipkinc(bad, zero).is_err());
        assert!(e::ellipeinc(bad, zero).is_err());
        assert!(e::ellippiinc(bad, zero, zero).is_err());
        assert!(e::ellipdinc(bad, zero).is_err());
        assert!(e::elliprf(bad, one, one).is_err());
        assert!(e::elliprd(bad, one, one).is_err());
        assert!(e::elliprg(bad, one, one).is_err());
        assert!(e::elliprj(bad, one, one, one).is_err());
        assert!(e::elliprc(bad, one).is_err());
        assert!(e::ellipke(bad).is_err());
        assert!(e::cel(bad, one, one, one).is_err());
        assert!(e::cel1(bad).is_err());
        assert!(e::cel2(bad, one, one).is_err());
        assert!(e::cel3(bad, one).is_err());
        assert!(e::el1(bad, one).is_err());
        assert!(e::el2(bad, one, one, one).is_err());
        assert!(e::el3(bad, one, one).is_err());
        assert!(e::jacobi_zeta(bad, zero).is_err());
        assert!(e::heuman_lambda(bad, cast(0.5)).is_err());
        assert!(e::ellippiinc_bulirsch(bad, zero, zero).is_err());
        assert!(e::ellipkm1(bad).is_err());
        assert!(stats::mean(&[bad]).is_err());
        assert!(stats::variance(&[bad], 0).is_err());
        assert!(stats::median(&[bad]).is_err());
        assert!(stats::zscore(&[bad], 0).is_err());
        assert!(i::barycentric_interpolate(&[zero, one], &[zero, one], &[bad]).is_err());
        assert!(i::krogh_interpolate(&[zero, one], &[zero, one], &[bad]).is_err());
        assert!(i::lagrange(&[zero, bad], &[zero, one]).is_err());
        assert!(i::pchip_interpolate(&[zero, one], &[zero, one], &[bad]).is_err());
    }
}

#[test]
fn binary32_boundaries_and_interoperability() {
    boundary_properties::<f32>();
}
#[test]
fn binary64_boundaries_and_interoperability() {
    boundary_properties::<f64>();
}

#[test]
fn binary32_complex_scaling_preserves_representable_components() {
    let value = c::exp(c::Complex {
        re: 90_f32,
        im: std::f32::consts::FRAC_PI_2,
    });
    let expected = 90_f64.exp() * f64::from(std::f32::consts::FRAC_PI_2).cos();
    assert!((f64::from(value.re) / expected - 1.).abs() < 2e-6);
    assert_eq!(value.im, f32::INFINITY);
    let tiny = f32::from_bits(1);
    let value = c::sin(c::Complex { re: tiny, im: 90. });
    let expected = f64::from(tiny) * 90_f64.sinh();
    assert!((f64::from(value.re) / expected - 1.).abs() < 2e-6);
    let z = c::Complex {
        re: 1e30_f32,
        im: 1e30_f32,
    };
    let inverse = c::atanh(z);
    assert!((inverse.re / 5e-31_f32 - 1.).abs() < 2e-6);
    assert_eq!(inverse.im, std::f32::consts::FRAC_PI_2);
}

#[test]
fn shared_storage_defaults_and_generic_variants() {
    let _: sci::Matrix = vec![vec![1_f64]];
    let _: sci::Matrix<f32> = vec![vec![1_f32]];
    let spline = sci::Spline {
        knots: vec![0_f32, 1.],
        coefficients: vec![2_f32],
        degree: 0,
    };
    let _: sci::ParametricSpline<f32> = sci::ParametricSpline {
        components: vec![spline],
        parameters: vec![0.],
    };
    let _: sci::SurfaceSpline<f32> = sci::SurfaceSpline {
        x_knots: vec![0., 1.],
        y_knots: vec![0., 1.],
        coefficients: vec![2.],
        x_degree: 0,
        y_degree: 0,
    };
    let _: sci::SparseMatrix<f32> = sci::SparseMatrix {
        rows: 1,
        cols: 1,
        indptr: vec![0, 1],
        indices: vec![0],
        data: vec![1.],
    };
    let _: sci::SparseArray<f32> = sci::SparseArray {
        shape: vec![1],
        coordinates: vec![vec![0]],
        data: vec![1.],
    };
    let _: c::Complex = c::Complex { re: 1_f64, im: 0. };
    let _: stats::Description<f32> = stats::describe(&[1_f32, 2., 3., 4.], 1, false).unwrap();
}
