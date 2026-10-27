//! Boundary and property tests that also run with the `stats` feature disabled.
mod common;
use common::finite_close;
use sci::{
    Error, Mode, Spline, cluster, complex, constants, differentiate, fft, integrate, interpolate,
    linalg, optimize, signal, spatial, special,
};

#[test]
fn complex_division_preserves_representable_extreme_quotients() {
    use complex::Complex;
    for scale in [1e-300, 1.0, 1e300] {
        let z = Complex {
            re: scale,
            im: -scale,
        };
        let quotient = complex::divide(z, z);
        finite_close(quotient.re, 1.0, 2e-12, 2e-12);
        finite_close(quotient.im, 0.0, 2e-12, 2e-12);
        let quotient = complex::divide(
            Complex {
                re: -scale,
                im: -scale,
            },
            z,
        );
        finite_close(quotient.re, 0.0, 2e-12, 2e-12);
        finite_close(quotient.im, -1.0, 2e-12, 2e-12);
    }
    let quotient = complex::divide(
        Complex {
            re: f64::MAX,
            im: f64::MAX,
        },
        Complex { re: 1.0, im: 1.0 },
    );
    assert_eq!(quotient.re, f64::MAX);
    assert_eq!(quotient.im, 0.0);
    let quotient = complex::divide(
        Complex { re: 1.0, im: 2.0 },
        Complex {
            re: f64::NAN,
            im: f64::NAN,
        },
    );
    assert!(quotient.re.is_nan() && quotient.im.is_nan());
}

#[test]
fn constants_search_is_sorted_unique_and_case_insensitive() {
    let all = constants::find(None);
    assert!(all.windows(2).all(|pair| pair[0] < pair[1]));
    assert_eq!(
        constants::find(Some("PLANCK")),
        constants::find(Some("planck"))
    );
    assert!(constants::find(Some("不存在")).is_empty());
    for key in constants::find(Some("mass")) {
        assert!(key.to_lowercase().contains("mass"));
        assert!(constants::value(key).unwrap().is_finite());
        // SciPy defines precision as uncertainty/value, including signed ratios.
        assert!(constants::precision(key).unwrap().is_finite());
    }
    for convert in [constants::lambda2nu, constants::nu2lambda] {
        assert_eq!(convert(0.0), f64::INFINITY);
        assert_eq!(convert(f64::INFINITY).to_bits(), 0.0_f64.to_bits());
        assert_eq!(convert(f64::NEG_INFINITY).to_bits(), (-0.0_f64).to_bits());
    }
}

#[test]
fn finite_difference_scales_before_an_overflowing_quotient() {
    // delta/h overflows, but delta/(2h) is representable.
    let slope = f64::MAX * 0.75;
    let f = |x: f64| slope * x;
    assert_eq!(differentiate::derivative(f, 0.0, 0.5).unwrap(), slope);
    assert_eq!(
        differentiate::gradient(|x| f(x[0]), &[0.0], 0.5).unwrap(),
        vec![slope]
    );
    assert_eq!(
        differentiate::jacobian(|x| vec![f(x[0])], &[0.0], 0.5).unwrap(),
        vec![vec![slope]]
    );
}

#[test]
fn optimization_validation_precedes_callbacks_and_preserves_late_errors() {
    use optimize::Options;
    let never = |_: &[f64]| -> f64 { panic!("invalid controls reached callback") };
    let never_gradient = |_: &[f64]| -> Vec<f64> { panic!("invalid controls reached gradient") };
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let options = Options {
            tolerance: bad,
            ..Default::default()
        };
        assert!(optimize::minimize(never, &[0.0], options).is_err());
        assert!(optimize::minimize_with_gradient(never, never_gradient, &[0.0], options).is_err());
        let options = Options {
            learning_rate: bad,
            ..Default::default()
        };
        assert!(optimize::minimize_with_gradient(never, never_gradient, &[0.0], options).is_err());
    }
    let options = Options {
        max_iterations: u64::MAX,
        ..Default::default()
    };
    assert!(optimize::minimize(never, &[0.0], options).is_err());
    assert!(optimize::minimize_with_gradient(never, never_gradient, &[0.0], options).is_err());
    let options = Options {
        max_iterations: 0,
        ..Default::default()
    };
    assert!(optimize::minimize(never, &vec![0.0; 1000], options).is_err());
    let error = Error::NumericalFailure("gradient callback failed");
    assert_eq!(
        optimize::try_minimize_with_gradient(
            |x| Ok(x[0] * x[0]),
            |_| Err(error.clone()),
            &[1.0],
            Default::default()
        ),
        Err(error)
    );
    let error = Error::InvalidInput("trial point rejected");
    assert_eq!(
        optimize::try_minimize(
            |x| if x[0] == 1.0 {
                Ok(1.0)
            } else {
                Err(error.clone())
            },
            &[1.0],
            Default::default()
        ),
        Err(error)
    );
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            optimize::minimize_with_gradient(|_| 1.0, |_| vec![bad], &[0.0], Default::default()),
            Err(Error::NonFiniteEvaluation)
        );
    }
}

#[test]
fn integration_rejects_overflow_and_respects_callback_counts() {
    use std::cell::Cell;
    for n in [1, 2, 16, 256] {
        let count = Cell::new(0);
        let result = integrate::fixed_quad(
            |_| {
                count.set(count.get() + 1);
                1.0
            },
            -1.0,
            1.0,
            n,
        )
        .unwrap();
        finite_close(result, 2.0, 2e-12, 2e-12);
        assert_eq!(count.get(), n);
    }
    assert_eq!(
        integrate::fixed_quad(|_| 0.0, -f64::MAX, f64::MAX, 2),
        Ok(0.0)
    );
    finite_close(
        integrate::fixed_quad(|_| f64::MAX, 0.0, 1.0, 2).unwrap(),
        f64::MAX,
        2e-12,
        2e-12,
    );
    assert_eq!(
        integrate::fixed_quad(|_| f64::MAX, 0.0, 2.0, 2),
        Err(Error::NonFiniteEvaluation)
    );
    assert_eq!(
        integrate::trapezoid(&[1.0, 1.0], Some(&[-f64::MAX, f64::MAX]), 1.0),
        Err(Error::NonFiniteEvaluation)
    );
    assert!(
        integrate::cumulative_simpson(&[1.0, 1.0, 1.0], None, 1.0, Some(f64::MAX))
            .unwrap()
            .iter()
            .all(|x| x.is_finite())
    );
    for integrate in [integrate::trapezoid, integrate::simpson] {
        assert!(integrate(&[f64::MAX; 3], None, f64::MAX).is_err());
    }
    assert!(integrate::romb(&[f64::MAX; 3], f64::MAX).is_err());
}

#[test]
fn spline_public_fields_are_validated_by_every_consumer() {
    use interpolate::SplineExtension;
    let valid = Spline {
        knots: vec![0.0, 0.0, 1.0, 1.0],
        coefficients: vec![0.0, 1.0],
        degree: 1,
    };
    let mut bad_knots = valid.clone();
    bad_knots.knots[2] = -1.0;
    let mut bad_coefficients = valid.clone();
    bad_coefficients.coefficients[0] = f64::NAN;
    let mut bad_degree = valid.clone();
    bad_degree.degree = usize::MAX;
    let mut bad_shape = valid.clone();
    bad_shape.coefficients.pop();
    for spline in [bad_knots, bad_coefficients, bad_degree, bad_shape] {
        assert!(interpolate::splev(&[0.5], &spline, 0, SplineExtension::Extrapolate).is_err());
        assert!(interpolate::splder(&spline, 0).is_err());
        assert!(interpolate::splantider(&spline, 0).is_err());
        assert!(interpolate::spalde(&[0.5], &spline).is_err());
        assert!(interpolate::splint(0.0, 1.0, &spline).is_err());
        assert!(interpolate::insert(0.5, &spline, 1).is_err());
    }
    assert_eq!(interpolate::splder(&valid, 0).unwrap(), valid);
    assert_eq!(interpolate::splantider(&valid, 0).unwrap(), valid);
    finite_close(
        interpolate::splint(1.0, 0.0, &valid).unwrap(),
        -0.5,
        2e-12,
        2e-12,
    );
}

#[test]
fn vector_quantization_handles_empty_clusters_and_distance_overflow() {
    use cluster::vq;
    let observations = vec![vec![0.0], vec![2.0]];
    let codes = vec![vec![1.0], vec![100.0]];
    let (retained, labels) = vq::kmeans2(&observations, &codes, 2).unwrap();
    assert_eq!(retained, codes);
    assert_eq!(labels, vec![0, 0]);
    let (dropped, distortion) = vq::kmeans(&observations, &codes, 1e-8, 4).unwrap();
    assert_eq!(dropped, vec![vec![1.0]]);
    finite_close(distortion, 1.0, 2e-12, 2e-12);
    assert_eq!(
        vq::kmeans(&observations, &[vec![100.0]], 0.0, 1),
        Err(Error::NumericalFailure(
            "k-means did not converge within iteration limit"
        ))
    );
    assert_eq!(
        vq::vq(&[vec![f64::MAX]], &[vec![-f64::MAX]]),
        Err(Error::NonFiniteEvaluation)
    );
    assert_eq!(
        vq::whiten(&[vec![3.0, 0.0], vec![3.0, 0.0]]).unwrap(),
        vec![vec![3.0, 0.0]; 2]
    );
}

#[test]
fn spatial_zero_weights_exclude_overflowing_coordinate_differences() {
    use spatial::distance as d;
    let u = [f64::MAX, 1.0];
    let v = [-f64::MAX, 4.0];
    let w = [0.0, 1.0];
    for metric in [d::euclidean, d::cityblock, d::chebyshev] {
        finite_close(metric(&u, &v, Some(&w)).unwrap(), 3.0, 2e-12, 2e-12);
        assert_eq!(metric(&u, &v, None), Err(Error::NonFiniteEvaluation));
    }
    finite_close(d::sqeuclidean(&u, &v, Some(&w)).unwrap(), 9.0, 2e-12, 2e-12);
    assert_eq!(
        d::sqeuclidean(&[1e200], &[0.0], None),
        Err(Error::NonFiniteEvaluation)
    );
    for weights in [[1.0, 2.0], [1e300, 2e300], [1e-300, 2e-300]] {
        finite_close(
            d::hamming(&[0.0, 1.0], &[0.0, 0.0], Some(&weights)).unwrap(),
            2.0 / 3.0,
            2e-12,
            2e-12,
        );
        finite_close(
            d::jaccard(&[0.0, 1.0], &[0.0, 0.0], Some(&weights)).unwrap(),
            1.0,
            2e-12,
            2e-12,
        );
    }
}

#[test]
fn special_reductions_handle_empty_and_nonfinite_normalization() {
    assert_eq!(special::logsumexp::<f64>(&[]), f64::NEG_INFINITY);
    assert!(special::softmax::<f64>(&[]).is_empty());
    assert!(special::log_softmax::<f64>(&[]).is_empty());
    assert_eq!(
        special::logsumexp(&[f64::NEG_INFINITY; 2]),
        f64::NEG_INFINITY
    );
    assert_eq!(special::logsumexp(&[0.0, f64::INFINITY]), f64::INFINITY);
    assert!(special::logsumexp(&[f64::INFINITY, f64::NAN]).is_nan());
    for input in [
        [f64::NEG_INFINITY; 2],
        [f64::INFINITY, 0.0],
        [0.0, f64::NAN],
    ] {
        assert!(special::softmax(&input).iter().all(|x| x.is_nan()));
        assert!(special::log_softmax(&input).iter().all(|x| x.is_nan()));
    }
    for shift in [-1000.0, 0.0, 1000.0] {
        let values = [shift - 2.0, shift, shift + 3.0];
        let probabilities = special::softmax(&values);
        let logs = special::log_softmax(&values);
        finite_close(probabilities.iter().sum(), 1.0, 2e-12, 2e-12);
        for (p, log) in probabilities.iter().zip(logs) {
            finite_close(*p, log.exp(), 2e-12, 2e-12);
        }
    }
}

#[test]
fn fft_tensor_axes_validate_shapes_and_empty_axes_preserve_input() {
    use complex::Complex;
    let input = vec![Complex { re: 1.0, im: -2.0 }; 6];
    assert_eq!(fft::fftn(&input, &[2, 3], &[]).unwrap(), input);
    assert_eq!(fft::ifftn(&input, &[2, 3], &[]).unwrap(), input);
    for axes in [&[0, 0][..], &[2], &[usize::MAX]] {
        assert!(fft::fftn(&input, &[2, 3], axes).is_err());
        assert!(fft::ifftn(&input, &[2, 3], axes).is_err());
        assert!(fft::rfftn(&[1.0; 6], &[2, 3], axes).is_err());
    }
    for shape in [&[usize::MAX][..], &[1_048_576, 2], &[2, 2], &[0, 3]] {
        assert!(fft::fftn(&input, shape, &[0]).is_err());
    }
    for transform in [fft::fft2, fft::ifft2] {
        assert!(transform(&[vec![input[0]], vec![]]).is_err());
    }
    // Tiny logarithmic spacing overflows the frequency before gamma evaluation.
    assert_eq!(
        fft::fhtoffset(f64::from_bits(1), 0.0, 0.0, 0.0),
        Err(Error::NonFiniteEvaluation)
    );
    for transform in [fft::fht, fft::ifht] {
        assert!(transform(&[1.0; 3], 1000.0, 0.0, 0.0, 1.0).is_err());
    }
}

#[test]
fn fft_zero_hyperbolic_numerators_and_overflow_are_explicit() {
    let x = [1.0, -2.0, 4.0];
    assert_eq!(
        fft::sc_diff(&x, 0.0, 1.0, std::f64::consts::TAU).unwrap(),
        vec![0.0; 3]
    );
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    for v in fft::ss_diff(&x, 0.0, 1.0, std::f64::consts::TAU).unwrap() {
        finite_close(v, 0.0 * mean, 2e-12, 2e-12);
    }
    assert!(fft::diff(&x, 64, f64::MIN_POSITIVE).is_err());
    assert!(fft::shift(&x, f64::MAX, 1.0).is_err());
    assert!(fft::sc_diff(&x, f64::MAX, 1.0, 1.0).is_err());
}

#[test]
fn linalg_array_slice_adapters_preserve_order_and_ieee_bits() {
    use linalg::{MatrixInput, VectorInput};
    let values = [[-0.0_f64, f64::NAN], [f64::INFINITY, f64::NEG_INFINITY]];
    let rows = [values[0].to_vec(), values[1].to_vec()];
    for input in [&values as &dyn MatrixInput, &&values[..], &rows] {
        assert_eq!(input.matrix_shape().unwrap(), (2, 2));
        for (i, row) in values.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                assert_eq!(input.matrix_entry(i, j).to_bits(), value.to_bits());
                assert_eq!(input.to_matrix().unwrap()[i][j].to_bits(), value.to_bits());
            }
        }
    }
    let v = vec![1.0, 2.0, 3.0];
    let array = [1.0, 2.0, 3.0];
    for input in [&v as &dyn VectorInput, &array, &&v[..]] {
        assert_eq!(input.vector_len(), 3);
        for (i, expected) in v.iter().enumerate() {
            assert_eq!(input.vector_entry(i), *expected);
        }
        let reference = &input;
        assert_eq!(reference.to_vector().unwrap(), v);
    }
}

#[test]
fn principal_matrix_functions_distinguish_zero_and_nilpotent_inputs() {
    let zero = [[0.0; 2]; 2];
    let root = linalg::sqrtm(&zero).unwrap();
    assert!(root.iter().flatten().all(|z| z.re == 0.0 && z.im == 0.0));
    assert!(linalg::logm(&zero).is_err());
    assert!(linalg::fractional_matrix_power(&zero, -1.0).is_err());
    assert_eq!(
        linalg::sqrtm(&[[0.0, 1.0], [0.0, 0.0]]),
        Err(Error::NumericalFailure(
            "matrix has no supported principal square root"
        ))
    );
    let identity = linalg::fractional_matrix_power(&zero, 0.0).unwrap();
    for (i, row) in identity.iter().enumerate() {
        for (j, z) in row.iter().enumerate() {
            assert_eq!(z.re, f64::from(i == j));
            assert_eq!(z.im, 0.0);
        }
    }
}

#[test]
fn signal_division_and_convolution_cover_short_dividends_and_operand_order() {
    let x = [2.0, -1.0];
    let h = [1.0, 3.0, -2.0, 4.0];
    assert_eq!(signal::deconvolve(&x, &h).unwrap(), (vec![], x.to_vec()));
    for mode in [Mode::Full, Mode::Valid] {
        let a = signal::convolve(&x, &h, mode).unwrap();
        assert_eq!(a, signal::convolve(&h, &x, mode).unwrap());
        let b = signal::fftconvolve(&x, &h, mode).unwrap();
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            finite_close(*a, b, 2e-12, 2e-12);
        }
    }
    // Same-mode output is sized to the first operand, including n < m.
    assert_eq!(signal::convolve(&x, &h, Mode::Same).unwrap().len(), x.len());
    let (quotient, remainder) =
        signal::deconvolve(&signal::convolve(&x, &h, Mode::Full).unwrap(), &h).unwrap();
    assert_eq!(quotient, x);
    assert!(remainder.iter().all(|v| *v == 0.0));
    assert!(signal::lfilter_with_state(&[1.0, 1.0], &[1.0], &x, &[0.0, 0.0]).is_err());
    assert!(signal::lfilter(&[f64::MAX], &[f64::MIN_POSITIVE], &x).is_err());
}

#[test]
fn elliptic_functions_reject_nonfinite_and_singular_parameters() {
    use special::elliptic as e;
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for f in [e::ellipk, e::ellipe, e::ellipd, e::cel1, e::ellipkm1] {
            assert!(f(bad).is_err());
        }
        assert!(e::ellipke(bad).is_err());
        for f in [
            e::ellipeinc,
            e::ellipdinc,
            e::jacobi_zeta,
            e::heuman_lambda,
            e::ellipkinc,
        ] {
            assert!(f(bad, 0.5).is_err());
            assert!(f(0.3, bad).is_err());
        }
        for f in [e::ellippi, e::elliprc, e::cel3, e::el1] {
            assert!(f(bad, 0.5).is_err());
            assert!(f(0.3, bad).is_err());
        }
        for f in [
            e::elliprf,
            e::elliprd,
            e::elliprg,
            e::cel2,
            e::el3,
            e::ellippiinc,
            e::ellippiinc_bulirsch,
        ] {
            for values in [[bad, 0.5, 0.5], [0.5, bad, 0.5], [0.5, 0.5, bad]] {
                assert!(f(values[0], values[1], values[2]).is_err());
            }
        }
        for f in [e::elliprj, e::cel, e::el2] {
            for i in 0..4 {
                let mut values = [0.5; 4];
                values[i] = bad;
                assert!(f(values[0], values[1], values[2], values[3]).is_err());
            }
        }
    }
    assert!(e::ellipd(1.0).is_err());
    assert!(e::ellippi(1.0, 0.5).is_err());
    assert!(e::elliprc(1.0, 0.0).is_err());
    assert!(e::elliprd(0.0, 0.0, 1.0).is_err());
    assert!(e::elliprj(1.0, 1.0, 1.0, 0.0).is_err());
    assert!(e::cel(0.0, 1.0, 1.0, 1.0).is_err());
    assert!(e::el3(1.0, 1.0, -1.0).is_err());
    assert!(e::heuman_lambda(0.3, -0.1).is_err());
}

#[test]
fn special_polynomials_and_sequences_bound_native_work() {
    for f in [
        special::legendre,
        special::laguerre,
        special::hermite,
        special::hermitenorm,
        special::chebyt,
        special::chebyu,
        special::chebyc,
        special::chebys,
        special::sh_chebyt,
        special::sh_chebyu,
        special::sh_legendre,
    ] {
        assert!(f(257, false).is_err());
        assert!(f(usize::MAX, true).is_err());
    }
    for f in [
        special::roots_legendre,
        special::roots_laguerre,
        special::roots_hermite,
        special::roots_hermitenorm,
        special::roots_chebyt,
        special::roots_chebyu,
        special::roots_chebyc,
        special::roots_chebys,
        special::roots_sh_chebyt,
        special::roots_sh_chebyu,
        special::roots_sh_legendre,
    ] {
        for n in [0, 257, usize::MAX] {
            assert!(f(n).is_err());
        }
    }
    for bad in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(special::jacobi(2, bad, 0.0, false).is_err());
        assert!(special::jacobi(2, 0.0, bad, false).is_err());
        assert!(special::roots_jacobi(2, bad, 0.0).is_err());
        assert!(special::roots_jacobi(2, 0.0, bad).is_err());
        assert!(special::genlaguerre(2, bad, false).is_err());
        assert!(special::roots_genlaguerre(2, bad).is_err());
        assert!(special::gegenbauer(2, bad, false).is_err());
        assert!(special::roots_gegenbauer(2, bad).is_err());
        assert!(special::sh_jacobi(2, 1.0, bad, false).is_err());
        assert!(special::roots_sh_jacobi(2, 1.0, bad).is_err());
    }
    assert!(special::gegenbauer(2, 0.0, true).is_err());
    for n in [0, 4097, usize::MAX] {
        assert!(special::fresnel_zeros(n).is_err());
        assert!(special::fresnels_zeros(n).is_err());
        assert!(special::fresnelc_zeros(n).is_err());
    }
    for n in [129, usize::MAX] {
        assert!(special::legendre_p_all(n, 0.3).is_err());
        assert!(special::assoc_legendre_p_all(n, 1, 0.3).is_err());
        assert!(special::sph_legendre_p_all(1, n, 0.3).is_err());
        assert!(special::sph_harm_y_all(n, 1, 0.3, 0.4).is_err());
        let z = special::sph_harm_y(n, 0, 0.3, 0.4);
        assert!(z.re.is_nan() && z.im.is_nan());
    }
    for n in [-1, 4097, i32::MAX] {
        for f in [
            special::eval_legendre,
            special::eval_laguerre,
            special::eval_chebyt,
            special::eval_chebyu,
            special::eval_hermite,
            special::eval_hermitenorm,
        ] {
            assert!(f(n, 0.3_f64).is_nan());
        }
        assert!(special::eval_jacobi(n, 0.0_f64, 0.0, 0.3).is_nan());
        assert!(special::eval_genlaguerre(n, 0.0_f64, 0.3).is_nan());
        assert!(special::eval_gegenbauer(n, 1.0_f64, 0.3).is_nan());
    }
}
