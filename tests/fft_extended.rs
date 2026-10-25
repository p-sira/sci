mod common;
use common::finite_close;
use common::vector;

use f::RealTransformType as T;
use sci::{complex::Complex, fft as f};
fn sizes(s: &str) -> Vec<usize> {
    vector(s).into_iter().map(|v| v as usize).collect()
}
fn complex(re: &[f64], im: &[f64]) -> Vec<Complex> {
    re.iter()
        .zip(im)
        .map(|(&re, &im)| Complex { re, im })
        .collect()
}
fn rows<T: Clone>(x: &[T], n: usize) -> Vec<Vec<T>> {
    x.chunks(n).map(<[T]>::to_vec).collect()
}
#[test]
fn public_fft_inventory_matches_scipy() {
    let inputs: Vec<_> = common::lines(include_str!("fixtures/fft_inputs.csv")).collect();
    let expected: Vec<_> = common::lines(include_str!("fixtures/fft_expected.csv")).collect();
    assert_eq!(inputs.len(), expected.len());
    let mut covered = std::collections::BTreeSet::new();
    for (input, expected) in inputs.iter().zip(expected) {
        let c: Vec<_> = common::fields(input);
        let e: Vec<_> = common::fields(expected);
        assert_eq!(c[0], e[0]);
        covered.insert(c[1]);
        let re = vector(c[2]);
        let im = vector(c[3]);
        let x = complex(&re, &im);
        let shape = sizes(c[4]);
        let axes = sizes(c[5]);
        let n = if c[6].is_empty() {
            None
        } else {
            Some(c[6].parse::<usize>().unwrap())
        };
        let kind = match c[10] {
            "1" => T::I,
            "2" => T::II,
            "3" => T::III,
            _ => T::IV,
        };
        let (got, got_shape) = match c[1] {
            "fht" | "ifht" | "fhtoffset" => {
                let dln = c[11].parse().unwrap();
                let mu = c[12].parse().unwrap();
                let offset = c[13].parse().unwrap();
                let bias = c[14].parse().unwrap();
                let v = match c[1] {
                    "fht" => f::fht(&re, dln, mu, offset, bias).unwrap(),
                    "ifht" => f::ifht(&re, dln, mu, offset, bias).unwrap(),
                    _ => vec![f::fhtoffset(dln, mu, offset, bias).unwrap()],
                };
                let len = v.len();
                (complex(&v, &vec![0.0; len]), vec![len])
            }
            "fft" => (f::fft(&x), vec![x.len()]),
            "ifft" => (f::ifft(&x), vec![x.len()]),
            "rfft" => {
                let v = f::rfft(&re);
                let len = v.len();
                (v, vec![len])
            }
            "ihfft" => {
                let v = f::ihfft(&re, n).unwrap();
                let len = v.len();
                (v, vec![len])
            }
            "irfft" | "hfft" => {
                let v = if c[1] == "irfft" {
                    f::irfft(&x, n)
                } else {
                    f::hfft(&x, n)
                }
                .unwrap();
                let len = v.len();
                (complex(&v, &vec![0.0; len]), vec![len])
            }
            "fftfreq" | "rfftfreq" => {
                let d = c[7].parse().unwrap();
                let v = if c[1] == "fftfreq" {
                    f::fftfreq(n.unwrap(), d)
                } else {
                    f::rfftfreq(n.unwrap(), d)
                }
                .unwrap();
                let len = v.len();
                (complex(&v, &vec![0.0; len]), vec![len])
            }
            "next_fast_len" | "prev_fast_len" => {
                let target = c[8].parse().unwrap();
                let real = c[9] == "1";
                let v = if c[1] == "next_fast_len" {
                    f::next_fast_len(target, real)
                } else {
                    f::prev_fast_len(target, real)
                }
                .unwrap();
                (
                    vec![Complex {
                        re: v as f64,
                        im: 0.0,
                    }],
                    vec![1],
                )
            }
            "fftshift" => (f::fftshift(&x).unwrap(), vec![x.len()]),
            "ifftshift" => (f::ifftshift(&x).unwrap(), vec![x.len()]),
            "fftn" => (f::fftn(&x, &shape, &axes).unwrap(), shape.clone()),
            "ifftn" => (f::ifftn(&x, &shape, &axes).unwrap(), shape.clone()),
            "rfftn" => f::rfftn(&re, &shape, &axes).unwrap(),
            "ihfftn" => f::ihfftn(&re, &shape, &axes).unwrap(),
            "irfftn" | "hfftn" => {
                let v = if c[1] == "irfftn" {
                    f::irfftn(&x, &shape, &axes)
                } else {
                    f::hfftn(&x, &shape, &axes)
                }
                .unwrap();
                (complex(&v, &vec![0.0; v.len()]), shape.clone())
            }
            "fft2" | "ifft2" => {
                let m = rows(&x, shape[1]);
                let v = if c[1] == "fft2" {
                    f::fft2(&m)
                } else {
                    f::ifft2(&m)
                }
                .unwrap();
                (v.into_iter().flatten().collect(), shape.clone())
            }
            "rfft2" | "ihfft2" => {
                let m = rows(&re, shape[1]);
                let v = if c[1] == "rfft2" {
                    f::rfft2(&m)
                } else {
                    f::ihfft2(&m)
                }
                .unwrap();
                let out_shape = vec![v.len(), v[0].len()];
                (v.into_iter().flatten().collect(), out_shape)
            }
            "irfft2" | "hfft2" => {
                let m = rows(&x, shape[1] / 2 + 1);
                let v = if c[1] == "irfft2" {
                    f::irfft2(&m, n.unwrap())
                } else {
                    f::hfft2(&m, n.unwrap())
                }
                .unwrap();
                let out_shape = vec![v.len(), v[0].len()];
                let v: Vec<_> = v.into_iter().flatten().collect();
                (complex(&v, &vec![0.0; v.len()]), out_shape)
            }
            "dct" | "idct" | "dst" | "idst" | "dctn" | "idctn" | "dstn" | "idstn" => {
                let v = match c[1] {
                    "dct" => f::dct(&re, kind),
                    "idct" => f::idct(&re, kind),
                    "dst" => f::dst(&re, kind),
                    "idst" => f::idst(&re, kind),
                    "dctn" => f::dctn(&re, &shape, &axes, kind),
                    "idctn" => f::idctn(&re, &shape, &axes, kind),
                    "dstn" => f::dstn(&re, &shape, &axes, kind),
                    _ => f::idstn(&re, &shape, &axes, kind),
                }
                .unwrap();
                (complex(&v, &vec![0.0; v.len()]), shape.clone())
            }
            _ => panic!("unknown API"),
        };
        assert_eq!(got_shape, sizes(e[1]), "{} shape", c[0]);
        let want = complex(&vector(e[2]), &vector(e[3]));
        assert_eq!(got.len(), want.len(), "{} length", c[0]);
        let tolerance: f64 = c[15].parse().unwrap();
        for (a, b) in got.into_iter().zip(want) {
            assert!(
                (a.re - b.re).abs() <= tolerance * (1.0 + b.re.abs()),
                "{}: expected {}, got {}",
                c[0],
                b.re,
                a.re
            );
            finite_close(a.im, b.im, tolerance, tolerance);
        }
    }
    assert_eq!(covered.len(), 35);
}
#[test]
fn inverse_and_interoperability_properties() {
    for n in [1, 2, 3, 5, 8, 17, 31] {
        let x: Vec<_> = (0..n).map(|i| (i as f64).sin()).collect();
        let bins = f::rfft(&x);
        let restored = f::irfft(&bins, Some(n)).unwrap();
        for (a, b) in restored.into_iter().zip(&x) {
            finite_close(a, *b, 3e-12, 3e-12);
        }
        let hermitian = f::ihfft(&x, None).unwrap();
        let restored = f::hfft(&hermitian, Some(n)).unwrap();
        for (a, b) in restored.into_iter().zip(&x) {
            finite_close(a, *b, 3e-12, 3e-12);
        }
        let full = complex(&x, &vec![0.0; n]);
        assert_eq!(f::ifftshift(&f::fftshift(&full).unwrap()).unwrap(), full);
        for kind in [T::I, T::II, T::III, T::IV] {
            if n > 1 || kind != T::I {
                let restored = f::idct(&f::dct(&x, kind).unwrap(), kind).unwrap();
                for (a, b) in restored.into_iter().zip(&x) {
                    finite_close(a, *b, 3e-12, 3e-12);
                }
            }
            let restored = f::idst(&f::dst(&x, kind).unwrap(), kind).unwrap();
            for (a, b) in restored.into_iter().zip(&x) {
                finite_close(a, *b, 3e-12, 3e-12);
            }
        }
    }
    let shape = [3, 2, 5];
    let x: Vec<_> = (0..30).map(|i| (i as f64 / 3.0).cos()).collect();
    for axes in [&[0, 1, 2][..], &[2, 0][..], &[1][..]] {
        let (bins, small) = f::rfftn(&x, &shape, axes).unwrap();
        assert_eq!(bins.len(), small.iter().product());
        for (a, b) in f::irfftn(&bins, &shape, axes).unwrap().iter().zip(&x) {
            finite_close(*a, *b, 3e-12, 3e-12);
        }
        let (bins, _) = f::ihfftn(&x, &shape, axes).unwrap();
        for (a, b) in f::hfftn(&bins, &shape, axes).unwrap().iter().zip(&x) {
            finite_close(*a, *b, 3e-12, 3e-12);
        }
        for kind in [T::I, T::II, T::III, T::IV] {
            let y = f::dctn(&x, &shape, axes, kind).unwrap();
            for (a, b) in f::idctn(&y, &shape, axes, kind).unwrap().iter().zip(&x) {
                finite_close(*a, *b, 3e-12, 3e-12);
            }
            let y = f::dstn(&x, &shape, axes, kind).unwrap();
            for (a, b) in f::idstn(&y, &shape, axes, kind).unwrap().iter().zip(&x) {
                finite_close(*a, *b, 3e-12, 3e-12);
            }
        }
    }
    let matrix = rows(&x[..10], 5);
    let spectrum = f::rfft2(&matrix).unwrap();
    let restored = f::irfft2(&spectrum, 5).unwrap();
    for (a, b) in restored.iter().flatten().zip(matrix.iter().flatten()) {
        finite_close(*a, *b, 3e-12, 3e-12);
    }
    let power: Vec<_> = spectrum
        .iter()
        .flatten()
        .map(|z| sci::complex::norm_sqr(*z))
        .collect();
    assert!(sci::stats::mean(&power).unwrap() > 0.0);
    assert_eq!(
        f::fftn(&complex(&x, &[0.0; 30]), &shape, &[]).unwrap(),
        complex(&x, &[0.0; 30])
    );
    for real in [false, true] {
        for target in [1, 17, 93059, f::prev_fast_len(1_048_576, real).unwrap()] {
            let next = f::next_fast_len(target, real).unwrap();
            let prev = f::prev_fast_len(target, real).unwrap();
            assert!(prev <= target && target <= next);
            assert_eq!(f::next_fast_len(next, real).unwrap(), next);
            assert_eq!(f::prev_fast_len(prev, real).unwrap(), prev);
        }
    }
}
#[test]
fn validation_and_special_values() {
    let c = Complex { re: 1.0, im: 2.0 };
    let bad = Complex {
        re: f64::NAN,
        im: 0.0,
    };
    for shape in [&[][..], &[0][..], &[2][..], &[1_048_577][..], &[1; 17][..]] {
        assert!(f::fftn(&[c], shape, &[0]).is_err());
        assert!(f::rfftn(&[1.0], shape, &[0]).is_err());
    }
    for axes in [&[1][..], &[0, 0][..]] {
        assert!(f::ifftn(&[c], &[1], axes).is_err());
        assert!(f::irfftn(&[c], &[1], axes).is_err());
    }
    assert!(f::rfftn(&[1.0], &[1], &[]).is_err());
    assert!(f::irfftn(&[c], &[1], &[]).is_err());
    assert!(f::fftn(&[bad], &[1], &[0]).is_err());
    assert!(f::rfftn(&[f64::INFINITY], &[1], &[0]).is_err());
    assert!(f::irfft(&[c], None).is_err());
    assert!(f::irfft(&[], None).is_err());
    assert!(f::irfft(&[bad], Some(1)).is_err());
    assert!(f::hfft(&[c], Some(0)).is_err());
    assert!(f::ihfft(&[f64::NAN], None).is_err());
    assert_eq!(f::irfft(&[], Some(3)).unwrap(), vec![0.0; 3]);
    assert_eq!(f::hfft(&[], Some(3)).unwrap(), vec![0.0; 3]);
    assert_eq!(f::ihfft(&[], Some(3)).unwrap().len(), 2);
    let a = f::irfft(
        &[
            Complex { re: 1.0, im: 100.0 },
            Complex { re: 2.0, im: 200.0 },
        ],
        Some(2),
    )
    .unwrap();
    assert_eq!(a, vec![1.5, -0.5]);
    assert!(f::fft2(&[vec![c], vec![]]).is_err());
    assert!(f::rfft2(&[vec![]]).is_err());
    assert!(f::irfft2(&[vec![c]], 0).is_err());
    assert!(f::irfft2(&[vec![c]], 4).is_err());
    for d in [0.0, f64::NAN, f64::INFINITY, f64::from_bits(1)] {
        assert!(f::fftfreq(2, d).is_err());
        assert!(f::rfftfreq(2, d).is_err());
    }
    assert!(f::fftfreq(0, 1.0).is_err());
    assert!(f::next_fast_len(1_048_577, false).is_err());
    assert!(f::prev_fast_len(usize::MAX, true).is_err());
    assert!(f::fftshift(&[bad]).is_err());
    assert!(f::ifftshift(&[bad]).is_err());
    assert!(f::fftshift(&[]).unwrap().is_empty());
    assert!(f::dct(&[1.0], T::I).is_err());
    assert!(f::idct(&[], T::II).is_err());
    assert!(f::dst(&[f64::INFINITY], T::III).is_err());
    assert!(f::idst(&[f64::NAN], T::IV).is_err());
    assert!(f::dctn(&[1.0], &[1], &[0], T::I).is_err());
    assert!(f::dstn(&[1.0], &[1], &[0, 0], T::II).is_err());
    assert!(f::dct(&[f64::MAX; 2], T::II).is_err());
    assert!(
        f::fftn(
            &[Complex {
                re: f64::MAX,
                im: 0.0
            }; 2],
            &[2],
            &[0]
        )
        .is_err()
    );
}

#[test]
fn fftlog_round_trips_and_domains() {
    for (dln, mu, bias) in [
        (0.1, 0.0, 0.0),
        (0.3, 1.5, 0.2),
        (-0.2, 0.4, 0.0),
        (0.5, -2.2, -0.1),
    ] {
        let offset = f::fhtoffset(dln, mu, 0.5, bias).unwrap();
        assert!((offset - 0.5).abs() <= dln.abs() / 2.0 + 1e-12);
        let x: Vec<_> = (0..31).map(|i| (i as f64 * 0.3).sin()).collect();
        let y = f::fht(&x, dln, mu, offset, bias).unwrap();
        let restored = f::ifht(&y, dln, mu, offset, bias).unwrap();
        for (a, b) in restored.iter().zip(&x) {
            assert!(approx::abs_diff_eq!(
                a,
                b,
                epsilon = 3e-11 * (1.0 + b.abs())
            ));
        }
    }
    for (dln, mu, offset, bias) in [
        (0.0, 0.0, 0.0, 0.0),
        (f64::NAN, 0.0, 0.0, 0.0),
        (0.1, f64::INFINITY, 0.0, 0.0),
        (0.1, 0.0, f64::NAN, 0.0),
        (0.1, 0.0, 0.0, f64::INFINITY),
    ] {
        assert!(f::fht(&[1.0, 2.0], dln, mu, offset, bias).is_err());
        assert!(f::ifht(&[1.0, 2.0], dln, mu, offset, bias).is_err());
        assert!(f::fhtoffset(dln, mu, offset, bias).is_err());
    }
    assert!(f::fht(&[], 0.1, 0.0, 0.0, 0.0).is_err());
    assert!(f::ifht(&[f64::NAN], 0.1, 0.0, 0.0, 0.0).is_err());
    assert!(f::fht(&[1.0, 2.0], 0.1, 0.0, 0.0, 1.0).is_err());
    assert!(f::ifht(&[1.0, 2.0], 0.1, 0.0, 0.0, 1.0).is_err());
    assert!(f::fht(&[1.0, 2.0], 1000.0, 0.0, 0.0, 2.0).is_err());
    assert!(f::ifht(&[1.0, 2.0], 1000.0, 0.0, 0.0, 2.0).is_err());
}

#[test]
fn fftlog_dc_coefficients_obey_gamma_recurrence() {
    // At n=1 and offset=0, the transform is multiplication by
    // 2^q Gamma((mu+1+q)/2) / Gamma((mu+1-q)/2).
    for mu in [-6.0, -2.0, -0.5, 0.5, 2.0, 6.0, 1e6] {
        for (bias, coefficient) in [(1.0, mu), (2.0, mu * mu - 1.0)] {
            let value = f::fht(&[1.0], 0.1, mu, 0.0, bias).unwrap();
            finite_close(value[0], coefficient, 2e-14, 2e-14);
            let inverse = f::ifht(&[coefficient], 0.1, mu, 0.0, bias).unwrap();
            finite_close(inverse[0], 1.0, 2e-14, 2e-14);
        }
    }
    // Extreme finite parameters must finish within the fixed coefficient work
    // limits and report arithmetic failure rather than entering long recurrences.
    assert!(f::fht(&[1.0], 0.1, 0.0, 0.0, 1e300).is_err());
    assert!(f::fhtoffset(f64::from_bits(1), 0.0, 0.0, 0.0).is_err());
}
