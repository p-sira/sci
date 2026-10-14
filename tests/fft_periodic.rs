mod common;
use common::vector;
use sci::fft as f;
#[track_caller]
fn close(a: f64, b: f64) {
    assert!(a.is_finite() && b.is_finite());
    common::assert_close(a, b, 3e-12, 3e-12, "numerical comparison");
}
#[test]
fn public_periodic_operators_match_scipy() {
    let inputs: Vec<_> = common::lines(include_str!("fixtures/fft_periodic_inputs.csv")).collect();
    let expected: Vec<_> =
        common::lines(include_str!("fixtures/fft_periodic_expected.csv")).collect();
    assert_eq!(inputs.len(), expected.len());
    let mut covered = std::collections::BTreeSet::new();
    for (input, expected) in inputs.iter().zip(expected) {
        let c: Vec<_> = common::fields(input);
        let e: Vec<_> = common::fields(expected);
        assert_eq!(c[0], e[0]);
        covered.insert(c[1]);
        let re = vector(c[2]);
        let got = match c[1] {
            "diff" => f::diff(&re, c[7].parse().unwrap(), c[6].parse().unwrap()),
            "hilbert" => f::hilbert(&re),
            "ihilbert" => f::ihilbert(&re),
            "tilbert" => f::tilbert(&re, c[5].parse().unwrap(), c[6].parse().unwrap()),
            "itilbert" => f::itilbert(&re, c[5].parse().unwrap(), c[6].parse().unwrap()),
            "shift" => f::shift(&re, c[3].parse().unwrap(), c[6].parse().unwrap()),
            "cc_diff" => f::cc_diff(
                &re,
                c[3].parse().unwrap(),
                c[4].parse().unwrap(),
                c[6].parse().unwrap(),
            ),
            "cs_diff" => f::cs_diff(
                &re,
                c[3].parse().unwrap(),
                c[4].parse().unwrap(),
                c[6].parse().unwrap(),
            ),
            "sc_diff" => f::sc_diff(
                &re,
                c[3].parse().unwrap(),
                c[4].parse().unwrap(),
                c[6].parse().unwrap(),
            ),
            "ss_diff" => f::ss_diff(
                &re,
                c[3].parse().unwrap(),
                c[4].parse().unwrap(),
                c[6].parse().unwrap(),
            ),
            _ => panic!("unknown API"),
        }
        .unwrap();
        assert_eq!(vector(e[1]), vec![re.len() as f64], "{} shape", c[0]);
        let want = vector(e[2]);
        assert_eq!(got.len(), want.len(), "{} length", c[0]);
        assert!(vector(e[3]).iter().all(|v| *v == 0.0));
        for (a, b) in got.into_iter().zip(want) {
            close(a, b);
        }
    }
    assert_eq!(covered.len(), 10);
}
#[test]
fn properties_and_interoperability() {
    let n = 17;
    let x: Vec<_> = (0..n)
        .map(|j| (j as f64 * std::f64::consts::TAU / n as f64).sin())
        .collect();
    let p = std::f64::consts::TAU;
    for (a, b) in f::diff(&x, 1, p).unwrap().iter().enumerate() {
        close(*b, (a as f64 * p / n as f64).cos());
    }
    for order in [-3, -2, -1, 1, 2, 3] {
        let y = f::diff(&f::diff(&x, order, p).unwrap(), -order, p).unwrap();
        for (a, b) in y.iter().zip(&x) {
            close(*a, *b);
        }
    }
    let y = f::hilbert(&x).unwrap();
    let analytic = sci::signal::hilbert(&x).unwrap();
    for (a, b) in y.iter().zip(&analytic) {
        close(*a, -b.im);
    }
    let z = f::ihilbert(&y).unwrap();
    for (a, b) in z.iter().zip(&x) {
        close(*a, *b);
    }
    let z = f::tilbert(&f::itilbert(&x, 0.3, p).unwrap(), 0.3, p).unwrap();
    for (a, b) in z.iter().zip(&x) {
        close(*a, *b);
    }
    for (a, b) in f::shift(&x, p / n as f64, p)
        .unwrap()
        .iter()
        .zip(x.iter().cycle().skip(1))
    {
        close(*a, *b);
    }
    let alternate: Vec<_> = (0..6)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let phase: f64 = 3.0 * (-0.2) * p / 10.0;
    let scale = phase.cos() + phase.sin();
    for (a, b) in f::shift(&alternate, -0.2, 10.0)
        .unwrap()
        .iter()
        .zip(&alternate)
    {
        close(*a, *b * scale);
    }
    for (a, b) in f::cc_diff(&[1.0, 2.0, 3.0], 1000.0, 1000.0, p)
        .unwrap()
        .iter()
        .zip([1.0, 2.0, 3.0])
    {
        close(*a, b);
    }
}
#[test]
fn invalid_domains_and_work_bounds() {
    for x in [
        vec![],
        vec![f64::NAN],
        vec![f64::INFINITY],
        vec![1.0; 1_048_577],
    ] {
        assert!(f::diff(&x, 1, 1.0).is_err());
        assert!(f::hilbert(&x).is_err());
        assert!(f::ihilbert(&x).is_err());
        assert!(f::tilbert(&x, 1.0, 1.0).is_err());
        assert!(f::itilbert(&x, 1.0, 1.0).is_err());
        assert!(f::shift(&x, 0.1, 1.0).is_err());
        assert!(f::cc_diff(&x, 1.0, 1.0, 1.0).is_err());
        assert!(f::cs_diff(&x, 1.0, 1.0, 1.0).is_err());
        assert!(f::sc_diff(&x, 1.0, 1.0, 1.0).is_err());
        assert!(f::ss_diff(&x, 1.0, 1.0, 1.0).is_err());
    }
    assert!(f::diff(&[1.0], 65, 1.0).is_err());
    assert!(f::diff(&[1.0], i32::MIN, 1.0).is_err());
    for period in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(f::diff(&[1.0], 0, period).is_err());
        assert!(f::shift(&[1.0], 0.0, period).is_err());
    }
    assert!(f::cs_diff(&[1.0], 1.0, 0.0, 1.0).is_err());
    assert!(f::ss_diff(&[1.0], 1.0, 0.0, 1.0).is_err());
    assert!(f::cc_diff(&[1.0], f64::NAN, 1.0, 1.0).is_err());
    assert!(f::sc_diff(&[1.0], 1.0, f64::INFINITY, 1.0).is_err());
    assert!(f::tilbert(&[1.0], 0.0, 1.0).is_err());
    assert!(f::itilbert(&[1.0], f64::NAN, 1.0).is_err());
    assert!(f::shift(&[1.0], f64::INFINITY, 1.0).is_err());
    assert!(f::cc_diff(&[1.0, 2.0, 3.0], 1000.0, 1.0, 1.0).is_err());
}
