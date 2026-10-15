mod common;
use common::finite_close;

use sci::{linalg, special as s};
use std::collections::BTreeSet;

fn inventory_names(inventory: &str) -> BTreeSet<&str> {
    inventory
        .lines()
        .skip_while(|line| *line != "#### Functions")
        .skip(1)
        .take_while(|line| !line.starts_with("#### "))
        .filter_map(|line| {
            line.strip_prefix("- ")
                .and_then(|line| line.split('`').nth(1))
        })
        .collect()
}

#[test]
fn inventory_is_covered_by_csv_or_explicit_policies() {
    let inventory = include_str!("../function_coverages/scipy_function_list/scipy.special.md");
    let expected = inventory_names(inventory);
    let unix = inventory.replace("\r\n", "\n");
    let windows = unix.replace('\n', "\r\n");
    assert_eq!(inventory_names(&unix), inventory_names(&windows));
    assert_eq!(expected.len(), 355);
    let mut covered: BTreeSet<_> = common::lines(include_str!("fixtures/special_extended.csv"))
        .map(|line| line.split(',').next().unwrap())
        .collect();
    covered.extend(["geterr", "seterr"]);
    assert!(
        expected.is_subset(&covered),
        "missing: {:?}",
        expected.difference(&covered).collect::<Vec<_>>()
    );
    assert_eq!(s::geterr(), s::ErrorPolicy::Ieee);
    assert!(s::seterr(f64::NAN, s::ErrorPolicy::Finite).is_err());
    assert!(s::seterr(f64::NAN, s::ErrorPolicy::Ieee).unwrap().is_nan());
    assert_eq!(s::geterr(), s::ErrorPolicy::Ieee);
}

#[test]
fn small_sequences_and_native_work_limits_are_safe() {
    for m in 0..=1 {
        for n in 0..=1 {
            let (values, derivatives) = s::lqmn(m, n, 0.3).unwrap();
            assert_eq!(values.len(), m + 1);
            assert_eq!(values[0].len(), n + 1);
            assert_eq!(derivatives.len(), m + 1);
            finite_close(values[0][0], 0.5 * (1.3_f64 / 0.7).ln(), 1e-14, 1e-14);
        }
    }
    assert_eq!(s::lqn(0, 0.3).unwrap().0.len(), 1);
    assert!(s::polygamma(u32::MAX, 1.0).is_nan());
    assert!(s::multigammaln(1.0, u32::MAX).is_nan());
    assert!(s::mathieu_a(u32::MAX, 0.5).is_nan());
    assert!(s::mathieu_even_coef(1, f64::MAX).is_err());
    assert!(s::pro_cv(1, 3, f64::MAX).is_nan());
    assert!(s::pro_cv_seq(1, 3, f64::MAX).is_err());
    assert!(s::chndtr(f64::INFINITY, 3.0, -1.0).is_nan());
    assert!(s::ncfdtr(3.0, 5.0, -1.0, f64::INFINITY).is_nan());
}

#[test]
fn distribution_inverses_round_trip() {
    for p in [0.01, 0.2, 0.5, 0.9, 0.99] {
        for nc in [0.0, 0.5, 5.0] {
            finite_close(s::chndtr(s::chndtrix(p, 3.0, nc), 3.0, nc), p, 2e-13, 2e-13);
            finite_close(
                s::ncfdtr(3.0, 5.0, nc, s::ncfdtri(3.0, 5.0, nc, p)),
                p,
                2e-13,
                2e-13,
            );
            finite_close(s::nctdtr(5.0, nc, s::nctdtrit(5.0, nc, p)), p, 2e-12, 2e-12);
        }
    }
    let p = s::ncfdtr(3.0, 5.0, 0.5, 0.5);
    finite_close(s::ncfdtrinc(3.0, 5.0, p, 0.5), 0.5, 2e-12, 2e-12);
    finite_close(s::ncfdtridfd(3.0, p, 0.5, 0.5), 5.0, 2e-12, 2e-12);
}

#[test]
fn quadrature_composes_with_polynomial_evaluation_and_linalg() {
    let (nodes, weights) = s::roots_legendre(16).unwrap();
    for degree in 0..=30 {
        let values: Vec<_> = nodes.iter().map(|x| x.powi(degree)).collect();
        let expected = if degree % 2 == 0 {
            2.0 / (degree + 1) as f64
        } else {
            0.0
        };
        finite_close(
            linalg::dot(&weights, &values).unwrap(),
            expected,
            3e-14,
            3e-14,
        );
    }
    let coefficients = s::legendre(7, false).unwrap();
    for x in [-0.9, 0.0, 0.3, 0.9] {
        let value = coefficients.iter().fold(0.0, |v, c| v * x + c);
        finite_close(value, s::eval_legendre(7, x), 2e-13, 2e-13);
    }
}

#[test]
fn elementary_limits_and_elliptic_conventions() {
    assert_eq!(s::kl_div(-1.0, 2.0), f64::INFINITY);
    assert_eq!(s::kl_div(0.0, f64::INFINITY), f64::INFINITY);
    assert_eq!(s::pseudo_huber(1.0, f64::INFINITY), f64::INFINITY);
    assert_eq!(s::pseudo_huber(f64::INFINITY, 2.0), 2.0);
    for m in [-2.0, 0.0, 0.5, 0.99] {
        let (k, e) = s::ellipke(m).unwrap();
        finite_close(k, s::ellipk(m).unwrap(), 1e-14, 1e-14);
        finite_close(e, s::ellipe(m).unwrap(), 1e-14, 1e-14);
        finite_close(
            s::cel3((1.0 - m).sqrt(), 0.8).unwrap(),
            s::ellippi(0.2, m).unwrap(),
            1e-14,
            1e-14,
        );
    }
    assert!(s::ellipkm1(1e-20).unwrap().is_finite());
    finite_close(
        s::logsumexp(&[1000.0, 1000.0]),
        1000.0 + 2.0_f64.ln(),
        1e-14,
        1e-14,
    );
}

#[test]
fn kelvin_zero_sequences_agree_and_remain_ordered_after_exponential_underflow() {
    let zeros = s::keip_zeros(4096).unwrap();
    assert_eq!(zeros.len(), 4096);
    assert!(zeros.iter().all(|x| x.is_finite() && *x > 0.));
    assert!(zeros.windows(2).all(|pair| pair[0] < pair[1]));
    for count in [1, 3, 5] {
        let grouped = s::kelvin_zeros(count).unwrap();
        assert_eq!(grouped[7], zeros[..count]);
        for &x in &grouped[7] {
            // The scalar Kelvin series has cancellation near x=9.4;
            // compare the derivative residual on its local envelope scale.
            let envelope = (-x / std::f64::consts::SQRT_2).exp() / x.sqrt();
            assert!(s::keip(x).abs() <= 2e-9 * envelope);
        }
    }
    assert!(s::keip_zeros(0).is_err());
    assert!(s::keip_zeros(4097).is_err());
}
