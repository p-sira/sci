//! Public-domain and work-limit regressions for the pure Rust migration.
use sci::special as s;

#[test]
fn polynomial_constructors_and_quadrature_reject_unsupported_sizes() {
    type Polynomial = fn(usize, bool) -> sci::Result<Vec<f64>>;
    type Quadrature = fn(usize) -> sci::Result<(Vec<f64>, Vec<f64>)>;
    let polynomials: [Polynomial; 12] = [
        s::legendre,
        s::laguerre,
        s::hermite,
        s::hermitenorm,
        s::chebyt,
        s::chebyu,
        s::chebyc,
        s::chebys,
        s::sh_chebyt,
        s::sh_chebyu,
        s::sh_legendre,
        |n, monic| s::gegenbauer(n, 0.5, monic),
    ];
    let quadratures: [Quadrature; 12] = [
        s::roots_legendre,
        s::roots_laguerre,
        s::roots_hermite,
        s::roots_hermitenorm,
        s::roots_chebyt,
        s::roots_chebyu,
        s::roots_chebyc,
        s::roots_chebys,
        s::roots_sh_chebyt,
        s::roots_sh_chebyu,
        s::roots_sh_legendre,
        |n| s::roots_gegenbauer(n, 0.5),
    ];
    for polynomial in polynomials {
        assert!(polynomial(257, false).is_err());
        assert_eq!(polynomial(0, true).unwrap(), [1.0]);
    }
    for quadrature in quadratures {
        assert!(quadrature(0).is_err());
        assert!(quadrature(257).is_err());
    }
    for parameter in [f64::NAN, -1.0, f64::INFINITY] {
        assert!(s::jacobi(2, parameter, 0.0, false).is_err());
        assert!(s::roots_jacobi(2, 0.0, parameter).is_err());
        assert!(s::genlaguerre(2, parameter, false).is_err());
        assert!(s::roots_genlaguerre(2, parameter).is_err());
        assert!(s::gegenbauer(2, parameter, false).is_err());
        assert!(s::roots_gegenbauer(2, parameter).is_err());
        assert!(s::sh_jacobi(2, 0.0, parameter, false).is_err());
        assert!(s::roots_sh_jacobi(2, parameter, 1.0).is_err());
    }
}

#[test]
fn wave_functions_validate_before_entering_recurrences() {
    type Angular = fn(u32, f64, f64) -> (f64, f64);
    for function in [
        s::mathieu_cem,
        s::mathieu_sem,
        s::mathieu_modcem1,
        s::mathieu_modcem2,
        s::mathieu_modsem1,
        s::mathieu_modsem2,
    ] as [Angular; 6]
    {
        for (m, q, x) in [
            (129, 1.0, 0.5),
            (1, f64::NAN, 0.5),
            (1, 101.0, 0.5),
            (1, 1.0, f64::INFINITY),
        ] {
            let value = function(m, q, x);
            assert!(value.0.is_nan() && value.1.is_nan());
        }
    }
    assert!(s::mathieu_b(129, 0.0).is_nan());
    assert!(s::mathieu_b(0, 0.5).is_nan());
    assert!(s::mathieu_odd_coef(0, 1.0).is_err());
    assert!(s::mathieu_odd_coef(129, 1.0).is_err());
    type Wave = fn(u64, u64, f64, f64) -> (f64, f64);
    for function in [
        s::pro_ang1,
        s::pro_rad1,
        s::pro_rad2,
        s::obl_ang1,
        s::obl_rad1,
        s::obl_rad2,
    ] as [Wave; 6]
    {
        for (m, n, c, x) in [
            (2, 1, 1.0, 2.0),
            (0, 129, 1.0, 2.0),
            (0, 1, 65.0, 2.0),
            (0, 1, 1.0, f64::NAN),
        ] {
            let value = function(m, n, c, x);
            assert!(value.0.is_nan() && value.1.is_nan());
        }
    }
    type WithCv = fn(u64, u64, f64, f64, f64) -> (f64, f64);
    for function in [
        s::pro_ang1_cv,
        s::pro_rad1_cv,
        s::pro_rad2_cv,
        s::obl_ang1_cv,
        s::obl_rad1_cv,
        s::obl_rad2_cv,
    ] as [WithCv; 6]
    {
        let value = function(2, 1, 1.0, 2.0, 2.0);
        assert!(value.0.is_nan() && value.1.is_nan());
        let value = function(0, 1, 1.0, f64::NAN, 2.0);
        assert!(value.0.is_nan() && value.1.is_nan());
    }
    assert!(s::obl_cv(0, 129, 1.0).is_nan());
    assert!(s::obl_cv_seq(2, 1, 1.0).is_err());
}

#[test]
fn zero_and_sequence_apis_enforce_work_limits() {
    type Zeros = fn(usize) -> sci::Result<Vec<f64>>;
    for function in [
        s::ber_zeros,
        s::bei_zeros,
        s::ker_zeros,
        s::kei_zeros,
        s::berp_zeros,
        s::beip_zeros,
        s::kerp_zeros,
        s::keip_zeros,
    ] as [Zeros; 8]
    {
        assert!(function(0).is_err());
        assert!(function(4097).is_err());
    }
    assert!(s::kelvin_zeros(0).is_err());
    assert!(s::bernoulli(257).is_err());
    assert!(s::euler(257).is_err());
    assert!(s::pbdv_seq(f64::NAN, 1.0).is_err());
    assert!(s::pbvv_seq(5000.0, 1.0).is_err());
    assert!(s::pbdn_seq(257, 1.0).is_err());
    assert!(s::riccati_jn(4097, 1.0).is_err());
    assert!(s::riccati_yn(1, 0.0).is_err());
    assert!(s::lmbda(f64::NAN, 1.0).is_err());
    assert!(s::jnjnp_zeros(129).is_err());
    assert!(s::lqn(4097, 0.5).is_err());
    assert!(s::lqmn(1, 4097, 0.5).is_err());
}

#[test]
fn elementary_limits_preserve_parity_and_overflow_conventions() {
    let tau = std::f64::consts::TAU;
    assert_eq!(s::diric(tau, 2), -1.0);
    assert_eq!(s::diric(tau, 3), 1.0);
    assert!(s::diric(0.0, 0).is_nan());
    assert_eq!(s::factorial2(-1), 0.0);
    assert_eq!(s::factorial2(301), f64::INFINITY);
    assert!(s::factorialk(1, 0).is_nan());
    assert_eq!(s::factorialk(-1, 3), 0.0);
    assert_eq!(s::factorialk(1000, 1), f64::INFINITY);
    assert!(s::perm(1.0, 100001).is_nan());
    assert_eq!(s::perm(1.0, 2), 0.0);
    assert!(s::stirling2(4097, 2).is_nan());
    assert_eq!(s::stirling2(2, 3), 0.0);
    assert!(s::jvp(1.0, 2.0, 1025).is_nan());
    for function in [
        s::spherical_jn,
        s::spherical_yn,
        s::spherical_in,
        s::spherical_kn,
    ] {
        assert!(function(i64::MAX, 1.0_f64, false).is_nan());
        assert!(function(i64::MAX, 1.0_f64, true).is_nan());
    }
}

#[test]
fn elliptic_real_paths_and_pole_errors_match_documented_domains() {
    use sci::special::elliptic as e;
    let pi = std::f64::consts::PI;
    for (phi, m) in [(pi, 2.0), (0.9, 2.0)] {
        assert!(e::ellipf(phi, m).is_err());
        assert!(e::ellipeinc(phi, m).is_err());
        assert!(e::ellipdinc(phi, m).is_err());
    }
    assert!(e::ellipf(pi, 1.0).is_err());
    assert!(e::ellippiinc(pi / 2.0, 1.0, 0.3).is_err());
    assert!(e::ellippiinc_bulirsch(pi / 2.0, 1.0, 0.3).is_err());
    assert!(e::el3(1.0, 0.8, -1.0).is_err());
    assert!(e::el3(0.3, 1.2, -1.0).is_err());
    assert!(e::cel3(0.8, 0.0).is_err());
    for phi in [-0.4_f64, 0.0, 0.4] {
        // Ellip's F(phi,1) = atanh(sin(phi)) before the first pole.
        let value = e::ellipf(phi, 1.0).unwrap();
        assert!((value - phi.sin().atanh()).abs() < 2e-15);
        assert_eq!(value, e::ellipkinc(phi, 1.0).unwrap());
    }
    assert_eq!(
        e::ellipf(0.4_f32, 2.0).unwrap(),
        e::ellipkinc(0.4_f32, 2.0).unwrap()
    );
}

#[test]
fn hyp1f1_cancellation_matches_independent_precision_checked_series() {
    for row in include_str!("fixtures/hyp1f1_cancellation.csv")
        .lines()
        .skip(1)
    {
        let fields: Vec<f64> = row.split(',').map(|value| value.parse().unwrap()).collect();
        let actual = s::hyp1f1(fields[0], fields[1], fields[2]);
        let expected = fields[3];
        // Relative accuracy matters for exponentially small functions as well.
        // No absolute floor hides cancellation or premature underflow to zero.
        assert!(
            (actual - expected).abs() <= 3e-12 * expected.abs(),
            "{row}: got {actual:e}"
        );
    }
}

#[test]
fn hyp1f1_continuation_identities_and_work_limits() {
    for (a, b, x) in [
        (65.1, -3.2, -20.0),
        (1000.0, 0.2, -100.0),
        (32.0, 3.2, -50.0),
    ] {
        let value = s::hyp1f1(a, b, x);
        let contiguous =
            s::hyp1f1(a, b + 1.0, x) + a * x / (b * (b + 1.0)) * s::hyp1f1(a + 1.0, b + 2.0, x);
        assert!((value - contiguous).abs() <= 3e-12 * value.abs());
    }
    for x in [-20.0_f64, -100.0, -1000.0] {
        let scale = (x * 0.5).exp();
        let expected = ((1.0 + x) * scale) * scale;
        assert_eq!(s::hyp1f1(2.0, 1.0, x), expected);
    }
    assert!(s::hyp1f1(100000.0, 0.2, -1000.0).is_nan()); // Continuation step budget.
    assert!(s::hyp1f1(65.1, -3.0, -20.0).is_infinite()); // Denominator pole unchanged.
    assert_eq!(s::hyp1f1(65.1, 65.1, -20.0), (-20.0_f64).exp());
}
