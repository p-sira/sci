mod common;

use common::{fields, integer, lines, vector};
use sci::stats::qmc::{QmcEngine, Sobol};

#[test]
fn sobol_matches_unscrambled_scipy_reference() {
    let cases: Vec<_> = lines(include_str!("scipy/sobol_cases.csv")).collect();
    let expected: Vec<_> = lines(include_str!("scipy/sobol_expected.csv")).collect();
    assert_eq!(cases.len(), 29);
    assert_eq!(cases.len(), expected.len());

    for (case, reference) in cases.iter().zip(&expected) {
        let c = fields(case);
        let e = fields(reference);
        assert_eq!(c.len(), 6, "malformed case: {case}");
        assert_eq!(e.len(), 4, "malformed reference: {reference}");
        let id = c[0];
        assert_eq!(id, e[0], "case/reference ordering mismatch");

        let d = integer(c[1]);
        let bits = integer(c[2]);
        let skip = integer(c[4]);
        let arg = integer(c[5]);
        let rows = integer(e[1]);
        let cols = integer(e[2]);
        let expected_points = vector(e[3]);
        assert_eq!(cols, d, "{id}: expected dimension mismatch");
        assert_eq!(expected_points.len(), rows * cols, "{id}: fixture shape");

        let mut engine = Sobol::new(d, bits).unwrap();
        if skip > 0 {
            engine.fast_forward(skip).unwrap();
        }
        let points = match c[3] {
            "random" => engine.random(arg).unwrap(),
            "base2" => engine.random_base2(arg).unwrap(),
            other => panic!("{id}: unknown method {other}"),
        };
        assert_eq!(points.len(), rows, "{id}: row count");
        for (i, row) in points.iter().enumerate() {
            assert_eq!(row.len(), d, "{id}: row {i} column count");
            for (j, &actual) in row.iter().enumerate() {
                let expected = expected_points[i * d + j];
                assert_eq!(actual, expected, "{id}: point [{i},{j}]");
                assert!(
                    (0.0..1.0).contains(&actual),
                    "{id}: point [{i},{j}] out of range"
                );
            }
        }
    }
}

#[test]
fn sobol_reset_chunking_and_fast_forward_agree() {
    let mut engine = Sobol::new(5, 10).unwrap();
    let complete = engine.random_base2(4).unwrap();
    assert_eq!(complete.len(), 16);

    engine.reset();
    let mut chunked = engine.random(5).unwrap();
    chunked.extend(engine.random(11).unwrap());
    assert_eq!(chunked, complete, "chunking changed sequence");

    engine.reset();
    engine.fast_forward(5).unwrap();
    assert_eq!(
        engine.random(11).unwrap(),
        complete[5..].to_vec(),
        "fast-forward mismatch"
    );

    engine.reset();
    let mut balanced = engine.random_base2(2).unwrap();
    balanced.extend(engine.random_base2(2).unwrap());
    assert_eq!(
        balanced,
        complete[..8].to_vec(),
        "base2 calls changed sequence"
    );
}

#[test]
fn sobol_first_coordinate_is_a_complete_dyadic_partition() {
    let points = Sobol::new(3, 5).unwrap().random_base2(5).unwrap();
    let mut first: Vec<_> = points.iter().map(|row| row[0]).collect();
    first.sort_by(f64::total_cmp);
    let expected: Vec<_> = (0..32).map(|n| n as f64 / 32.0).collect();
    assert_eq!(first, expected);
}

#[test]
fn sobol_validates_parameters_and_period() {
    assert!(Sobol::new(0, 8).is_err());
    assert!(Sobol::new(1, 0).is_err());
    assert!(Sobol::new(1, 65).is_err());

    let mut engine = Sobol::new(2, 3).unwrap();
    assert!(
        engine.random_base2(4).is_err(),
        "exponent exceeds bit width"
    );
    assert_eq!(engine.random(8).unwrap().len(), 8);
    assert!(
        engine.random(1).is_err(),
        "generator exceeded 2^bits capacity"
    );

    let mut engine = Sobol::new(2, 3).unwrap();
    assert!(
        engine.fast_forward(9).is_err(),
        "fast-forward exceeded capacity"
    );

    let mut engine = Sobol::new(2, 6).unwrap();
    engine.random_base2(2).unwrap();
    assert!(
        engine.random_base2(1).is_err(),
        "base2 cumulative count is not a power of two"
    );
}

#[test]
fn sobol_scrambled_is_reproducible_and_stratified() {
    let mut a = Sobol::scrambled(3, 8, 12345).unwrap();
    let mut b = Sobol::scrambled(3, 8, 12345).unwrap();
    let mut c = Sobol::scrambled(3, 8, 67890).unwrap();
    let first = a.random_base2(8).unwrap();
    assert_eq!(first, b.random_base2(8).unwrap());
    assert_ne!(first, c.random_base2(8).unwrap());
    assert_eq!(a.num_generated(), 256);
    assert_eq!(a.dimensions(), 3);
    assert_eq!(a.bits(), 8);

    for axis in 0..3 {
        let mut strata: Vec<_> = first.iter().map(|row| row[axis]).collect();
        strata.sort_by(f64::total_cmp);
        let expected: Vec<_> = (0..256).map(|n| n as f64 / 256.0).collect();
        assert_eq!(strata, expected, "scramble lost an axis's dyadic partition");
    }

    a.reset();
    assert_eq!(a.num_generated(), 0);
    a.fast_forward(100).unwrap();
    assert_eq!(a.random(5).unwrap(), first[100..105]);
}

#[test]
fn sobol_max_dimensions_and_large_64_bit_indices() {
    let mut high_dim = Sobol::new(21_201, 2).unwrap();
    let first = high_dim.random_base2(1).unwrap();
    assert_eq!(first.len(), 2);
    assert!(first[0].iter().all(|&x| x == 0.0));
    assert!(first[1].iter().all(|&x| x == 0.5));
    assert!(Sobol::new(21_202, 2).is_err());
    assert!(Sobol::new(21_201, 64).unwrap().random(50).is_err());

    #[cfg(target_pointer_width = "64")]
    {
        let mut engine = Sobol::new(1, 64).unwrap();
        engine.fast_forward(usize::MAX).unwrap();
        assert_eq!(engine.random(1).unwrap(), vec![vec![2_f64.powi(-64)]]);
        assert!(engine.random(1).is_err());

        let mut almost_one = Sobol::new(1, 64).unwrap();
        let max_gray_index = 0xaaaa_aaaa_aaaa_aaaa_u64 as usize;
        almost_one.fast_forward(max_gray_index).unwrap();
        let point = almost_one.random(1).unwrap()[0][0];
        assert_eq!(point, f64::from_bits(1.0_f64.to_bits() - 1));
    }
}

#[test]
fn qmc_integer_mapping_respects_bounds_and_stream_state() {
    let mut generator = Sobol::new(2, 8).unwrap();
    assert_eq!(
        QmcEngine::integers(&mut generator, 4, &[0, 10], &[4, 14], false).unwrap(),
        vec![vec![0, 10], vec![2, 12], vec![3, 11], vec![1, 13]]
    );
    assert_eq!(QmcEngine::num_generated(&generator), 4);
    assert!(QmcEngine::integers(&mut generator, 1, &[0], &[4], false).is_err());
    assert!(QmcEngine::integers(&mut generator, 1, &[4, 10], &[4, 14], false).is_err());
    assert_eq!(QmcEngine::num_generated(&generator), 4);

    QmcEngine::reset(&mut generator);
    let wide = QmcEngine::integers(
        &mut generator,
        2,
        &[i64::MIN, i64::MIN],
        &[i64::MAX, i64::MAX],
        true,
    )
    .unwrap();
    assert_eq!(wide[0], [i64::MIN, i64::MIN]);
    assert_eq!(wide[1], [0, 0]);
}
