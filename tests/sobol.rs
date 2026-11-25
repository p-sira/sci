mod common;

use common::{fields, integer, lines, vector};
use sci::stats::qmc::{QmcEngine, Sobol};

#[test]
fn sobol_flat_buffer_reuse_and_invalid_calls() {
    let mut stream = Sobol::scrambled(6, 30, 71).unwrap();
    let mut comparison = stream.clone();
    let mut buffer = Vec::with_capacity(4096);
    for count in [8, 128, 1, 0, 256, 7] {
        stream.random_flat_into(count, &mut buffer).unwrap();
        assert_eq!(buffer, comparison.random_flat(count).unwrap());
    }
    let original = buffer.clone();
    let position = stream.num_generated();
    assert!(stream.random_flat_into(1_048_577, &mut buffer).is_err());
    assert_eq!(buffer, original);
    assert_eq!(stream.num_generated(), position);
    stream.random_flat_into(17, &mut buffer).unwrap();
    assert_eq!(buffer, comparison.random_flat(17).unwrap());
}

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

/// Frozen before sci-stats 0.3.0's Sobol performance changes. Checking bitwise
/// outputs across seeds, dimensions and precisions catches scramble and state
/// changes that can survive ordinary numerical-tolerance comparisons.
#[test]
fn sobol_preoptimization_digital_streams_are_unchanged() {
    fn fingerprint(points: &[Vec<f64>]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for row in points {
            for &value in row {
                hash = (hash ^ value.to_bits()).wrapping_mul(0x100_0000_01b3);
            }
        }
        hash
    }

    // (dimensions, bits, seed, number of points, expected FNV-1a over f64 bits)
    for (d, bits, seed, n, expected) in [
        (1, 1, 0, 2, 0xd552_8807_b4eb_6fed),
        (6, 30, 42, 64, 0x7b2e_be5f_1cee_8125),
        (6, 30, 123_456_789, 1024, 0x6c6a_0965_2148_0325),
        (6, 64, 42, 64, 0x457f_8edb_8a43_6609),
        (16, 32, 123, 1024, 0x9f3a_154b_01c7_2325),
        (64, 53, 7, 64, 0x687b_ec14_46bb_0ced),
        (1024, 30, 100, 8, 0x63c4_dc2f_7f54_a325),
    ] {
        let mut engine = Sobol::scrambled(d, bits, seed).unwrap();
        let actual = fingerprint(&engine.random(n).unwrap());
        assert_eq!(
            actual, expected,
            "scrambled d={d}, bits={bits}, seed={seed}"
        );
    }

    for (d, bits, n, expected) in [
        (6, 30, 64, 0x40f9_c345_c0ee_8125),
        (64, 53, 64, 0xfcd7_0c83_ce3b_6325),
        (1024, 30, 8, 0xbb11_03fd_6854_a325),
    ] {
        let mut engine = Sobol::new(d, bits).unwrap();
        let actual = fingerprint(&engine.random(n).unwrap());
        assert_eq!(actual, expected, "unscrambled d={d}, bits={bits}");
    }
}

fn row_major_bits(rows: Vec<Vec<f64>>) -> Vec<u64> {
    rows.into_iter()
        .flat_map(|row| row.into_iter().map(f64::to_bits))
        .collect()
}

fn flat_bits(values: Vec<f64>) -> Vec<u64> {
    values.into_iter().map(f64::to_bits).collect()
}

#[test]
fn sobol_random_flat_matches_row_major_random_bit_for_bit() {
    for (d, bits, seed, n) in [
        (1, 1, None, 2),
        (2, 8, None, 64),
        (6, 30, Some(42), 64),
        (6, 32, Some(65_537), 1024),
        (16, 53, Some(123), 128),
        (64, 64, Some(42), 64),
        (256, 30, None, 8),
    ] {
        let make_engine = || match seed {
            Some(seed) => Sobol::scrambled(d, bits, seed).unwrap(),
            None => Sobol::new(d, bits).unwrap(),
        };
        let mut nested = make_engine();
        let mut flat = make_engine();
        assert!(flat.random_flat(0).unwrap().is_empty());
        assert_eq!(flat.num_generated(), 0);
        let expected = row_major_bits(nested.random(n).unwrap());
        let actual = flat_bits(flat.random_flat(n).unwrap());
        assert_eq!(actual.len(), n * d, "flat shape for d={d}, bits={bits}");
        assert_eq!(actual, expected, "d={d}, bits={bits}, seed={seed:?}");
        assert_eq!(flat.num_generated(), nested.num_generated());
    }
}

#[test]
fn sobol_large_row_batches_preserve_flat_bits_and_next_state() {
    for (d, bits, seed, n) in [
        (4, 30, None, 8192),
        (4, 53, Some(42), 8192),
        (16, 64, Some(65_537), 4096),
        (256, 30, None, 256),
    ] {
        let make = || match seed {
            Some(seed) => Sobol::scrambled(d, bits, seed).unwrap(),
            None => Sobol::new(d, bits).unwrap(),
        };
        let mut nested = make();
        let mut flat = make();
        let first = n / 3;
        let second = n - first;
        for count in [first, second] {
            let points = nested.random(count).unwrap();
            assert_eq!(points.len(), count);
            assert!(points.iter().all(|row| row.len() == d));
            assert_eq!(
                row_major_bits(points),
                flat_bits(flat.random_flat(count).unwrap()),
                "large batch differs for d={d}, bits={bits}, seed={seed:?}"
            );
        }
        assert_eq!(nested.num_generated(), n as u128);
        assert_eq!(flat.num_generated(), n as u128);

        // A full batch must leave the next Gray-code state identical.
        nested.fast_forward(23).unwrap();
        flat.fast_forward(23).unwrap();
        assert_eq!(
            row_major_bits(nested.random(17).unwrap()),
            flat_bits(flat.random_flat(17).unwrap())
        );
    }
}

#[test]
#[cfg(target_pointer_width = "64")]
fn sobol_final_period_rows_have_identical_nested_and_flat_bits() {
    for bits in [1, 8, 30, 53, 54, 64] {
        let remaining = (1_u128 << bits) - 2;
        let mut nested = Sobol::scrambled(4, bits, 123).unwrap();
        let mut flat = nested.clone();
        nested.fast_forward(remaining as usize).unwrap();
        flat.fast_forward(remaining as usize).unwrap();
        assert_eq!(
            row_major_bits(nested.random(2).unwrap()),
            flat_bits(flat.random_flat(2).unwrap()),
            "last two rows differ at bits={bits}"
        );
        assert_eq!(nested.num_generated(), 1_u128 << bits);
        assert_eq!(flat.num_generated(), 1_u128 << bits);
        assert!(nested.random(1).is_err());
        assert!(flat.random_flat(1).is_err());
        assert!(nested.random(0).unwrap().is_empty());
        assert!(flat.random_flat(0).unwrap().is_empty());
    }
}

#[test]
fn sobol_random_flat_chunking_mixed_draws_and_fast_forward_agree() {
    let mut reference = Sobol::scrambled(6, 30, 65_537).unwrap();
    let expected = row_major_bits(reference.random(128).unwrap());
    let d = 6;

    let mut flat = Sobol::scrambled(d, 30, 65_537).unwrap();
    let mut first = flat_bits(flat.random_flat(5).unwrap());
    assert!(flat.random_flat(0).unwrap().is_empty());
    first.extend(flat_bits(flat.random_flat(11).unwrap()));
    assert_eq!(first, expected[..16 * d]);

    // Existing row-oriented reads and the new flat reads must share one state.
    assert_eq!(
        row_major_bits(flat.random(7).unwrap()),
        expected[16 * d..23 * d]
    );
    flat.fast_forward(8).unwrap();
    assert_eq!(flat.num_generated(), 31);
    assert_eq!(
        flat_bits(flat.random_flat(10).unwrap()),
        expected[31 * d..41 * d]
    );
    assert_eq!(flat.num_generated(), 41);

    flat.reset();
    assert_eq!(flat.num_generated(), 0);
    assert_eq!(flat_bits(flat.random_flat(16).unwrap()), expected[..16 * d]);

    let mut skipped = Sobol::scrambled(d, 30, 65_537).unwrap();
    skipped.fast_forward(31).unwrap();
    assert_eq!(
        flat_bits(skipped.random_flat(10).unwrap()),
        expected[31 * d..41 * d]
    );
}

#[test]
fn sobol_random_flat_rejects_invalid_batches_without_advancing() {
    let mut flat = Sobol::scrambled(6, 8, 7).unwrap();
    let _ = flat.random_flat(3).unwrap();
    assert_eq!(flat.num_generated(), 3);

    // Reject both scalar-output-limit violations and period exhaustion.
    assert!(flat.random_flat(1_048_576 / 6 + 1).is_err());
    assert!(flat.random_flat(usize::MAX).is_err());
    assert!(flat.random_flat(254).is_err());
    assert_eq!(flat.num_generated(), 3);
    assert!(flat.random_flat(0).unwrap().is_empty());
    assert_eq!(flat.num_generated(), 3);

    let mut reference = Sobol::scrambled(6, 8, 7).unwrap();
    reference.fast_forward(3).unwrap();
    assert_eq!(
        flat_bits(flat.random_flat(5).unwrap()),
        row_major_bits(reference.random(5).unwrap())
    );

    let mut exhausted = Sobol::new(2, 3).unwrap();
    assert_eq!(exhausted.random_flat(8).unwrap().len(), 16);
    assert_eq!(exhausted.num_generated(), 8);
    assert!(exhausted.random_flat(1).is_err());
    assert_eq!(exhausted.num_generated(), 8);
    assert!(exhausted.random_flat(0).unwrap().is_empty());
}

#[test]
#[cfg(target_pointer_width = "64")]
fn sobol_random_flat_handles_large_64_bit_indices() {
    let mut last = Sobol::new(1, 64).unwrap();
    last.fast_forward(usize::MAX).unwrap();
    assert_eq!(
        last.random_flat(1)
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        vec![2_f64.powi(-64).to_bits()]
    );
    assert_eq!(last.num_generated(), 1_u128 << 64);
    assert!(last.random_flat(1).is_err());
    assert_eq!(last.num_generated(), 1_u128 << 64);

    let index = 0xaaaa_aaaa_aaaa_aaaa_u64 as usize;
    let mut flat = Sobol::new(1, 64).unwrap();
    let mut nested = Sobol::new(1, 64).unwrap();
    flat.fast_forward(index).unwrap();
    nested.fast_forward(index).unwrap();
    assert_eq!(
        flat_bits(flat.random_flat(1).unwrap()),
        row_major_bits(nested.random(1).unwrap())
    );
    assert_eq!(flat.num_generated(), index as u128 + 1);
    assert_eq!(flat.random_flat(0).unwrap(), Vec::<f64>::new());
}
