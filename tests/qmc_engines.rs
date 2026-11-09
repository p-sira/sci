use sci::stats::qmc::{
    Halton, LatinHypercube, MultinomialQMC, MultivariateNormalQMC, PoissonDisk, QmcEngine, Sobol,
};

#[test]
fn public_qmc_engine_trait_interoperates_across_generators() {
    let engines: Vec<Box<dyn QmcEngine>> = vec![
        Box::new(Sobol::new(2, 10).unwrap()),
        Box::new(Halton::new(2).unwrap()),
        Box::new(LatinHypercube::new(2, 145).unwrap()),
    ];
    for mut engine in engines {
        assert_eq!(engine.dimensions(), 2);
        let initial = engine.random(8).unwrap();
        assert_eq!(initial.len(), 8);
        assert!(initial.iter().all(|row| row.len() == 2));
        assert!(initial.iter().flatten().all(|x| (0.0..1.0).contains(x)));
        assert_eq!(engine.num_generated(), 8);

        assert!(engine.integers(1, &[0], &[4], false).is_err());
        assert_eq!(engine.num_generated(), 8);
        let integers = engine.integers(4, &[0, 10], &[4, 14], false).unwrap();
        assert_eq!(integers.len(), 4);
        assert!(
            integers
                .iter()
                .all(|p| (0..4).contains(&p[0]) && (10..14).contains(&p[1]))
        );

        engine.reset();
        assert_eq!(engine.num_generated(), 0);
        assert_eq!(engine.random(8).unwrap(), initial);
    }
}

#[test]
fn latin_hypercube_preserves_strata_and_skip_state() {
    let mut sampler = LatinHypercube::new(4, 556).unwrap();
    let points = sampler.random(32).unwrap();
    for column in 0..4 {
        let mut strata: Vec<_> = points
            .iter()
            .map(|point| (point[column] * 32.0).floor() as usize)
            .collect();
        strata.sort_unstable();
        assert_eq!(strata, (0..32).collect::<Vec<_>>());
    }

    let mut skip = LatinHypercube::new(4, 556).unwrap();
    skip.fast_forward(32).unwrap();
    assert_eq!(sampler.random(8).unwrap(), skip.random(8).unwrap());
}

#[test]
fn normal_and_multinomial_qmc_use_shared_sobol_sequences() {
    let mut normal = MultivariateNormalQMC::new(&[1.0, -2.0], None).unwrap();
    let points = normal.random(4).unwrap();
    assert_eq!(points[1], vec![1.0, -2.0]);
    assert!((points[2][0] - 1.6744897501174103).abs() < 1e-12);
    assert!((points[2][1] - (-2.6744897501174103)).abs() < 1e-12);
    normal.reset();
    assert_eq!(points, normal.random(4).unwrap());

    let mut multinomial = MultinomialQMC::new(&[0.0, 0.25, 0.75], 4).unwrap();
    let counts = multinomial.random(8).unwrap();
    assert_eq!(counts[0], vec![0, 1, 3]);
    assert!(
        counts
            .iter()
            .all(|row| row[0] == 0 && row.iter().sum::<usize>() == 4)
    );
    multinomial.reset();
    assert_eq!(counts, multinomial.random(8).unwrap());
}

#[test]
fn public_poisson_disk_samples_are_separated_and_replayable() {
    let mut sampler = PoissonDisk::new(2, 0.12, 42).unwrap();
    let points = sampler.random(64).unwrap();
    assert!(points.len() >= 10);
    assert!(points.iter().flatten().all(|x| (0.0..1.0).contains(x)));
    for (i, first) in points.iter().enumerate() {
        for second in points.iter().take(i) {
            let squared_distance: f64 = first
                .iter()
                .zip(second)
                .map(|(&x, &y)| (x - y).powi(2))
                .sum();
            assert!(squared_distance + 1e-13 >= 0.12_f64.powi(2));
        }
    }

    let mut chunks = PoissonDisk::new(2, 0.12, 42).unwrap();
    let mut split = chunks.random(10).unwrap();
    split.extend(chunks.random(54).unwrap());
    assert_eq!(split, points);

    let mut skipped = PoissonDisk::new(2, 0.12, 42).unwrap();
    skipped.fast_forward(10).unwrap();
    assert_eq!(skipped.random(54).unwrap(), points[10..]);
    skipped.reset();
    assert_eq!(skipped.random(64).unwrap(), points);
}
