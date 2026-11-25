mod common;

use common::{fields, integer, lines, vector};
use sci::stats::qmc::Halton;

#[test]
fn halton_matches_unscrambled_scipy() {
    let cases: Vec<_> = lines(include_str!("scipy/halton_cases.csv")).collect();
    let expected: Vec<_> = lines(include_str!("scipy/halton_expected.csv")).collect();
    assert_eq!(cases.len(), 16);
    assert_eq!(cases.len(), expected.len());

    for (input, reference) in cases.iter().zip(&expected) {
        let c = fields(input);
        let e = fields(reference);
        assert_eq!(c.len(), 4, "malformed input: {input}");
        assert_eq!(e.len(), 4, "malformed reference: {reference}");
        let id = c[0];
        assert_eq!(id, e[0], "case/reference order differs");
        let d = integer(c[1]);
        let skip = integer(c[2]);
        let n = integer(c[3]);
        assert_eq!(n, integer(e[1]), "{id}: row count in fixture");
        assert_eq!(d, integer(e[2]), "{id}: column count in fixture");
        let target = vector(e[3]);
        assert_eq!(target.len(), n * d, "{id}: flattened fixture shape");

        let mut sampler = Halton::new(d).unwrap();
        sampler.fast_forward(skip).unwrap();
        assert_eq!(sampler.num_generated(), skip as u128);
        let points = sampler.random(n).unwrap();
        assert_eq!(sampler.num_generated(), (skip + n) as u128);
        assert_eq!(points.len(), n, "{id}: row count");
        for (i, row) in points.iter().enumerate() {
            assert_eq!(row.len(), d, "{id}: row {i} dimension");
            for (j, &actual) in row.iter().enumerate() {
                let expected = target[i * d + j];
                assert!(
                    (actual - expected).abs() <= 2.0e-15,
                    "{id}: [{i},{j}] = {actual}, SciPy = {expected}"
                );
                assert!((0.0..1.0).contains(&actual), "{id}: point outside [0,1)");
            }
        }
    }
}

#[test]
fn halton_skip_reset_and_partition() {
    let mut sampler = Halton::new(7).unwrap();
    let expected = sampler.random(47).unwrap();
    sampler.reset();
    let mut split = sampler.random(8).unwrap();
    split.extend(sampler.random(39).unwrap());
    assert_eq!(expected, split);
    sampler.reset();
    sampler.fast_forward(8).unwrap();
    assert_eq!(expected[8..].to_vec(), sampler.random(39).unwrap());
}

#[test]
fn halton_scrambled_is_seeded_and_bounded() {
    let mut a = Halton::scrambled(65, 17).unwrap();
    let mut b = Halton::scrambled(65, 17).unwrap();
    let points = a.random(12).unwrap();
    assert_eq!(points, b.random(12).unwrap());
    assert!(points.iter().flatten().all(|x| (0.0..1.0).contains(x)));
    a.reset();
    assert_eq!(points, a.random(12).unwrap());
    assert_ne!(
        points,
        Halton::scrambled(65, 18).unwrap().random(12).unwrap()
    );
}
