use sci::{
    Error, Matrix,
    filter::{Gaussian, SigmaPoints, Update, kalman::*, particle::*},
};
mod common;
use common::finite_close;

fn v(s: &str) -> Vec<f64> {
    common::vector(s)
}
fn m(s: &str) -> Matrix {
    s.split(':').map(v).collect()
}
fn g(a: &[&str]) -> Gaussian {
    Gaussian {
        mean: v(a[0]),
        covariance: m(a[1]),
    }
}
fn p(a: &[&str]) -> ParticleSet {
    ParticleSet {
        states: m(a[0]),
        weights: v(a[1]),
    }
}
fn gaussian_values(g: Gaussian) -> Vec<f64> {
    assert_eq!(g.covariance.len(), g.mean.len());
    assert!(g.covariance.iter().all(|r| r.len() == g.mean.len()));
    g.mean
        .into_iter()
        .chain(g.covariance.into_iter().flatten())
        .collect()
}

#[test]
fn identity_prediction_preserves_correlated_covariance() -> sci::Result<()> {
    let prior = Gaussian {
        mean: vec![1., -2., 3.],
        covariance: vec![vec![4., 1., 1.], vec![1., 3., 0.5], vec![1., 0.5, 2.]],
    };
    let identity = vec![vec![1., 0., 0.], vec![0., 1., 0.], vec![0., 0., 1.]];
    let result = kalman_predict(&prior, &identity, &vec![vec![0.; 3]; 3])?;
    assert_eq!(result, prior);
    Ok(())
}
fn update_values(u: Update, size: usize) -> Vec<f64> {
    assert_eq!(u.innovation.len(), size);
    gaussian_values(u.estimate)
        .into_iter()
        .chain(u.innovation)
        .collect()
}
fn particle_values(p: ParticleSet, count: usize, dim: usize) -> Vec<f64> {
    assert_eq!(p.states.len(), count);
    assert!(p.states.iter().all(|s| s.len() == dim));
    assert_eq!(p.weights.len(), count);
    p.states.into_iter().flatten().chain(p.weights).collect()
}
fn model(x: &[f64], name: &str) -> sci::Result<Vec<f64>> {
    Ok(match name {
        "linear" if x.len() == 1 => x.to_vec(),
        "linear" => vec![x[0] + 0.2 * x[1], x[1]],
        "square" => x.iter().map(|v| v * v).collect(),
        "coupled" => vec![x[0] + 0.1 * x[1] * x[1], x[1]],
        "range" => vec![x[0].hypot(x[1])],
        _ => panic!("unknown model {name}"),
    })
}
fn jacobian(x: &[f64], name: &str) -> sci::Result<Matrix> {
    Ok(match name {
        "square" => (0..x.len())
            .map(|i| {
                (0..x.len())
                    .map(|j| if i == j { 2. * x[i] } else { 0. })
                    .collect()
            })
            .collect(),
        "coupled" => vec![vec![1., 0.2 * x[1]], vec![0., 1.]],
        "range" => {
            let r = x[0].hypot(x[1]);
            vec![vec![x[0] / r, x[1] / r]]
        }
        _ => panic!("unknown Jacobian {name}"),
    })
}

#[test]
fn independent_scipy_identities_cover_every_public_function() {
    let inputs: Vec<_> = common::rows(include_str!("fixtures/filter_inputs.csv")).collect();
    let expected: Vec<_> = common::rows(include_str!("fixtures/filter_expected.csv")).collect();
    assert_eq!(inputs.len(), expected.len());
    for (input, expected) in inputs.into_iter().zip(expected) {
        assert_eq!(input[0], expected[0]);
        let a = &input[2..];
        let actual = match input[1] {
            "kalman_predict" => gaussian_values(kalman_predict(&g(a), &m(a[2]), &m(a[3])).unwrap()),
            "kalman_update" => update_values(
                kalman_update(&g(a), &v(a[2]), &m(a[3]), &m(a[4])).unwrap(),
                v(a[2]).len(),
            ),
            "extended_kalman_predict" => gaussian_values(
                extended_kalman_predict(&g(a), &m(a[2]), |x| model(x, a[3]), |x| jacobian(x, a[3]))
                    .unwrap(),
            ),
            "extended_kalman_update" => update_values(
                extended_kalman_update(
                    &g(a),
                    &v(a[2]),
                    &m(a[3]),
                    |x| model(x, a[4]),
                    |x| jacobian(x, a[4]),
                )
                .unwrap(),
                v(a[2]).len(),
            ),
            "unscented_kalman_step" => {
                let points = v(a[5]);
                update_values(
                    unscented_kalman_step(
                        &g(a),
                        &m(a[2]),
                        &v(a[3]),
                        &m(a[4]),
                        SigmaPoints {
                            alpha: points[0],
                            beta: points[1],
                            kappa: points[2],
                        },
                        |x| model(x, a[6]),
                        |x| {
                            if a[7] == "linear" {
                                Ok(x.to_vec())
                            } else {
                                model(x, a[7])
                            }
                        },
                    )
                    .unwrap(),
                    v(a[3]).len(),
                )
            }
            "particle_predict" => {
                let p = p(a);
                particle_values(
                    particle_predict(&p, |x| model(x, a[2])).unwrap(),
                    p.states.len(),
                    p.states[0].len(),
                )
            }
            "particle_update" => {
                let p = p(a);
                let mut ll = v(a[2]).into_iter();
                particle_values(
                    particle_update(&p, |_| Ok(ll.next().unwrap())).unwrap(),
                    p.states.len(),
                    p.states[0].len(),
                )
            }
            "particle_estimate" => gaussian_values(particle_estimate(&p(a)).unwrap()),
            "effective_sample_size" => vec![effective_sample_size(&p(a)).unwrap()],
            "systematic_resample" => {
                let p = p(a);
                particle_values(
                    systematic_resample(&p, a[2].parse().unwrap()).unwrap(),
                    p.states.len(),
                    p.states[0].len(),
                )
            }
            name => panic!("unknown fixture {name}"),
        };
        let expected = v(expected[1]);
        assert_eq!(
            actual.len(),
            expected.len(),
            "shape for {} {}",
            input[0],
            input[1]
        );
        for (a, b) in actual.into_iter().zip(expected) {
            finite_close(a, b, 2e-11, 2e-11);
        }
    }
}

fn scalar(mean: f64, variance: f64) -> Gaussian {
    Gaussian {
        mean: vec![mean],
        covariance: vec![vec![variance]],
    }
}
fn cloud() -> ParticleSet {
    ParticleSet {
        states: vec![vec![0.], vec![1.], vec![2.]],
        weights: vec![1., 2., 1.],
    }
}

#[test]
fn linear_models_agree_across_all_kalman_variants_and_streaming() {
    let f = vec![vec![1., 0.2], vec![0., 1.]];
    let q = vec![vec![0.1, 0.], vec![0., 0.01]];
    let h = vec![vec![1., 0.]];
    let r = vec![vec![0.2]];
    let mut state = Gaussian {
        mean: vec![0., 1.],
        covariance: vec![vec![1., 0.2], vec![0.2, 0.5]],
    };
    for z in [0.1, 0.5, 0.7, 1.0, 1.1] {
        let prediction = kalman_predict(&state, &f, &q).unwrap();
        let ek =
            extended_kalman_predict(&state, &q, |x| model(x, "linear"), |_| Ok(f.clone())).unwrap();
        assert_eq!(prediction, ek);
        let u = kalman_update(&prediction, &[z], &h, &r).unwrap();
        let eu =
            extended_kalman_update(&ek, &[z], &r, |x| Ok(vec![x[0]]), |_| Ok(h.clone())).unwrap();
        assert_eq!(u, eu);
        let uu = unscented_kalman_step(
            &state,
            &q,
            &[z],
            &r,
            SigmaPoints::default(),
            |x| model(x, "linear"),
            |x| Ok(vec![x[0]]),
        )
        .unwrap();
        for (a, b) in gaussian_values(u.estimate.clone())
            .into_iter()
            .zip(gaussian_values(uu.estimate))
        {
            finite_close(a, b, 2e-11, 2e-11);
        }
        finite_close(u.innovation[0], uu.innovation[0], 2e-11, 2e-11);
        assert!(u.estimate.covariance[0][0] < prediction.covariance[0][0]);
        state = u.estimate;
    }
}

#[test]
fn gaussian_validation_and_numerical_failures_are_explicit() {
    let identity = vec![vec![1.]];
    let zero = vec![vec![0.]];
    for x in [
        scalar(f64::NAN, 1.),
        scalar(f64::INFINITY, 1.),
        scalar(0., -1.),
        scalar(0., f64::NAN),
        Gaussian {
            mean: vec![],
            covariance: vec![],
        },
        Gaussian {
            mean: vec![0.],
            covariance: vec![vec![]],
        },
        Gaussian {
            mean: vec![0.; 33],
            covariance: vec![vec![1.; 33]; 33],
        },
    ] {
        assert!(kalman_predict(&x, &identity, &zero).is_err());
        assert!(kalman_update(&x, &[0.], &identity, &identity).is_err());
        let never = |_: &[f64]| -> sci::Result<Vec<f64>> { panic!("invalid input reached model") };
        let never_j =
            |_: &[f64]| -> sci::Result<Matrix> { panic!("invalid input reached Jacobian") };
        assert!(extended_kalman_predict(&x, &zero, never, never_j).is_err());
        assert!(extended_kalman_update(&x, &[0.], &identity, never, never_j).is_err());
        assert!(
            unscented_kalman_step(
                &x,
                &zero,
                &[0.],
                &identity,
                SigmaPoints::default(),
                never,
                never
            )
            .is_err()
        );
    }
    let x = scalar(0., 1.);
    for bad in [
        vec![],
        vec![vec![]],
        vec![vec![-1.]],
        vec![vec![f64::INFINITY]],
    ] {
        assert!(kalman_predict(&x, &identity, &bad).is_err());
        assert!(kalman_update(&x, &[0.], &identity, &bad).is_err());
    }
    assert!(kalman_predict(&x, &[vec![f64::MAX]], &zero).is_err());
    assert!(kalman_update(&scalar(0., 0.), &[0.], &identity, &zero).is_err());
    assert_eq!(
        kalman_update(&x, &[f64::NAN], &identity, &identity),
        Err(Error::InvalidInput("filter inputs must be finite"))
    );
    assert!(kalman_update(&x, &[], &identity, &identity).is_err());
    assert!(kalman_update(&x, &[0.], &[vec![]], &identity).is_err());
    for p in [
        vec![vec![1., 0.2], vec![0., 1.]],
        vec![vec![1., 2.], vec![2., 1.]],
    ] {
        let x = Gaussian {
            mean: vec![0., 0.],
            covariance: p,
        };
        assert!(
            kalman_predict(
                &x,
                &[vec![1., 0.], vec![0., 1.]],
                &[vec![0., 0.], vec![0., 0.]]
            )
            .is_err()
        );
    }
    assert!(kalman_predict(&scalar(0., 1e-300), &identity, &zero).is_ok());
    assert!(kalman_predict(&scalar(0., -1e-300), &identity, &zero).is_err());
}

#[test]
fn callback_errors_dimensions_and_special_values_are_checked() {
    let x = scalar(0., 1.);
    let q = vec![vec![0.1]];
    let r = vec![vec![1.]];
    let failure = Error::NumericalFailure("user model failed");
    assert_eq!(
        extended_kalman_predict(
            &x,
            &q,
            |_| Err(failure.clone()),
            |_| panic!("late callback")
        ),
        Err(failure.clone())
    );
    assert_eq!(
        extended_kalman_update(&x, &[0.], &r, |_| Ok(vec![0.]), |_| Err(failure.clone())),
        Err(failure.clone())
    );
    for bad in [vec![], vec![0., 0.], vec![f64::NAN], vec![f64::INFINITY]] {
        assert!(extended_kalman_predict(&x, &q, |_| Ok(bad.clone()), |_| Ok(r.clone())).is_err());
        assert!(
            extended_kalman_update(&x, &[0.], &r, |_| Ok(bad.clone()), |_| Ok(r.clone())).is_err()
        );
        assert!(
            unscented_kalman_step(
                &x,
                &q,
                &[0.],
                &r,
                SigmaPoints::default(),
                |_| Ok(bad.clone()),
                |x| Ok(x.to_vec())
            )
            .is_err()
        );
        assert!(
            unscented_kalman_step(
                &x,
                &q,
                &[0.],
                &r,
                SigmaPoints::default(),
                |x| Ok(x.to_vec()),
                |_| Ok(bad.clone())
            )
            .is_err()
        );
    }
    for bad in [vec![], vec![vec![]], vec![vec![f64::NAN]]] {
        assert!(extended_kalman_predict(&x, &q, |x| Ok(x.to_vec()), |_| Ok(bad.clone())).is_err());
        assert!(
            extended_kalman_update(&x, &[0.], &r, |x| Ok(x.to_vec()), |_| Ok(bad.clone())).is_err()
        );
    }
    let calls = std::cell::Cell::new(0);
    assert_eq!(
        unscented_kalman_step(
            &x,
            &q,
            &[0.],
            &r,
            SigmaPoints::default(),
            |_| {
                calls.set(calls.get() + 1);
                Err(failure.clone())
            },
            |_| panic!("late callback")
        ),
        Err(failure)
    );
    assert_eq!(calls.get(), 1);
    for points in [
        SigmaPoints {
            alpha: 0.,
            ..Default::default()
        },
        SigmaPoints {
            alpha: 1e-8,
            ..Default::default()
        },
        SigmaPoints {
            beta: -1.,
            ..Default::default()
        },
        SigmaPoints {
            kappa: -1.,
            ..Default::default()
        },
        SigmaPoints {
            alpha: f64::NAN,
            ..Default::default()
        },
    ] {
        assert!(
            unscented_kalman_step(
                &x,
                &q,
                &[0.],
                &r,
                points,
                |_| panic!("invalid scaling"),
                |_| panic!("invalid scaling")
            )
            .is_err()
        );
    }
    assert!(
        unscented_kalman_step(
            &scalar(0., 0.),
            &q,
            &[0.],
            &r,
            SigmaPoints::default(),
            |x| Ok(x.to_vec()),
            |x| Ok(x.to_vec())
        )
        .is_err()
    );
}

#[test]
fn particles_validate_every_entry_point_and_preserve_prior() {
    let good = cloud();
    for bad in [
        ParticleSet {
            states: vec![],
            weights: vec![],
        },
        ParticleSet {
            states: vec![vec![]],
            weights: vec![1.],
        },
        ParticleSet {
            weights: vec![0.; 3],
            ..good.clone()
        },
        ParticleSet {
            weights: vec![-1., 1., 1.],
            ..good.clone()
        },
        ParticleSet {
            weights: vec![f64::NAN; 3],
            ..good.clone()
        },
        ParticleSet {
            weights: vec![f64::INFINITY; 3],
            ..good.clone()
        },
        ParticleSet {
            weights: vec![1.],
            ..good.clone()
        },
        ParticleSet {
            states: vec![vec![f64::NAN]; 3],
            ..good.clone()
        },
        ParticleSet {
            states: vec![vec![0.; 33]],
            weights: vec![1.],
        },
        ParticleSet {
            states: vec![vec![0.; 32]; 16385],
            weights: vec![1.; 16385],
        },
        ParticleSet {
            states: vec![vec![0.]; 65537],
            weights: vec![1.; 65537],
        },
    ] {
        assert!(particle_predict(&bad, |_| panic!("invalid particles")).is_err());
        assert!(particle_update(&bad, |_| panic!("invalid particles")).is_err());
        assert!(particle_estimate(&bad).is_err());
        assert!(effective_sample_size(&bad).is_err());
        assert!(systematic_resample(&bad, 0.5).is_err());
    }
    for offset in [-1., 1., f64::NAN, f64::INFINITY] {
        assert!(systematic_resample(&good, offset).is_err());
    }
    for bad in [vec![], vec![0., 0.], vec![f64::NAN]] {
        assert!(particle_predict(&good, |_| Ok(bad.clone())).is_err());
    }
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(particle_update(&good, |_| Ok(bad)).is_err());
    }
    let failure = Error::NumericalFailure("user likelihood failed");
    assert_eq!(
        particle_update(&good, |_| Err(failure.clone())),
        Err(failure.clone())
    );
    assert_eq!(
        particle_predict(&good, |_| Err(failure.clone())),
        Err(failure)
    );
    assert_eq!(good, cloud());
    let extreme = ParticleSet {
        states: vec![vec![-f64::MAX], vec![f64::MAX]],
        weights: vec![1., 1.],
    };
    assert!(particle_estimate(&extreme).is_err());
}

#[test]
fn particle_identities_and_resampling_boundaries() {
    let p = cloud();
    let unchanged = particle_update(&p, |_| Ok(0.)).unwrap();
    assert_eq!(unchanged.weights, vec![0.25, 0.5, 0.25]);
    for scale in [1e-300, 1e300] {
        let scaled = ParticleSet {
            weights: p.weights.iter().map(|w| w * scale).collect(),
            ..p.clone()
        };
        assert_eq!(
            particle_estimate(&p).unwrap(),
            particle_estimate(&scaled).unwrap()
        );
        finite_close(
            effective_sample_size(&p).unwrap(),
            effective_sample_size(&scaled).unwrap(),
            2e-11,
            2e-11,
        );
    }
    let zero = ParticleSet {
        states: vec![vec![-100.], vec![5.], vec![100.]],
        weights: vec![0., 1., 0.],
    };
    for offset in [0., 0.5, 1. - f64::EPSILON] {
        let q = systematic_resample(&zero, offset).unwrap();
        assert_eq!(q.states, vec![vec![5.]; 3]);
        finite_close(effective_sample_size(&q).unwrap(), 3., 2e-11, 2e-11);
    }
    let uniform = ParticleSet {
        states: p.states.clone(),
        weights: vec![1.; 3],
    };
    assert_eq!(
        systematic_resample(&uniform, 0.).unwrap().states,
        uniform.states
    );
    let mut count = 0;
    let moved = particle_predict(&p, |x| {
        count += 1;
        Ok(vec![x[0] + 2.])
    })
    .unwrap();
    assert_eq!(count, 3);
    finite_close(
        particle_estimate(&moved).unwrap().mean[0],
        particle_estimate(&p).unwrap().mean[0] + 2.,
        2e-11,
        2e-11,
    );
    assert_eq!(
        particle_estimate(&moved).unwrap().covariance,
        particle_estimate(&p).unwrap().covariance
    );
}

#[test]
fn particle_callback_failure_stops_later_evaluations() {
    let prior = cloud();
    let failure = Error::NumericalFailure("second particle failed");
    for invalid_output in [false, true] {
        let mut calls = 0;
        let result = particle_predict(&prior, |x| {
            calls += 1;
            if calls == 2 {
                if invalid_output {
                    return Ok(vec![f64::NAN]);
                }
                return Err(failure.clone());
            }
            Ok(x.iter().map(|v| v + 1.).collect())
        });
        assert_eq!(
            result,
            Err(if invalid_output {
                Error::NonFiniteEvaluation
            } else {
                failure.clone()
            })
        );
        assert_eq!(calls, 2);
        assert_eq!(prior, cloud());
    }
    let mut calls = 0;
    assert_eq!(
        particle_update(&prior, |_| {
            calls += 1;
            if calls == 2 {
                Err(failure.clone())
            } else {
                Ok(0.)
            }
        }),
        Err(failure)
    );
    assert_eq!(calls, 2);
    assert_eq!(prior, cloud());
}
