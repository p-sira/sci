use sci::{
    filter::{Gaussian, ParticleSet, kalman::kalman_update, particle::particle_estimate},
    linalg, stats,
};

#[test]
fn estimates_share_matrix_conventions_with_linear_algebra_and_statistics() {
    let particles = ParticleSet {
        states: vec![vec![-1., 2.], vec![1., 4.], vec![3., 6.]],
        weights: vec![1.; 3],
    };
    let summary = particle_estimate(&particles).unwrap();
    assert!((summary.mean[0] - stats::descriptive::mean(&[-1., 1., 3.]).unwrap()).abs() < 1e-14);
    assert!(
        (summary.covariance[0][0] - stats::descriptive::variance(&[-1., 1., 3.], 0).unwrap()).abs()
            < 1e-14
    );
    let predicted = linalg::basic::matvec(&summary.covariance, &[1., 0.]).unwrap();
    assert_eq!(
        predicted,
        vec![summary.covariance[0][0], summary.covariance[1][0]]
    );
    let prior = Gaussian {
        mean: summary.mean,
        covariance: summary.covariance,
    };
    let posterior = kalman_update(&prior, &[1.5], &[vec![1., 0.]], &[vec![1.]]).unwrap();
    assert_eq!(posterior.estimate.covariance.len(), 2);
    assert!(posterior.estimate.mean[0] > 1.);
}
