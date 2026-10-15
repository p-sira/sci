use super::*;

#[test]
fn nonfinite_pole_steps_are_errors() {
    for z in [Complex64::new(f64::NAN, 1.), Complex64::new(0., 0.)] {
        assert!(matches!(
            refine_bessel_pole(12, z),
            Err(Error::NonFiniteEvaluation)
        ));
    }
}

#[test]
fn unsuitable_initial_poles_exhaust_the_iteration_bound() {
    // A guess far from the order-12 poles must not cause unbounded Newton work.
    assert!(matches!(
        refine_bessel_pole(12, Complex64::new(-0.01, 0.1)),
        Err(Error::NumericalFailure(
            "Bessel pole refinement did not converge"
        ))
    ));
}
