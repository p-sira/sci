use super::*;

#[test]
fn nonfinite_and_excessive_kelvin_steps_are_errors() {
    for x in [f64::NAN, 0., 1.] {
        assert!(matches!(
            refine_keip_zero(x),
            Err(Error::NumericalFailure("Kelvin zero refinement failed"))
        ));
    }
}

#[test]
fn unsuitable_initial_guesses_exhaust_the_iteration_bound() {
    // A guess near the singular origin is outside the asymptotic seed range.
    assert!(matches!(
        refine_keip_zero(1e-8),
        Err(Error::NumericalFailure(
            "Kelvin zero refinement did not converge"
        ))
    ));
}
