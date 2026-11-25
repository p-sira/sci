// Preserve original high-precision independent numerical reference literals.
#![allow(clippy::excessive_precision)]
// Integration regressions extracted from the sci-special numerical backends.
use sci_special::hypergeometric::hyp0f1;

#[test]
fn negative_argument_shortcut_matches_independent_high_precision() {
    // mpmath 1.3.0, mp.dps = 80, hyper([], [v], z), rounded to f64.
    // The cases include both sides of the shortcut threshold, small v,
    // values close to zeros, and large alternating-series cancellation.
    const REFERENCES: &[(f64, f64, f64)] = &[
        (0.125, -3.01, -2.8993537576671815),
        (0.125, -5.0, 2.2799868130002063),
        (0.125, -10.0, 5.143622558744288),
        (0.125, -30.0, 4.3540733947902295),
        (0.5, -3.00001, -0.9484413659322454),
        (0.5, -5.551652475612764, -9.468420460028252e-17),
        (0.5, -5.55165, -1.0506828067508704e-6),
        (0.5, -9.0, 0.960170286650366),
        (0.5, -25.0, -0.8390715290764524),
        (1.0, -3.0, -0.37492595624331004),
        (1.0, -7.617815585, -1.1286788840317782e-10),
        (1.0, -7.6178, -1.9214742434014665e-6),
        (1.0, -12.0, 0.29896586116360446),
        (1.0, -50.0, 0.15055578737182204),
        (2.5, -3.01, 0.2127112447528678),
        (2.5, -5.0, 0.0031146110444027386),
        (2.5, -10.0, -0.074445380648445),
        (2.5, -15.0, 0.0010271017807084596),
        (2.5, -50.0, 0.0011351770111090956),
        (5.0, -3.01, 0.5292643399033946),
        (5.0, -5.001, 0.33129163930217),
        (5.0, -8.0, 0.14602211172333576),
        (5.0, -25.0, -0.008432743146317128),
        (5.0, -50.0, 0.0004564044063749215),
        (10.0, -3.01, 0.7369299995911754),
        (10.0, -11.0, 0.3125290020302696),
        (10.0, -40.0, 0.002932256211165023),
        (0.1, -100.0, -2.968942964377085),
        (0.5, -100.0, 0.40808206181339196),
        (2.5, -100.0, -0.0027182609945775797),
        (10.0, -100.0, 4.5405815286664117e-5),
        (3.25, 0.5, 1.1631890479552405),
        (3.25, 10.0, 11.228357500393701),
        (3.25, 100.0, 548442.9914829632),
        (-0.3, -3.01, -0.835042183928332),
        (-2.3, -5.0, 7.25048701318888),
        (-0.3, 4.0, -94.14213052614711),
        (-2.3, 1.0, 0.47197024339118),
    ];
    for &(v, z, expected) in REFERENCES {
        let actual = hyp0f1(v, z);
        assert!(
            (actual - expected).abs() <= 3e-12 * expected.abs().max(1.0),
            "0F1(;{v};{z}): got {actual}, reference {expected}"
        );
    }
}

#[test]
fn negative_argument_shortcut_respects_recurrence_and_poles() {
    // Contiguous relation from the defining Pochhammer series:
    // F(;b-1;z) = F(;b;z) + z F(;b+1;z)/(b(b-1)).
    for &v in &[0.5_f64, 1.125, 2.5, 5.0, 10.0, 25.0] {
        for &z in &[-100.0, -25.0, -10.0, -3.0001, -3.0, -2.9999, 2.0, 20.0] {
            let lhs = hyp0f1(v - 1.0, z);
            let rhs = hyp0f1(v, z) + z / (v * (v - 1.0)) * hyp0f1(v + 1.0, z);
            assert!(
                (lhs - rhs).abs() <= 2e-10 * lhs.abs().max(rhs.abs()).max(1.0),
                "0F1 recurrence v={v}, z={z}: {lhs} vs {rhs}"
            );
        }
    }
    for &v in &[0.0_f64, -1.0, -2.0, -3.0] {
        assert!(hyp0f1(v, -10.0).is_nan());
    }
    assert!(hyp0f1(1.0, f64::NAN).is_nan());
}
