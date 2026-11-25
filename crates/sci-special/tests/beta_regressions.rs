// Preserve original high-precision independent numerical reference literals.
#![allow(clippy::excessive_precision)]
// Public log-beta regression test migrated from the Cephes kernel.

#[test]
fn moderate_positive_logbeta_matches_high_precision_references() {
    // Independently evaluated with 80-digit mpmath log(beta(a,b)).
    // Includes exact CodSpeed input pairs from positive_values(128,1/2),
    // the small-shape edge, and the upper boundary of the fast path.
    let cases = [
        (0.1, 0.1, 2.9813614810376274),
        (0.25, 9.7, 0.7297356961285787),
        (5.1, 5.1, -6.594738407285101),
        (8.9, 1.37, -3.139775074596382),
        (10.1, 9.9, -13.735177544346377),
        (15.0, 14.99, -20.86749393422148),
        (16.0, 16.0, -22.293680785633527),
        (0.1, 31.9, 1.9078685837064263),
        (5.194074428837206, 9.271161254706481, -9.107256962634744),
        (6.583593939634305, 7.013954653016276, -9.092255165988822),
        (3.928633905082601, 3.7451057732121957, -4.6908536302730615),
        (8.497261096476889, 2.066586641881247, -4.514737247441051),
        (6.316081841493, 0.6387656150871024, -0.8180420809505264),
        (8.69635623326492, 5.884713080910398, -9.524858388782952),
        (1.6859440249728328, 3.925613346611085, -2.5383231548682404),
    ];
    for (a, b, reference) in cases {
        for (a, b) in [(a, b), (b, a)] {
            let result = sci_special::beta::betaln(a, b);
            assert!(
                (result - reference).abs() <= 7e-14 * reference.abs().max(1.0),
                "sci_special::beta::betaln({a:.17e},{b:.17e})={result:.17e}, 80-digit reference={reference:.17e}"
            );
        }
    }
}
