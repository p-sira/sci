//! Public Faddeeva regression tests focused on algorithm-switch boundaries.
//! These avoid reconstructing a small result by subtracting two large ones.
use sci::{complex::Complex, special as s};

type C = Complex<f64>;

fn c(re: f64, im: f64) -> C {
    C { re, im }
}

fn mul(a: C, b: C) -> C {
    c(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

fn close(actual: f64, expected: f64, relative: f64, absolute: f64) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= relative * expected.abs() + absolute,
        "actual={actual:.17e}, expected={expected:.17e}, rtol={relative:e}, atol={absolute:e}"
    );
}

fn close_complex(actual: C, expected: C, relative: f64, absolute: f64) {
    close(actual.re, expected.re, relative, absolute);
    close(actual.im, expected.im, relative, absolute);
}

#[test]
fn complex_scaled_erfcx_matches_large_real_asymptotic_across_cutovers() {
    // Real Faddeeva erfcx switches to a continued fraction at x=50 and to
    // its leading term at x=5e7. The scalar public API uses an independent
    // Cephes implementation, so it also checks the magnitude and scaling.
    // erfcx(x) ~ [1 - 1/(2x²) + 3/(4x⁴) - 15/(8x⁶) + 105/(16x⁸)]/(sqrt(pi)x).
    for x in [49.999, 50.001, 5.0e7 - 1.0, 5.0e7 + 1.0, 1.0e9] {
        let q = 1.0 / (x * x);
        let expected = (1.0 + q * (-0.5 + q * (0.75 + q * (-1.875 + q * 6.5625))))
            / (std::f64::consts::PI.sqrt() * x);
        let complex = s::complex::erfcx(c(x, 0.0)).unwrap();
        let w = s::complex::wofz(c(0.0, x)).unwrap();
        close(complex.re, expected, 3e-13, 1e-300);
        close(w.re, expected, 3e-13, 1e-300);
        close(s::erfcx(x), expected, 3e-12, 1e-300);
        assert_eq!(complex.im, 0.0);
        assert_eq!(w.im, 0.0);
    }

    // The negative axis grows exponentially. In this range the subtracted
    // erfcx(+x) is small compared to 2 exp(x²), so reflection is well
    // conditioned; beyond x~26.6 the complex result is nonrepresentable.
    for x in [3.0_f64, 6.1, 12.0] {
        let expected = 2.0 * (x * x).exp() - s::erfcx(x);
        close_complex(
            s::complex::erfcx(c(-x, 0.0)).unwrap(),
            c(expected, 0.0),
            5e-13,
            1e-300,
        );
    }
    assert!(s::complex::erfcx(c(-27.0, 0.0)).is_err());
    assert!(s::erfcx(-27.0_f64).is_infinite());
}

#[test]
fn wofz_reflection_and_upper_half_plane_symmetries_across_regions() {
    // w(-conj(z))=conj(w(z)) involves no cancellation, even when
    // Re(w(z)) is tiny relative to Im(w(z)). These coordinates straddle
    // the small-x Taylor, algorithm-816, and continued-fraction cutovers.
    //
    // Independent 150-digit decimal Taylor recurrence:
    //   w(0)=1, w'(0)=2i/sqrt(pi), w'(z)=-2zw(z)+2i/sqrt(pi),
    //   (n+1)c[n+1]=-2c[n-1].
    // For x >= 10^4 use the independent large-z asymptotic expansion
    // instead of summing exponentially large, canceling Taylor terms.
    for (z, expected) in [
        (
            c(4.9e-4, 0.25),
            c(0.7703464073817383, 0.0003641708408529479),
        ),
        (
            c(5.1e-4, 0.25),
            c(0.7703463956908393, 0.00037903495274605673),
        ),
        (c(6.1, 0.09), c(0.0014233672263959613, 0.09376479656310802)),
        (c(6.1, 0.11), c(0.0017394637287773862, 0.09375396641318717)),
        (c(8.5, 1e-8), c(7.976873706375801e-11, 0.06684447298834638)),
        (c(9.9, 1e-12), c(5.846880473740769e-15, 0.05728414381662895)),
        (
            c(10.1, 1e-12),
            c(5.614119133449045e-15, 0.05613828149648325),
        ),
        (c(4.0, 7.01), c(0.06069653217923638, 0.0341148281139277)),
    ] {
        let value = s::complex::wofz(z).unwrap();
        let reflected = s::complex::wofz(c(-z.re, z.im)).unwrap();
        // Absolute error is scaled to |w|, not separately to components:
        // near the real axis Re(w) can be ~1e-15 while Im(w) is ~1e-2.
        let scale = expected.re.abs().max(expected.im.abs());
        close(value.re, expected.re, 3e-11, 3e-11 * scale);
        close(value.im, expected.im, 3e-11, 3e-11 * scale);
        close(reflected.re, expected.re, 3e-11, 3e-11 * scale);
        close(reflected.im, -expected.im, 3e-11, 3e-11 * scale);
        close_complex(reflected, c(value.re, -value.im), 2e-11, 1e-300);
        if z.im < 1e-7 {
            // Re w(x+iy) = exp(-x²) + (2y/sqrt(pi))*(2x*Dawson(x)-1)
            // + O(y²). Here the real part is 1e-11..1e-15, much smaller
            // than the imaginary part; the norm-scaled check above alone
            // would not constrain it. The public scalar Dawson path uses
            // an independent Cephes kernel.
            let near_axis = (-z.re * z.re).exp()
                + std::f64::consts::FRAC_2_SQRT_PI * z.im * (2.0 * z.re * s::dawsn(z.re) - 1.0);
            close(value.re, near_axis, 2e-7, 1e-300);
        }
    }
    // For very large z with Im(z)>0, w(z)=i/(sqrt(pi)z)*(1+O(|z|^-2)).
    for z in [c(10000.0, 0.5), c(1e8, 0.5)] {
        let value = s::complex::wofz(z).unwrap();
        let reflected = s::complex::wofz(c(-z.re, z.im)).unwrap();
        let denom = std::f64::consts::PI.sqrt() * (z.re * z.re + z.im * z.im);
        let expected = c(z.im / denom, z.re / denom);
        close_complex(value, expected, 2e-8, 1e-300);
        close_complex(reflected, c(expected.re, -expected.im), 2e-8, 1e-300);
    }

    // At moderate arguments neither side is exponentially negligible, so
    // the entire-function reflection formula is well conditioned.
    for z in [c(0.25, 0.5), c(1.0, 0.5), c(2.0, 0.5)] {
        let plus = s::complex::wofz(z).unwrap();
        let minus = s::complex::wofz(c(-z.re, -z.im)).unwrap();
        let minus_z2 = mul(c(-z.re, -z.im), z);
        let expected = sci::complex::exp(minus_z2);
        close_complex(
            c(plus.re + minus.re, plus.im + minus.im),
            c(2.0 * expected.re, 2.0 * expected.im),
            5e-12,
            1e-14,
        );
    }
}

#[test]
fn dawson_negative_small_imaginary_and_huge_real_paths() {
    // F'(x)=1-2xF(x) gives an independent first-order check of the
    // negative-small-imaginary Taylor expansion without subtracting
    // exponentials of opposite magnitudes.
    for (x, y) in [(1.4, -1e-5), (12.0, -1e-5), (41.0, -2e-5)] {
        let got = s::complex::dawsn(c(x, y)).unwrap();
        let real = s::dawsn(x);
        let slope = 1.0 - 2.0 * x * real;
        close(got.re, real, 2e-7, 2e-9);
        close(got.im / y, slope, 3e-5, 2e-9);
    }

    // At |x| > 5e7 the complex kernel has a dedicated algebraic branch.
    // F(x+iy)~1/(2(x+iy)), with O(x^-3) corrections. Compare the tiny
    // imaginary component RELATIVELY, not against a large absolute floor.
    let x = 5.01e7;
    let y = -2e-11;
    let value = s::complex::dawsn(c(x, y)).unwrap();
    close(value.re, 0.5 / x, 2e-12, 1e-300);
    close(value.im, -y / (2.0 * x * x), 2e-8, 1e-300);

    // The small complex Taylor series, including the negative-imaginary
    // branch, is independently characterized by z - 2z³/3 + 4z⁵/15.
    let z = c(0.002, -0.002);
    let z2 = mul(z, z);
    let z3 = mul(z2, z);
    let z5 = mul(z3, z2);
    let expected = c(
        z.re - (2.0 / 3.0) * z3.re + (4.0 / 15.0) * z5.re,
        z.im - (2.0 / 3.0) * z3.im + (4.0 / 15.0) * z5.im,
    );
    close_complex(s::complex::dawsn(z).unwrap(), expected, 2e-13, 1e-300);
}

#[test]
fn complex_erf_taylor_covers_small_real_both_signs() {
    // Direct entire-function series, independent of the Faddeeva rational/
    // Fourier approximation. The complex backend has separate Taylor paths
    // for small positive and negative real parts and small imaginary parts.
    for z in [
        c(0.003, 0.25),
        c(-0.003, 0.25),
        c(0.03, 0.005),
        c(-0.03, 0.005),
        c(0.003, -0.25),
        c(-0.003, -0.25),
    ] {
        let negative_z2 = {
            let sq = mul(z, z);
            c(-sq.re, -sq.im)
        };
        let mut term = z;
        let mut sum = z;
        for n in 1..32 {
            let t = mul(term, negative_z2);
            term = c(t.re / n as f64, t.im / n as f64);
            let d = (2 * n + 1) as f64;
            sum = c(sum.re + term.re / d, sum.im + term.im / d);
        }
        let scale = std::f64::consts::FRAC_2_SQRT_PI;
        let expected = c(scale * sum.re, scale * sum.im);
        close_complex(s::complex::erf(z).unwrap(), expected, 6e-13, 1e-300);
    }
}
