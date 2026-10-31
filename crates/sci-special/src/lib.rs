//! # Sci-Special
//!
//! Special mathematical functions.
//!
//! Elliptic integrals, scalar gamma/error/Bessel functions, polynomial evaluators,
//! and stable reductions accept `T: sci_core::Float`. Sequence and wave APIs
//! currently use `f64`. Specialized kernels generally evaluate in binary64 and
//! convert to the requested type; elliptic integrals compute in the selected type.
//!
//! Part of [Sci](https://github.com/p-sira/sci), a unified scientific computing library for Rust.
//!
//! Gamma, Bessel, elliptic, probability, polynomial, and combinatorial functions
//! are organized into public mathematical-family modules. For example,
//! [`gamma::gamma`] computes the gamma function and [`bessel::jv`] computes a
//! cylindrical Bessel function. No numerical functions are exported at the
//! crate root. Specialized algorithms use private pure Rust kernels and ellip
//! backends; elementary formulas use the standard library.
//! Use `value.is_finite()` to check scalar results for NaN and infinity.
//! Complex gamma (including log-gamma, reciprocal gamma and digamma),
//! error/Bessel/Hankel/Airy/Fresnel APIs are available under
//! `complex` using shared generic complex values and errors.
//! There is no broadcasting, implicit complex-input overloading, exact integer
//! mode, or derivative-axis output for array functions. Polynomial constructors
//! return descending coefficients. Read each function's documented real domain,
//! output ordering, and bounded order/size limits.
//! Scalar functions return documented IEEE values; fallible integrals and
//! sequences return the shared [`crate::Result`].
//!
//! Elliptic integrals use Ellip's notation: `ellipf(phi, m)` computes `F(phi, m)`,
//! with radians and elliptic parameter `m = k²`; incomplete third-kind arguments
//! are `(phi, n, m)`. Supported principal values are documented per function.
//! SciPy is an independent reference; complete SciPy parity is not required.

//! ## Examples
//!
//! ```
//! use sci_special::{elliptic, gamma};
//! assert_eq!(gamma::gamma(5.0_f64), 24.0);
//! let k = elliptic::ellipk(0.5_f64)?;
//! assert!((k - 1.8540746773013719).abs() < 1e-14);
//! # Ok::<(), sci_core::Error>(())
//! ```
//!
//! ## References
//!
//! - [NIST DLMF, gamma functions](https://dlmf.nist.gov/5).
//! - [NIST DLMF, elliptic integrals](https://dlmf.nist.gov/19).
//!
//! ## Acknowledgments
//!
//! We thank `ellip`, `nalgebra`, `num-complex`, `complex-bessel`, and `lambert_w`
//! for their specialized numerical algorithms. Retained third-party license
//! notices accompany the private translated kernels in `LICENSES`.

#![forbid(unsafe_code)]
#![deny(missing_docs, rustdoc::broken_intra_doc_links)]

pub use sci_core::{Error, Result};
mod backend;

pub mod complex;
#[path = "mod.rs"]
mod implementation;

// Public numerical APIs are grouped by mathematical family. The implementation
// layer remains private, so each function has a single public namespace.

/// Gamma, incomplete gamma, digamma and related special functions.
pub mod gamma {
    pub use crate::implementation::{digamma, digammainv, gamma, gamma_ratio, gammainc, gammaincc};
    pub use crate::implementation::{
        gammainccinv, gammaincinv, gammaln, gammasgn, loggamma, multigammaln,
    };
    pub use crate::implementation::{polygamma, rgamma, trigamma};
}

/// Beta and incomplete-beta functions.
pub mod beta {
    pub use crate::implementation::{beta, betainc, betaincc, betainccinv, betaincinv, betaln};
}

/// Bessel, Hankel, spherical Bessel, Struve and Kelvin functions.
pub mod bessel {
    pub use crate::implementation::yvp;
    pub use crate::implementation::{bei, bei_zeros, beip, beip_zeros, ber, ber_zeros};
    pub use crate::implementation::{berp, berp_zeros, besselpoly, h1vp, h2vp, hankel1};
    pub use crate::implementation::{hankel1e, hankel2, hankel2e, i0, i0e, i1};
    pub use crate::implementation::{i1e, it2i0k0, it2j0y0, it2struve0, iti0k0, itj0y0};
    pub use crate::implementation::{
        itmodstruve0, itstruve0, iv, iv_ratio, iv_ratio_c, iv_ratioinv,
    };
    pub use crate::implementation::{ive, ivp, j0, j1, jn, jn_zeros};
    pub use crate::implementation::{jnjnp_zeros, jnp_zeros, jnyn_zeros, jv, jve, jvp};
    pub use crate::implementation::{k0, k0e, k1, k1e, kei, kei_zeros};
    pub use crate::implementation::{keip, keip_zeros, kelvin, kelvin_zeros, ker, ker_zeros};
    pub use crate::implementation::{kerp, kerp_zeros, kn, kv, kve, kvp};
    pub use crate::implementation::{
        modstruve, riccati_jn, riccati_yn, spherical_in, spherical_jn, spherical_kn,
    };
    pub use crate::implementation::{spherical_yn, struve, y0, y0_zeros, y1, y1_zeros};
    pub use crate::implementation::{y1p_zeros, yn, yn_zeros, ynp_zeros, yv, yve};
}

/// Orthogonal polynomial construction, evaluation, quadrature and Legendre functions.
pub mod orthogonal {
    /// Values and first derivatives for associated Legendre Q.
    pub use crate::implementation::ValueDerivativeMatrices;
    pub use crate::implementation::{
        assoc_legendre_p, assoc_legendre_p_all, chebyc, chebys, chebyt, chebyu,
    };
    pub use crate::implementation::{
        eval_chebyc, eval_chebys, eval_chebyt, eval_chebyu, eval_gegenbauer, eval_genlaguerre,
    };
    pub use crate::implementation::{
        eval_hermite, eval_hermitenorm, eval_jacobi, eval_laguerre, eval_legendre, eval_sh_chebyt,
    };
    pub use crate::implementation::{
        eval_sh_chebyu, eval_sh_jacobi, eval_sh_legendre, gegenbauer, genlaguerre, hermite,
    };
    pub use crate::implementation::{
        hermitenorm, jacobi, laguerre, legendre, legendre_p, legendre_p_all,
    };
    pub use crate::implementation::{lpmv, lqmn, lqn, roots_chebyc, roots_chebys, roots_chebyt};
    pub use crate::implementation::{
        roots_chebyu, roots_gegenbauer, roots_genlaguerre, roots_hermite, roots_hermitenorm,
        roots_jacobi,
    };
    pub use crate::implementation::{
        roots_laguerre, roots_legendre, roots_sh_chebyt, roots_sh_chebyu, roots_sh_jacobi,
        roots_sh_legendre,
    };
    pub use crate::implementation::{
        sh_chebyt, sh_chebyu, sh_jacobi, sh_legendre, sph_legendre_p, sph_legendre_p_all,
    };
}

/// Confluent and Gauss hypergeometric and Wright functions.
pub mod hypergeometric {
    pub use crate::implementation::{
        hyp0f1, hyp1f1, hyp2f1, hyperu, log_wright_bessel, wright_bessel,
    };
}

/// Airy, Mathieu, spheroidal, parabolic-cylinder and related wave functions.
pub mod wave {
    pub use crate::implementation::sph_harm_y_all;
    pub use crate::implementation::{ai_zeros, airy, airye, bi_zeros, ellip_harm, ellip_harm_2};
    pub use crate::implementation::{
        ellip_normal, itairy, lmbda, mathieu_a, mathieu_b, mathieu_cem,
    };
    pub use crate::implementation::{
        mathieu_even_coef, mathieu_modcem1, mathieu_modcem2, mathieu_modsem1, mathieu_modsem2,
        mathieu_odd_coef,
    };
    pub use crate::implementation::{
        mathieu_sem, obl_ang1, obl_ang1_cv, obl_cv, obl_cv_seq, obl_rad1,
    };
    pub use crate::implementation::{obl_rad1_cv, obl_rad2, obl_rad2_cv, pbdn_seq, pbdv, pbdv_seq};
    pub use crate::implementation::{pbvv, pbvv_seq, pbwa, pro_ang1, pro_ang1_cv, pro_cv};
    pub use crate::implementation::{
        pro_cv_seq, pro_rad1, pro_rad1_cv, pro_rad2, pro_rad2_cv, sph_harm_y,
    };
}

/// Exponential, trigonometric, Fresnel, and related special integrals.
pub mod integrals {
    pub use crate::implementation::spence;
    pub use crate::implementation::{exp1, expi, expn, fresnel, fresnel_zeros, fresnelc_zeros};
    pub use crate::implementation::{
        fresnels_zeros, modfresnelm, modfresnelp, scaled_exp1, shichi, sici,
    };
}

/// Riemann and Hurwitz zeta functions.
pub mod zeta {
    pub use crate::implementation::{zeta, zetac};
}

/// Special probability functions, distribution integrals and inverses.
pub mod probability {
    pub use crate::implementation::{bdtr, bdtrc, bdtri, bdtrik, bdtrin, boxcox};
    pub use crate::implementation::{boxcox1p, btdtria, btdtrib, chdtr, chdtrc, chdtri};
    pub use crate::implementation::{chdtriv, chndtr, chndtridf, chndtrinc, chndtrix, fdtr};
    pub use crate::implementation::{fdtrc, fdtri, fdtridfd, gdtr, gdtrc, gdtria};
    pub use crate::implementation::{
        gdtrib, gdtrix, inv_boxcox, inv_boxcox1p, kolmogi, kolmogorov,
    };
    pub use crate::implementation::{log_ndtr, nbdtr, nbdtrc, nbdtri, nbdtrik, nbdtrin};
    pub use crate::implementation::{ncfdtr, ncfdtri, ncfdtridfd, ncfdtridfn, ncfdtrinc, nctdtr};
    pub use crate::implementation::{nctdtridf, nctdtrinc, nctdtrit, ndtr, ndtri, ndtri_exp};
    pub use crate::implementation::{nrdtrimn, nrdtrisd, owens_t, pdtr, pdtrc, pdtri};
    pub use crate::implementation::{pdtrik, smirnov, smirnovi, stdtr, stdtridf, stdtrit};
    pub use crate::implementation::{tklmbda, voigt_profile};
}

/// Error functions, stable scalar transforms, reductions and related helpers.
pub mod elementary {
    pub use crate::implementation::{agm, cosdg, cosm1, cotdg, dawsn, diric};
    pub use crate::implementation::{entr, erf, erf_zeros, erfc, erfcinv, erfcx};
    pub use crate::implementation::{erfi, erfinv, expit, exprel, huber, kl_div};
    pub use crate::implementation::{lambertw, lambertwm1, log_expit, log_softmax, log1pmx, logit};
    pub use crate::implementation::{logsumexp, powm1, pseudo_huber, radian, rel_entr, sinc};
    pub use crate::implementation::{sindg, softmax, softplus, tandg, wofz, wrightomega};
    pub use crate::implementation::{xlog1py, xlogy};
}

/// Factorials, binomial coefficients, harmonic numbers and related sequences.
pub mod combinatorics {
    pub use crate::implementation::{bernoulli, binom, comb, euler, factorial, factorial2};
    pub use crate::implementation::{
        factorialk, gen_harmonic, harmonic, ln_binomial, ln_factorial, ln_multinomial,
    };
    pub use crate::implementation::{multinomial, normalized_gen_harmonic, perm, poch, stirling2};
}

/// Legendre, Carlson, Bulirsch, and Jacobi elliptic functions.
pub mod elliptic {
    pub use crate::implementation::ellipj;
    pub use crate::implementation::elliptic::*;
}
