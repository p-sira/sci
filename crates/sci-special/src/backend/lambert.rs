//! Real branches of Lambert W, backed by the `lambert_w` dependency used by `special`.
use crate::backend::StrErr;
use crate::backend::utils::real_kernel;
use sci_core::Float;

real_kernel! {
    /// Principal real Lambert W branch, for x >= -1/e.
    ///
    /// Inverts w*exp(w) for w >= -1. Preserves signed zero.
    lambertw0(x) {
        if x == lambert_w::NEG_INV_E {
            -1.0
        } else {
            lambert_w::lambert_w0(x)
        }
    }
}

real_kernel! {
    /// Lower real Lambert W branch, for -1/e <= x <= 0.
    ///
    /// Inverts w*exp(w) for w <= -1. At either signed zero returns -infinity.
    lambertwm1(x) {
        if x == lambert_w::NEG_INV_E {
            -1.0
        } else {
            lambert_w::lambert_wm1(x)
        }
    }
}
