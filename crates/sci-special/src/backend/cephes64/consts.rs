// Rust translation adapted from spec_math 0.1.6 (MIT); see LICENSES.
#![allow(clippy::excessive_precision)]

#[allow(unused_imports)]
use sci_core::Float;

pub const M_E: f64 = core::f64::consts::E;
pub const M_2_PI: f64 = core::f64::consts::FRAC_2_PI; /* 2/pi */
pub const M_PI: f64 = core::f64::consts::PI;
pub const M_PI_2: f64 = core::f64::consts::FRAC_PI_2;
pub const M_PI_4: f64 = core::f64::consts::FRAC_PI_4;

pub const MACHEP: f64 = 1.11022302462515654042E-16;

// #ifdef DENORMAL
pub const MAXLOG: f64 = 7.09782712893383996732E2; /* log(DBL_MAX) */
pub const MINLOG: f64 = -7.451332191019412076235E2; /* log(2**-1075) */
// #else
// pub const MAXLOG: f64 = 7.08396418532264106224E2;	/* log 2**1022 */
// pub const MINLOG: f64 = -7.08396418532264106224E2;	/* log 2**-1022 */
// #endif

pub const SQ2OPI: f64 = 7.9788456080286535587989E-1; /* sqrt( 2/pi ) */
pub const LOGSQ2: f64 = 3.46573590279972654709E-1; /* log(2)/2 */
pub const THPIO4: f64 = 2.35619449019234492885; /* 3*pi/4 */

pub const M_2_SQRTPI: f64 = core::f64::consts::FRAC_2_SQRT_PI;
pub const M_SQRT2: f64 = core::f64::consts::SQRT_2;
pub const M_SQRT1_2: f64 = core::f64::consts::FRAC_1_SQRT_2;

pub const EULER: f64 = 0.577215664901532860606512090082402431; /* Euler constant */
