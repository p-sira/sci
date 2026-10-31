// Rust translation adapted from spec_math 0.1.6 (MIT); see LICENSES.
/*  (C) Copyright John Maddock 2006.
*  Use, modification and distribution are subject to the
*  Boost Software License, Version 1.0. (See accompanying file
*  LICENSE_1_0.txt or copy at https://www.boost.org/LICENSE_1_0.txt)
*/

/* Both lanczos.h and lanczos.c were formed from Boost's lanczos.hpp
*
* Scipy changes:
* - 06-22-2016: Removed all code not related to double precision and
*   ported to c for use in Cephes. Note that the order of the
*   coefficients is reversed to match the behavior of polevl.
*/

/*
* Optimal values for G for each N are taken from
* https://web.viu.ca/pughg/phdThesis/phdThesis.pdf,
* as are the theoretical error bounds.
*
* Constants calculated using the method described by Godfrey
* https://my.fit.edu/~gabdo/gamma.txt and elaborated by Toth at
* https://www.rskey.org/gamma.htm using NTL::RR at 1000 bit precision.
*/

/*
* Lanczos Coefficients for N=13 G=6.024680040776729583740234375
* Max experimental error (with arbitrary precision arithmetic) 1.196214e-17
* Generated with compiler: Microsoft Visual C++ version 8.0 on Win32 at Mar 23 2006
*
* Use for double precision.
*/

#![allow(clippy::excessive_precision)]

#[allow(unused_imports)]
use sci_core::Float;

const LANCZOS_SUM_EXPG_SCALED_NUM: [f64; 13] = [
    0.006061842346248906525783753964555936883222,
    0.5098416655656676188125178644804694509993,
    19.51992788247617482847860966235652136208,
    449.9445569063168119446858607650988409623,
    6955.999602515376140356310115515198987526,
    75999.29304014542649875303443598909137092,
    601859.6171681098786670226533699352302507,
    3481712.15498064590882071018964774556468,
    14605578.08768506808414169982791359218571,
    43338889.32467613834773723740590533316085,
    86363131.28813859145546927288977868422342,
    103794043.1163445451906271053616070238554,
    56906521.91347156388090791033559122686859,
];

const LANCZOS_SUM_EXPG_SCALED_DENOM: [f64; 13] = [
    1.0,
    66.0,
    1925.0,
    32670.0,
    357423.0,
    2637558.0,
    13339535.0,
    45995730.0,
    105258076.0,
    150917976.0,
    120543840.0,
    39916800.0,
    0.0,
];

pub const LANCZOS_G: f64 = 6.024680040776729583740234375;

/* Scipy changes:
 * - 06-22-2016: Removed all code not related to double precision and
 *   ported to c for use in Cephes
 */

//#include "mconf.h"
//#include "lanczos.h"
use crate::backend::cephes64::polevl::ratevl;

pub fn lanczos_sum_expg_scaled(x: f64) -> f64 {
    ratevl(
        x,
        &LANCZOS_SUM_EXPG_SCALED_NUM,
        LANCZOS_SUM_EXPG_SCALED_NUM.len() as isize - 1,
        &LANCZOS_SUM_EXPG_SCALED_DENOM,
        LANCZOS_SUM_EXPG_SCALED_DENOM.len() as isize - 1,
    )
}
