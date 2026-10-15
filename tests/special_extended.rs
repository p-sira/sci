mod common;
use common::{pair_call, parsed_call, scalar_call};

use sci::{complex::Complex, special};

#[test]
fn special_families_match_scipy() {
    for row in common::lines(include_str!("fixtures/special_extended.csv")) {
        let fields: Vec<_> = common::fields(row);
        let args: Vec<_> = fields[1].split(';').collect();
        let expected: Vec<f64> = common::vector(fields[2]);
        let actual: Vec<f64> = match fields[0] {
            "agm" => scalar_call!(special::agm; args[0], args[1]),
            "airy" => parsed_call!(special::airy; args[0]).to_vec(),
            "airye" => parsed_call!(special::airye; args[0]).to_vec(),
            "itairy" => parsed_call!(special::itairy; args[0]).to_vec(),
            "ellipj" => {
                let (a, b, c, d) = parsed_call!(special::ellipj; args[0], args[1]);
                vec![a, b, c, d]
            }
            "binom" => scalar_call!(special::binom; args[0], args[1]),
            "erfcx" => scalar_call!(special::erfcx; args[0]),
            "erfi" => scalar_call!(special::erfi; args[0]),
            "erfcinv" => scalar_call!(special::erfcinv; args[0]),
            "dawsn" => scalar_call!(special::dawsn; args[0]),
            "gammasgn" => scalar_call!(special::gammasgn; args[0]),
            "gammaincc" => scalar_call!(special::gammaincc; args[0], args[1]),
            "gammaincinv" => scalar_call!(special::gammaincinv; args[0], args[1]),
            "gammainccinv" => scalar_call!(special::gammainccinv; args[0], args[1]),
            "loggamma" => scalar_call!(special::loggamma; args[0]),
            "rgamma" => scalar_call!(special::rgamma; args[0]),
            "polygamma" => scalar_call!(special::polygamma; args[0], args[1]),
            "multigammaln" => scalar_call!(special::multigammaln; args[0], args[1]),
            "jv" => scalar_call!(special::jv; args[0], args[1]),
            "jve" => scalar_call!(special::jve; args[0], args[1]),
            "yv" => scalar_call!(special::yv; args[0], args[1]),
            "yve" => scalar_call!(special::yve; args[0], args[1]),
            "iv" => scalar_call!(special::iv; args[0], args[1]),
            "ive" => scalar_call!(special::ive; args[0], args[1]),
            "kv" => scalar_call!(special::kv; args[0], args[1]),
            "kve" => scalar_call!(special::kve; args[0], args[1]),
            "j0" => scalar_call!(special::j0; args[0]),
            "j1" => scalar_call!(special::j1; args[0]),
            "y0" => scalar_call!(special::y0; args[0]),
            "y1" => scalar_call!(special::y1; args[0]),
            "i0" => scalar_call!(special::i0; args[0]),
            "i1" => scalar_call!(special::i1; args[0]),
            "i0e" => scalar_call!(special::i0e; args[0]),
            "i1e" => scalar_call!(special::i1e; args[0]),
            "k0" => scalar_call!(special::k0; args[0]),
            "k1" => scalar_call!(special::k1; args[0]),
            "k0e" => scalar_call!(special::k0e; args[0]),
            "k1e" => scalar_call!(special::k1e; args[0]),
            "besselpoly" => scalar_call!(special::besselpoly; args[0], args[1], args[2]),
            "itj0y0" => pair_call!(special::itj0y0; args[0]),
            "it2j0y0" => pair_call!(special::it2j0y0; args[0]),
            "iti0k0" => pair_call!(special::iti0k0; args[0]),
            "it2i0k0" => pair_call!(special::it2i0k0; args[0]),
            "struve" => scalar_call!(special::struve; args[0], args[1]),
            "modstruve" => scalar_call!(special::modstruve; args[0], args[1]),
            "itstruve0" => scalar_call!(special::itstruve0; args[0]),
            "it2struve0" => scalar_call!(special::it2struve0; args[0]),
            "itmodstruve0" => scalar_call!(special::itmodstruve0; args[0]),
            "hyp0f1" => scalar_call!(special::hyp0f1; args[0], args[1]),
            "hyp1f1" => scalar_call!(special::hyp1f1; args[0], args[1], args[2]),
            "hyp2f1" => scalar_call!(special::hyp2f1; args[0], args[1], args[2], args[3]),
            "hyperu" => scalar_call!(special::hyperu; args[0], args[1], args[2]),
            "exp1" => scalar_call!(special::exp1; args[0]),
            "expi" => scalar_call!(special::expi; args[0]),
            "expn" => scalar_call!(special::expn; args[0], args[1]),
            "sici" => pair_call!(special::sici; args[0]),
            "shichi" => pair_call!(special::shichi; args[0]),
            "fresnel" => pair_call!(special::fresnel; args[0]),
            "spence" => scalar_call!(special::spence; args[0]),
            "zeta" => scalar_call!(special::zeta; args[0], args[1]),
            "zetac" => scalar_call!(special::zetac; args[0]),
            "wrightomega" => scalar_call!(special::wrightomega; args[0]),
            "wright_bessel" => scalar_call!(special::wright_bessel; args[0], args[1], args[2]),
            "log_wright_bessel" => {
                scalar_call!(special::log_wright_bessel; args[0], args[1], args[2])
            }
            "voigt_profile" => scalar_call!(special::voigt_profile; args[0], args[1], args[2]),
            "bdtr" => scalar_call!(special::bdtr; args[0], args[1], args[2]),
            "bdtrc" => scalar_call!(special::bdtrc; args[0], args[1], args[2]),
            "bdtri" => scalar_call!(special::bdtri; args[0], args[1], args[2]),
            "chdtr" => scalar_call!(special::chdtr; args[0], args[1]),
            "chdtrc" => scalar_call!(special::chdtrc; args[0], args[1]),
            "chdtri" => scalar_call!(special::chdtri; args[0], args[1]),
            "fdtr" => scalar_call!(special::fdtr; args[0], args[1], args[2]),
            "fdtrc" => scalar_call!(special::fdtrc; args[0], args[1], args[2]),
            "fdtri" => scalar_call!(special::fdtri; args[0], args[1], args[2]),
            "gdtr" => scalar_call!(special::gdtr; args[0], args[1], args[2]),
            "gdtrc" => scalar_call!(special::gdtrc; args[0], args[1], args[2]),
            "gdtrib" => scalar_call!(special::gdtrib; args[0], args[1], args[2]),
            "nbdtr" => scalar_call!(special::nbdtr; args[0], args[1], args[2]),
            "nbdtrc" => scalar_call!(special::nbdtrc; args[0], args[1], args[2]),
            "nbdtri" => scalar_call!(special::nbdtri; args[0], args[1], args[2]),
            "ndtr" => scalar_call!(special::ndtr; args[0]),
            "ndtri" => scalar_call!(special::ndtri; args[0]),
            "log_ndtr" => scalar_call!(special::log_ndtr; args[0]),
            "ndtri_exp" => scalar_call!(special::ndtri_exp; args[0]),
            "pdtr" => scalar_call!(special::pdtr; args[0], args[1]),
            "pdtrc" => scalar_call!(special::pdtrc; args[0], args[1]),
            "pdtri" => scalar_call!(special::pdtri; args[0], args[1]),
            "stdtr" => scalar_call!(special::stdtr; args[0], args[1]),
            "stdtrit" => scalar_call!(special::stdtrit; args[0], args[1]),
            "kolmogorov" => scalar_call!(special::kolmogorov; args[0]),
            "kolmogi" => scalar_call!(special::kolmogi; args[0]),
            "smirnov" => scalar_call!(special::smirnov; args[0], args[1]),
            "smirnovi" => scalar_call!(special::smirnovi; args[0], args[1]),
            "owens_t" => scalar_call!(special::owens_t; args[0], args[1]),
            "tklmbda" => scalar_call!(special::tklmbda; args[0], args[1]),
            "pbdv" => pair_call!(special::pbdv; args[0], args[1]),
            "pbvv" => pair_call!(special::pbvv; args[0], args[1]),
            "pbwa" => pair_call!(special::pbwa; args[0], args[1]),
            "mathieu_a" => scalar_call!(special::mathieu_a; args[0], args[1]),
            "mathieu_b" => scalar_call!(special::mathieu_b; args[0], args[1]),
            "mathieu_cem" => pair_call!(special::mathieu_cem; args[0], args[1], args[2]),
            "mathieu_sem" => pair_call!(special::mathieu_sem; args[0], args[1], args[2]),
            "mathieu_modcem1" => pair_call!(special::mathieu_modcem1; args[0], args[1], args[2]),
            "mathieu_modcem2" => pair_call!(special::mathieu_modcem2; args[0], args[1], args[2]),
            "mathieu_modsem1" => pair_call!(special::mathieu_modsem1; args[0], args[1], args[2]),
            "mathieu_modsem2" => pair_call!(special::mathieu_modsem2; args[0], args[1], args[2]),
            "ber" => scalar_call!(special::ber; args[0]),
            "bei" => scalar_call!(special::bei; args[0]),
            "ker" => scalar_call!(special::ker; args[0]),
            "kei" => scalar_call!(special::kei; args[0]),
            "berp" => scalar_call!(special::berp; args[0]),
            "beip" => scalar_call!(special::beip; args[0]),
            "kerp" => scalar_call!(special::kerp; args[0]),
            "keip" => scalar_call!(special::keip; args[0]),
            "ber_zeros" => parsed_call!(special::ber_zeros; args[0]).unwrap(),
            "bei_zeros" => parsed_call!(special::bei_zeros; args[0]).unwrap(),
            "ker_zeros" => parsed_call!(special::ker_zeros; args[0]).unwrap(),
            "kei_zeros" => parsed_call!(special::kei_zeros; args[0]).unwrap(),
            "berp_zeros" => parsed_call!(special::berp_zeros; args[0]).unwrap(),
            "beip_zeros" => parsed_call!(special::beip_zeros; args[0]).unwrap(),
            "kerp_zeros" => parsed_call!(special::kerp_zeros; args[0]).unwrap(),
            "keip_zeros" => parsed_call!(special::keip_zeros; args[0]).unwrap(),
            "kelvin_zeros" => parsed_call!(special::kelvin_zeros; args[0])
                .unwrap()
                .into_iter()
                .flatten()
                .collect(),
            "lpmv" => scalar_call!(special::lpmv; args[0], args[1], args[2]),
            "legendre_p" => scalar_call!(special::legendre_p; args[0], args[1]),
            "assoc_legendre_p" => {
                scalar_call!(special::assoc_legendre_p; args[0], args[1], args[2])
            }
            "sph_legendre_p" => scalar_call!(special::sph_legendre_p; args[0], args[1], args[2]),
            "lqn" => {
                let (a, b) = parsed_call!(special::lqn; args[0], args[1]).unwrap();
                [a, b].concat()
            }
            "lqmn" => {
                let (a, b) = parsed_call!(special::lqmn; args[0], args[1], args[2]).unwrap();
                [a, b].into_iter().flatten().flatten().collect()
            }
            "cbrt" => scalar_call!(special::cbrt; args[0]),
            "exp2" => scalar_call!(special::exp2; args[0]),
            "exp10" => scalar_call!(special::exp10; args[0]),
            "expm1" => scalar_call!(special::expm1; args[0]),
            "log1p" => scalar_call!(special::log1p; args[0]),
            "expit" => scalar_call!(special::expit; args[0]),
            "logit" => scalar_call!(special::logit; args[0]),
            "softplus" => scalar_call!(special::softplus; args[0]),
            "log_expit" => scalar_call!(special::log_expit; args[0]),
            "exprel" => scalar_call!(special::exprel; args[0]),
            "sinc" => scalar_call!(special::sinc; args[0]),
            "cosm1" => scalar_call!(special::cosm1; args[0]),
            "sindg" => scalar_call!(special::sindg; args[0]),
            "cosdg" => scalar_call!(special::cosdg; args[0]),
            "tandg" => scalar_call!(special::tandg; args[0]),
            "cotdg" => scalar_call!(special::cotdg; args[0]),
            "radian" => scalar_call!(special::radian; args[0], args[1], args[2]),
            "round" => scalar_call!(special::round; args[0]),
            "entr" => scalar_call!(special::entr; args[0]),
            "rel_entr" => scalar_call!(special::rel_entr; args[0], args[1]),
            "kl_div" => scalar_call!(special::kl_div; args[0], args[1]),
            "huber" => scalar_call!(special::huber; args[0], args[1]),
            "pseudo_huber" => scalar_call!(special::pseudo_huber; args[0], args[1]),
            "xlogy" => scalar_call!(special::xlogy; args[0], args[1]),
            "xlog1py" => scalar_call!(special::xlog1py; args[0], args[1]),
            "powm1" => scalar_call!(special::powm1; args[0], args[1]),
            "boxcox" => scalar_call!(special::boxcox; args[0], args[1]),
            "boxcox1p" => scalar_call!(special::boxcox1p; args[0], args[1]),
            "inv_boxcox" => scalar_call!(special::inv_boxcox; args[0], args[1]),
            "inv_boxcox1p" => scalar_call!(special::inv_boxcox1p; args[0], args[1]),
            "psi" => scalar_call!(special::psi; args[0]),
            "betaincc" => scalar_call!(special::betaincc; args[0], args[1], args[2]),
            "betainccinv" => scalar_call!(special::betainccinv; args[0], args[1], args[2]),
            "nrdtrimn" => scalar_call!(special::nrdtrimn; args[0], args[1], args[2]),
            "nrdtrisd" => scalar_call!(special::nrdtrisd; args[0], args[1], args[2]),
            "gdtria" => scalar_call!(special::gdtria; args[0], args[1], args[2]),
            "gdtrix" => scalar_call!(special::gdtrix; args[0], args[1], args[2]),
            "chdtriv" => scalar_call!(special::chdtriv; args[0], args[1]),
            "jn" => scalar_call!(special::jn; args[0], args[1]),
            "yn" => scalar_call!(special::yn; args[0], args[1]),
            "kn" => scalar_call!(special::kn; args[0], args[1]),
            "eval_jacobi" => scalar_call!(special::eval_jacobi; args[0], args[1], args[2], args[3]),
            "eval_legendre" => scalar_call!(special::eval_legendre; args[0], args[1]),
            "eval_gegenbauer" => scalar_call!(special::eval_gegenbauer; args[0], args[1], args[2]),
            "eval_genlaguerre" => {
                scalar_call!(special::eval_genlaguerre; args[0], args[1], args[2])
            }
            "eval_laguerre" => scalar_call!(special::eval_laguerre; args[0], args[1]),
            "eval_chebyt" => scalar_call!(special::eval_chebyt; args[0], args[1]),
            "eval_chebyu" => scalar_call!(special::eval_chebyu; args[0], args[1]),
            "eval_hermite" => scalar_call!(special::eval_hermite; args[0], args[1]),
            "eval_hermitenorm" => scalar_call!(special::eval_hermitenorm; args[0], args[1]),
            "pro_cv" => scalar_call!(special::pro_cv; args[0], args[1], args[2]),
            "pro_ang1" => pair_call!(special::pro_ang1; args[0], args[1], args[2], args[3]),
            "pro_ang1_cv" => {
                pair_call!(special::pro_ang1_cv; args[0], args[1], args[2], args[3], args[4])
            }
            "pro_rad1" => pair_call!(special::pro_rad1; args[0], args[1], args[2], args[3]),
            "pro_rad1_cv" => {
                pair_call!(special::pro_rad1_cv; args[0], args[1], args[2], args[3], args[4])
            }
            "pro_rad2" => pair_call!(special::pro_rad2; args[0], args[1], args[2], args[3]),
            "pro_rad2_cv" => {
                pair_call!(special::pro_rad2_cv; args[0], args[1], args[2], args[3], args[4])
            }
            "obl_cv" => scalar_call!(special::obl_cv; args[0], args[1], args[2]),
            "obl_ang1" => pair_call!(special::obl_ang1; args[0], args[1], args[2], args[3]),
            "obl_ang1_cv" => {
                pair_call!(special::obl_ang1_cv; args[0], args[1], args[2], args[3], args[4])
            }
            "obl_rad1" => pair_call!(special::obl_rad1; args[0], args[1], args[2], args[3]),
            "obl_rad1_cv" => {
                pair_call!(special::obl_rad1_cv; args[0], args[1], args[2], args[3], args[4])
            }
            "obl_rad2" => pair_call!(special::obl_rad2; args[0], args[1], args[2], args[3]),
            "obl_rad2_cv" => {
                pair_call!(special::obl_rad2_cv; args[0], args[1], args[2], args[3], args[4])
            }
            "jacobi" => parsed_call!(special::jacobi; args[0], args[1], args[2], args[3]).unwrap(),
            "roots_jacobi" => {
                let (a, b) =
                    parsed_call!(special::roots_jacobi; args[0], args[1], args[2]).unwrap();
                [a, b].concat()
            }
            "legendre" => parsed_call!(special::legendre; args[0], args[1]).unwrap(),
            "roots_legendre" => {
                let (a, b) = parsed_call!(special::roots_legendre; args[0]).unwrap();
                [a, b].concat()
            }
            "gegenbauer" => parsed_call!(special::gegenbauer; args[0], args[1], args[2]).unwrap(),
            "roots_gegenbauer" => {
                let (a, b) = parsed_call!(special::roots_gegenbauer; args[0], args[1]).unwrap();
                [a, b].concat()
            }
            "genlaguerre" => parsed_call!(special::genlaguerre; args[0], args[1], args[2]).unwrap(),
            "roots_genlaguerre" => {
                let (a, b) = parsed_call!(special::roots_genlaguerre; args[0], args[1]).unwrap();
                [a, b].concat()
            }
            "laguerre" => parsed_call!(special::laguerre; args[0], args[1]).unwrap(),
            "roots_laguerre" => {
                let (a, b) = parsed_call!(special::roots_laguerre; args[0]).unwrap();
                [a, b].concat()
            }
            "hermite" => parsed_call!(special::hermite; args[0], args[1]).unwrap(),
            "roots_hermite" => {
                let (a, b) = parsed_call!(special::roots_hermite; args[0]).unwrap();
                [a, b].concat()
            }
            "hermitenorm" => parsed_call!(special::hermitenorm; args[0], args[1]).unwrap(),
            "roots_hermitenorm" => {
                let (a, b) = parsed_call!(special::roots_hermitenorm; args[0]).unwrap();
                [a, b].concat()
            }
            "chebyt" => parsed_call!(special::chebyt; args[0], args[1]).unwrap(),
            "roots_chebyt" => {
                let (a, b) = parsed_call!(special::roots_chebyt; args[0]).unwrap();
                [a, b].concat()
            }
            "chebyu" => parsed_call!(special::chebyu; args[0], args[1]).unwrap(),
            "roots_chebyu" => {
                let (a, b) = parsed_call!(special::roots_chebyu; args[0]).unwrap();
                [a, b].concat()
            }
            "chebyc" => parsed_call!(special::chebyc; args[0], args[1]).unwrap(),
            "roots_chebyc" => {
                let (a, b) = parsed_call!(special::roots_chebyc; args[0]).unwrap();
                [a, b].concat()
            }
            "chebys" => parsed_call!(special::chebys; args[0], args[1]).unwrap(),
            "roots_chebys" => {
                let (a, b) = parsed_call!(special::roots_chebys; args[0]).unwrap();
                [a, b].concat()
            }
            "sh_chebyt" => parsed_call!(special::sh_chebyt; args[0], args[1]).unwrap(),
            "roots_sh_chebyt" => {
                let (a, b) = parsed_call!(special::roots_sh_chebyt; args[0]).unwrap();
                [a, b].concat()
            }
            "sh_chebyu" => parsed_call!(special::sh_chebyu; args[0], args[1]).unwrap(),
            "roots_sh_chebyu" => {
                let (a, b) = parsed_call!(special::roots_sh_chebyu; args[0]).unwrap();
                [a, b].concat()
            }
            "sh_legendre" => parsed_call!(special::sh_legendre; args[0], args[1]).unwrap(),
            "roots_sh_legendre" => {
                let (a, b) = parsed_call!(special::roots_sh_legendre; args[0]).unwrap();
                [a, b].concat()
            }
            "sh_jacobi" => {
                parsed_call!(special::sh_jacobi; args[0], args[1], args[2], args[3]).unwrap()
            }
            "roots_sh_jacobi" => {
                let (a, b) =
                    parsed_call!(special::roots_sh_jacobi; args[0], args[1], args[2]).unwrap();
                [a, b].concat()
            }
            "hankel1" => {
                let z = parsed_call!(special::hankel1; args[0], args[1]);
                vec![z.re, z.im]
            }
            "hankel1e" => {
                let z = parsed_call!(special::hankel1e; args[0], args[1]);
                vec![z.re, z.im]
            }
            "hankel2" => {
                let z = parsed_call!(special::hankel2; args[0], args[1]);
                vec![z.re, z.im]
            }
            "hankel2e" => {
                let z = parsed_call!(special::hankel2e; args[0], args[1]);
                vec![z.re, z.im]
            }
            "wofz" => {
                let z = special::wofz({
                    let p: Vec<f64> = args[0].split('/').map(|v| v.parse().unwrap()).collect();
                    Complex { re: p[0], im: p[1] }
                });
                vec![z.re, z.im]
            }
            "kelvin" => parsed_call!(special::kelvin; args[0])
                .into_iter()
                .flat_map(|z| [z.re, z.im])
                .collect(),
            "modfresnelp" => {
                let (a, b) = parsed_call!(special::modfresnelp; args[0]);
                vec![a.re, a.im, b.re, b.im]
            }
            "modfresnelm" => {
                let (a, b) = parsed_call!(special::modfresnelm; args[0]);
                vec![a.re, a.im, b.re, b.im]
            }
            "fresnel_zeros" => {
                let (a, b) = parsed_call!(special::fresnel_zeros; args[0]).unwrap();
                [a, b]
                    .into_iter()
                    .flatten()
                    .flat_map(|z| [z.re, z.im])
                    .collect()
            }
            "fresnels_zeros" => parsed_call!(special::fresnels_zeros; args[0])
                .unwrap()
                .into_iter()
                .flat_map(|z| [z.re, z.im])
                .collect(),
            "fresnelc_zeros" => parsed_call!(special::fresnelc_zeros; args[0])
                .unwrap()
                .into_iter()
                .flat_map(|z| [z.re, z.im])
                .collect(),
            "sph_harm_y" => {
                let z = parsed_call!(special::sph_harm_y; args[0], args[1], args[2], args[3]);
                vec![z.re, z.im]
            }
            "sph_harm_y_all" => {
                parsed_call!(special::sph_harm_y_all; args[0], args[1], args[2], args[3])
                    .unwrap()
                    .into_iter()
                    .flatten()
                    .flat_map(|z| [z.re, z.im])
                    .collect()
            }
            "legendre_p_all" => parsed_call!(special::legendre_p_all; args[0], args[1]).unwrap(),
            "assoc_legendre_p_all" => {
                parsed_call!(special::assoc_legendre_p_all; args[0], args[1], args[2])
                    .unwrap()
                    .into_iter()
                    .flatten()
                    .collect()
            }
            "sph_legendre_p_all" => {
                parsed_call!(special::sph_legendre_p_all; args[0], args[1], args[2])
                    .unwrap()
                    .into_iter()
                    .flatten()
                    .collect()
            }
            "mathieu_even_coef" => {
                parsed_call!(special::mathieu_even_coef; args[0], args[1]).unwrap()
            }
            "mathieu_odd_coef" => {
                parsed_call!(special::mathieu_odd_coef; args[0], args[1]).unwrap()
            }
            "pro_cv_seq" => parsed_call!(special::pro_cv_seq; args[0], args[1], args[2]).unwrap(),
            "obl_cv_seq" => parsed_call!(special::obl_cv_seq; args[0], args[1], args[2]).unwrap(),
            "jvp" => scalar_call!(special::jvp; args[0], args[1], args[2]),
            "yvp" => scalar_call!(special::yvp; args[0], args[1], args[2]),
            "ivp" => scalar_call!(special::ivp; args[0], args[1], args[2]),
            "kvp" => scalar_call!(special::kvp; args[0], args[1], args[2]),
            "h1vp" => {
                let z = parsed_call!(special::h1vp; args[0], args[1], args[2]);
                vec![z.re, z.im]
            }
            "h2vp" => {
                let z = parsed_call!(special::h2vp; args[0], args[1], args[2]);
                vec![z.re, z.im]
            }
            "spherical_jn" => scalar_call!(special::spherical_jn; args[0], args[1], args[2]),
            "spherical_yn" => scalar_call!(special::spherical_yn; args[0], args[1], args[2]),
            "spherical_in" => scalar_call!(special::spherical_in; args[0], args[1], args[2]),
            "spherical_kn" => scalar_call!(special::spherical_kn; args[0], args[1], args[2]),
            "assoc_laguerre" => scalar_call!(special::assoc_laguerre; args[0], args[1], args[2]),
            "eval_chebyc" => scalar_call!(special::eval_chebyc; args[0], args[1]),
            "eval_chebys" => scalar_call!(special::eval_chebys; args[0], args[1]),
            "eval_sh_chebyt" => scalar_call!(special::eval_sh_chebyt; args[0], args[1]),
            "eval_sh_chebyu" => scalar_call!(special::eval_sh_chebyu; args[0], args[1]),
            "eval_sh_legendre" => scalar_call!(special::eval_sh_legendre; args[0], args[1]),
            "eval_sh_jacobi" => {
                scalar_call!(special::eval_sh_jacobi; args[0], args[1], args[2], args[3])
            }
            "diric" => scalar_call!(special::diric; args[0], args[1]),
            "factorial" => scalar_call!(special::factorial; args[0]),
            "factorial2" => scalar_call!(special::factorial2; args[0]),
            "factorialk" => scalar_call!(special::factorialk; args[0], args[1]),
            "comb" => scalar_call!(special::comb; args[0], args[1]),
            "perm" => scalar_call!(special::perm; args[0], args[1]),
            "poch" => scalar_call!(special::poch; args[0], args[1]),
            "stirling2" => scalar_call!(special::stirling2; args[0], args[1]),
            "bernoulli" => parsed_call!(special::bernoulli; args[0]).unwrap(),
            "euler" => parsed_call!(special::euler; args[0]).unwrap(),
            "pbdv_seq" => {
                let (a, b) = parsed_call!(special::pbdv_seq; args[0], args[1]).unwrap();
                [a, b].concat()
            }
            "pbvv_seq" => {
                let (a, b) = parsed_call!(special::pbvv_seq; args[0], args[1]).unwrap();
                [a, b].concat()
            }
            "pbdn_seq" => {
                let (a, b) = parsed_call!(special::pbdn_seq; args[0], args[1]).unwrap();
                [a, b].concat()
            }
            "riccati_jn" => {
                let (a, b) = parsed_call!(special::riccati_jn; args[0], args[1]).unwrap();
                [a, b].concat()
            }
            "riccati_yn" => {
                let (a, b) = parsed_call!(special::riccati_yn; args[0], args[1]).unwrap();
                [a, b].concat()
            }
            "lmbda" => {
                let (a, b) = parsed_call!(special::lmbda; args[0], args[1]).unwrap();
                [a, b].concat()
            }
            "logsumexp" => vec![special::logsumexp(
                &args[0]
                    .split('/')
                    .map(|v| v.parse::<f64>().unwrap())
                    .collect::<Vec<_>>(),
            )],
            "softmax" => special::softmax(
                &args[0]
                    .split('/')
                    .map(|v| v.parse::<f64>().unwrap())
                    .collect::<Vec<_>>(),
            )
            .to_vec(),
            "log_softmax" => special::log_softmax(
                &args[0]
                    .split('/')
                    .map(|v| v.parse::<f64>().unwrap())
                    .collect::<Vec<_>>(),
            )
            .to_vec(),
            "c_roots" => {
                let (a, b) = parsed_call!(special::c_roots; args[0]).unwrap();
                [a, b].concat()
            }
            "cg_roots" => {
                let (a, b) = parsed_call!(special::cg_roots; args[0], args[1]).unwrap();
                [a, b].concat()
            }
            "h_roots" => {
                let (a, b) = parsed_call!(special::h_roots; args[0]).unwrap();
                [a, b].concat()
            }
            "he_roots" => {
                let (a, b) = parsed_call!(special::he_roots; args[0]).unwrap();
                [a, b].concat()
            }
            "j_roots" => {
                let (a, b) = parsed_call!(special::j_roots; args[0], args[1], args[2]).unwrap();
                [a, b].concat()
            }
            "js_roots" => {
                let (a, b) = parsed_call!(special::js_roots; args[0], args[1], args[2]).unwrap();
                [a, b].concat()
            }
            "l_roots" => {
                let (a, b) = parsed_call!(special::l_roots; args[0]).unwrap();
                [a, b].concat()
            }
            "la_roots" => {
                let (a, b) = parsed_call!(special::la_roots; args[0], args[1]).unwrap();
                [a, b].concat()
            }
            "p_roots" => {
                let (a, b) = parsed_call!(special::p_roots; args[0]).unwrap();
                [a, b].concat()
            }
            "ps_roots" => {
                let (a, b) = parsed_call!(special::ps_roots; args[0]).unwrap();
                [a, b].concat()
            }
            "s_roots" => {
                let (a, b) = parsed_call!(special::s_roots; args[0]).unwrap();
                [a, b].concat()
            }
            "t_roots" => {
                let (a, b) = parsed_call!(special::t_roots; args[0]).unwrap();
                [a, b].concat()
            }
            "ts_roots" => {
                let (a, b) = parsed_call!(special::ts_roots; args[0]).unwrap();
                [a, b].concat()
            }
            "u_roots" => {
                let (a, b) = parsed_call!(special::u_roots; args[0]).unwrap();
                [a, b].concat()
            }
            "us_roots" => {
                let (a, b) = parsed_call!(special::us_roots; args[0]).unwrap();
                [a, b].concat()
            }
            "btdtria" => scalar_call!(special::btdtria; args[0], args[1], args[2]),
            "btdtrib" => scalar_call!(special::btdtrib; args[0], args[1], args[2]),
            "bdtrik" => scalar_call!(special::bdtrik; args[0], args[1], args[2]),
            "bdtrin" => scalar_call!(special::bdtrin; args[0], args[1], args[2]),
            "nbdtrik" => scalar_call!(special::nbdtrik; args[0], args[1], args[2]),
            "nbdtrin" => scalar_call!(special::nbdtrin; args[0], args[1], args[2]),
            "pdtrik" => scalar_call!(special::pdtrik; args[0], args[1]),
            "stdtridf" => scalar_call!(special::stdtridf; args[0], args[1]),
            "fdtridfd" => scalar_call!(special::fdtridfd; args[0], args[1], args[2]),
            "chndtr" => scalar_call!(special::chndtr; args[0], args[1], args[2]),
            "chndtrix" => scalar_call!(special::chndtrix; args[0], args[1], args[2]),
            "chndtridf" => scalar_call!(special::chndtridf; args[0], args[1], args[2]),
            "chndtrinc" => scalar_call!(special::chndtrinc; args[0], args[1], args[2]),
            "ncfdtr" => scalar_call!(special::ncfdtr; args[0], args[1], args[2], args[3]),
            "ncfdtri" => scalar_call!(special::ncfdtri; args[0], args[1], args[2], args[3]),
            "ncfdtridfd" => scalar_call!(special::ncfdtridfd; args[0], args[1], args[2], args[3]),
            "ncfdtridfn" => scalar_call!(special::ncfdtridfn; args[0], args[1], args[2], args[3]),
            "ncfdtrinc" => scalar_call!(special::ncfdtrinc; args[0], args[1], args[2], args[3]),
            "nctdtr" => scalar_call!(special::nctdtr; args[0], args[1], args[2]),
            "nctdtrit" => scalar_call!(special::nctdtrit; args[0], args[1], args[2]),
            "nctdtrinc" => scalar_call!(special::nctdtrinc; args[0], args[1], args[2]),
            "nctdtridf" => scalar_call!(special::nctdtridf; args[0], args[1], args[2]),
            "jn_zeros" => parsed_call!(special::jn_zeros; args[0], args[1]).unwrap(),
            "jnp_zeros" => parsed_call!(special::jnp_zeros; args[0], args[1]).unwrap(),
            "yn_zeros" => parsed_call!(special::yn_zeros; args[0], args[1]).unwrap(),
            "ynp_zeros" => parsed_call!(special::ynp_zeros; args[0], args[1]).unwrap(),
            "jnyn_zeros" => parsed_call!(special::jnyn_zeros; args[0], args[1])
                .unwrap()
                .into_iter()
                .flatten()
                .collect(),
            "y0_zeros" => {
                let (a, b) = parsed_call!(special::y0_zeros; args[0]).unwrap();
                [a, b].concat()
            }
            "y1_zeros" => {
                let (a, b) = parsed_call!(special::y1_zeros; args[0]).unwrap();
                [a, b].concat()
            }
            "y1p_zeros" => {
                let (a, b) = parsed_call!(special::y1p_zeros; args[0]).unwrap();
                [a, b].concat()
            }
            "ai_zeros" => parsed_call!(special::ai_zeros; args[0])
                .unwrap()
                .into_iter()
                .flatten()
                .collect(),
            "bi_zeros" => parsed_call!(special::bi_zeros; args[0])
                .unwrap()
                .into_iter()
                .flatten()
                .collect(),
            "erf_zeros" => parsed_call!(special::erf_zeros; args[0])
                .unwrap()
                .into_iter()
                .flat_map(|z| [z.re, z.im])
                .collect(),
            "jnjnp_zeros" => parsed_call!(special::jnjnp_zeros; args[0])
                .unwrap()
                .into_iter()
                .flatten()
                .collect(),
            "ellip_harm" => {
                vec![parsed_call!(special::ellip_harm; args[0], args[1], args[2], args[3], args[4], args[5], args[6]).unwrap()]
            }
            "ellip_harm_2" => {
                vec![parsed_call!(special::ellip_harm_2; args[0], args[1], args[2], args[3], args[4]).unwrap()]
            }
            "ellip_normal" => {
                vec![
                    parsed_call!(special::ellip_normal; args[0], args[1], args[2], args[3])
                        .unwrap(),
                ]
            }
            "gamma" => scalar_call!(special::gamma; args[0]),
            "gammaln" => scalar_call!(special::gammaln; args[0]),
            "digamma" => scalar_call!(special::digamma; args[0]),
            "erf" => scalar_call!(special::erf; args[0]),
            "erfc" => scalar_call!(special::erfc; args[0]),
            "erfinv" => scalar_call!(special::erfinv; args[0]),
            "beta" => scalar_call!(special::beta; args[0], args[1]),
            "betaln" => scalar_call!(special::betaln; args[0], args[1]),
            "betainc" => scalar_call!(special::betainc; args[0], args[1], args[2]),
            "betaincinv" => scalar_call!(special::betaincinv; args[0], args[1], args[2]),
            "gammainc" => scalar_call!(special::gammainc; args[0], args[1]),
            "lambertw" => scalar_call!(special::lambertw; args[0]),
            "ellipk" => vec![parsed_call!(special::ellipk; args[0]).unwrap()],
            "ellipe" => vec![parsed_call!(special::ellipe; args[0]).unwrap()],
            "ellipeinc" => vec![parsed_call!(special::ellipeinc; args[0], args[1]).unwrap()],
            "elliprf" => vec![parsed_call!(special::elliprf; args[0], args[1], args[2]).unwrap()],
            "elliprd" => vec![parsed_call!(special::elliprd; args[0], args[1], args[2]).unwrap()],
            "elliprg" => vec![parsed_call!(special::elliprg; args[0], args[1], args[2]).unwrap()],
            "elliprj" => {
                vec![parsed_call!(special::elliprj; args[0], args[1], args[2], args[3]).unwrap()]
            }
            "elliprc" => vec![parsed_call!(special::elliprc; args[0], args[1]).unwrap()],
            "ellipkinc" => vec![parsed_call!(special::ellipkinc; args[0], args[1]).unwrap()],
            "ellipkm1" => vec![parsed_call!(special::ellipkm1; args[0]).unwrap()],
            "trigamma" => scalar_call!(special::trigamma; args[0]),
            "lambertwm1" => scalar_call!(special::lambertwm1; args[0]),
            "ellipke" => {
                let (a, b) = parsed_call!(special::ellipke; args[0]).unwrap();
                vec![a, b]
            }
            "ellipf" => vec![parsed_call!(special::ellipf; args[0], args[1]).unwrap()],
            "ellippi" => vec![parsed_call!(special::ellippi; args[0], args[1]).unwrap()],
            "ellippiinc" => {
                vec![parsed_call!(special::ellippiinc; args[0], args[1], args[2]).unwrap()]
            }
            "ellipd" => vec![parsed_call!(special::ellipd; args[0]).unwrap()],
            "ellipdinc" => vec![parsed_call!(special::ellipdinc; args[0], args[1]).unwrap()],
            "cel" => vec![parsed_call!(special::cel; args[0], args[1], args[2], args[3]).unwrap()],
            "cel1" => vec![parsed_call!(special::cel1; args[0]).unwrap()],
            "cel2" => vec![parsed_call!(special::cel2; args[0], args[1], args[2]).unwrap()],
            "cel3" => vec![parsed_call!(special::cel3; args[0], args[1]).unwrap()],
            "el1" => vec![parsed_call!(special::el1; args[0], args[1]).unwrap()],
            "el2" => vec![parsed_call!(special::el2; args[0], args[1], args[2], args[3]).unwrap()],
            "el3" => vec![parsed_call!(special::el3; args[0], args[1], args[2]).unwrap()],
            "jacobi_zeta" => vec![parsed_call!(special::jacobi_zeta; args[0], args[1]).unwrap()],
            "heuman_lambda" => {
                vec![parsed_call!(special::heuman_lambda; args[0], args[1]).unwrap()]
            }
            "ellippiinc_bulirsch" => {
                vec![parsed_call!(special::ellippiinc_bulirsch; args[0], args[1], args[2]).unwrap()]
            }
            name => panic!("missing Rust dispatch for {name}"),
        };
        assert_eq!(actual.len(), expected.len(), "{} shape", fields[0]);
        for (i, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            if expected.is_nan() {
                assert!(actual.is_nan(), "{}[{i}]", fields[0]);
            } else if expected.is_infinite() {
                assert_eq!(actual, expected, "{}[{i}]", fields[0]);
            } else {
                let rtol: f64 = fields[3].parse().unwrap();
                let atol: f64 = fields[4].parse().unwrap();
                assert!(
                    approx::abs_diff_eq!(
                        actual,
                        expected,
                        epsilon = atol.max(rtol * expected.abs())
                    ),
                    "{}[{i}]: {actual} != {expected}",
                    fields[0]
                );
            }
        }
    }
}
