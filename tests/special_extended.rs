mod common;
use common::{concat_pair, pair_call, parsed_call, scalar_call};

use sci::{complex::Complex, special};

#[test]
fn special_families_match_scipy() {
    for row in common::lines(include_str!("fixtures/special_extended.csv")) {
        let fields: Vec<_> = common::fields(row);
        let args: Vec<_> = fields[1].split(';').collect();
        let expected: Vec<f64> = common::vector(fields[2]);
        let actual: Vec<f64> = match fields[0] {
            "agm" => scalar_call!(special::elementary::agm; args[0], args[1]),
            "airy" => parsed_call!(special::wave::airy; args[0]).to_vec(),
            "airye" => parsed_call!(special::wave::airye; args[0]).to_vec(),
            "itairy" => parsed_call!(special::wave::itairy; args[0]).to_vec(),
            "ellipj" => {
                let (a, b, c, d) = parsed_call!(special::elliptic::ellipj; args[0], args[1]);
                vec![a, b, c, d]
            }
            "binom" => scalar_call!(special::combinatorics::binom; args[0], args[1]),
            "erfcx" => scalar_call!(special::elementary::erfcx; args[0]),
            "erfi" => scalar_call!(special::elementary::erfi; args[0]),
            "erfcinv" => scalar_call!(special::elementary::erfcinv; args[0]),
            "dawsn" => scalar_call!(special::elementary::dawsn; args[0]),
            "gammasgn" => scalar_call!(special::gamma::gammasgn; args[0]),
            "gammaincc" => scalar_call!(special::gamma::gammaincc; args[0], args[1]),
            "gammaincinv" => scalar_call!(special::gamma::gammaincinv; args[0], args[1]),
            "gammainccinv" => scalar_call!(special::gamma::gammainccinv; args[0], args[1]),
            "loggamma" => scalar_call!(special::gamma::loggamma; args[0]),
            "rgamma" => scalar_call!(special::gamma::rgamma; args[0]),
            "polygamma" => scalar_call!(special::gamma::polygamma; args[0], args[1]),
            "multigammaln" => scalar_call!(special::gamma::multigammaln; args[0], args[1]),
            "jv" => scalar_call!(special::bessel::jv; args[0], args[1]),
            "jve" => scalar_call!(special::bessel::jve; args[0], args[1]),
            "yv" => scalar_call!(special::bessel::yv; args[0], args[1]),
            "yve" => scalar_call!(special::bessel::yve; args[0], args[1]),
            "iv" => scalar_call!(special::bessel::iv; args[0], args[1]),
            "ive" => scalar_call!(special::bessel::ive; args[0], args[1]),
            "kv" => scalar_call!(special::bessel::kv; args[0], args[1]),
            "kve" => scalar_call!(special::bessel::kve; args[0], args[1]),
            "j0" => scalar_call!(special::bessel::j0; args[0]),
            "j1" => scalar_call!(special::bessel::j1; args[0]),
            "y0" => scalar_call!(special::bessel::y0; args[0]),
            "y1" => scalar_call!(special::bessel::y1; args[0]),
            "i0" => scalar_call!(special::bessel::i0; args[0]),
            "i1" => scalar_call!(special::bessel::i1; args[0]),
            "i0e" => scalar_call!(special::bessel::i0e; args[0]),
            "i1e" => scalar_call!(special::bessel::i1e; args[0]),
            "k0" => scalar_call!(special::bessel::k0; args[0]),
            "k1" => scalar_call!(special::bessel::k1; args[0]),
            "k0e" => scalar_call!(special::bessel::k0e; args[0]),
            "k1e" => scalar_call!(special::bessel::k1e; args[0]),
            "besselpoly" => scalar_call!(special::bessel::besselpoly; args[0], args[1], args[2]),
            "itj0y0" => pair_call!(special::bessel::itj0y0; args[0]),
            "it2j0y0" => pair_call!(special::bessel::it2j0y0; args[0]),
            "iti0k0" => pair_call!(special::bessel::iti0k0; args[0]),
            "it2i0k0" => pair_call!(special::bessel::it2i0k0; args[0]),
            "struve" => scalar_call!(special::bessel::struve; args[0], args[1]),
            "modstruve" => scalar_call!(special::bessel::modstruve; args[0], args[1]),
            "itstruve0" => scalar_call!(special::bessel::itstruve0; args[0]),
            "it2struve0" => scalar_call!(special::bessel::it2struve0; args[0]),
            "itmodstruve0" => scalar_call!(special::bessel::itmodstruve0; args[0]),
            "hyp0f1" => scalar_call!(special::hypergeometric::hyp0f1; args[0], args[1]),
            "hyp1f1" => scalar_call!(special::hypergeometric::hyp1f1; args[0], args[1], args[2]),
            "hyp2f1" => {
                scalar_call!(special::hypergeometric::hyp2f1; args[0], args[1], args[2], args[3])
            }
            "hyperu" => scalar_call!(special::hypergeometric::hyperu; args[0], args[1], args[2]),
            "exp1" => scalar_call!(special::integrals::exp1; args[0]),
            "expi" => scalar_call!(special::integrals::expi; args[0]),
            "expn" => scalar_call!(special::integrals::expn; args[0], args[1]),
            "sici" => pair_call!(special::integrals::sici; args[0]),
            "shichi" => pair_call!(special::integrals::shichi; args[0]),
            "fresnel" => pair_call!(special::integrals::fresnel; args[0]),
            "spence" => scalar_call!(special::integrals::spence; args[0]),
            "zeta" => scalar_call!(special::zeta::zeta; args[0], args[1]),
            "zetac" => scalar_call!(special::zeta::zetac; args[0]),
            "wrightomega" => scalar_call!(special::elementary::wrightomega; args[0]),
            "wright_bessel" => {
                scalar_call!(special::hypergeometric::wright_bessel; args[0], args[1], args[2])
            }
            "log_wright_bessel" => {
                scalar_call!(special::hypergeometric::log_wright_bessel; args[0], args[1], args[2])
            }
            "voigt_profile" => {
                scalar_call!(special::probability::voigt_profile; args[0], args[1], args[2])
            }
            "bdtr" => scalar_call!(special::probability::bdtr; args[0], args[1], args[2]),
            "bdtrc" => scalar_call!(special::probability::bdtrc; args[0], args[1], args[2]),
            "bdtri" => scalar_call!(special::probability::bdtri; args[0], args[1], args[2]),
            "chdtr" => scalar_call!(special::probability::chdtr; args[0], args[1]),
            "chdtrc" => scalar_call!(special::probability::chdtrc; args[0], args[1]),
            "chdtri" => scalar_call!(special::probability::chdtri; args[0], args[1]),
            "fdtr" => scalar_call!(special::probability::fdtr; args[0], args[1], args[2]),
            "fdtrc" => scalar_call!(special::probability::fdtrc; args[0], args[1], args[2]),
            "fdtri" => scalar_call!(special::probability::fdtri; args[0], args[1], args[2]),
            "gdtr" => scalar_call!(special::probability::gdtr; args[0], args[1], args[2]),
            "gdtrc" => scalar_call!(special::probability::gdtrc; args[0], args[1], args[2]),
            "gdtrib" => scalar_call!(special::probability::gdtrib; args[0], args[1], args[2]),
            "nbdtr" => scalar_call!(special::probability::nbdtr; args[0], args[1], args[2]),
            "nbdtrc" => scalar_call!(special::probability::nbdtrc; args[0], args[1], args[2]),
            "nbdtri" => scalar_call!(special::probability::nbdtri; args[0], args[1], args[2]),
            "ndtr" => scalar_call!(special::probability::ndtr; args[0]),
            "ndtri" => scalar_call!(special::probability::ndtri; args[0]),
            "log_ndtr" => scalar_call!(special::probability::log_ndtr; args[0]),
            "ndtri_exp" => scalar_call!(special::probability::ndtri_exp; args[0]),
            "pdtr" => scalar_call!(special::probability::pdtr; args[0], args[1]),
            "pdtrc" => scalar_call!(special::probability::pdtrc; args[0], args[1]),
            "pdtri" => scalar_call!(special::probability::pdtri; args[0], args[1]),
            "stdtr" => scalar_call!(special::probability::stdtr; args[0], args[1]),
            "stdtrit" => scalar_call!(special::probability::stdtrit; args[0], args[1]),
            "kolmogorov" => scalar_call!(special::probability::kolmogorov; args[0]),
            "kolmogi" => scalar_call!(special::probability::kolmogi; args[0]),
            "smirnov" => scalar_call!(special::probability::smirnov; args[0], args[1]),
            "smirnovi" => scalar_call!(special::probability::smirnovi; args[0], args[1]),
            "owens_t" => scalar_call!(special::probability::owens_t; args[0], args[1]),
            "tklmbda" => scalar_call!(special::probability::tklmbda; args[0], args[1]),
            "pbdv" => pair_call!(special::wave::pbdv; args[0], args[1]),
            "pbvv" => pair_call!(special::wave::pbvv; args[0], args[1]),
            "pbwa" => pair_call!(special::wave::pbwa; args[0], args[1]),
            "mathieu_a" => scalar_call!(special::wave::mathieu_a; args[0], args[1]),
            "mathieu_b" => scalar_call!(special::wave::mathieu_b; args[0], args[1]),
            "mathieu_cem" => pair_call!(special::wave::mathieu_cem; args[0], args[1], args[2]),
            "mathieu_sem" => pair_call!(special::wave::mathieu_sem; args[0], args[1], args[2]),
            "mathieu_modcem1" => {
                pair_call!(special::wave::mathieu_modcem1; args[0], args[1], args[2])
            }
            "mathieu_modcem2" => {
                pair_call!(special::wave::mathieu_modcem2; args[0], args[1], args[2])
            }
            "mathieu_modsem1" => {
                pair_call!(special::wave::mathieu_modsem1; args[0], args[1], args[2])
            }
            "mathieu_modsem2" => {
                pair_call!(special::wave::mathieu_modsem2; args[0], args[1], args[2])
            }
            "ber" => scalar_call!(special::bessel::ber; args[0]),
            "bei" => scalar_call!(special::bessel::bei; args[0]),
            "ker" => scalar_call!(special::bessel::ker; args[0]),
            "kei" => scalar_call!(special::bessel::kei; args[0]),
            "berp" => scalar_call!(special::bessel::berp; args[0]),
            "beip" => scalar_call!(special::bessel::beip; args[0]),
            "kerp" => scalar_call!(special::bessel::kerp; args[0]),
            "keip" => scalar_call!(special::bessel::keip; args[0]),
            "ber_zeros" => parsed_call!(special::bessel::ber_zeros; args[0]).unwrap(),
            "bei_zeros" => parsed_call!(special::bessel::bei_zeros; args[0]).unwrap(),
            "ker_zeros" => parsed_call!(special::bessel::ker_zeros; args[0]).unwrap(),
            "kei_zeros" => parsed_call!(special::bessel::kei_zeros; args[0]).unwrap(),
            "berp_zeros" => parsed_call!(special::bessel::berp_zeros; args[0]).unwrap(),
            "beip_zeros" => parsed_call!(special::bessel::beip_zeros; args[0]).unwrap(),
            "kerp_zeros" => parsed_call!(special::bessel::kerp_zeros; args[0]).unwrap(),
            "keip_zeros" => parsed_call!(special::bessel::keip_zeros; args[0]).unwrap(),
            "kelvin_zeros" => parsed_call!(special::bessel::kelvin_zeros; args[0])
                .unwrap()
                .into_iter()
                .flatten()
                .collect(),
            "lpmv" => scalar_call!(special::orthogonal::lpmv; args[0], args[1], args[2]),
            "legendre_p" => scalar_call!(special::orthogonal::legendre_p; args[0], args[1]),
            "assoc_legendre_p" => {
                scalar_call!(special::orthogonal::assoc_legendre_p; args[0], args[1], args[2])
            }
            "sph_legendre_p" => {
                scalar_call!(special::orthogonal::sph_legendre_p; args[0], args[1], args[2])
            }
            "lqn" => {
                concat_pair!(parsed_call!(special::orthogonal::lqn; args[0], args[1]).unwrap())
            }
            "lqmn" => {
                let (a, b) =
                    parsed_call!(special::orthogonal::lqmn; args[0], args[1], args[2]).unwrap();
                [a, b].into_iter().flatten().flatten().collect()
            }
            "expit" => scalar_call!(special::elementary::expit; args[0]),
            "logit" => scalar_call!(special::elementary::logit; args[0]),
            "softplus" => scalar_call!(special::elementary::softplus; args[0]),
            "log_expit" => scalar_call!(special::elementary::log_expit; args[0]),
            "exprel" => scalar_call!(special::elementary::exprel; args[0]),
            "sinc" => scalar_call!(special::elementary::sinc; args[0]),
            "cosm1" => scalar_call!(special::elementary::cosm1; args[0]),
            "sindg" => scalar_call!(special::elementary::sindg; args[0]),
            "cosdg" => scalar_call!(special::elementary::cosdg; args[0]),
            "tandg" => scalar_call!(special::elementary::tandg; args[0]),
            "cotdg" => scalar_call!(special::elementary::cotdg; args[0]),
            "radian" => scalar_call!(special::elementary::radian; args[0], args[1], args[2]),
            "entr" => scalar_call!(special::elementary::entr; args[0]),
            "rel_entr" => scalar_call!(special::elementary::rel_entr; args[0], args[1]),
            "kl_div" => scalar_call!(special::elementary::kl_div; args[0], args[1]),
            "huber" => scalar_call!(special::elementary::huber; args[0], args[1]),
            "pseudo_huber" => scalar_call!(special::elementary::pseudo_huber; args[0], args[1]),
            "xlogy" => scalar_call!(special::elementary::xlogy; args[0], args[1]),
            "xlog1py" => scalar_call!(special::elementary::xlog1py; args[0], args[1]),
            "powm1" => scalar_call!(special::elementary::powm1; args[0], args[1]),
            "boxcox" => scalar_call!(special::probability::boxcox; args[0], args[1]),
            "boxcox1p" => scalar_call!(special::probability::boxcox1p; args[0], args[1]),
            "inv_boxcox" => scalar_call!(special::probability::inv_boxcox; args[0], args[1]),
            "inv_boxcox1p" => scalar_call!(special::probability::inv_boxcox1p; args[0], args[1]),
            "psi" => scalar_call!(special::gamma::digamma; args[0]),
            "betaincc" => scalar_call!(special::beta::betaincc; args[0], args[1], args[2]),
            "betainccinv" => scalar_call!(special::beta::betainccinv; args[0], args[1], args[2]),
            "nrdtrimn" => scalar_call!(special::probability::nrdtrimn; args[0], args[1], args[2]),
            "nrdtrisd" => scalar_call!(special::probability::nrdtrisd; args[0], args[1], args[2]),
            "gdtria" => scalar_call!(special::probability::gdtria; args[0], args[1], args[2]),
            "gdtrix" => scalar_call!(special::probability::gdtrix; args[0], args[1], args[2]),
            "chdtriv" => scalar_call!(special::probability::chdtriv; args[0], args[1]),
            "jn" => scalar_call!(special::bessel::jn; args[0], args[1]),
            "yn" => scalar_call!(special::bessel::yn; args[0], args[1]),
            "kn" => scalar_call!(special::bessel::kn; args[0], args[1]),
            "eval_jacobi" => {
                scalar_call!(special::orthogonal::eval_jacobi; args[0], args[1], args[2], args[3])
            }
            "eval_legendre" => scalar_call!(special::orthogonal::eval_legendre; args[0], args[1]),
            "eval_gegenbauer" => {
                scalar_call!(special::orthogonal::eval_gegenbauer; args[0], args[1], args[2])
            }
            "eval_genlaguerre" => {
                scalar_call!(special::orthogonal::eval_genlaguerre; args[0], args[1], args[2])
            }
            "eval_laguerre" => scalar_call!(special::orthogonal::eval_laguerre; args[0], args[1]),
            "eval_chebyt" => scalar_call!(special::orthogonal::eval_chebyt; args[0], args[1]),
            "eval_chebyu" => scalar_call!(special::orthogonal::eval_chebyu; args[0], args[1]),
            "eval_hermite" => scalar_call!(special::orthogonal::eval_hermite; args[0], args[1]),
            "eval_hermitenorm" => {
                scalar_call!(special::orthogonal::eval_hermitenorm; args[0], args[1])
            }
            "pro_cv" => scalar_call!(special::wave::pro_cv; args[0], args[1], args[2]),
            "pro_ang1" => pair_call!(special::wave::pro_ang1; args[0], args[1], args[2], args[3]),
            "pro_ang1_cv" => {
                pair_call!(special::wave::pro_ang1_cv; args[0], args[1], args[2], args[3], args[4])
            }
            "pro_rad1" => pair_call!(special::wave::pro_rad1; args[0], args[1], args[2], args[3]),
            "pro_rad1_cv" => {
                pair_call!(special::wave::pro_rad1_cv; args[0], args[1], args[2], args[3], args[4])
            }
            "pro_rad2" => pair_call!(special::wave::pro_rad2; args[0], args[1], args[2], args[3]),
            "pro_rad2_cv" => {
                pair_call!(special::wave::pro_rad2_cv; args[0], args[1], args[2], args[3], args[4])
            }
            "obl_cv" => scalar_call!(special::wave::obl_cv; args[0], args[1], args[2]),
            "obl_ang1" => pair_call!(special::wave::obl_ang1; args[0], args[1], args[2], args[3]),
            "obl_ang1_cv" => {
                pair_call!(special::wave::obl_ang1_cv; args[0], args[1], args[2], args[3], args[4])
            }
            "obl_rad1" => pair_call!(special::wave::obl_rad1; args[0], args[1], args[2], args[3]),
            "obl_rad1_cv" => {
                pair_call!(special::wave::obl_rad1_cv; args[0], args[1], args[2], args[3], args[4])
            }
            "obl_rad2" => pair_call!(special::wave::obl_rad2; args[0], args[1], args[2], args[3]),
            "obl_rad2_cv" => {
                pair_call!(special::wave::obl_rad2_cv; args[0], args[1], args[2], args[3], args[4])
            }
            "jacobi" => {
                parsed_call!(special::orthogonal::jacobi; args[0], args[1], args[2], args[3])
                    .unwrap()
            }
            "roots_jacobi" => concat_pair!(
                parsed_call!(special::orthogonal::roots_jacobi; args[0], args[1], args[2]).unwrap()
            ),
            "legendre" => parsed_call!(special::orthogonal::legendre; args[0], args[1]).unwrap(),
            "roots_legendre" => {
                concat_pair!(parsed_call!(special::orthogonal::roots_legendre; args[0]).unwrap())
            }
            "gegenbauer" => {
                parsed_call!(special::orthogonal::gegenbauer; args[0], args[1], args[2]).unwrap()
            }
            "roots_gegenbauer" => {
                concat_pair!(
                    parsed_call!(special::orthogonal::roots_gegenbauer; args[0], args[1]).unwrap()
                )
            }
            "genlaguerre" => {
                parsed_call!(special::orthogonal::genlaguerre; args[0], args[1], args[2]).unwrap()
            }
            "roots_genlaguerre" => {
                concat_pair!(
                    parsed_call!(special::orthogonal::roots_genlaguerre; args[0], args[1]).unwrap()
                )
            }
            "laguerre" => parsed_call!(special::orthogonal::laguerre; args[0], args[1]).unwrap(),
            "roots_laguerre" => {
                concat_pair!(parsed_call!(special::orthogonal::roots_laguerre; args[0]).unwrap())
            }
            "hermite" => parsed_call!(special::orthogonal::hermite; args[0], args[1]).unwrap(),
            "roots_hermite" => {
                concat_pair!(parsed_call!(special::orthogonal::roots_hermite; args[0]).unwrap())
            }
            "hermitenorm" => {
                parsed_call!(special::orthogonal::hermitenorm; args[0], args[1]).unwrap()
            }
            "roots_hermitenorm" => {
                concat_pair!(parsed_call!(special::orthogonal::roots_hermitenorm; args[0]).unwrap())
            }
            "chebyt" => parsed_call!(special::orthogonal::chebyt; args[0], args[1]).unwrap(),
            "roots_chebyt" => {
                concat_pair!(parsed_call!(special::orthogonal::roots_chebyt; args[0]).unwrap())
            }
            "chebyu" => parsed_call!(special::orthogonal::chebyu; args[0], args[1]).unwrap(),
            "roots_chebyu" => {
                concat_pair!(parsed_call!(special::orthogonal::roots_chebyu; args[0]).unwrap())
            }
            "chebyc" => parsed_call!(special::orthogonal::chebyc; args[0], args[1]).unwrap(),
            "roots_chebyc" => {
                concat_pair!(parsed_call!(special::orthogonal::roots_chebyc; args[0]).unwrap())
            }
            "chebys" => parsed_call!(special::orthogonal::chebys; args[0], args[1]).unwrap(),
            "roots_chebys" => {
                concat_pair!(parsed_call!(special::orthogonal::roots_chebys; args[0]).unwrap())
            }
            "sh_chebyt" => parsed_call!(special::orthogonal::sh_chebyt; args[0], args[1]).unwrap(),
            "roots_sh_chebyt" => {
                concat_pair!(parsed_call!(special::orthogonal::roots_sh_chebyt; args[0]).unwrap())
            }
            "sh_chebyu" => parsed_call!(special::orthogonal::sh_chebyu; args[0], args[1]).unwrap(),
            "roots_sh_chebyu" => {
                concat_pair!(parsed_call!(special::orthogonal::roots_sh_chebyu; args[0]).unwrap())
            }
            "sh_legendre" => {
                parsed_call!(special::orthogonal::sh_legendre; args[0], args[1]).unwrap()
            }
            "roots_sh_legendre" => {
                concat_pair!(parsed_call!(special::orthogonal::roots_sh_legendre; args[0]).unwrap())
            }
            "sh_jacobi" => {
                parsed_call!(special::orthogonal::sh_jacobi; args[0], args[1], args[2], args[3])
                    .unwrap()
            }
            "roots_sh_jacobi" => concat_pair!(
                parsed_call!(special::orthogonal::roots_sh_jacobi; args[0], args[1], args[2])
                    .unwrap()
            ),
            "hankel1" => {
                let z = parsed_call!(special::bessel::hankel1; args[0], args[1]);
                vec![z.re, z.im]
            }
            "hankel1e" => {
                let z = parsed_call!(special::bessel::hankel1e; args[0], args[1]);
                vec![z.re, z.im]
            }
            "hankel2" => {
                let z = parsed_call!(special::bessel::hankel2; args[0], args[1]);
                vec![z.re, z.im]
            }
            "hankel2e" => {
                let z = parsed_call!(special::bessel::hankel2e; args[0], args[1]);
                vec![z.re, z.im]
            }
            "wofz" => {
                let z = special::elementary::wofz({
                    let p: Vec<f64> = args[0].split('/').map(|v| v.parse().unwrap()).collect();
                    Complex { re: p[0], im: p[1] }
                });
                vec![z.re, z.im]
            }
            "kelvin" => parsed_call!(special::bessel::kelvin; args[0])
                .into_iter()
                .flat_map(|z| [z.re, z.im])
                .collect(),
            "modfresnelp" => {
                let (a, b) = parsed_call!(special::integrals::modfresnelp; args[0]);
                vec![a.re, a.im, b.re, b.im]
            }
            "modfresnelm" => {
                let (a, b) = parsed_call!(special::integrals::modfresnelm; args[0]);
                vec![a.re, a.im, b.re, b.im]
            }
            "fresnel_zeros" => {
                let (a, b) = parsed_call!(special::integrals::fresnel_zeros; args[0]).unwrap();
                [a, b]
                    .into_iter()
                    .flatten()
                    .flat_map(|z| [z.re, z.im])
                    .collect()
            }
            "fresnels_zeros" => parsed_call!(special::integrals::fresnels_zeros; args[0])
                .unwrap()
                .into_iter()
                .flat_map(|z| [z.re, z.im])
                .collect(),
            "fresnelc_zeros" => parsed_call!(special::integrals::fresnelc_zeros; args[0])
                .unwrap()
                .into_iter()
                .flat_map(|z| [z.re, z.im])
                .collect(),
            "sph_harm_y" => {
                let z = parsed_call!(special::wave::sph_harm_y; args[0], args[1], args[2], args[3]);
                vec![z.re, z.im]
            }
            "sph_harm_y_all" => {
                parsed_call!(special::wave::sph_harm_y_all; args[0], args[1], args[2], args[3])
                    .unwrap()
                    .into_iter()
                    .flatten()
                    .flat_map(|z| [z.re, z.im])
                    .collect()
            }
            "legendre_p_all" => {
                parsed_call!(special::orthogonal::legendre_p_all; args[0], args[1]).unwrap()
            }
            "assoc_legendre_p_all" => {
                parsed_call!(special::orthogonal::assoc_legendre_p_all; args[0], args[1], args[2])
                    .unwrap()
                    .into_iter()
                    .flatten()
                    .collect()
            }
            "sph_legendre_p_all" => {
                parsed_call!(special::orthogonal::sph_legendre_p_all; args[0], args[1], args[2])
                    .unwrap()
                    .into_iter()
                    .flatten()
                    .collect()
            }
            "mathieu_even_coef" => {
                parsed_call!(special::wave::mathieu_even_coef; args[0], args[1]).unwrap()
            }
            "mathieu_odd_coef" => {
                parsed_call!(special::wave::mathieu_odd_coef; args[0], args[1]).unwrap()
            }
            "pro_cv_seq" => {
                parsed_call!(special::wave::pro_cv_seq; args[0], args[1], args[2]).unwrap()
            }
            "obl_cv_seq" => {
                parsed_call!(special::wave::obl_cv_seq; args[0], args[1], args[2]).unwrap()
            }
            "jvp" => scalar_call!(special::bessel::jvp; args[0], args[1], args[2]),
            "yvp" => scalar_call!(special::bessel::yvp; args[0], args[1], args[2]),
            "ivp" => scalar_call!(special::bessel::ivp; args[0], args[1], args[2]),
            "kvp" => scalar_call!(special::bessel::kvp; args[0], args[1], args[2]),
            "h1vp" => {
                let z = parsed_call!(special::bessel::h1vp; args[0], args[1], args[2]);
                vec![z.re, z.im]
            }
            "h2vp" => {
                let z = parsed_call!(special::bessel::h2vp; args[0], args[1], args[2]);
                vec![z.re, z.im]
            }
            "spherical_jn" => {
                scalar_call!(special::bessel::spherical_jn; args[0], args[1], args[2])
            }
            "spherical_yn" => {
                scalar_call!(special::bessel::spherical_yn; args[0], args[1], args[2])
            }
            "spherical_in" => {
                scalar_call!(special::bessel::spherical_in; args[0], args[1], args[2])
            }
            "spherical_kn" => {
                scalar_call!(special::bessel::spherical_kn; args[0], args[1], args[2])
            }
            "assoc_laguerre" => {
                scalar_call!(special::orthogonal::eval_genlaguerre; args[1], args[2], args[0])
            }
            "eval_chebyc" => scalar_call!(special::orthogonal::eval_chebyc; args[0], args[1]),
            "eval_chebys" => scalar_call!(special::orthogonal::eval_chebys; args[0], args[1]),
            "eval_sh_chebyt" => scalar_call!(special::orthogonal::eval_sh_chebyt; args[0], args[1]),
            "eval_sh_chebyu" => scalar_call!(special::orthogonal::eval_sh_chebyu; args[0], args[1]),
            "eval_sh_legendre" => {
                scalar_call!(special::orthogonal::eval_sh_legendre; args[0], args[1])
            }
            "eval_sh_jacobi" => {
                scalar_call!(special::orthogonal::eval_sh_jacobi; args[0], args[1], args[2], args[3])
            }
            "diric" => scalar_call!(special::elementary::diric; args[0], args[1]),
            "factorial" => scalar_call!(special::combinatorics::factorial; args[0]),
            "factorial2" => scalar_call!(special::combinatorics::factorial2; args[0]),
            "factorialk" => scalar_call!(special::combinatorics::factorialk; args[0], args[1]),
            "comb" => scalar_call!(special::combinatorics::comb; args[0], args[1]),
            "perm" => scalar_call!(special::combinatorics::perm; args[0], args[1]),
            "poch" => scalar_call!(special::combinatorics::poch; args[0], args[1]),
            "stirling2" => scalar_call!(special::combinatorics::stirling2; args[0], args[1]),
            "bernoulli" => parsed_call!(special::combinatorics::bernoulli; args[0]).unwrap(),
            "euler" => parsed_call!(special::combinatorics::euler; args[0]).unwrap(),
            "pbdv_seq" => {
                concat_pair!(parsed_call!(special::wave::pbdv_seq; args[0], args[1]).unwrap())
            }
            "pbvv_seq" => {
                concat_pair!(parsed_call!(special::wave::pbvv_seq; args[0], args[1]).unwrap())
            }
            "pbdn_seq" => {
                concat_pair!(parsed_call!(special::wave::pbdn_seq; args[0], args[1]).unwrap())
            }
            "riccati_jn" => {
                concat_pair!(parsed_call!(special::bessel::riccati_jn; args[0], args[1]).unwrap())
            }
            "riccati_yn" => {
                concat_pair!(parsed_call!(special::bessel::riccati_yn; args[0], args[1]).unwrap())
            }
            "lmbda" => concat_pair!(parsed_call!(special::wave::lmbda; args[0], args[1]).unwrap()),
            "logsumexp" => vec![special::elementary::logsumexp(
                &args[0]
                    .split('/')
                    .map(|v| v.parse::<f64>().unwrap())
                    .collect::<Vec<_>>(),
            )],
            "softmax" => special::elementary::softmax(
                &args[0]
                    .split('/')
                    .map(|v| v.parse::<f64>().unwrap())
                    .collect::<Vec<_>>(),
            )
            .to_vec(),
            "log_softmax" => special::elementary::log_softmax(
                &args[0]
                    .split('/')
                    .map(|v| v.parse::<f64>().unwrap())
                    .collect::<Vec<_>>(),
            )
            .to_vec(),
            "c_roots" => {
                concat_pair!(parsed_call!(special::orthogonal::roots_chebyc; args[0]).unwrap())
            }
            "cg_roots" => {
                concat_pair!(
                    parsed_call!(special::orthogonal::roots_gegenbauer; args[0], args[1]).unwrap()
                )
            }
            "h_roots" => {
                concat_pair!(parsed_call!(special::orthogonal::roots_hermite; args[0]).unwrap())
            }
            "he_roots" => {
                concat_pair!(parsed_call!(special::orthogonal::roots_hermitenorm; args[0]).unwrap())
            }
            "j_roots" => {
                concat_pair!(
                    parsed_call!(special::orthogonal::roots_jacobi; args[0], args[1], args[2])
                        .unwrap()
                )
            }
            "js_roots" => {
                concat_pair!(
                    parsed_call!(special::orthogonal::roots_sh_jacobi; args[0], args[1], args[2])
                        .unwrap()
                )
            }
            "l_roots" => {
                concat_pair!(parsed_call!(special::orthogonal::roots_laguerre; args[0]).unwrap())
            }
            "la_roots" => {
                concat_pair!(
                    parsed_call!(special::orthogonal::roots_genlaguerre; args[0], args[1]).unwrap()
                )
            }
            "p_roots" => {
                concat_pair!(parsed_call!(special::orthogonal::roots_legendre; args[0]).unwrap())
            }
            "ps_roots" => {
                concat_pair!(parsed_call!(special::orthogonal::roots_sh_legendre; args[0]).unwrap())
            }
            "s_roots" => {
                concat_pair!(parsed_call!(special::orthogonal::roots_chebys; args[0]).unwrap())
            }
            "t_roots" => {
                concat_pair!(parsed_call!(special::orthogonal::roots_chebyt; args[0]).unwrap())
            }
            "ts_roots" => {
                concat_pair!(parsed_call!(special::orthogonal::roots_sh_chebyt; args[0]).unwrap())
            }
            "u_roots" => {
                concat_pair!(parsed_call!(special::orthogonal::roots_chebyu; args[0]).unwrap())
            }
            "us_roots" => {
                concat_pair!(parsed_call!(special::orthogonal::roots_sh_chebyu; args[0]).unwrap())
            }
            "btdtria" => scalar_call!(special::probability::btdtria; args[0], args[1], args[2]),
            "btdtrib" => scalar_call!(special::probability::btdtrib; args[0], args[1], args[2]),
            "bdtrik" => scalar_call!(special::probability::bdtrik; args[0], args[1], args[2]),
            "bdtrin" => scalar_call!(special::probability::bdtrin; args[0], args[1], args[2]),
            "nbdtrik" => scalar_call!(special::probability::nbdtrik; args[0], args[1], args[2]),
            "nbdtrin" => scalar_call!(special::probability::nbdtrin; args[0], args[1], args[2]),
            "pdtrik" => scalar_call!(special::probability::pdtrik; args[0], args[1]),
            "stdtridf" => scalar_call!(special::probability::stdtridf; args[0], args[1]),
            "fdtridfd" => scalar_call!(special::probability::fdtridfd; args[0], args[1], args[2]),
            "chndtr" => scalar_call!(special::probability::chndtr; args[0], args[1], args[2]),
            "chndtrix" => scalar_call!(special::probability::chndtrix; args[0], args[1], args[2]),
            "chndtridf" => scalar_call!(special::probability::chndtridf; args[0], args[1], args[2]),
            "chndtrinc" => scalar_call!(special::probability::chndtrinc; args[0], args[1], args[2]),
            "ncfdtr" => {
                scalar_call!(special::probability::ncfdtr; args[0], args[1], args[2], args[3])
            }
            "ncfdtri" => {
                scalar_call!(special::probability::ncfdtri; args[0], args[1], args[2], args[3])
            }
            "ncfdtridfd" => {
                scalar_call!(special::probability::ncfdtridfd; args[0], args[1], args[2], args[3])
            }
            "ncfdtridfn" => {
                scalar_call!(special::probability::ncfdtridfn; args[0], args[1], args[2], args[3])
            }
            "ncfdtrinc" => {
                scalar_call!(special::probability::ncfdtrinc; args[0], args[1], args[2], args[3])
            }
            "nctdtr" => scalar_call!(special::probability::nctdtr; args[0], args[1], args[2]),
            "nctdtrit" => scalar_call!(special::probability::nctdtrit; args[0], args[1], args[2]),
            "nctdtrinc" => scalar_call!(special::probability::nctdtrinc; args[0], args[1], args[2]),
            "nctdtridf" => scalar_call!(special::probability::nctdtridf; args[0], args[1], args[2]),
            "jn_zeros" => parsed_call!(special::bessel::jn_zeros; args[0], args[1]).unwrap(),
            "jnp_zeros" => parsed_call!(special::bessel::jnp_zeros; args[0], args[1]).unwrap(),
            "yn_zeros" => parsed_call!(special::bessel::yn_zeros; args[0], args[1]).unwrap(),
            "ynp_zeros" => parsed_call!(special::bessel::ynp_zeros; args[0], args[1]).unwrap(),
            "jnyn_zeros" => parsed_call!(special::bessel::jnyn_zeros; args[0], args[1])
                .unwrap()
                .into_iter()
                .flatten()
                .collect(),
            "y0_zeros" => concat_pair!(parsed_call!(special::bessel::y0_zeros; args[0]).unwrap()),
            "y1_zeros" => concat_pair!(parsed_call!(special::bessel::y1_zeros; args[0]).unwrap()),
            "y1p_zeros" => concat_pair!(parsed_call!(special::bessel::y1p_zeros; args[0]).unwrap()),
            "ai_zeros" => parsed_call!(special::wave::ai_zeros; args[0])
                .unwrap()
                .into_iter()
                .flatten()
                .collect(),
            "bi_zeros" => parsed_call!(special::wave::bi_zeros; args[0])
                .unwrap()
                .into_iter()
                .flatten()
                .collect(),
            "erf_zeros" => parsed_call!(special::elementary::erf_zeros; args[0])
                .unwrap()
                .into_iter()
                .flat_map(|z| [z.re, z.im])
                .collect(),
            "jnjnp_zeros" => parsed_call!(special::bessel::jnjnp_zeros; args[0])
                .unwrap()
                .into_iter()
                .flatten()
                .collect(),
            "ellip_harm" => {
                vec![parsed_call!(special::wave::ellip_harm; args[0], args[1], args[2], args[3], args[4], args[5], args[6]).unwrap()]
            }
            "ellip_harm_2" => {
                vec![parsed_call!(special::wave::ellip_harm_2; args[0], args[1], args[2], args[3], args[4]).unwrap()]
            }
            "ellip_normal" => {
                vec![
                    parsed_call!(special::wave::ellip_normal; args[0], args[1], args[2], args[3])
                        .unwrap(),
                ]
            }
            "gamma" => scalar_call!(special::gamma::gamma; args[0]),
            "gammaln" => scalar_call!(special::gamma::gammaln; args[0]),
            "digamma" => scalar_call!(special::gamma::digamma; args[0]),
            "erf" => scalar_call!(special::elementary::erf; args[0]),
            "erfc" => scalar_call!(special::elementary::erfc; args[0]),
            "erfinv" => scalar_call!(special::elementary::erfinv; args[0]),
            "beta" => scalar_call!(special::beta::beta; args[0], args[1]),
            "betaln" => scalar_call!(special::beta::betaln; args[0], args[1]),
            "betainc" => scalar_call!(special::beta::betainc; args[0], args[1], args[2]),
            "betaincinv" => scalar_call!(special::beta::betaincinv; args[0], args[1], args[2]),
            "gammainc" => scalar_call!(special::gamma::gammainc; args[0], args[1]),
            "lambertw" => scalar_call!(special::elementary::lambertw; args[0]),
            "ellipk" => vec![parsed_call!(special::elliptic::ellipk; args[0]).unwrap()],
            "ellipe" => vec![parsed_call!(special::elliptic::ellipe; args[0]).unwrap()],
            "ellipeinc" => {
                vec![parsed_call!(special::elliptic::ellipeinc; args[0], args[1]).unwrap()]
            }
            "elliprf" => {
                vec![parsed_call!(special::elliptic::elliprf; args[0], args[1], args[2]).unwrap()]
            }
            "elliprd" => {
                vec![parsed_call!(special::elliptic::elliprd; args[0], args[1], args[2]).unwrap()]
            }
            "elliprg" => {
                vec![parsed_call!(special::elliptic::elliprg; args[0], args[1], args[2]).unwrap()]
            }
            "elliprj" => {
                vec![
                    parsed_call!(special::elliptic::elliprj; args[0], args[1], args[2], args[3])
                        .unwrap(),
                ]
            }
            "elliprc" => vec![parsed_call!(special::elliptic::elliprc; args[0], args[1]).unwrap()],
            "ellipkinc" => {
                vec![parsed_call!(special::elliptic::ellipkinc; args[0], args[1]).unwrap()]
            }
            "ellipkm1" => vec![parsed_call!(special::elliptic::ellipkm1; args[0]).unwrap()],
            "trigamma" => scalar_call!(special::gamma::trigamma; args[0]),
            "lambertwm1" => scalar_call!(special::elementary::lambertwm1; args[0]),
            "ellipke" => {
                let (a, b) = parsed_call!(special::elliptic::ellipke; args[0]).unwrap();
                vec![a, b]
            }
            "ellipf" => vec![parsed_call!(special::elliptic::ellipf; args[0], args[1]).unwrap()],
            "ellippi" => vec![parsed_call!(special::elliptic::ellippi; args[0], args[1]).unwrap()],
            "ellippiinc" => {
                vec![
                    parsed_call!(special::elliptic::ellippiinc; args[0], args[1], args[2]).unwrap(),
                ]
            }
            "ellipd" => vec![parsed_call!(special::elliptic::ellipd; args[0]).unwrap()],
            "ellipdinc" => {
                vec![parsed_call!(special::elliptic::ellipdinc; args[0], args[1]).unwrap()]
            }
            "cel" => vec![
                parsed_call!(special::elliptic::cel; args[0], args[1], args[2], args[3]).unwrap(),
            ],
            "cel1" => vec![parsed_call!(special::elliptic::cel1; args[0]).unwrap()],
            "cel2" => {
                vec![parsed_call!(special::elliptic::cel2; args[0], args[1], args[2]).unwrap()]
            }
            "cel3" => vec![parsed_call!(special::elliptic::cel3; args[0], args[1]).unwrap()],
            "el1" => vec![parsed_call!(special::elliptic::el1; args[0], args[1]).unwrap()],
            "el2" => vec![
                parsed_call!(special::elliptic::el2; args[0], args[1], args[2], args[3]).unwrap(),
            ],
            "el3" => vec![parsed_call!(special::elliptic::el3; args[0], args[1], args[2]).unwrap()],
            "jacobi_zeta" => {
                vec![parsed_call!(special::elliptic::jacobi_zeta; args[0], args[1]).unwrap()]
            }
            "heuman_lambda" => {
                vec![parsed_call!(special::elliptic::heuman_lambda; args[0], args[1]).unwrap()]
            }
            "ellippiinc_bulirsch" => {
                vec![
                    parsed_call!(special::elliptic::ellippiinc_bulirsch; args[0], args[1], args[2])
                        .unwrap(),
                ]
            }
            name => panic!("missing Rust dispatch for {name}"),
        };
        assert_eq!(actual.len(), expected.len(), "{} shape", fields[0]);
        for (i, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            if expected.is_nan() {
                assert!(
                    actual.is_nan(),
                    "{}({})[{i}]: {actual} != NaN",
                    fields[0],
                    fields[1]
                );
            } else if expected.is_infinite() {
                assert_eq!(actual, expected, "{}({})[{i}]", fields[0], fields[1]);
            } else {
                let rtol: f64 = fields[3].parse().unwrap();
                let atol: f64 = fields[4].parse().unwrap();
                assert!(
                    approx::abs_diff_eq!(
                        actual,
                        expected,
                        epsilon = atol.max(rtol * expected.abs())
                    ),
                    "{}({})[{i}]: {actual} != {expected}",
                    fields[0],
                    fields[1]
                );
            }
        }
    }
}
