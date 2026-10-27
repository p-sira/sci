# Changelog

## 0.3.0

- Make sampled integration and fixed quadrature generic over `Float`; round backend-generated Gaussian nodes and weights to the selected precision.

- Update shared Sci dependency requirements to the new minor versions.

## 0.2.0

Changes since the repository tag `v0.1.0`.

- Require `sci-core 0.2` and update Sci dependency requirements to matching releases. Shared error and numerical types are incompatible with those from `sci-core 0.1`.
- Use exact rational tabulations for uniform Newton–Cotes rules of orders 1 through 12, improving weights and error coefficients. Nonuniform rules retain the bounded Vandermonde solve.

## 0.1.0

- Initial release of numerical quadrature and sampled-data integration.
