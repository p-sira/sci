# Changelog

## 0.2.1

- Inline private scaling and interval arithmetic helpers.

## 0.2.0

Changes since the repository tag `v0.1.0`.

- Require `sci-core 0.2` and update Sci dependency requirements to matching releases. Shared error and numerical types are incompatible with those from `sci-core 0.1`.
- Use exact rational tabulations for uniform Newton–Cotes rules of orders 1 through 12, improving weights and error coefficients. Nonuniform rules retain the bounded Vandermonde solve.

## 0.1.0

- Initial release of numerical quadrature and sampled-data integration.
