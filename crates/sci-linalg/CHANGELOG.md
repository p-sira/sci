# Changelog

## 0.2.0

Changes since the repository tag `v0.1.0`.

- Require `sci-core 0.2` and update Sci dependency requirements to matching releases. Shared error and numerical types are incompatible with those from `sci-core 0.1`.
- Refine dense `solve` results with compensated product residuals and up to four updates that reuse the LU factors. Preserve explicit singular-system errors and finite-result checks.

## 0.1.0

- Initial release of dense linear algebra, factorizations, matrix functions, and input adapters.
