# Changelog

## 0.3.0

- Skip redundant corners of singleton axes in multidimensional interpolation and reuse Cartesian B-spline bases in surface evaluation; compute repeated spline derivatives incrementally.

- Solve well-conditioned B-spline interpolation systems with pivoted LU
  instead of an unconditional SVD, retaining pivot/residual checks and the
  SVD fallback for ill-conditioned systems. The native cubic interpolation
  spline benchmark improved approximately 12.5×.
- Make barycentric, Krogh, Lagrange, and PCHIP interpolation generic over `Float`.

- Remove splrep and splprep aliases; use make_splrep and make_splprep.
- Reuse spline knots and coefficient buffers during tensor-product surface evaluation.

## 0.2.0

Changes since the repository tag `v0.1.0`.

- Require `sci-core 0.2` and update Sci dependency requirements to matching releases. Shared error and numerical types are incompatible with those from `sci-core 0.1`.

## 0.1.0

- Initial release of interpolation and shared spline APIs.
