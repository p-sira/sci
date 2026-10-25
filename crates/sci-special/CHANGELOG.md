# Changelog

## 0.2.1

- Upgrade the private `special` backend to 0.14.2, including its updated Lambert W implementation. Disable its default features and select `std` and `lambert_w`; elliptic integrals continue to use `ellip` directly.
- Avoid principal Lambert W cancellation near zero with a bounded Taylor expansion, preserving signed zero and the relative accuracy of tiny inputs.

## 0.2.0

Changes since the repository tag `v0.1.0`.

- Require `sci-core 0.2` and update Sci dependency requirements to matching releases. Shared error and numerical types are incompatible with those from `sci-core 0.1`.
- Improve Student-t `stdtr` and `stdtrit` accuracy near the median and use stable elementary formulas for one and two degrees of freedom.
- Use the private XSF Box–Cox kernels to preserve finite results for tiny parameters and large powers, while explicitly retaining endpoint conventions.

## 0.1.0

- Initial release of special mathematical functions.
