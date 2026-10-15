# Sci-Constants

Physical constants and unit conversions.

Part of [Sci](https://github.com/p-sira/sci), a unified scientific computing library for Rust.

Physical constants and unit conversions using a pinned CODATA 2022 snapshot.
Table entries include SI units and standard uncertainties. Historical lookup
keys are retained for reproducibility; values do not depend on a host
installation or runtime data file.

Constants are expressed in SI units unless a function documents another
unit. Conversion functions multiply by the exact scale represented by their
arguments and return `f64`; they do not attach or infer unit metadata.
