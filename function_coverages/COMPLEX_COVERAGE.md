# `complex` coverage and SciPy adaptations

The `SCIPY_FUNCTIONS.md` inventory and its top-level `scipy` inventory contain no
`scipy.complex` namespace. Elementary complex arithmetic belongs to NumPy;
SciPy special functions, FFTs, and matrix operations remain in their respective
Sci modules. There are no namespace aliases or Cython duplicates to add
for this module. This is a scalar API using the existing shared `Complex` type.

All 34 exported numerical functions have committed independent reference cases
and runnable documentation examples:

| Group | Public functions |
| --- | --- |
| Arithmetic | `add`, `subtract`, `multiply`, `divide`, `negate`, `reciprocal`, `conjugate` |
| Components and magnitude | `real`, `imag`, `abs`, `norm_sqr`, `arg` |
| Polar conversion | `to_polar`, `from_polar` |
| Exponential and roots | `exp`, `ln`, `sqrt`, `cbrt`, `powf`, `powc`, `powi`, `powu` |
| Trigonometric | `sin`, `cos`, `tan`, `asin`, `acos`, `atan` |
| Hyperbolic | `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` |

## Conventions and adaptations

- Cartesian values use `Complex { re, im }`; angles are radians. Polar output is
  `(radius, phase)`, with phase in `[-π, π]`. `from_polar` returns the shared
  `Result` and rejects negative/nonfinite radius and nonfinite phase.
- Elementary inverses, logarithm, square root, and noninteger powers use
  principal branches. Signed zero selects the branch-cut side.
- Scalar operations return IEEE values directly. Division by zero returns two
  NaNs; NumPy division can instead produce an infinite component. Reciprocal
  zero follows NumPy's two-NaN convention. Zero to zero is one; zero to positive
  real powers is zero, and negative powers of zero return NaNs.
- `powi` accepts all `i32` values and uses at most 32 iterations. Large integer
  powers of unit roots use exact cycle references: NumPy's general complex
  power loses about `3e-7` in the real component of `i^(2^31−1)`.
- Existing `Complex`, `add`, `multiply`, `conjugate`, `abs`, and `arg` remain
  source compatible. No backend type or new dependency is exposed.

## Numerical limits and unsupported behavior

Magnitude, division, logarithm, square root, and tangent scale inputs to avoid
common intermediate overflow/underflow. Square root computes the smaller
component by division, preserving values near the negative real axis. Logarithm
uses `log1p` near the unit circle, preserving small real parts. Exponential,
sine, cosine and their hyperbolic counterparts split large exponential factors
before multiplication, preserving representable components and signed zeros.
Tangent uses a scaled sum of squares near poles, avoiding cancellation in the
double-angle denominator, and retains the leading exponential coefficient at
the subnormal transition. Real powers scale overflowing magnitudes, reuse
square roots for half powers, and use a near-positive-axis limit to retain tiny
imaginary components. Complex powers with real exponents reuse this path.
Multiplication and squared magnitude combine binary-scaled products using
fused multiply-add residuals before restoring the exponent. This avoids
spurious NaN from overflowing products and premature underflow of a
representable sum of squares; it also retains small cancellation residuals.
True component overflow and true subnormal underflow remain. Finite inverse
functions share rationalized first-quadrant distances and
`log1p` formulas, with explicit branch-axis handling and scaled large-argument
limits. This avoids intermediate overflow and preserves tiny branch components.
Binary64 rounding and subnormal underflow remain; ill-conditioned branch points
do not carry a uniform relative accuracy guarantee.
Nonfinite inverse inputs and several nonfinite elementary combinations do not
promise NumPy's componentwise IEEE conventions. The module documentation
records these limitations. Correct rounding is not guaranteed.

Array broadcasting, NumPy's complete mathematical namespace, complex-input
special-function overloads, and arithmetic operator traits are outside this
module's scalar scope. No SciPy complex namespace entries remain unimplemented,
because no such namespace exists in the inventory; this is not a claim of
complete complex-input coverage across SciPy.

## Verification

`tests/scipy/generate_complex.py` pins NumPy 2.5.4 and SciPy 1.18.1 and regenerates
`complex_expected.csv` from the separate `complex_inputs.csv`. References come
from NumPy (extended-range division avoids complex128 intermediate overflow)
and documented polar/unit-root/square-root identities; extreme real powers use
extended precision to avoid complex128 logarithm rounding. References never
come from Sci or num-complex. Finite multiplication and squared magnitude use
exact Python rational arithmetic on binary64 inputs to independently resolve
cancellation below longdouble precision and overflowing/underflowing products. Rust tests exercise all 32 functions with 3,299 reference cases,
including extreme-scale inverses, points adjacent to branch endpoints,
near-unit logarithms, large exponentials, tangent pole neighbors and subnormal
transitions, extreme real powers and tiny axis components, branch cuts, signed
zero, nonfinite classifications, polar-domain validation, extreme scales, and integer exponent boundaries. The comparison
uses `3e-14` absolute and relative tolerances for moderate values and relative
comparisons for nonzero results below `1e-6`. Additional tests check inverse round
trips, arithmetic identities, polar ordering, and FFT Parseval interoperability.

Run the coverage audit with `python tests/scipy/audit_complex.py`.

References: [NumPy mathematical functions](https://numpy.org/doc/stable/reference/routines.math.html),
[NIST DLMF §4](https://dlmf.nist.gov/4), and
[num-complex 0.4.6](https://docs.rs/num-complex/0.4.6/num_complex/struct.Complex.html).

Unsigned powers and principal cube roots have additional independent fixtures
and property tests in `tests/dependency_extras.rs`; see the
[dependency audit](DEPENDENCY_EXTRAS_COVERAGE.md).
