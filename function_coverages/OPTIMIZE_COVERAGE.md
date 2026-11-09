# `optimize` coverage and SciPy adaptations

Inventory: SciPy 1.18.1, `scipy_function_list/scipy.optimize.md`. Eight of the 57
main namespace names have tested supported native signatures. Compatibility
namespaces repeat these algorithms and introduce no separate coverage claim.
The four elementwise and four Cython-only root solvers remain absent.

## Supported signatures

The existing `minimize` adapts SciPy to explicit unconstrained Nelder–Mead
controls and a shared `Minimum` result. Its gradient-descent variant and
fallible callback variants are native extras, not additional SciPy names.

`rosen`, `rosen_der`, `rosen_hess`, and `rosen_hess_prod` implement analytical
Rosenbrock polynomial formulas. Inputs are one-dimensional shared f64 slices;
SciPy multidimensional broadcasting is absent. NaN/infinity and arithmetic
overflow return shared errors. `rosen` supports the empty sum for fewer than
two coordinates; derivatives require at least two. Objective, gradient, and
Hessian-vector product sizes are bounded at 1,000,000 coordinates. Dense
Hessians are bounded at 256 coordinates. Products reuse tridiagonal curvature
formulas in O(n) time, without dense allocation. All output ordering follows
coordinate ordering and interoperates with differentiation and linear algebra.

`approx_fprime(function, x, epsilon)` returns a shared m × n Jacobian.
Callbacks return vectors; scalar objectives use a one-element vector. Steps
may be a singleton slice for broadcasting or one per coordinate and may be
positive or negative. They must be finite, nonzero, and representable as a
coordinate increment. The denominator is the actual floating-point increment,
as in SciPy. Collapsed increments return an error instead of silently choosing
SciPy’s fallback step. Inputs and outputs are bounded at 512 dimensions each;
callback dimension changes and nonfinite evaluations return shared errors.
There are exactly n+1 evaluations; callback execution cost remains caller-controlled.
Captured closures replace Python positional args.

`check_grad` compares a scalar objective’s analytical gradient against forward
differences using an overflow-resistant Euclidean norm. It implements SciPy’s
all-coordinate direction. Random-direction/rng options are absent. The same
steps, work bounds, and finite-evaluation rules apply. Reported discrepancy
includes forward-difference truncation and cancellation error.

## Dogbox signatures and adaptations

`least_squares::dogbox` adapts only the dense, linear-loss dogbox method of
SciPy's `least_squares`; the compatibility `minpack.least_squares` repeats
that entry. There is no separate `dogbox` inventory name. The four public
functions cover numerical or analytical Jacobians and infallible or fallible
callbacks. Coordinates use shared `&[f64]` slices, residuals use vectors, and
analytical Jacobians use `sci_core::Matrix`, with rows in residual order and
columns in coordinate order. Results include cost, residuals, gradient,
projected optimality, active bounds and evaluation counts.

The private nalgebra SVD supplies minimum-norm Gauss–Newton solves; singular
values at or below machine epsilon times the largest are discarded. Native
code handles the rectangular dogleg, constrained Cauchy fallback and trust
region. Bound intersections are snapped exactly before evaluation. Numerical
differences prefer steps away from zero and adjust to feasible coordinates.
Nonfinite trial values shrink the region; initial and differentiation values
must be finite. Callback errors propagate unchanged. SVD convergence failures
use `NumericalFailure`; nonfinite arithmetic uses `NonFiniteEvaluation`.

Bounds must be strictly ordered; fixed variables require caller elimination.
Dogbox admits `m*n*(max_iterations+1) <= 16,777,216`, excluding callback work.
This dimension-based limit is not an operation count: each SVD costs roughly
`O(m*n*min(m,n))`, with a finite iteration budget. Zero attempts still evaluates
initial residuals and the Jacobian. Unlike SciPy's `max_nfev`, attempts count
rejected steps, and the residual evaluation count includes differentiation.
Finite positive tolerances are mandatory. Robust losses, variable scaling,
sparse/iterative solves, TRF and Levenberg–Marquardt remain unsupported.

## Dogbox quality review against sci-special

The baseline is sci-special's family namespaces, shared types/errors, private
kernels, explicit mathematical contracts and independent numerical evidence.

| Area | Assessment |
| --- | --- |
| Public API | Matches the functional family namespace and shared containers/errors. All four variants reuse one solver. Binary64 is explicit, as in sci-special's sequence APIs; generic scalar precision is not implemented. |
| Backend boundary | nalgebra types stay private. SVD is delegated to an established dependency; bounds and stopping logic remain local. Callers need no backend traits. |
| Contracts and validation | Definitions, units, shapes, special values, stopping rules, work limits and SciPy adaptations are documented. Callback errors are preserved, with explicit shape and finite checks. |
| Numerical evidence | Pinned SciPy references exercise both Jacobian paths. Independent checks cover stationarity, exact bounds, callback counts, shared Matrix use, invalid inputs and work limits. |

Remaining limits: the solver recomputes the same SVD after rejected trials;
caching its Gauss–Newton solution and Cauchy coefficients could reduce retry
cost. The fixed difference interval can exceed small parameter scales, so a
step/cost stop does not establish a small true gradient; analytical Jacobians
are the supported remedy. There is no broad randomized or high-precision
accuracy study comparable to sci-special's wider numerical audit. These are
scope/performance limits rather than additional SciPy coverage claims.

## Verification

Input-only `tests/scipy/optimize_cases.jsonl` drives the pinned SciPy 1.18.1 /
NumPy 2.5.4 generator. Separate checked-in CSVs support ordinary Rust tests.
Analytical comparisons use 3e-12 absolute plus relative tolerance; forward
differences use 3e-9 because subtraction amplifies last-bit evaluation
roundoff by 1/epsilon (tested steps are at least 2^-20). No tolerance is a
general derivative-accuracy guarantee. Independent tests check analytical
vs central derivatives, Hessian symmetry, dense vs tridiagonal products,
minimizer interoperability, callback counts, validation, and work limits.

Dogbox uses a separate pinned generator and 16 input/reference pairs, each
checked with numerical and analytical Jacobians. Regressions include symmetric
finite-difference cancellation and rounded upper/lower bound intersections.
The scaled-square case additionally requires cost below `1e-14` and relative
root error below `1e-7`; its raw gradient is compared with SciPy's attainable
value because the default difference interval exceeds the root scale. The
unscaled square can stop on cost decrease near a flat region, but must take
a step rather than falsely claim initial gradient stationarity.

```console
uv run tests/scipy/generate_optimize.py
uv run tests/scipy/generate_dogbox.py
python tests/scipy/audit_optimize.py
cargo fmt --all -- --check
cargo test
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --no-deps
```

## Remaining main namespace names

`anderson`, `basinhopping`, `bisect`, `bracket`, `brent`, `brenth`, `brentq`, `broyden1`, `broyden2`, `brute`, `curve_fit`, `diagbroyden`, `differential_evolution`, `direct`, `dual_annealing`, `excitingmixing`, `fixed_point`, `fmin`, `fmin_bfgs`, `fmin_cg`, `fmin_cobyla`, `fmin_l_bfgs_b`, `fmin_ncg`, `fmin_powell`, `fmin_slsqp`, `fmin_tnc`, `fminbound`, `fsolve`, `golden`, `isotonic_regression`, `leastsq`, `line_search`, `linear_sum_assignment`, `linearmixing`, `linprog`, `linprog_verbose_callback`, `lsq_linear`, `milp`, `minimize_scalar`, `newton`, `newton_krylov`, `nnls`, `quadratic_assignment`, `ridder`, `root`, `root_scalar`, `shgo`, `show_options`, `toms748`.

## Minimizer storage and work bounds

Minimizers allow at most 1,000,000 coordinates. Nelder–Mead additionally bounds
its n*(n+1) simplex coordinates at 1,000,000 before initializing the backend.
Coordinate-iteration budgets, including initialization, are at most 16,777,216:
(max_iterations+1)*n*(n+1) for Nelder–Mead and (max_iterations+1)*n for gradient
descent. Overflowing budgets fail before callbacks. Callback execution time
remains controlled by the caller; a zero iteration budget still initializes.

Gradient convergence uses hypot accumulation without squaring the norm or
tolerance, preventing false convergence through overflow or underflow. Native
quadratic regression fixtures verify the update identity, status and iteration
count across extreme objective scales, without additional callback evaluations.
