# `optimize` coverage and SciPy adaptations

Inventory: SciPy 1.18.1, `scipy-func-list/scipy.optimize.md`. Seven of the 57
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

## Verification

Input-only `tests/scipy/optimize_cases.jsonl` drives the pinned SciPy 1.18.1 /
NumPy 2.5.4 generator. Separate checked-in CSVs support ordinary Rust tests.
Analytical comparisons use 3e-12 absolute plus relative tolerance; forward
differences use 3e-9 because subtraction amplifies last-bit evaluation
roundoff by 1/epsilon (tested steps are at least 2^-20). No tolerance is a
general derivative-accuracy guarantee. Independent tests check analytical
vs central derivatives, Hessian symmetry, dense vs tridiagonal products,
minimizer interoperability, callback counts, validation, and work limits.

```console
uv run tests/scipy/generate_optimize.py
python tests/scipy/audit_optimize.py
cargo fmt --all -- --check
cargo test
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --no-deps
```

## Remaining main namespace names

`anderson`, `basinhopping`, `bisect`, `bracket`, `brent`, `brenth`, `brentq`, `broyden1`, `broyden2`, `brute`, `curve_fit`, `diagbroyden`, `differential_evolution`, `direct`, `dual_annealing`, `excitingmixing`, `fixed_point`, `fmin`, `fmin_bfgs`, `fmin_cg`, `fmin_cobyla`, `fmin_l_bfgs_b`, `fmin_ncg`, `fmin_powell`, `fmin_slsqp`, `fmin_tnc`, `fminbound`, `fsolve`, `golden`, `isotonic_regression`, `least_squares`, `leastsq`, `line_search`, `linear_sum_assignment`, `linearmixing`, `linprog`, `linprog_verbose_callback`, `lsq_linear`, `milp`, `minimize_scalar`, `newton`, `newton_krylov`, `nnls`, `quadratic_assignment`, `ridder`, `root`, `root_scalar`, `shgo`, `show_options`, `toms748`.

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
