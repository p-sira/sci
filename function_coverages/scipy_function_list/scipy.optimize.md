# scipy.optimize

[Back to SciPy function index](../SCIPY_FUNCTIONS.md)

## Namespaces

- [`scipy.optimize`](#scipy-optimize)
- [`scipy.optimize.cobyla`](#scipy-optimize-cobyla) (additional / compatibility)
- [`scipy.optimize.cython_optimize`](#scipy-optimize-cython_optimize)
- [`scipy.optimize.elementwise`](#scipy-optimize-elementwise)
- [`scipy.optimize.lbfgsb`](#scipy-optimize-lbfgsb) (additional / compatibility)
- [`scipy.optimize.linesearch`](#scipy-optimize-linesearch) (additional / compatibility)
- [`scipy.optimize.minpack`](#scipy-optimize-minpack) (additional / compatibility)
- [`scipy.optimize.minpack2`](#scipy-optimize-minpack2) (additional / compatibility)
- [`scipy.optimize.moduleTNC`](#scipy-optimize-moduletnc) (additional / compatibility)
- [`scipy.optimize.nonlin`](#scipy-optimize-nonlin) (additional / compatibility)
- [`scipy.optimize.optimize`](#scipy-optimize-optimize) (additional / compatibility)
- [`scipy.optimize.slsqp`](#scipy-optimize-slsqp) (additional / compatibility)
- [`scipy.optimize.tnc`](#scipy-optimize-tnc) (additional / compatibility)
- [`scipy.optimize.zeros`](#scipy-optimize-zeros) (additional / compatibility)

## Public API namespaces

### scipy.optimize

[Module reference](https://docs.scipy.org/doc/scipy/reference/optimize.html#module-scipy.optimize)

Function entries: **57**. Callable objects: **0**. Classes: **14**.

#### Functions

- [`anderson`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.anderson.html#scipy.optimize.anderson)
- [`approx_fprime`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.approx_fprime.html#scipy.optimize.approx_fprime)
- [`basinhopping`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html#scipy.optimize.basinhopping)
- [`bisect`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.bisect.html#scipy.optimize.bisect)
- [`bracket`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.bracket.html#scipy.optimize.bracket)
- [`brent`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brent.html#scipy.optimize.brent)
- [`brenth`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brenth.html#scipy.optimize.brenth)
- [`brentq`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html#scipy.optimize.brentq)
- [`broyden1`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.broyden1.html#scipy.optimize.broyden1)
- [`broyden2`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.broyden2.html#scipy.optimize.broyden2)
- [`brute`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brute.html#scipy.optimize.brute)
- [`check_grad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.check_grad.html#scipy.optimize.check_grad)
- [`curve_fit`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html#scipy.optimize.curve_fit)
- [`diagbroyden`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.diagbroyden.html#scipy.optimize.diagbroyden)
- [`differential_evolution`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html#scipy.optimize.differential_evolution)
- [`direct`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.direct.html#scipy.optimize.direct)
- [`dual_annealing`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html#scipy.optimize.dual_annealing)
- [`excitingmixing`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.excitingmixing.html#scipy.optimize.excitingmixing)
- [`fixed_point`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fixed_point.html#scipy.optimize.fixed_point)
- [`fmin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html#scipy.optimize.fmin)
- [`fmin_bfgs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_bfgs.html#scipy.optimize.fmin_bfgs)
- [`fmin_cg`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_cg.html#scipy.optimize.fmin_cg)
- [`fmin_cobyla`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_cobyla.html#scipy.optimize.fmin_cobyla)
- [`fmin_l_bfgs_b`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html#scipy.optimize.fmin_l_bfgs_b)
- [`fmin_ncg`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_ncg.html#scipy.optimize.fmin_ncg)
- [`fmin_powell`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_powell.html#scipy.optimize.fmin_powell)
- [`fmin_slsqp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_slsqp.html#scipy.optimize.fmin_slsqp)
- [`fmin_tnc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_tnc.html#scipy.optimize.fmin_tnc)
- [`fminbound`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fminbound.html#scipy.optimize.fminbound)
- [`fsolve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html#scipy.optimize.fsolve)
- [`golden`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.golden.html#scipy.optimize.golden)
- [`isotonic_regression`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.isotonic_regression.html#scipy.optimize.isotonic_regression)
- [`least_squares`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares)
- [`leastsq`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html#scipy.optimize.leastsq)
- [`line_search`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.line_search.html#scipy.optimize.line_search)
- [`linear_sum_assignment`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html#scipy.optimize.linear_sum_assignment)
- [`linearmixing`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linearmixing.html#scipy.optimize.linearmixing)
- [`linprog`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html#scipy.optimize.linprog)
- [`linprog_verbose_callback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog_verbose_callback.html#scipy.optimize.linprog_verbose_callback)
- [`lsq_linear`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html#scipy.optimize.lsq_linear)
- [`milp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html#scipy.optimize.milp)
- [`minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize)
- [`minimize_scalar`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html#scipy.optimize.minimize_scalar)
- [`newton`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html#scipy.optimize.newton)
- [`newton_krylov`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton_krylov.html#scipy.optimize.newton_krylov)
- [`nnls`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html#scipy.optimize.nnls)
- [`quadratic_assignment`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.quadratic_assignment.html#scipy.optimize.quadratic_assignment)
- [`ridder`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.ridder.html#scipy.optimize.ridder)
- [`root`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html#scipy.optimize.root)
- [`root_scalar`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root_scalar.html#scipy.optimize.root_scalar)
- [`rosen`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.rosen.html#scipy.optimize.rosen)
- [`rosen_der`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.rosen_der.html#scipy.optimize.rosen_der)
- [`rosen_hess`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.rosen_hess.html#scipy.optimize.rosen_hess)
- [`rosen_hess_prod`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.rosen_hess_prod.html#scipy.optimize.rosen_hess_prod)
- [`shgo`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.shgo.html#scipy.optimize.shgo)
- [`show_options`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.show_options.html#scipy.optimize.show_options)
- [`toms748`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.toms748.html#scipy.optimize.toms748)

#### Classes and their public methods

- [`BFGS`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.BFGS.html#scipy.optimize.bfgs)
  - [`BFGS.dot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.BFGS.dot.html#scipy.optimize.bfgs.dot)
  - [`BFGS.get_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.BFGS.get_matrix.html#scipy.optimize.bfgs.get_matrix)
  - [`BFGS.initialize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.BFGS.initialize.html#scipy.optimize.bfgs.initialize)
  - [`BFGS.update`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.BFGS.update.html#scipy.optimize.bfgs.update)
- [`Bounds`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.Bounds.html#scipy.optimize.bounds)
  - [`Bounds.residual`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.Bounds.residual.html#scipy.optimize.bounds.residual)
- [`BroydenFirst`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.BroydenFirst.html#scipy.optimize.broydenfirst)
  - [`BroydenFirst.aspreconditioner`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.BroydenFirst.aspreconditioner.html#scipy.optimize.broydenfirst.aspreconditioner)
  - [`BroydenFirst.matvec`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.BroydenFirst.matvec.html#scipy.optimize.broydenfirst.matvec)
  - [`BroydenFirst.rmatvec`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.BroydenFirst.rmatvec.html#scipy.optimize.broydenfirst.rmatvec)
  - [`BroydenFirst.rsolve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.BroydenFirst.rsolve.html#scipy.optimize.broydenfirst.rsolve)
  - [`BroydenFirst.setup`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.BroydenFirst.setup.html#scipy.optimize.broydenfirst.setup)
  - [`BroydenFirst.solve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.BroydenFirst.solve.html#scipy.optimize.broydenfirst.solve)
  - [`BroydenFirst.todense`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.BroydenFirst.todense.html#scipy.optimize.broydenfirst.todense)
  - [`BroydenFirst.update`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.BroydenFirst.update.html#scipy.optimize.broydenfirst.update)
- [`HessianUpdateStrategy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.HessianUpdateStrategy.html#scipy.optimize.hessianupdatestrategy)
  - [`HessianUpdateStrategy.dot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.HessianUpdateStrategy.dot.html#scipy.optimize.hessianupdatestrategy.dot)
  - [`HessianUpdateStrategy.get_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.HessianUpdateStrategy.get_matrix.html#scipy.optimize.hessianupdatestrategy.get_matrix)
  - [`HessianUpdateStrategy.initialize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.HessianUpdateStrategy.initialize.html#scipy.optimize.hessianupdatestrategy.initialize)
  - [`HessianUpdateStrategy.update`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.HessianUpdateStrategy.update.html#scipy.optimize.hessianupdatestrategy.update)
- [`InverseJacobian`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.InverseJacobian.html#scipy.optimize.inversejacobian)
- [`KrylovJacobian`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.KrylovJacobian.html#scipy.optimize.krylovjacobian)
  - [`KrylovJacobian.aspreconditioner`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.KrylovJacobian.aspreconditioner.html#scipy.optimize.krylovjacobian.aspreconditioner)
  - [`KrylovJacobian.matvec`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.KrylovJacobian.matvec.html#scipy.optimize.krylovjacobian.matvec)
  - [`KrylovJacobian.setup`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.KrylovJacobian.setup.html#scipy.optimize.krylovjacobian.setup)
  - [`KrylovJacobian.solve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.KrylovJacobian.solve.html#scipy.optimize.krylovjacobian.solve)
  - [`KrylovJacobian.update`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.KrylovJacobian.update.html#scipy.optimize.krylovjacobian.update)
- [`LbfgsInvHessProduct`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.LbfgsInvHessProduct.html#scipy.optimize.lbfgsinvhessproduct)
  - [`LbfgsInvHessProduct.adjoint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.LbfgsInvHessProduct.adjoint.html#scipy.optimize.lbfgsinvhessproduct.adjoint)
  - [`LbfgsInvHessProduct.dot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.LbfgsInvHessProduct.dot.html#scipy.optimize.lbfgsinvhessproduct.dot)
  - [`LbfgsInvHessProduct.matmat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.LbfgsInvHessProduct.matmat.html#scipy.optimize.lbfgsinvhessproduct.matmat)
  - [`LbfgsInvHessProduct.matvec`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.LbfgsInvHessProduct.matvec.html#scipy.optimize.lbfgsinvhessproduct.matvec)
  - [`LbfgsInvHessProduct.rdot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.LbfgsInvHessProduct.rdot.html#scipy.optimize.lbfgsinvhessproduct.rdot)
  - [`LbfgsInvHessProduct.rmatmat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.LbfgsInvHessProduct.rmatmat.html#scipy.optimize.lbfgsinvhessproduct.rmatmat)
  - [`LbfgsInvHessProduct.rmatvec`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.LbfgsInvHessProduct.rmatvec.html#scipy.optimize.lbfgsinvhessproduct.rmatvec)
  - [`LbfgsInvHessProduct.todense`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.LbfgsInvHessProduct.todense.html#scipy.optimize.lbfgsinvhessproduct.todense)
  - [`LbfgsInvHessProduct.transpose`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.LbfgsInvHessProduct.transpose.html#scipy.optimize.lbfgsinvhessproduct.transpose)
- [`LinearConstraint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.LinearConstraint.html#scipy.optimize.linearconstraint)
  - [`LinearConstraint.residual`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.LinearConstraint.residual.html#scipy.optimize.linearconstraint.residual)
- [`NoConvergence`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.NoConvergence.html#scipy.optimize.noconvergence)
  - [`NoConvergence.add_note`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.NoConvergence.html#scipy.optimize.noconvergence.add_note)
  - [`NoConvergence.with_traceback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.NoConvergence.html#scipy.optimize.noconvergence.with_traceback)
- [`NonlinearConstraint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.NonlinearConstraint.html#scipy.optimize.nonlinearconstraint)
- [`OptimizeResult`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.optimizeresult)
  - [`OptimizeResult.clear`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.clear.html#scipy.optimize.optimizeresult.clear)
  - [`OptimizeResult.copy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.copy.html#scipy.optimize.optimizeresult.copy)
  - [`OptimizeResult.fromkeys`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.fromkeys.html#scipy.optimize.optimizeresult.fromkeys)
  - [`OptimizeResult.get`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.get.html#scipy.optimize.optimizeresult.get)
  - [`OptimizeResult.items`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.items.html#scipy.optimize.optimizeresult.items)
  - [`OptimizeResult.keys`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.keys.html#scipy.optimize.optimizeresult.keys)
  - [`OptimizeResult.pop`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.pop.html#scipy.optimize.optimizeresult.pop)
  - [`OptimizeResult.popitem`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.popitem.html#scipy.optimize.optimizeresult.popitem)
  - [`OptimizeResult.setdefault`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.setdefault.html#scipy.optimize.optimizeresult.setdefault)
  - [`OptimizeResult.update`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.update.html#scipy.optimize.optimizeresult.update)
  - [`OptimizeResult.values`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.values.html#scipy.optimize.optimizeresult.values)
- [`OptimizeWarning`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeWarning.html#scipy.optimize.optimizewarning)
  - [`OptimizeWarning.add_note`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeWarning.html#scipy.optimize.optimizewarning.add_note)
  - [`OptimizeWarning.with_traceback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeWarning.html#scipy.optimize.optimizewarning.with_traceback)
- [`RootResults`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.RootResults.html#scipy.optimize.rootresults)
  - [`RootResults.clear`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.RootResults.clear.html#scipy.optimize.rootresults.clear)
  - [`RootResults.copy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.RootResults.copy.html#scipy.optimize.rootresults.copy)
  - [`RootResults.fromkeys`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.RootResults.fromkeys.html#scipy.optimize.rootresults.fromkeys)
  - [`RootResults.get`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.RootResults.get.html#scipy.optimize.rootresults.get)
  - [`RootResults.items`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.RootResults.items.html#scipy.optimize.rootresults.items)
  - [`RootResults.keys`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.RootResults.keys.html#scipy.optimize.rootresults.keys)
  - [`RootResults.pop`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.RootResults.pop.html#scipy.optimize.rootresults.pop)
  - [`RootResults.popitem`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.RootResults.popitem.html#scipy.optimize.rootresults.popitem)
  - [`RootResults.setdefault`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.RootResults.setdefault.html#scipy.optimize.rootresults.setdefault)
  - [`RootResults.update`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.RootResults.update.html#scipy.optimize.rootresults.update)
  - [`RootResults.values`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.RootResults.values.html#scipy.optimize.rootresults.values)
- [`SR1`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.SR1.html#scipy.optimize.sr1)
  - [`SR1.dot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.SR1.dot.html#scipy.optimize.sr1.dot)
  - [`SR1.get_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.SR1.get_matrix.html#scipy.optimize.sr1.get_matrix)
  - [`SR1.initialize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.SR1.initialize.html#scipy.optimize.sr1.initialize)
  - [`SR1.update`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.SR1.update.html#scipy.optimize.sr1.update)

### scipy.optimize.cython_optimize

[Module reference](https://docs.scipy.org/doc/scipy/reference/optimize.cython_optimize.html#module-scipy.optimize.cython_optimize)

Function entries: **4**. Callable objects: **0**. Classes: **0**.

#### Cython-only functions

These functions are declared in the Cython interface and are accessed with `cimport`.

- `bisect`
- `brenth`
- `brentq`
- `ridder`

### scipy.optimize.elementwise

[Module reference](https://docs.scipy.org/doc/scipy/reference/optimize.elementwise.html#module-scipy.optimize.elementwise)

Function entries: **4**. Callable objects: **0**. Classes: **0**.

#### Functions

- [`bracket_minimum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.elementwise.bracket_minimum.html#scipy.optimize.elementwise.bracket_minimum)
- [`bracket_root`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.elementwise.bracket_root.html#scipy.optimize.elementwise.bracket_root)
- [`find_minimum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.elementwise.find_minimum.html#scipy.optimize.elementwise.find_minimum)
- [`find_root`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.elementwise.find_root.html#scipy.optimize.elementwise.find_root)

## Additional and compatibility namespaces

Many of these paths are deprecated compatibility aliases. Class and object method lists are omitted in these sections to avoid repeating the public API inventory.

### scipy.optimize.cobyla

Function entries: **1**. Callable objects: **0**. Classes: **1**.

#### Functions

- `fmin_cobyla`

#### Classes

- `OptimizeResult`

### scipy.optimize.lbfgsb

Function entries: **2**. Callable objects: **0**. Classes: **2**.

#### Functions

- `fmin_l_bfgs_b`
- `zeros`

#### Classes

- `LbfgsInvHessProduct`
- `OptimizeResult`

### scipy.optimize.linesearch

Function entries: **1**. Callable objects: **0**. Classes: **0**.

#### Functions

- `line_search`

### scipy.optimize.minpack

Function entries: **6**. Callable objects: **0**. Classes: **2**.

#### Functions

- `curve_fit`
- `fixed_point`
- `fsolve`
- `least_squares`
- `leastsq`
- `zeros`

#### Classes

- `OptimizeResult`
- `OptimizeWarning`

### scipy.optimize.minpack2

Function entries: **0**. Callable objects: **0**. Classes: **0**.

This namespace contains submodules or has no exported callable entries in this runtime.

### scipy.optimize.moduleTNC

Function entries: **0**. Callable objects: **0**. Classes: **0**.

This namespace contains submodules or has no exported callable entries in this runtime.

### scipy.optimize.nonlin

Function entries: **7**. Callable objects: **0**. Classes: **3**.

#### Functions

- `anderson`
- `broyden1`
- `broyden2`
- `diagbroyden`
- `excitingmixing`
- `linearmixing`
- `newton_krylov`

#### Classes

- `BroydenFirst`
- `InverseJacobian`
- `KrylovJacobian`

### scipy.optimize.optimize

Function entries: **19**. Callable objects: **0**. Classes: **2**.

#### Functions

- `approx_fprime`
- `bracket`
- `brent`
- `brute`
- `check_grad`
- `fmin`
- `fmin_bfgs`
- `fmin_cg`
- `fmin_ncg`
- `fmin_powell`
- `fminbound`
- `golden`
- `line_search`
- `rosen`
- `rosen_der`
- `rosen_hess`
- `rosen_hess_prod`
- `show_options`
- `zeros`

#### Classes

- `OptimizeResult`
- `OptimizeWarning`

### scipy.optimize.slsqp

Function entries: **2**. Callable objects: **0**. Classes: **1**.

#### Functions

- `fmin_slsqp`
- `slsqp`

#### Classes

- `OptimizeResult`

### scipy.optimize.tnc

Function entries: **2**. Callable objects: **0**. Classes: **1**.

#### Functions

- `fmin_tnc`
- `zeros`

#### Classes

- `OptimizeResult`

### scipy.optimize.zeros

Function entries: **6**. Callable objects: **0**. Classes: **1**.

#### Functions

- `bisect`
- `brenth`
- `brentq`
- `newton`
- `ridder`
- `toms748`

#### Classes

- `RootResults`
