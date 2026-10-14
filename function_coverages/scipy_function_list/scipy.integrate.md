# scipy.integrate

[Back to SciPy function index](../SCIPY_FUNCTIONS.md)

## Namespaces

- [`scipy.integrate`](#scipy-integrate)
- [`scipy.integrate.dop`](#scipy-integrate-dop) (additional / compatibility)
- [`scipy.integrate.lsoda`](#scipy-integrate-lsoda) (additional / compatibility)
- [`scipy.integrate.odepack`](#scipy-integrate-odepack) (additional / compatibility)
- [`scipy.integrate.quadpack`](#scipy-integrate-quadpack) (additional / compatibility)
- [`scipy.integrate.vode`](#scipy-integrate-vode) (additional / compatibility)

## Public API namespaces

### scipy.integrate

[Module reference](https://docs.scipy.org/doc/scipy/reference/integrate.html#module-scipy.integrate)

Function entries: **20**. Callable objects: **0**. Classes: **13**.

#### Functions

- [`cubature`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cubature.html#scipy.integrate.cubature)
- [`cumulative_simpson`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cumulative_simpson.html#scipy.integrate.cumulative_simpson)
- [`cumulative_trapezoid`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cumulative_trapezoid.html#scipy.integrate.cumulative_trapezoid)
- [`dblquad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.dblquad.html#scipy.integrate.dblquad)
- [`fixed_quad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.fixed_quad.html#scipy.integrate.fixed_quad)
- [`lebedev_rule`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.lebedev_rule.html#scipy.integrate.lebedev_rule)
- [`newton_cotes`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.newton_cotes.html#scipy.integrate.newton_cotes)
- [`nquad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.nquad.html#scipy.integrate.nquad)
- [`nsum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.nsum.html#scipy.integrate.nsum)
- [`odeint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html#scipy.integrate.odeint)
- [`qmc_quad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.qmc_quad.html#scipy.integrate.qmc_quad)
- [`quad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html#scipy.integrate.quad)
- [`quad_vec`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad_vec.html#scipy.integrate.quad_vec)
- [`romb`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.romb.html#scipy.integrate.romb)
- [`simpson`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.simpson.html#scipy.integrate.simpson)
- [`solve_bvp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html#scipy.integrate.solve_bvp)
- [`solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp)
- [`tanhsinh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.tanhsinh.html#scipy.integrate.tanhsinh)
- [`tplquad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.tplquad.html#scipy.integrate.tplquad)
- [`trapezoid`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.trapezoid.html#scipy.integrate.trapezoid)

#### Classes and their public methods

- [`BDF`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.BDF.html#scipy.integrate.bdf)
  - [`BDF.dense_output`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.BDF.dense_output.html#scipy.integrate.bdf.dense_output)
  - [`BDF.step`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.BDF.step.html#scipy.integrate.bdf.step)
- [`DOP853`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.DOP853.html#scipy.integrate.dop853)
  - [`DOP853.dense_output`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.DOP853.dense_output.html#scipy.integrate.dop853.dense_output)
  - [`DOP853.step`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.DOP853.step.html#scipy.integrate.dop853.step)
- [`DenseOutput`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.DenseOutput.html#scipy.integrate.denseoutput)
- [`IntegrationWarning`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.IntegrationWarning.html#scipy.integrate.integrationwarning)
  - [`IntegrationWarning.add_note`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.IntegrationWarning.html#scipy.integrate.integrationwarning.add_note)
  - [`IntegrationWarning.with_traceback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.IntegrationWarning.html#scipy.integrate.integrationwarning.with_traceback)
- [`LSODA`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.LSODA.html#scipy.integrate.lsoda)
  - [`LSODA.dense_output`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.LSODA.dense_output.html#scipy.integrate.lsoda.dense_output)
  - [`LSODA.step`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.LSODA.step.html#scipy.integrate.lsoda.step)
- [`ODEintWarning`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ODEintWarning.html#scipy.integrate.odeintwarning)
  - [`ODEintWarning.add_note`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ODEintWarning.html#scipy.integrate.odeintwarning.add_note)
  - [`ODEintWarning.with_traceback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ODEintWarning.html#scipy.integrate.odeintwarning.with_traceback)
- [`OdeSolution`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.OdeSolution.html#scipy.integrate.odesolution)
- [`OdeSolver`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.OdeSolver.html#scipy.integrate.odesolver)
  - [`OdeSolver.dense_output`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.OdeSolver.dense_output.html#scipy.integrate.odesolver.dense_output)
  - [`OdeSolver.step`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.OdeSolver.step.html#scipy.integrate.odesolver.step)
- [`RK23`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK23.html#scipy.integrate.rk23)
  - [`RK23.dense_output`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK23.dense_output.html#scipy.integrate.rk23.dense_output)
  - [`RK23.step`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK23.step.html#scipy.integrate.rk23.step)
- [`RK45`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK45.html#scipy.integrate.rk45)
  - [`RK45.dense_output`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK45.dense_output.html#scipy.integrate.rk45.dense_output)
  - [`RK45.step`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK45.step.html#scipy.integrate.rk45.step)
- [`Radau`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.Radau.html#scipy.integrate.radau)
  - [`Radau.dense_output`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.Radau.dense_output.html#scipy.integrate.radau.dense_output)
  - [`Radau.step`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.Radau.step.html#scipy.integrate.radau.step)
- [`complex_ode`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.complex_ode.html#scipy.integrate.complex_ode)
  - [`complex_ode.get_return_code`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.complex_ode.get_return_code.html#scipy.integrate.complex_ode.get_return_code)
  - [`complex_ode.integrate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.complex_ode.integrate.html#scipy.integrate.complex_ode.integrate)
  - [`complex_ode.set_f_params`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.complex_ode.set_f_params.html#scipy.integrate.complex_ode.set_f_params)
  - [`complex_ode.set_initial_value`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.complex_ode.set_initial_value.html#scipy.integrate.complex_ode.set_initial_value)
  - [`complex_ode.set_integrator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.complex_ode.set_integrator.html#scipy.integrate.complex_ode.set_integrator)
  - [`complex_ode.set_jac_params`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.complex_ode.set_jac_params.html#scipy.integrate.complex_ode.set_jac_params)
  - [`complex_ode.set_solout`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.complex_ode.set_solout.html#scipy.integrate.complex_ode.set_solout)
  - [`complex_ode.successful`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.complex_ode.successful.html#scipy.integrate.complex_ode.successful)
- [`ode`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html#scipy.integrate.ode)
  - [`ode.get_return_code`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.get_return_code.html#scipy.integrate.ode.get_return_code)
  - [`ode.integrate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.integrate.html#scipy.integrate.ode.integrate)
  - [`ode.set_f_params`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.set_f_params.html#scipy.integrate.ode.set_f_params)
  - [`ode.set_initial_value`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.set_initial_value.html#scipy.integrate.ode.set_initial_value)
  - [`ode.set_integrator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.set_integrator.html#scipy.integrate.ode.set_integrator)
  - [`ode.set_jac_params`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.set_jac_params.html#scipy.integrate.ode.set_jac_params)
  - [`ode.set_solout`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.set_solout.html#scipy.integrate.ode.set_solout)
  - [`ode.successful`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.successful.html#scipy.integrate.ode.successful)

## Additional and compatibility namespaces

Many of these paths are deprecated compatibility aliases. Class and object method lists are omitted in these sections to avoid repeating the public API inventory.

### scipy.integrate.dop

Function entries: **0**. Callable objects: **0**. Classes: **0**.

This namespace contains submodules or has no exported callable entries in this runtime.

### scipy.integrate.lsoda

Function entries: **1**. Callable objects: **0**. Classes: **0**.

#### Functions

- `lsoda`

### scipy.integrate.odepack

Function entries: **1**. Callable objects: **0**. Classes: **1**.

#### Functions

- `odeint`

#### Classes

- `ODEintWarning`

### scipy.integrate.quadpack

Function entries: **4**. Callable objects: **0**. Classes: **1**.

#### Functions

- `dblquad`
- `nquad`
- `quad`
- `tplquad`

#### Classes

- `IntegrationWarning`

### scipy.integrate.vode

Function entries: **0**. Callable objects: **0**. Classes: **0**.

This namespace contains submodules or has no exported callable entries in this runtime.
