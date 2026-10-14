# scipy.interpolate

[Back to SciPy function index](../SCIPY_FUNCTIONS.md)

## Namespaces

- [`scipy.interpolate`](#scipy-interpolate)
- [`scipy.interpolate.dfitpack`](#scipy-interpolate-dfitpack) (additional / compatibility)
- [`scipy.interpolate.fitpack`](#scipy-interpolate-fitpack) (additional / compatibility)
- [`scipy.interpolate.fitpack2`](#scipy-interpolate-fitpack2) (additional / compatibility)
- [`scipy.interpolate.interpnd`](#scipy-interpolate-interpnd) (additional / compatibility)
- [`scipy.interpolate.interpolate`](#scipy-interpolate-interpolate) (additional / compatibility)
- [`scipy.interpolate.ndgriddata`](#scipy-interpolate-ndgriddata) (additional / compatibility)
- [`scipy.interpolate.polyint`](#scipy-interpolate-polyint) (additional / compatibility)
- [`scipy.interpolate.rbf`](#scipy-interpolate-rbf) (additional / compatibility)

## Public API namespaces

### scipy.interpolate

[Module reference](https://docs.scipy.org/doc/scipy/reference/interpolate.html#module-scipy.interpolate)

Function entries: **25**. Callable objects: **0**. Classes: **31**.

#### Functions

- [`approximate_taylor_polynomial`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.approximate_taylor_polynomial.html#scipy.interpolate.approximate_taylor_polynomial)
- [`barycentric_interpolate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.barycentric_interpolate.html#scipy.interpolate.barycentric_interpolate)
- [`bisplev`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.bisplev.html#scipy.interpolate.bisplev)
- [`bisplrep`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.bisplrep.html#scipy.interpolate.bisplrep)
- [`generate_knots`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.generate_knots.html#scipy.interpolate.generate_knots)
- [`griddata`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html#scipy.interpolate.griddata)
- [`insert`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.insert.html#scipy.interpolate.insert)
- [`interpn`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html#scipy.interpolate.interpn)
- [`krogh_interpolate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.krogh_interpolate.html#scipy.interpolate.krogh_interpolate)
- [`lagrange`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.lagrange.html#scipy.interpolate.lagrange)
- [`make_interp_spline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_interp_spline.html#scipy.interpolate.make_interp_spline)
- [`make_lsq_spline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_lsq_spline.html#scipy.interpolate.make_lsq_spline)
- [`make_smoothing_spline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_smoothing_spline.html#scipy.interpolate.make_smoothing_spline)
- [`make_splprep`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_splprep.html#scipy.interpolate.make_splprep)
- [`make_splrep`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_splrep.html#scipy.interpolate.make_splrep)
- [`pade`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.pade.html#scipy.interpolate.pade)
- [`pchip_interpolate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.pchip_interpolate.html#scipy.interpolate.pchip_interpolate)
- [`spalde`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.spalde.html#scipy.interpolate.spalde)
- [`splantider`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splantider.html#scipy.interpolate.splantider)
- [`splder`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splder.html#scipy.interpolate.splder)
- [`splev`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splev.html#scipy.interpolate.splev)
- [`splint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splint.html#scipy.interpolate.splint)
- [`splprep`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html#scipy.interpolate.splprep)
- [`splrep`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splrep.html#scipy.interpolate.splrep)
- [`sproot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.sproot.html#scipy.interpolate.sproot)

#### Classes and their public methods

- [`AAA`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.AAA.html#scipy.interpolate.aaa)
  - [`AAA.clean_up`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.AAA.clean_up.html#scipy.interpolate.aaa.clean_up)
  - [`AAA.poles`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.AAA.poles.html#scipy.interpolate.aaa.poles)
  - [`AAA.residues`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.AAA.residues.html#scipy.interpolate.aaa.residues)
  - [`AAA.roots`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.AAA.roots.html#scipy.interpolate.aaa.roots)
- [`Akima1DInterpolator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.html#scipy.interpolate.akima1dinterpolator)
  - [`Akima1DInterpolator.antiderivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.antiderivative.html#scipy.interpolate.akima1dinterpolator.antiderivative)
  - [`Akima1DInterpolator.construct_fast`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.construct_fast.html#scipy.interpolate.akima1dinterpolator.construct_fast)
  - [`Akima1DInterpolator.derivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.derivative.html#scipy.interpolate.akima1dinterpolator.derivative)
  - [`Akima1DInterpolator.extend`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.extend.html#scipy.interpolate.akima1dinterpolator.extend)
  - [`Akima1DInterpolator.from_bernstein_basis`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.from_bernstein_basis.html#scipy.interpolate.akima1dinterpolator.from_bernstein_basis)
  - [`Akima1DInterpolator.from_spline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.from_spline.html#scipy.interpolate.akima1dinterpolator.from_spline)
  - [`Akima1DInterpolator.integrate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.integrate.html#scipy.interpolate.akima1dinterpolator.integrate)
  - [`Akima1DInterpolator.roots`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.roots.html#scipy.interpolate.akima1dinterpolator.roots)
  - [`Akima1DInterpolator.solve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.solve.html#scipy.interpolate.akima1dinterpolator.solve)
- [`BPoly`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BPoly.html#scipy.interpolate.bpoly)
  - [`BPoly.antiderivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BPoly.antiderivative.html#scipy.interpolate.bpoly.antiderivative)
  - [`BPoly.construct_fast`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BPoly.construct_fast.html#scipy.interpolate.bpoly.construct_fast)
  - [`BPoly.derivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BPoly.derivative.html#scipy.interpolate.bpoly.derivative)
  - [`BPoly.extend`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BPoly.extend.html#scipy.interpolate.bpoly.extend)
  - [`BPoly.from_derivatives`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BPoly.from_derivatives.html#scipy.interpolate.bpoly.from_derivatives)
  - [`BPoly.from_power_basis`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BPoly.from_power_basis.html#scipy.interpolate.bpoly.from_power_basis)
  - [`BPoly.integrate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BPoly.integrate.html#scipy.interpolate.bpoly.integrate)
- [`BSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html#scipy.interpolate.bspline)
  - [`BSpline.antiderivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.antiderivative.html#scipy.interpolate.bspline.antiderivative)
  - [`BSpline.basis_element`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.basis_element.html#scipy.interpolate.bspline.basis_element)
  - [`BSpline.construct_fast`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.construct_fast.html#scipy.interpolate.bspline.construct_fast)
  - [`BSpline.derivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.derivative.html#scipy.interpolate.bspline.derivative)
  - [`BSpline.design_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.design_matrix.html#scipy.interpolate.bspline.design_matrix)
  - [`BSpline.from_power_basis`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.from_power_basis.html#scipy.interpolate.bspline.from_power_basis)
  - [`BSpline.insert_knot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.insert_knot.html#scipy.interpolate.bspline.insert_knot)
  - [`BSpline.integrate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.integrate.html#scipy.interpolate.bspline.integrate)
- [`BarycentricInterpolator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BarycentricInterpolator.html#scipy.interpolate.barycentricinterpolator)
  - [`BarycentricInterpolator.add_xi`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BarycentricInterpolator.add_xi.html#scipy.interpolate.barycentricinterpolator.add_xi)
  - [`BarycentricInterpolator.derivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BarycentricInterpolator.derivative.html#scipy.interpolate.barycentricinterpolator.derivative)
  - [`BarycentricInterpolator.derivatives`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BarycentricInterpolator.derivatives.html#scipy.interpolate.barycentricinterpolator.derivatives)
  - [`BarycentricInterpolator.set_yi`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BarycentricInterpolator.set_yi.html#scipy.interpolate.barycentricinterpolator.set_yi)
- [`BivariateSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BivariateSpline.html#scipy.interpolate.bivariatespline)
  - [`BivariateSpline.ev`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BivariateSpline.ev.html#scipy.interpolate.bivariatespline.ev)
  - [`BivariateSpline.get_coeffs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BivariateSpline.get_coeffs.html#scipy.interpolate.bivariatespline.get_coeffs)
  - [`BivariateSpline.get_knots`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BivariateSpline.get_knots.html#scipy.interpolate.bivariatespline.get_knots)
  - [`BivariateSpline.get_residual`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BivariateSpline.get_residual.html#scipy.interpolate.bivariatespline.get_residual)
  - [`BivariateSpline.integral`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BivariateSpline.integral.html#scipy.interpolate.bivariatespline.integral)
  - [`BivariateSpline.partial_derivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BivariateSpline.partial_derivative.html#scipy.interpolate.bivariatespline.partial_derivative)
- [`CloughTocher2DInterpolator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CloughTocher2DInterpolator.html#scipy.interpolate.cloughtocher2dinterpolator)
- [`CubicHermiteSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicHermiteSpline.html#scipy.interpolate.cubichermitespline)
  - [`CubicHermiteSpline.antiderivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicHermiteSpline.antiderivative.html#scipy.interpolate.cubichermitespline.antiderivative)
  - [`CubicHermiteSpline.construct_fast`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicHermiteSpline.construct_fast.html#scipy.interpolate.cubichermitespline.construct_fast)
  - [`CubicHermiteSpline.derivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicHermiteSpline.derivative.html#scipy.interpolate.cubichermitespline.derivative)
  - [`CubicHermiteSpline.extend`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicHermiteSpline.extend.html#scipy.interpolate.cubichermitespline.extend)
  - [`CubicHermiteSpline.from_bernstein_basis`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicHermiteSpline.from_bernstein_basis.html#scipy.interpolate.cubichermitespline.from_bernstein_basis)
  - [`CubicHermiteSpline.from_spline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicHermiteSpline.from_spline.html#scipy.interpolate.cubichermitespline.from_spline)
  - [`CubicHermiteSpline.integrate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicHermiteSpline.integrate.html#scipy.interpolate.cubichermitespline.integrate)
  - [`CubicHermiteSpline.roots`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicHermiteSpline.roots.html#scipy.interpolate.cubichermitespline.roots)
  - [`CubicHermiteSpline.solve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicHermiteSpline.solve.html#scipy.interpolate.cubichermitespline.solve)
- [`CubicSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html#scipy.interpolate.cubicspline)
  - [`CubicSpline.antiderivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.antiderivative.html#scipy.interpolate.cubicspline.antiderivative)
  - [`CubicSpline.construct_fast`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.construct_fast.html#scipy.interpolate.cubicspline.construct_fast)
  - [`CubicSpline.derivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.derivative.html#scipy.interpolate.cubicspline.derivative)
  - [`CubicSpline.extend`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.extend.html#scipy.interpolate.cubicspline.extend)
  - [`CubicSpline.from_bernstein_basis`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.from_bernstein_basis.html#scipy.interpolate.cubicspline.from_bernstein_basis)
  - [`CubicSpline.from_spline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.from_spline.html#scipy.interpolate.cubicspline.from_spline)
  - [`CubicSpline.integrate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.integrate.html#scipy.interpolate.cubicspline.integrate)
  - [`CubicSpline.roots`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.roots.html#scipy.interpolate.cubicspline.roots)
  - [`CubicSpline.solve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.solve.html#scipy.interpolate.cubicspline.solve)
- [`FloaterHormannInterpolator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.FloaterHormannInterpolator.html#scipy.interpolate.floaterhormanninterpolator)
  - [`FloaterHormannInterpolator.poles`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.FloaterHormannInterpolator.poles.html#scipy.interpolate.floaterhormanninterpolator.poles)
  - [`FloaterHormannInterpolator.residues`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.FloaterHormannInterpolator.residues.html#scipy.interpolate.floaterhormanninterpolator.residues)
  - [`FloaterHormannInterpolator.roots`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.FloaterHormannInterpolator.roots.html#scipy.interpolate.floaterhormanninterpolator.roots)
- [`InterpolatedUnivariateSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.InterpolatedUnivariateSpline.html#scipy.interpolate.interpolatedunivariatespline)
  - [`InterpolatedUnivariateSpline.antiderivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.InterpolatedUnivariateSpline.antiderivative.html#scipy.interpolate.interpolatedunivariatespline.antiderivative)
  - [`InterpolatedUnivariateSpline.derivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.InterpolatedUnivariateSpline.derivative.html#scipy.interpolate.interpolatedunivariatespline.derivative)
  - [`InterpolatedUnivariateSpline.derivatives`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.InterpolatedUnivariateSpline.derivatives.html#scipy.interpolate.interpolatedunivariatespline.derivatives)
  - [`InterpolatedUnivariateSpline.get_coeffs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.InterpolatedUnivariateSpline.get_coeffs.html#scipy.interpolate.interpolatedunivariatespline.get_coeffs)
  - [`InterpolatedUnivariateSpline.get_knots`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.InterpolatedUnivariateSpline.get_knots.html#scipy.interpolate.interpolatedunivariatespline.get_knots)
  - [`InterpolatedUnivariateSpline.get_residual`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.InterpolatedUnivariateSpline.get_residual.html#scipy.interpolate.interpolatedunivariatespline.get_residual)
  - [`InterpolatedUnivariateSpline.integral`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.InterpolatedUnivariateSpline.integral.html#scipy.interpolate.interpolatedunivariatespline.integral)
  - [`InterpolatedUnivariateSpline.roots`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.InterpolatedUnivariateSpline.roots.html#scipy.interpolate.interpolatedunivariatespline.roots)
  - [`InterpolatedUnivariateSpline.set_smoothing_factor`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.InterpolatedUnivariateSpline.set_smoothing_factor.html#scipy.interpolate.interpolatedunivariatespline.set_smoothing_factor)
  - [`InterpolatedUnivariateSpline.validate_input`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.InterpolatedUnivariateSpline.validate_input.html#scipy.interpolate.interpolatedunivariatespline.validate_input)
- [`KroghInterpolator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.KroghInterpolator.html#scipy.interpolate.kroghinterpolator)
  - [`KroghInterpolator.derivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.KroghInterpolator.derivative.html#scipy.interpolate.kroghinterpolator.derivative)
  - [`KroghInterpolator.derivatives`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.KroghInterpolator.derivatives.html#scipy.interpolate.kroghinterpolator.derivatives)
- [`LSQBivariateSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQBivariateSpline.html#scipy.interpolate.lsqbivariatespline)
  - [`LSQBivariateSpline.ev`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQBivariateSpline.ev.html#scipy.interpolate.lsqbivariatespline.ev)
  - [`LSQBivariateSpline.get_coeffs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQBivariateSpline.get_coeffs.html#scipy.interpolate.lsqbivariatespline.get_coeffs)
  - [`LSQBivariateSpline.get_knots`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQBivariateSpline.get_knots.html#scipy.interpolate.lsqbivariatespline.get_knots)
  - [`LSQBivariateSpline.get_residual`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQBivariateSpline.get_residual.html#scipy.interpolate.lsqbivariatespline.get_residual)
  - [`LSQBivariateSpline.integral`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQBivariateSpline.integral.html#scipy.interpolate.lsqbivariatespline.integral)
  - [`LSQBivariateSpline.partial_derivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQBivariateSpline.partial_derivative.html#scipy.interpolate.lsqbivariatespline.partial_derivative)
- [`LSQSphereBivariateSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQSphereBivariateSpline.html#scipy.interpolate.lsqspherebivariatespline)
  - [`LSQSphereBivariateSpline.ev`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQSphereBivariateSpline.ev.html#scipy.interpolate.lsqspherebivariatespline.ev)
  - [`LSQSphereBivariateSpline.get_coeffs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQSphereBivariateSpline.get_coeffs.html#scipy.interpolate.lsqspherebivariatespline.get_coeffs)
  - [`LSQSphereBivariateSpline.get_knots`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQSphereBivariateSpline.get_knots.html#scipy.interpolate.lsqspherebivariatespline.get_knots)
  - [`LSQSphereBivariateSpline.get_residual`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQSphereBivariateSpline.get_residual.html#scipy.interpolate.lsqspherebivariatespline.get_residual)
  - [`LSQSphereBivariateSpline.partial_derivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQSphereBivariateSpline.partial_derivative.html#scipy.interpolate.lsqspherebivariatespline.partial_derivative)
- [`LSQUnivariateSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQUnivariateSpline.html#scipy.interpolate.lsqunivariatespline)
  - [`LSQUnivariateSpline.antiderivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQUnivariateSpline.antiderivative.html#scipy.interpolate.lsqunivariatespline.antiderivative)
  - [`LSQUnivariateSpline.derivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQUnivariateSpline.derivative.html#scipy.interpolate.lsqunivariatespline.derivative)
  - [`LSQUnivariateSpline.derivatives`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQUnivariateSpline.derivatives.html#scipy.interpolate.lsqunivariatespline.derivatives)
  - [`LSQUnivariateSpline.get_coeffs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQUnivariateSpline.get_coeffs.html#scipy.interpolate.lsqunivariatespline.get_coeffs)
  - [`LSQUnivariateSpline.get_knots`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQUnivariateSpline.get_knots.html#scipy.interpolate.lsqunivariatespline.get_knots)
  - [`LSQUnivariateSpline.get_residual`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQUnivariateSpline.get_residual.html#scipy.interpolate.lsqunivariatespline.get_residual)
  - [`LSQUnivariateSpline.integral`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQUnivariateSpline.integral.html#scipy.interpolate.lsqunivariatespline.integral)
  - [`LSQUnivariateSpline.roots`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQUnivariateSpline.roots.html#scipy.interpolate.lsqunivariatespline.roots)
  - [`LSQUnivariateSpline.set_smoothing_factor`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQUnivariateSpline.set_smoothing_factor.html#scipy.interpolate.lsqunivariatespline.set_smoothing_factor)
  - [`LSQUnivariateSpline.validate_input`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQUnivariateSpline.validate_input.html#scipy.interpolate.lsqunivariatespline.validate_input)
- [`LinearNDInterpolator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html#scipy.interpolate.linearndinterpolator)
- [`NdBSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.NdBSpline.html#scipy.interpolate.ndbspline)
  - [`NdBSpline.derivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.NdBSpline.derivative.html#scipy.interpolate.ndbspline.derivative)
  - [`NdBSpline.design_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.NdBSpline.design_matrix.html#scipy.interpolate.ndbspline.design_matrix)
- [`NdPPoly`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.NdPPoly.html#scipy.interpolate.ndppoly)
  - [`NdPPoly.antiderivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.NdPPoly.antiderivative.html#scipy.interpolate.ndppoly.antiderivative)
  - [`NdPPoly.construct_fast`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.NdPPoly.construct_fast.html#scipy.interpolate.ndppoly.construct_fast)
  - [`NdPPoly.derivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.NdPPoly.derivative.html#scipy.interpolate.ndppoly.derivative)
  - [`NdPPoly.integrate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.NdPPoly.integrate.html#scipy.interpolate.ndppoly.integrate)
  - [`NdPPoly.integrate_1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.NdPPoly.integrate_1d.html#scipy.interpolate.ndppoly.integrate_1d)
- [`NearestNDInterpolator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.NearestNDInterpolator.html#scipy.interpolate.nearestndinterpolator)
- [`PPoly`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PPoly.html#scipy.interpolate.ppoly)
  - [`PPoly.antiderivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PPoly.antiderivative.html#scipy.interpolate.ppoly.antiderivative)
  - [`PPoly.construct_fast`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PPoly.construct_fast.html#scipy.interpolate.ppoly.construct_fast)
  - [`PPoly.derivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PPoly.derivative.html#scipy.interpolate.ppoly.derivative)
  - [`PPoly.extend`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PPoly.extend.html#scipy.interpolate.ppoly.extend)
  - [`PPoly.from_bernstein_basis`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PPoly.from_bernstein_basis.html#scipy.interpolate.ppoly.from_bernstein_basis)
  - [`PPoly.from_spline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PPoly.from_spline.html#scipy.interpolate.ppoly.from_spline)
  - [`PPoly.integrate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PPoly.integrate.html#scipy.interpolate.ppoly.integrate)
  - [`PPoly.roots`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PPoly.roots.html#scipy.interpolate.ppoly.roots)
  - [`PPoly.solve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PPoly.solve.html#scipy.interpolate.ppoly.solve)
- [`PchipInterpolator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html#scipy.interpolate.pchipinterpolator)
  - [`PchipInterpolator.antiderivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.antiderivative.html#scipy.interpolate.pchipinterpolator.antiderivative)
  - [`PchipInterpolator.construct_fast`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.construct_fast.html#scipy.interpolate.pchipinterpolator.construct_fast)
  - [`PchipInterpolator.derivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.derivative.html#scipy.interpolate.pchipinterpolator.derivative)
  - [`PchipInterpolator.extend`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.extend.html#scipy.interpolate.pchipinterpolator.extend)
  - [`PchipInterpolator.from_bernstein_basis`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.from_bernstein_basis.html#scipy.interpolate.pchipinterpolator.from_bernstein_basis)
  - [`PchipInterpolator.from_spline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.from_spline.html#scipy.interpolate.pchipinterpolator.from_spline)
  - [`PchipInterpolator.integrate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.integrate.html#scipy.interpolate.pchipinterpolator.integrate)
  - [`PchipInterpolator.roots`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.roots.html#scipy.interpolate.pchipinterpolator.roots)
  - [`PchipInterpolator.solve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.solve.html#scipy.interpolate.pchipinterpolator.solve)
- [`RBFInterpolator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html#scipy.interpolate.rbfinterpolator)
- [`Rbf`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html#scipy.interpolate.rbf)
- [`RectBivariateSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectBivariateSpline.html#scipy.interpolate.rectbivariatespline)
  - [`RectBivariateSpline.ev`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectBivariateSpline.ev.html#scipy.interpolate.rectbivariatespline.ev)
  - [`RectBivariateSpline.get_coeffs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectBivariateSpline.get_coeffs.html#scipy.interpolate.rectbivariatespline.get_coeffs)
  - [`RectBivariateSpline.get_knots`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectBivariateSpline.get_knots.html#scipy.interpolate.rectbivariatespline.get_knots)
  - [`RectBivariateSpline.get_residual`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectBivariateSpline.get_residual.html#scipy.interpolate.rectbivariatespline.get_residual)
  - [`RectBivariateSpline.integral`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectBivariateSpline.integral.html#scipy.interpolate.rectbivariatespline.integral)
  - [`RectBivariateSpline.partial_derivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectBivariateSpline.partial_derivative.html#scipy.interpolate.rectbivariatespline.partial_derivative)
- [`RectSphereBivariateSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectSphereBivariateSpline.html#scipy.interpolate.rectspherebivariatespline)
  - [`RectSphereBivariateSpline.ev`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectSphereBivariateSpline.ev.html#scipy.interpolate.rectspherebivariatespline.ev)
  - [`RectSphereBivariateSpline.get_coeffs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectSphereBivariateSpline.get_coeffs.html#scipy.interpolate.rectspherebivariatespline.get_coeffs)
  - [`RectSphereBivariateSpline.get_knots`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectSphereBivariateSpline.get_knots.html#scipy.interpolate.rectspherebivariatespline.get_knots)
  - [`RectSphereBivariateSpline.get_residual`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectSphereBivariateSpline.get_residual.html#scipy.interpolate.rectspherebivariatespline.get_residual)
  - [`RectSphereBivariateSpline.partial_derivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectSphereBivariateSpline.partial_derivative.html#scipy.interpolate.rectspherebivariatespline.partial_derivative)
- [`RegularGridInterpolator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html#scipy.interpolate.regulargridinterpolator)
- [`SmoothBivariateSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.SmoothBivariateSpline.html#scipy.interpolate.smoothbivariatespline)
  - [`SmoothBivariateSpline.ev`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.SmoothBivariateSpline.ev.html#scipy.interpolate.smoothbivariatespline.ev)
  - [`SmoothBivariateSpline.get_coeffs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.SmoothBivariateSpline.get_coeffs.html#scipy.interpolate.smoothbivariatespline.get_coeffs)
  - [`SmoothBivariateSpline.get_knots`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.SmoothBivariateSpline.get_knots.html#scipy.interpolate.smoothbivariatespline.get_knots)
  - [`SmoothBivariateSpline.get_residual`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.SmoothBivariateSpline.get_residual.html#scipy.interpolate.smoothbivariatespline.get_residual)
  - [`SmoothBivariateSpline.integral`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.SmoothBivariateSpline.integral.html#scipy.interpolate.smoothbivariatespline.integral)
  - [`SmoothBivariateSpline.partial_derivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.SmoothBivariateSpline.partial_derivative.html#scipy.interpolate.smoothbivariatespline.partial_derivative)
- [`SmoothSphereBivariateSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.SmoothSphereBivariateSpline.html#scipy.interpolate.smoothspherebivariatespline)
  - [`SmoothSphereBivariateSpline.ev`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.SmoothSphereBivariateSpline.ev.html#scipy.interpolate.smoothspherebivariatespline.ev)
  - [`SmoothSphereBivariateSpline.get_coeffs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.SmoothSphereBivariateSpline.get_coeffs.html#scipy.interpolate.smoothspherebivariatespline.get_coeffs)
  - [`SmoothSphereBivariateSpline.get_knots`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.SmoothSphereBivariateSpline.get_knots.html#scipy.interpolate.smoothspherebivariatespline.get_knots)
  - [`SmoothSphereBivariateSpline.get_residual`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.SmoothSphereBivariateSpline.get_residual.html#scipy.interpolate.smoothspherebivariatespline.get_residual)
  - [`SmoothSphereBivariateSpline.partial_derivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.SmoothSphereBivariateSpline.partial_derivative.html#scipy.interpolate.smoothspherebivariatespline.partial_derivative)
- [`UnivariateSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html#scipy.interpolate.univariatespline)
  - [`UnivariateSpline.antiderivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.antiderivative.html#scipy.interpolate.univariatespline.antiderivative)
  - [`UnivariateSpline.derivative`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.derivative.html#scipy.interpolate.univariatespline.derivative)
  - [`UnivariateSpline.derivatives`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.derivatives.html#scipy.interpolate.univariatespline.derivatives)
  - [`UnivariateSpline.get_coeffs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.get_coeffs.html#scipy.interpolate.univariatespline.get_coeffs)
  - [`UnivariateSpline.get_knots`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.get_knots.html#scipy.interpolate.univariatespline.get_knots)
  - [`UnivariateSpline.get_residual`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.get_residual.html#scipy.interpolate.univariatespline.get_residual)
  - [`UnivariateSpline.integral`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.integral.html#scipy.interpolate.univariatespline.integral)
  - [`UnivariateSpline.roots`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.roots.html#scipy.interpolate.univariatespline.roots)
  - [`UnivariateSpline.set_smoothing_factor`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.set_smoothing_factor.html#scipy.interpolate.univariatespline.set_smoothing_factor)
  - [`UnivariateSpline.validate_input`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.validate_input.html#scipy.interpolate.univariatespline.validate_input)
- [`interp1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d)
- [`interp2d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp2d.html#scipy.interpolate.interp2d)

## Additional and compatibility namespaces

Many of these paths are deprecated compatibility aliases. Class and object method lists are omitted in these sections to avoid repeating the public API inventory.

### scipy.interpolate.dfitpack

Function entries: **5**. Callable objects: **0**. Classes: **0**.

#### Functions

- `spalde`
- `splder`
- `splev`
- `splint`
- `sproot`

### scipy.interpolate.fitpack

Function entries: **11**. Callable objects: **0**. Classes: **1**.

#### Functions

- `bisplev`
- `bisplrep`
- `insert`
- `spalde`
- `splantider`
- `splder`
- `splev`
- `splint`
- `splprep`
- `splrep`
- `sproot`

#### Classes

- `BSpline`

### scipy.interpolate.fitpack2

Function entries: **0**. Callable objects: **0**. Classes: **10**.

#### Classes

- `BivariateSpline`
- `InterpolatedUnivariateSpline`
- `LSQBivariateSpline`
- `LSQSphereBivariateSpline`
- `LSQUnivariateSpline`
- `RectBivariateSpline`
- `RectSphereBivariateSpline`
- `SmoothBivariateSpline`
- `SmoothSphereBivariateSpline`
- `UnivariateSpline`

### scipy.interpolate.interpnd

Function entries: **0**. Callable objects: **0**. Classes: **2**.

#### Classes

- `CloughTocher2DInterpolator`
- `LinearNDInterpolator`

### scipy.interpolate.interpolate

Function entries: **3**. Callable objects: **0**. Classes: **8**.

#### Functions

- `interpn`
- `lagrange`
- `make_interp_spline`

#### Classes

- `BPoly`
- `BSpline`
- `NdPPoly`
- `PPoly`
- `RectBivariateSpline`
- `RegularGridInterpolator`
- `interp1d`
- `interp2d`

### scipy.interpolate.ndgriddata

Function entries: **1**. Callable objects: **0**. Classes: **3**.

#### Functions

- `griddata`

#### Classes

- `CloughTocher2DInterpolator`
- `LinearNDInterpolator`
- `NearestNDInterpolator`

### scipy.interpolate.polyint

Function entries: **3**. Callable objects: **0**. Classes: **2**.

#### Functions

- `approximate_taylor_polynomial`
- `barycentric_interpolate`
- `krogh_interpolate`

#### Classes

- `BarycentricInterpolator`
- `KroghInterpolator`

### scipy.interpolate.rbf

Function entries: **0**. Callable objects: **0**. Classes: **1**.

#### Classes

- `Rbf`
