# scipy.odr

[Back to SciPy function index](../SCIPY_FUNCTIONS.md)

## Namespaces

- [`scipy.odr`](#scipy-odr)
- [`scipy.odr.models`](#scipy-odr-models) (additional / compatibility)
- [`scipy.odr.odrpack`](#scipy-odr-odrpack) (additional / compatibility)

## Public API namespaces

### scipy.odr

[Module reference](https://docs.scipy.org/doc/scipy/reference/odr.html#module-scipy.odr)

Function entries: **2**. Callable objects: **0**. Classes: **8**.

#### Functions

- [`odr`](https://docs.scipy.org/doc/scipy/reference/generated/odr-function.html#scipy.odr.odr)
- [`polynomial`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.polynomial.html#scipy.odr.polynomial)

#### Classes and their public methods

- [`Data`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.Data.html#scipy.odr.data)
  - [`Data.set_meta`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.Data.set_meta.html#scipy.odr.data.set_meta)
- [`Model`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.Model.html#scipy.odr.model)
  - [`Model.set_meta`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.Model.set_meta.html#scipy.odr.model.set_meta)
- [`ODR`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.ODR.html#scipy.odr.odr)
  - [`ODR.restart`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.ODR.restart.html#scipy.odr.odr.restart)
  - [`ODR.run`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.ODR.run.html#scipy.odr.odr.run)
  - [`ODR.set_iprint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.ODR.set_iprint.html#scipy.odr.odr.set_iprint)
  - [`ODR.set_job`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.ODR.set_job.html#scipy.odr.odr.set_job)
- [`OdrError`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.OdrError.html#scipy.odr.odrerror)
  - [`OdrError.add_note`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.OdrError.html#scipy.odr.odrerror.add_note)
  - [`OdrError.with_traceback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.OdrError.html#scipy.odr.odrerror.with_traceback)
- [`OdrStop`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.OdrStop.html#scipy.odr.odrstop)
  - [`OdrStop.add_note`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.OdrStop.html#scipy.odr.odrstop.add_note)
  - [`OdrStop.with_traceback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.OdrStop.html#scipy.odr.odrstop.with_traceback)
- [`OdrWarning`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.OdrWarning.html#scipy.odr.odrwarning)
  - [`OdrWarning.add_note`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.OdrWarning.html#scipy.odr.odrwarning.add_note)
  - [`OdrWarning.with_traceback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.OdrWarning.html#scipy.odr.odrwarning.with_traceback)
- [`Output`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.Output.html#scipy.odr.output)
  - [`Output.pprint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.Output.pprint.html#scipy.odr.output.pprint)
- [`RealData`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.RealData.html#scipy.odr.realdata)
  - [`RealData.set_meta`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.RealData.set_meta.html#scipy.odr.realdata.set_meta)

#### Constants and other non-callable exports

- [`exponential`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.exponential.html#scipy.odr.exponential)
- [`multilinear`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.multilinear.html#scipy.odr.multilinear)
- [`quadratic`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.quadratic.html#scipy.odr.quadratic)
- [`unilinear`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.unilinear.html#scipy.odr.unilinear)

## Additional and compatibility namespaces

Many of these paths are deprecated compatibility aliases. Class and object method lists are omitted in these sections to avoid repeating the public API inventory.

### scipy.odr.models

Function entries: **1**. Callable objects: **0**. Classes: **1**.

#### Functions

- `polynomial`

#### Classes

- `Model`

#### Constants and other non-callable exports

- `exponential`
- `multilinear`
- `quadratic`
- `unilinear`

### scipy.odr.odrpack

Function entries: **1**. Callable objects: **0**. Classes: **10**.

#### Functions

- `odr`

#### Classes

- `Data`
- `Model`
- `ODR`
- `OdrError`
- `OdrStop`
- `OdrWarning`
- `Output`
- `RealData`
- `odr_error`
- `odr_stop`
