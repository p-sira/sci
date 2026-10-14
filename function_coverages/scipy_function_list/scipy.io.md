# scipy.io

[Back to SciPy function index](../SCIPY_FUNCTIONS.md)

## Namespaces

- [`scipy.io`](#scipy-io)
- [`scipy.io.arff`](#scipy-io-arff)
- [`scipy.io.arff.arffread`](#scipy-io-arff-arffread) (additional / compatibility)
- [`scipy.io.harwell_boeing`](#scipy-io-harwell_boeing) (additional / compatibility)
- [`scipy.io.idl`](#scipy-io-idl) (additional / compatibility)
- [`scipy.io.matlab`](#scipy-io-matlab)
- [`scipy.io.matlab.byteordercodes`](#scipy-io-matlab-byteordercodes) (additional / compatibility)
- [`scipy.io.matlab.mio`](#scipy-io-matlab-mio) (additional / compatibility)
- [`scipy.io.matlab.mio4`](#scipy-io-matlab-mio4) (additional / compatibility)
- [`scipy.io.matlab.mio5`](#scipy-io-matlab-mio5) (additional / compatibility)
- [`scipy.io.matlab.mio5_params`](#scipy-io-matlab-mio5_params) (additional / compatibility)
- [`scipy.io.matlab.mio5_utils`](#scipy-io-matlab-mio5_utils) (additional / compatibility)
- [`scipy.io.matlab.mio_utils`](#scipy-io-matlab-mio_utils) (additional / compatibility)
- [`scipy.io.matlab.miobase`](#scipy-io-matlab-miobase) (additional / compatibility)
- [`scipy.io.matlab.streams`](#scipy-io-matlab-streams) (additional / compatibility)
- [`scipy.io.mmio`](#scipy-io-mmio) (additional / compatibility)
- [`scipy.io.netcdf`](#scipy-io-netcdf) (additional / compatibility)
- [`scipy.io.wavfile`](#scipy-io-wavfile)

## Public API namespaces

### scipy.io

[Module reference](https://docs.scipy.org/doc/scipy/reference/io.html#module-scipy.io)

Function entries: **9**. Callable objects: **0**. Classes: **5**.

#### Functions

- [`hb_read`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.hb_read.html#scipy.io.hb_read)
- [`hb_write`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.hb_write.html#scipy.io.hb_write)
- [`loadmat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html#scipy.io.loadmat)
- [`mminfo`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.mminfo.html#scipy.io.mminfo)
- [`mmread`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.mmread.html#scipy.io.mmread)
- [`mmwrite`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.mmwrite.html#scipy.io.mmwrite)
- [`readsav`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.readsav.html#scipy.io.readsav)
- [`savemat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.savemat.html#scipy.io.savemat)
- [`whosmat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.whosmat.html#scipy.io.whosmat)

#### Classes and their public methods

- [`FortranEOFError`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.FortranEOFError.html#scipy.io.fortraneoferror)
  - [`FortranEOFError.add_note`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.FortranEOFError.html#scipy.io.fortraneoferror.add_note)
  - [`FortranEOFError.with_traceback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.FortranEOFError.html#scipy.io.fortraneoferror.with_traceback)
- [`FortranFile`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.FortranFile.html#scipy.io.fortranfile)
  - [`FortranFile.close`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.FortranFile.close.html#scipy.io.fortranfile.close)
  - [`FortranFile.read_ints`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.FortranFile.read_ints.html#scipy.io.fortranfile.read_ints)
  - [`FortranFile.read_reals`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.FortranFile.read_reals.html#scipy.io.fortranfile.read_reals)
  - [`FortranFile.read_record`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.FortranFile.read_record.html#scipy.io.fortranfile.read_record)
  - [`FortranFile.write_record`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.FortranFile.write_record.html#scipy.io.fortranfile.write_record)
- [`FortranFormattingError`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.FortranFormattingError.html#scipy.io.fortranformattingerror)
  - [`FortranFormattingError.add_note`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.FortranFormattingError.html#scipy.io.fortranformattingerror.add_note)
  - [`FortranFormattingError.with_traceback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.FortranFormattingError.html#scipy.io.fortranformattingerror.with_traceback)
- [`netcdf_file`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.netcdf_file.html#scipy.io.netcdf_file)
  - [`netcdf_file.close`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.netcdf_file.close.html#scipy.io.netcdf_file.close)
  - [`netcdf_file.createDimension`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.netcdf_file.createDimension.html#scipy.io.netcdf_file.createdimension)
  - [`netcdf_file.createVariable`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.netcdf_file.createVariable.html#scipy.io.netcdf_file.createvariable)
  - [`netcdf_file.flush`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.netcdf_file.flush.html#scipy.io.netcdf_file.flush)
  - [`netcdf_file.sync`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.netcdf_file.sync.html#scipy.io.netcdf_file.sync)
- [`netcdf_variable`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.netcdf_variable.html#scipy.io.netcdf_variable)
  - [`netcdf_variable.assignValue`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.netcdf_variable.assignValue.html#scipy.io.netcdf_variable.assignvalue)
  - [`netcdf_variable.getValue`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.netcdf_variable.getValue.html#scipy.io.netcdf_variable.getvalue)
  - [`netcdf_variable.itemsize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.netcdf_variable.itemsize.html#scipy.io.netcdf_variable.itemsize)
  - [`netcdf_variable.typecode`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.netcdf_variable.typecode.html#scipy.io.netcdf_variable.typecode)

### scipy.io.arff

[Module reference](https://docs.scipy.org/doc/scipy/reference/io.html#module-scipy.io.arff)

Function entries: **1**. Callable objects: **0**. Classes: **3**.

#### Functions

- [`loadarff`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.arff.loadarff.html#scipy.io.arff.loadarff)

#### Classes and their public methods

- [`ArffError`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.arff.ArffError.html#scipy.io.arff.arfferror)
  - [`ArffError.add_note`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.arff.ArffError.html#scipy.io.arff.arfferror.add_note)
  - [`ArffError.with_traceback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.arff.ArffError.html#scipy.io.arff.arfferror.with_traceback)
- [`MetaData`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.arff.MetaData.html#scipy.io.arff.metadata)
  - [`MetaData.names`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.arff.MetaData.names.html#scipy.io.arff.metadata.names)
  - [`MetaData.types`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.arff.MetaData.types.html#scipy.io.arff.metadata.types)
- [`ParseArffError`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.arff.ParseArffError.html#scipy.io.arff.parsearfferror)
  - [`ParseArffError.add_note`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.arff.ParseArffError.html#scipy.io.arff.parsearfferror.add_note)
  - [`ParseArffError.with_traceback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.arff.ParseArffError.html#scipy.io.arff.parsearfferror.with_traceback)

### scipy.io.matlab

[Module reference](https://docs.scipy.org/doc/scipy/reference/io.matlab.html#module-scipy.io.matlab)

Function entries: **5**. Callable objects: **0**. Classes: **8**.

#### Functions

- [`loadmat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.matlab.loadmat.html#scipy.io.matlab.loadmat)
- [`matfile_version`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.matlab.matfile_version.html#scipy.io.matlab.matfile_version)
- [`savemat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.matlab.savemat.html#scipy.io.matlab.savemat)
- [`varmats_from_mat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.matlab.varmats_from_mat.html#scipy.io.matlab.varmats_from_mat)
- [`whosmat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.matlab.whosmat.html#scipy.io.matlab.whosmat)

#### Classes and their public methods

- [`MatReadError`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.matlab.MatReadError.html#scipy.io.matlab.matreaderror)
  - [`MatReadError.add_note`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.matlab.MatReadError.html#scipy.io.matlab.matreaderror.add_note)
  - [`MatReadError.with_traceback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.matlab.MatReadError.html#scipy.io.matlab.matreaderror.with_traceback)
- [`MatReadWarning`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.matlab.MatReadWarning.html#scipy.io.matlab.matreadwarning)
  - [`MatReadWarning.add_note`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.matlab.MatReadWarning.html#scipy.io.matlab.matreadwarning.add_note)
  - [`MatReadWarning.with_traceback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.matlab.MatReadWarning.html#scipy.io.matlab.matreadwarning.with_traceback)
- [`MatWriteError`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.matlab.MatWriteError.html#scipy.io.matlab.matwriteerror)
  - [`MatWriteError.add_note`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.matlab.MatWriteError.html#scipy.io.matlab.matwriteerror.add_note)
  - [`MatWriteError.with_traceback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.matlab.MatWriteError.html#scipy.io.matlab.matwriteerror.with_traceback)
- [`MatWriteWarning`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.matlab.MatWriteWarning.html#scipy.io.matlab.matwritewarning)
  - [`MatWriteWarning.add_note`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.matlab.MatWriteWarning.html#scipy.io.matlab.matwritewarning.add_note)
  - [`MatWriteWarning.with_traceback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.matlab.MatWriteWarning.html#scipy.io.matlab.matwritewarning.with_traceback)
- [`MatlabFunction`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.matlab.MatlabFunction.html#scipy.io.matlab.matlabfunction)
  - `MatlabFunction.all`
  - `MatlabFunction.any`
  - `MatlabFunction.argmax`
  - `MatlabFunction.argmin`
  - `MatlabFunction.argpartition`
  - `MatlabFunction.argsort`
  - `MatlabFunction.astype`
  - `MatlabFunction.byteswap`
  - `MatlabFunction.choose`
  - `MatlabFunction.clip`
  - `MatlabFunction.compress`
  - `MatlabFunction.conj`
  - `MatlabFunction.conjugate`
  - `MatlabFunction.copy`
  - `MatlabFunction.cumprod`
  - `MatlabFunction.cumsum`
  - `MatlabFunction.diagonal`
  - `MatlabFunction.dot`
  - `MatlabFunction.dump`
  - `MatlabFunction.dumps`
  - `MatlabFunction.fill`
  - `MatlabFunction.flatten`
  - `MatlabFunction.getfield`
  - `MatlabFunction.item`
  - `MatlabFunction.max`
  - `MatlabFunction.mean`
  - `MatlabFunction.min`
  - `MatlabFunction.nonzero`
  - `MatlabFunction.partition`
  - `MatlabFunction.prod`
  - `MatlabFunction.put`
  - `MatlabFunction.ravel`
  - `MatlabFunction.repeat`
  - `MatlabFunction.reshape`
  - `MatlabFunction.resize`
  - `MatlabFunction.round`
  - `MatlabFunction.searchsorted`
  - `MatlabFunction.setfield`
  - `MatlabFunction.setflags`
  - `MatlabFunction.sort`
  - `MatlabFunction.squeeze`
  - `MatlabFunction.std`
  - `MatlabFunction.sum`
  - `MatlabFunction.swapaxes`
  - `MatlabFunction.take`
  - `MatlabFunction.to_device`
  - `MatlabFunction.tobytes`
  - `MatlabFunction.tofile`
  - `MatlabFunction.tolist`
  - `MatlabFunction.trace`
  - `MatlabFunction.transpose`
  - `MatlabFunction.var`
  - `MatlabFunction.view`
- [`MatlabObject`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.matlab.MatlabObject.html#scipy.io.matlab.matlabobject)
  - `MatlabObject.all`
  - `MatlabObject.any`
  - `MatlabObject.argmax`
  - `MatlabObject.argmin`
  - `MatlabObject.argpartition`
  - `MatlabObject.argsort`
  - `MatlabObject.astype`
  - `MatlabObject.byteswap`
  - `MatlabObject.choose`
  - `MatlabObject.clip`
  - `MatlabObject.compress`
  - `MatlabObject.conj`
  - `MatlabObject.conjugate`
  - `MatlabObject.copy`
  - `MatlabObject.cumprod`
  - `MatlabObject.cumsum`
  - `MatlabObject.diagonal`
  - `MatlabObject.dot`
  - `MatlabObject.dump`
  - `MatlabObject.dumps`
  - `MatlabObject.fill`
  - `MatlabObject.flatten`
  - `MatlabObject.getfield`
  - `MatlabObject.item`
  - `MatlabObject.max`
  - `MatlabObject.mean`
  - `MatlabObject.min`
  - `MatlabObject.nonzero`
  - `MatlabObject.partition`
  - `MatlabObject.prod`
  - `MatlabObject.put`
  - `MatlabObject.ravel`
  - `MatlabObject.repeat`
  - `MatlabObject.reshape`
  - `MatlabObject.resize`
  - `MatlabObject.round`
  - `MatlabObject.searchsorted`
  - `MatlabObject.setfield`
  - `MatlabObject.setflags`
  - `MatlabObject.sort`
  - `MatlabObject.squeeze`
  - `MatlabObject.std`
  - `MatlabObject.sum`
  - `MatlabObject.swapaxes`
  - `MatlabObject.take`
  - `MatlabObject.to_device`
  - `MatlabObject.tobytes`
  - `MatlabObject.tofile`
  - `MatlabObject.tolist`
  - `MatlabObject.trace`
  - `MatlabObject.transpose`
  - `MatlabObject.var`
  - `MatlabObject.view`
- [`MatlabOpaque`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.matlab.MatlabOpaque.html#scipy.io.matlab.matlabopaque)
  - `MatlabOpaque.all`
  - `MatlabOpaque.any`
  - `MatlabOpaque.argmax`
  - `MatlabOpaque.argmin`
  - `MatlabOpaque.argpartition`
  - `MatlabOpaque.argsort`
  - `MatlabOpaque.astype`
  - `MatlabOpaque.byteswap`
  - `MatlabOpaque.choose`
  - `MatlabOpaque.clip`
  - `MatlabOpaque.compress`
  - `MatlabOpaque.conj`
  - `MatlabOpaque.conjugate`
  - `MatlabOpaque.copy`
  - `MatlabOpaque.cumprod`
  - `MatlabOpaque.cumsum`
  - `MatlabOpaque.diagonal`
  - `MatlabOpaque.dot`
  - `MatlabOpaque.dump`
  - `MatlabOpaque.dumps`
  - `MatlabOpaque.fill`
  - `MatlabOpaque.flatten`
  - `MatlabOpaque.getfield`
  - `MatlabOpaque.item`
  - `MatlabOpaque.max`
  - `MatlabOpaque.mean`
  - `MatlabOpaque.min`
  - `MatlabOpaque.nonzero`
  - `MatlabOpaque.partition`
  - `MatlabOpaque.prod`
  - `MatlabOpaque.put`
  - `MatlabOpaque.ravel`
  - `MatlabOpaque.repeat`
  - `MatlabOpaque.reshape`
  - `MatlabOpaque.resize`
  - `MatlabOpaque.round`
  - `MatlabOpaque.searchsorted`
  - `MatlabOpaque.setfield`
  - `MatlabOpaque.setflags`
  - `MatlabOpaque.sort`
  - `MatlabOpaque.squeeze`
  - `MatlabOpaque.std`
  - `MatlabOpaque.sum`
  - `MatlabOpaque.swapaxes`
  - `MatlabOpaque.take`
  - `MatlabOpaque.to_device`
  - `MatlabOpaque.tobytes`
  - `MatlabOpaque.tofile`
  - `MatlabOpaque.tolist`
  - `MatlabOpaque.trace`
  - `MatlabOpaque.transpose`
  - `MatlabOpaque.var`
  - `MatlabOpaque.view`
- [`mat_struct`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.matlab.mat_struct.html#scipy.io.matlab.mat_struct)

### scipy.io.wavfile

[Module reference](https://docs.scipy.org/doc/scipy/reference/io.html#module-scipy.io.wavfile)

Function entries: **2**. Callable objects: **0**. Classes: **2**.

#### Functions

- [`read`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html#scipy.io.wavfile.read)
- [`write`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html#scipy.io.wavfile.write)

#### Classes and their public methods

- `WAVE_FORMAT`
  - `WAVE_FORMAT.as_integer_ratio`
  - `WAVE_FORMAT.bit_count`
  - `WAVE_FORMAT.bit_length`
  - `WAVE_FORMAT.conjugate`
  - `WAVE_FORMAT.from_bytes`
  - `WAVE_FORMAT.is_integer`
  - `WAVE_FORMAT.to_bytes`
- [`WavFileWarning`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.WavFileWarning.html#scipy.io.wavfile.wavfilewarning)
  - [`WavFileWarning.add_note`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.WavFileWarning.html#scipy.io.wavfile.wavfilewarning.add_note)
  - [`WavFileWarning.with_traceback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.WavFileWarning.html#scipy.io.wavfile.wavfilewarning.with_traceback)

## Additional and compatibility namespaces

Many of these paths are deprecated compatibility aliases. Class and object method lists are omitted in these sections to avoid repeating the public API inventory.

### scipy.io.arff.arffread

Function entries: **1**. Callable objects: **0**. Classes: **3**.

#### Functions

- `loadarff`

#### Classes

- `ArffError`
- `MetaData`
- `ParseArffError`

### scipy.io.harwell_boeing

Function entries: **2**. Callable objects: **0**. Classes: **0**.

#### Functions

- `hb_read`
- `hb_write`

### scipy.io.idl

Function entries: **1**. Callable objects: **0**. Classes: **0**.

#### Functions

- `readsav`

### scipy.io.matlab.byteordercodes

Function entries: **0**. Callable objects: **0**. Classes: **0**.

This namespace contains submodules or has no exported callable entries in this runtime.

### scipy.io.matlab.mio

Function entries: **3**. Callable objects: **0**. Classes: **0**.

#### Functions

- `loadmat`
- `savemat`
- `whosmat`

### scipy.io.matlab.mio4

Function entries: **0**. Callable objects: **0**. Classes: **0**.

This namespace contains submodules or has no exported callable entries in this runtime.

### scipy.io.matlab.mio5

Function entries: **1**. Callable objects: **0**. Classes: **6**.

#### Functions

- `varmats_from_mat`

#### Classes

- `MatReadError`
- `MatReadWarning`
- `MatWriteError`
- `MatlabFunction`
- `MatlabObject`
- `mat_struct`

### scipy.io.matlab.mio5_params

Function entries: **0**. Callable objects: **0**. Classes: **4**.

#### Classes

- `MatlabFunction`
- `MatlabObject`
- `MatlabOpaque`
- `mat_struct`

### scipy.io.matlab.mio5_utils

Function entries: **0**. Callable objects: **0**. Classes: **0**.

This namespace contains submodules or has no exported callable entries in this runtime.

### scipy.io.matlab.mio_utils

Function entries: **0**. Callable objects: **0**. Classes: **0**.

This namespace contains submodules or has no exported callable entries in this runtime.

### scipy.io.matlab.miobase

Function entries: **0**. Callable objects: **0**. Classes: **3**.

#### Classes

- `MatReadError`
- `MatReadWarning`
- `MatWriteError`

### scipy.io.matlab.streams

Function entries: **0**. Callable objects: **0**. Classes: **0**.

This namespace contains submodules or has no exported callable entries in this runtime.

### scipy.io.mmio

Function entries: **3**. Callable objects: **0**. Classes: **0**.

#### Functions

- `mminfo`
- `mmread`
- `mmwrite`

### scipy.io.netcdf

Function entries: **0**. Callable objects: **0**. Classes: **2**.

#### Classes

- `netcdf_file`
- `netcdf_variable`
