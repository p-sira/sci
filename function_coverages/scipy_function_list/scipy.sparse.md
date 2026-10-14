# scipy.sparse

[Back to SciPy function index](../SCIPY_FUNCTIONS.md)

## Namespaces

- [`scipy.sparse`](#scipy-sparse)
- [`scipy.sparse.base`](#scipy-sparse-base) (additional / compatibility)
- [`scipy.sparse.bsr`](#scipy-sparse-bsr) (additional / compatibility)
- [`scipy.sparse.compressed`](#scipy-sparse-compressed) (additional / compatibility)
- [`scipy.sparse.construct`](#scipy-sparse-construct) (additional / compatibility)
- [`scipy.sparse.coo`](#scipy-sparse-coo) (additional / compatibility)
- [`scipy.sparse.csc`](#scipy-sparse-csc) (additional / compatibility)
- [`scipy.sparse.csgraph`](#scipy-sparse-csgraph)
- [`scipy.sparse.csr`](#scipy-sparse-csr) (additional / compatibility)
- [`scipy.sparse.data`](#scipy-sparse-data) (additional / compatibility)
- [`scipy.sparse.dia`](#scipy-sparse-dia) (additional / compatibility)
- [`scipy.sparse.dok`](#scipy-sparse-dok) (additional / compatibility)
- [`scipy.sparse.extract`](#scipy-sparse-extract) (additional / compatibility)
- [`scipy.sparse.lil`](#scipy-sparse-lil) (additional / compatibility)
- [`scipy.sparse.linalg`](#scipy-sparse-linalg)
- [`scipy.sparse.linalg.dsolve`](#scipy-sparse-linalg-dsolve) (additional / compatibility)
- [`scipy.sparse.linalg.eigen`](#scipy-sparse-linalg-eigen) (additional / compatibility)
- [`scipy.sparse.linalg.interface`](#scipy-sparse-linalg-interface) (additional / compatibility)
- [`scipy.sparse.linalg.isolve`](#scipy-sparse-linalg-isolve) (additional / compatibility)
- [`scipy.sparse.linalg.matfuncs`](#scipy-sparse-linalg-matfuncs) (additional / compatibility)
- [`scipy.sparse.sparsetools`](#scipy-sparse-sparsetools) (additional / compatibility)
- [`scipy.sparse.spfuncs`](#scipy-sparse-spfuncs) (additional / compatibility)
- [`scipy.sparse.sputils`](#scipy-sparse-sputils) (additional / compatibility)

## Public API namespaces

### scipy.sparse

[Module reference](https://docs.scipy.org/doc/scipy/reference/sparse.html#module-scipy.sparse)

Function entries: **36**. Callable objects: **0**. Classes: **18**.

#### Functions

- [`block_array`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.block_array.html#scipy.sparse.block_array)
- [`block_diag`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.block_diag.html#scipy.sparse.block_diag)
- [`bmat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bmat.html#scipy.sparse.bmat)
- [`diags`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.diags.html#scipy.sparse.diags)
- [`diags_array`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.diags_array.html#scipy.sparse.diags_array)
- [`expand_dims`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.expand_dims.html#scipy.sparse.expand_dims)
- [`eye`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.eye.html#scipy.sparse.eye)
- [`eye_array`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.eye_array.html#scipy.sparse.eye_array)
- [`find`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.find.html#scipy.sparse.find)
- [`get_index_dtype`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.get_index_dtype.html#scipy.sparse.get_index_dtype)
- [`hstack`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.hstack.html#scipy.sparse.hstack)
- [`identity`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.identity.html#scipy.sparse.identity)
- [`issparse`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.issparse.html#scipy.sparse.issparse)
- [`isspmatrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.isspmatrix.html#scipy.sparse.isspmatrix)
- [`isspmatrix_bsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.isspmatrix_bsr.html#scipy.sparse.isspmatrix_bsr)
- [`isspmatrix_coo`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.isspmatrix_coo.html#scipy.sparse.isspmatrix_coo)
- [`isspmatrix_csc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.isspmatrix_csc.html#scipy.sparse.isspmatrix_csc)
- [`isspmatrix_csr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.isspmatrix_csr.html#scipy.sparse.isspmatrix_csr)
- [`isspmatrix_dia`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.isspmatrix_dia.html#scipy.sparse.isspmatrix_dia)
- [`isspmatrix_dok`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.isspmatrix_dok.html#scipy.sparse.isspmatrix_dok)
- [`isspmatrix_lil`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.isspmatrix_lil.html#scipy.sparse.isspmatrix_lil)
- [`kron`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.kron.html#scipy.sparse.kron)
- [`kronsum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.kronsum.html#scipy.sparse.kronsum)
- [`load_npz`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.load_npz.html#scipy.sparse.load_npz)
- [`matrix_transpose`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.matrix_transpose.html#scipy.sparse.matrix_transpose)
- [`permute_dims`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.permute_dims.html#scipy.sparse.permute_dims)
- [`rand`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.rand.html#scipy.sparse.rand)
- [`random`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.random.html#scipy.sparse.random)
- [`random_array`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.random_array.html#scipy.sparse.random_array)
- [`safely_cast_index_arrays`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.safely_cast_index_arrays.html#scipy.sparse.safely_cast_index_arrays)
- [`save_npz`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.save_npz.html#scipy.sparse.save_npz)
- [`spdiags`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.spdiags.html#scipy.sparse.spdiags)
- [`swapaxes`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.swapaxes.html#scipy.sparse.swapaxes)
- [`tril`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.tril.html#scipy.sparse.tril)
- [`triu`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.triu.html#scipy.sparse.triu)
- [`vstack`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.vstack.html#scipy.sparse.vstack)

#### Classes and their public methods

- [`SparseEfficiencyWarning`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.SparseEfficiencyWarning.html#scipy.sparse.sparseefficiencywarning)
  - [`SparseEfficiencyWarning.add_note`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.SparseEfficiencyWarning.html#scipy.sparse.sparseefficiencywarning.add_note)
  - [`SparseEfficiencyWarning.with_traceback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.SparseEfficiencyWarning.html#scipy.sparse.sparseefficiencywarning.with_traceback)
- [`SparseWarning`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.SparseWarning.html#scipy.sparse.sparsewarning)
  - [`SparseWarning.add_note`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.SparseWarning.html#scipy.sparse.sparsewarning.add_note)
  - [`SparseWarning.with_traceback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.SparseWarning.html#scipy.sparse.sparsewarning.with_traceback)
- [`bsr_array`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.html#scipy.sparse.bsr_array)
  - [`bsr_array.arcsin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.arcsin.html#scipy.sparse.bsr_array.arcsin)
  - [`bsr_array.arcsinh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.arcsinh.html#scipy.sparse.bsr_array.arcsinh)
  - [`bsr_array.arctan`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.arctan.html#scipy.sparse.bsr_array.arctan)
  - [`bsr_array.arctanh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.arctanh.html#scipy.sparse.bsr_array.arctanh)
  - [`bsr_array.argmax`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.argmax.html#scipy.sparse.bsr_array.argmax)
  - [`bsr_array.argmin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.argmin.html#scipy.sparse.bsr_array.argmin)
  - [`bsr_array.asformat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.asformat.html#scipy.sparse.bsr_array.asformat)
  - [`bsr_array.astype`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.astype.html#scipy.sparse.bsr_array.astype)
  - [`bsr_array.ceil`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.ceil.html#scipy.sparse.bsr_array.ceil)
  - [`bsr_array.check_format`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.check_format.html#scipy.sparse.bsr_array.check_format)
  - [`bsr_array.conj`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.conj.html#scipy.sparse.bsr_array.conj)
  - [`bsr_array.conjugate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.conjugate.html#scipy.sparse.bsr_array.conjugate)
  - [`bsr_array.copy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.copy.html#scipy.sparse.bsr_array.copy)
  - [`bsr_array.count_nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.count_nonzero.html#scipy.sparse.bsr_array.count_nonzero)
  - [`bsr_array.deg2rad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.deg2rad.html#scipy.sparse.bsr_array.deg2rad)
  - [`bsr_array.diagonal`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.diagonal.html#scipy.sparse.bsr_array.diagonal)
  - [`bsr_array.dot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.dot.html#scipy.sparse.bsr_array.dot)
  - [`bsr_array.eliminate_zeros`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.eliminate_zeros.html#scipy.sparse.bsr_array.eliminate_zeros)
  - [`bsr_array.expm1`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.expm1.html#scipy.sparse.bsr_array.expm1)
  - [`bsr_array.floor`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.floor.html#scipy.sparse.bsr_array.floor)
  - [`bsr_array.log1p`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.log1p.html#scipy.sparse.bsr_array.log1p)
  - [`bsr_array.max`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.max.html#scipy.sparse.bsr_array.max)
  - [`bsr_array.maximum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.maximum.html#scipy.sparse.bsr_array.maximum)
  - [`bsr_array.mean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.mean.html#scipy.sparse.bsr_array.mean)
  - [`bsr_array.min`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.min.html#scipy.sparse.bsr_array.min)
  - [`bsr_array.minimum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.minimum.html#scipy.sparse.bsr_array.minimum)
  - [`bsr_array.multiply`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.multiply.html#scipy.sparse.bsr_array.multiply)
  - [`bsr_array.nanmax`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.nanmax.html#scipy.sparse.bsr_array.nanmax)
  - [`bsr_array.nanmin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.nanmin.html#scipy.sparse.bsr_array.nanmin)
  - [`bsr_array.nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.nonzero.html#scipy.sparse.bsr_array.nonzero)
  - [`bsr_array.power`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.power.html#scipy.sparse.bsr_array.power)
  - [`bsr_array.prune`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.prune.html#scipy.sparse.bsr_array.prune)
  - [`bsr_array.rad2deg`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.rad2deg.html#scipy.sparse.bsr_array.rad2deg)
  - [`bsr_array.reshape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.reshape.html#scipy.sparse.bsr_array.reshape)
  - [`bsr_array.resize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.resize.html#scipy.sparse.bsr_array.resize)
  - [`bsr_array.rint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.rint.html#scipy.sparse.bsr_array.rint)
  - [`bsr_array.setdiag`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.setdiag.html#scipy.sparse.bsr_array.setdiag)
  - [`bsr_array.sign`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.sign.html#scipy.sparse.bsr_array.sign)
  - [`bsr_array.sin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.sin.html#scipy.sparse.bsr_array.sin)
  - [`bsr_array.sinh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.sinh.html#scipy.sparse.bsr_array.sinh)
  - [`bsr_array.sort_indices`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.sort_indices.html#scipy.sparse.bsr_array.sort_indices)
  - [`bsr_array.sorted_indices`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.sorted_indices.html#scipy.sparse.bsr_array.sorted_indices)
  - [`bsr_array.sqrt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.sqrt.html#scipy.sparse.bsr_array.sqrt)
  - [`bsr_array.sum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.sum.html#scipy.sparse.bsr_array.sum)
  - [`bsr_array.sum_duplicates`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.sum_duplicates.html#scipy.sparse.bsr_array.sum_duplicates)
  - [`bsr_array.tan`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.tan.html#scipy.sparse.bsr_array.tan)
  - [`bsr_array.tanh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.tanh.html#scipy.sparse.bsr_array.tanh)
  - [`bsr_array.toarray`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.toarray.html#scipy.sparse.bsr_array.toarray)
  - [`bsr_array.tobsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.tobsr.html#scipy.sparse.bsr_array.tobsr)
  - [`bsr_array.tocoo`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.tocoo.html#scipy.sparse.bsr_array.tocoo)
  - [`bsr_array.tocsc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.tocsc.html#scipy.sparse.bsr_array.tocsc)
  - [`bsr_array.tocsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.tocsr.html#scipy.sparse.bsr_array.tocsr)
  - [`bsr_array.todense`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.todense.html#scipy.sparse.bsr_array.todense)
  - [`bsr_array.todia`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.todia.html#scipy.sparse.bsr_array.todia)
  - [`bsr_array.todok`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.todok.html#scipy.sparse.bsr_array.todok)
  - [`bsr_array.tolil`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.tolil.html#scipy.sparse.bsr_array.tolil)
  - [`bsr_array.trace`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.trace.html#scipy.sparse.bsr_array.trace)
  - [`bsr_array.transpose`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.transpose.html#scipy.sparse.bsr_array.transpose)
  - [`bsr_array.trunc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_array.trunc.html#scipy.sparse.bsr_array.trunc)
- [`bsr_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.html#scipy.sparse.bsr_matrix)
  - [`bsr_matrix.arcsin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.arcsin.html#scipy.sparse.bsr_matrix.arcsin)
  - [`bsr_matrix.arcsinh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.arcsinh.html#scipy.sparse.bsr_matrix.arcsinh)
  - [`bsr_matrix.arctan`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.arctan.html#scipy.sparse.bsr_matrix.arctan)
  - [`bsr_matrix.arctanh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.arctanh.html#scipy.sparse.bsr_matrix.arctanh)
  - [`bsr_matrix.argmax`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.argmax.html#scipy.sparse.bsr_matrix.argmax)
  - [`bsr_matrix.argmin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.argmin.html#scipy.sparse.bsr_matrix.argmin)
  - [`bsr_matrix.asformat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.asformat.html#scipy.sparse.bsr_matrix.asformat)
  - [`bsr_matrix.asfptype`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.asfptype.html#scipy.sparse.bsr_matrix.asfptype)
  - [`bsr_matrix.astype`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.astype.html#scipy.sparse.bsr_matrix.astype)
  - [`bsr_matrix.ceil`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.ceil.html#scipy.sparse.bsr_matrix.ceil)
  - [`bsr_matrix.check_format`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.check_format.html#scipy.sparse.bsr_matrix.check_format)
  - [`bsr_matrix.conj`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.conj.html#scipy.sparse.bsr_matrix.conj)
  - [`bsr_matrix.conjugate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.conjugate.html#scipy.sparse.bsr_matrix.conjugate)
  - [`bsr_matrix.copy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.copy.html#scipy.sparse.bsr_matrix.copy)
  - [`bsr_matrix.count_nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.count_nonzero.html#scipy.sparse.bsr_matrix.count_nonzero)
  - [`bsr_matrix.deg2rad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.deg2rad.html#scipy.sparse.bsr_matrix.deg2rad)
  - [`bsr_matrix.diagonal`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.diagonal.html#scipy.sparse.bsr_matrix.diagonal)
  - [`bsr_matrix.dot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.dot.html#scipy.sparse.bsr_matrix.dot)
  - [`bsr_matrix.eliminate_zeros`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.eliminate_zeros.html#scipy.sparse.bsr_matrix.eliminate_zeros)
  - [`bsr_matrix.expm1`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.expm1.html#scipy.sparse.bsr_matrix.expm1)
  - [`bsr_matrix.floor`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.floor.html#scipy.sparse.bsr_matrix.floor)
  - [`bsr_matrix.getH`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.getH.html#scipy.sparse.bsr_matrix.geth)
  - [`bsr_matrix.get_shape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.get_shape.html#scipy.sparse.bsr_matrix.get_shape)
  - [`bsr_matrix.getcol`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.getcol.html#scipy.sparse.bsr_matrix.getcol)
  - [`bsr_matrix.getformat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.getformat.html#scipy.sparse.bsr_matrix.getformat)
  - [`bsr_matrix.getmaxprint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.getmaxprint.html#scipy.sparse.bsr_matrix.getmaxprint)
  - [`bsr_matrix.getnnz`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.getnnz.html#scipy.sparse.bsr_matrix.getnnz)
  - [`bsr_matrix.getrow`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.getrow.html#scipy.sparse.bsr_matrix.getrow)
  - [`bsr_matrix.log1p`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.log1p.html#scipy.sparse.bsr_matrix.log1p)
  - [`bsr_matrix.max`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.max.html#scipy.sparse.bsr_matrix.max)
  - [`bsr_matrix.maximum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.maximum.html#scipy.sparse.bsr_matrix.maximum)
  - [`bsr_matrix.mean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.mean.html#scipy.sparse.bsr_matrix.mean)
  - [`bsr_matrix.min`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.min.html#scipy.sparse.bsr_matrix.min)
  - [`bsr_matrix.minimum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.minimum.html#scipy.sparse.bsr_matrix.minimum)
  - [`bsr_matrix.multiply`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.multiply.html#scipy.sparse.bsr_matrix.multiply)
  - [`bsr_matrix.nanmax`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.nanmax.html#scipy.sparse.bsr_matrix.nanmax)
  - [`bsr_matrix.nanmin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.nanmin.html#scipy.sparse.bsr_matrix.nanmin)
  - [`bsr_matrix.nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.nonzero.html#scipy.sparse.bsr_matrix.nonzero)
  - [`bsr_matrix.power`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.power.html#scipy.sparse.bsr_matrix.power)
  - [`bsr_matrix.prune`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.prune.html#scipy.sparse.bsr_matrix.prune)
  - [`bsr_matrix.rad2deg`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.rad2deg.html#scipy.sparse.bsr_matrix.rad2deg)
  - [`bsr_matrix.reshape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.reshape.html#scipy.sparse.bsr_matrix.reshape)
  - [`bsr_matrix.resize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.resize.html#scipy.sparse.bsr_matrix.resize)
  - [`bsr_matrix.rint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.rint.html#scipy.sparse.bsr_matrix.rint)
  - [`bsr_matrix.set_shape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.set_shape.html#scipy.sparse.bsr_matrix.set_shape)
  - [`bsr_matrix.setdiag`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.setdiag.html#scipy.sparse.bsr_matrix.setdiag)
  - [`bsr_matrix.sign`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.sign.html#scipy.sparse.bsr_matrix.sign)
  - [`bsr_matrix.sin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.sin.html#scipy.sparse.bsr_matrix.sin)
  - [`bsr_matrix.sinh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.sinh.html#scipy.sparse.bsr_matrix.sinh)
  - [`bsr_matrix.sort_indices`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.sort_indices.html#scipy.sparse.bsr_matrix.sort_indices)
  - [`bsr_matrix.sorted_indices`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.sorted_indices.html#scipy.sparse.bsr_matrix.sorted_indices)
  - [`bsr_matrix.sqrt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.sqrt.html#scipy.sparse.bsr_matrix.sqrt)
  - [`bsr_matrix.sum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.sum.html#scipy.sparse.bsr_matrix.sum)
  - [`bsr_matrix.sum_duplicates`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.sum_duplicates.html#scipy.sparse.bsr_matrix.sum_duplicates)
  - [`bsr_matrix.tan`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.tan.html#scipy.sparse.bsr_matrix.tan)
  - [`bsr_matrix.tanh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.tanh.html#scipy.sparse.bsr_matrix.tanh)
  - [`bsr_matrix.toarray`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.toarray.html#scipy.sparse.bsr_matrix.toarray)
  - [`bsr_matrix.tobsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.tobsr.html#scipy.sparse.bsr_matrix.tobsr)
  - [`bsr_matrix.tocoo`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.tocoo.html#scipy.sparse.bsr_matrix.tocoo)
  - [`bsr_matrix.tocsc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.tocsc.html#scipy.sparse.bsr_matrix.tocsc)
  - [`bsr_matrix.tocsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.tocsr.html#scipy.sparse.bsr_matrix.tocsr)
  - [`bsr_matrix.todense`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.todense.html#scipy.sparse.bsr_matrix.todense)
  - [`bsr_matrix.todia`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.todia.html#scipy.sparse.bsr_matrix.todia)
  - [`bsr_matrix.todok`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.todok.html#scipy.sparse.bsr_matrix.todok)
  - [`bsr_matrix.tolil`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.tolil.html#scipy.sparse.bsr_matrix.tolil)
  - [`bsr_matrix.trace`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.trace.html#scipy.sparse.bsr_matrix.trace)
  - [`bsr_matrix.transpose`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.transpose.html#scipy.sparse.bsr_matrix.transpose)
  - [`bsr_matrix.trunc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.trunc.html#scipy.sparse.bsr_matrix.trunc)
- [`coo_array`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.html#scipy.sparse.coo_array)
  - [`coo_array.arcsin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.arcsin.html#scipy.sparse.coo_array.arcsin)
  - [`coo_array.arcsinh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.arcsinh.html#scipy.sparse.coo_array.arcsinh)
  - [`coo_array.arctan`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.arctan.html#scipy.sparse.coo_array.arctan)
  - [`coo_array.arctanh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.arctanh.html#scipy.sparse.coo_array.arctanh)
  - [`coo_array.argmax`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.argmax.html#scipy.sparse.coo_array.argmax)
  - [`coo_array.argmin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.argmin.html#scipy.sparse.coo_array.argmin)
  - [`coo_array.asformat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.asformat.html#scipy.sparse.coo_array.asformat)
  - [`coo_array.astype`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.astype.html#scipy.sparse.coo_array.astype)
  - [`coo_array.ceil`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.ceil.html#scipy.sparse.coo_array.ceil)
  - [`coo_array.conj`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.conj.html#scipy.sparse.coo_array.conj)
  - [`coo_array.conjugate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.conjugate.html#scipy.sparse.coo_array.conjugate)
  - [`coo_array.copy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.copy.html#scipy.sparse.coo_array.copy)
  - [`coo_array.count_nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.count_nonzero.html#scipy.sparse.coo_array.count_nonzero)
  - [`coo_array.deg2rad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.deg2rad.html#scipy.sparse.coo_array.deg2rad)
  - [`coo_array.diagonal`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.diagonal.html#scipy.sparse.coo_array.diagonal)
  - [`coo_array.dot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.dot.html#scipy.sparse.coo_array.dot)
  - [`coo_array.eliminate_zeros`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.eliminate_zeros.html#scipy.sparse.coo_array.eliminate_zeros)
  - [`coo_array.expm1`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.expm1.html#scipy.sparse.coo_array.expm1)
  - [`coo_array.floor`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.floor.html#scipy.sparse.coo_array.floor)
  - [`coo_array.log1p`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.log1p.html#scipy.sparse.coo_array.log1p)
  - [`coo_array.max`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.max.html#scipy.sparse.coo_array.max)
  - [`coo_array.maximum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.maximum.html#scipy.sparse.coo_array.maximum)
  - [`coo_array.mean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.mean.html#scipy.sparse.coo_array.mean)
  - [`coo_array.min`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.min.html#scipy.sparse.coo_array.min)
  - [`coo_array.minimum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.minimum.html#scipy.sparse.coo_array.minimum)
  - [`coo_array.multiply`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.multiply.html#scipy.sparse.coo_array.multiply)
  - [`coo_array.nanmax`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.nanmax.html#scipy.sparse.coo_array.nanmax)
  - [`coo_array.nanmin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.nanmin.html#scipy.sparse.coo_array.nanmin)
  - [`coo_array.nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.nonzero.html#scipy.sparse.coo_array.nonzero)
  - [`coo_array.power`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.power.html#scipy.sparse.coo_array.power)
  - [`coo_array.rad2deg`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.rad2deg.html#scipy.sparse.coo_array.rad2deg)
  - [`coo_array.reshape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.reshape.html#scipy.sparse.coo_array.reshape)
  - [`coo_array.resize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.resize.html#scipy.sparse.coo_array.resize)
  - [`coo_array.rint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.rint.html#scipy.sparse.coo_array.rint)
  - [`coo_array.setdiag`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.setdiag.html#scipy.sparse.coo_array.setdiag)
  - [`coo_array.sign`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.sign.html#scipy.sparse.coo_array.sign)
  - [`coo_array.sin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.sin.html#scipy.sparse.coo_array.sin)
  - [`coo_array.sinh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.sinh.html#scipy.sparse.coo_array.sinh)
  - [`coo_array.sqrt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.sqrt.html#scipy.sparse.coo_array.sqrt)
  - [`coo_array.sum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.sum.html#scipy.sparse.coo_array.sum)
  - [`coo_array.sum_duplicates`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.sum_duplicates.html#scipy.sparse.coo_array.sum_duplicates)
  - [`coo_array.tan`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.tan.html#scipy.sparse.coo_array.tan)
  - [`coo_array.tanh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.tanh.html#scipy.sparse.coo_array.tanh)
  - [`coo_array.tensordot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.tensordot.html#scipy.sparse.coo_array.tensordot)
  - [`coo_array.toarray`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.toarray.html#scipy.sparse.coo_array.toarray)
  - [`coo_array.tobsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.tobsr.html#scipy.sparse.coo_array.tobsr)
  - [`coo_array.tocoo`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.tocoo.html#scipy.sparse.coo_array.tocoo)
  - [`coo_array.tocsc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.tocsc.html#scipy.sparse.coo_array.tocsc)
  - [`coo_array.tocsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.tocsr.html#scipy.sparse.coo_array.tocsr)
  - [`coo_array.todense`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.todense.html#scipy.sparse.coo_array.todense)
  - [`coo_array.todia`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.todia.html#scipy.sparse.coo_array.todia)
  - [`coo_array.todok`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.todok.html#scipy.sparse.coo_array.todok)
  - [`coo_array.tolil`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.tolil.html#scipy.sparse.coo_array.tolil)
  - [`coo_array.trace`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.trace.html#scipy.sparse.coo_array.trace)
  - [`coo_array.transpose`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.transpose.html#scipy.sparse.coo_array.transpose)
  - [`coo_array.trunc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.trunc.html#scipy.sparse.coo_array.trunc)
- [`coo_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html#scipy.sparse.coo_matrix)
  - [`coo_matrix.arcsin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.arcsin.html#scipy.sparse.coo_matrix.arcsin)
  - [`coo_matrix.arcsinh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.arcsinh.html#scipy.sparse.coo_matrix.arcsinh)
  - [`coo_matrix.arctan`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.arctan.html#scipy.sparse.coo_matrix.arctan)
  - [`coo_matrix.arctanh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.arctanh.html#scipy.sparse.coo_matrix.arctanh)
  - [`coo_matrix.argmax`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.argmax.html#scipy.sparse.coo_matrix.argmax)
  - [`coo_matrix.argmin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.argmin.html#scipy.sparse.coo_matrix.argmin)
  - [`coo_matrix.asformat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.asformat.html#scipy.sparse.coo_matrix.asformat)
  - [`coo_matrix.asfptype`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.asfptype.html#scipy.sparse.coo_matrix.asfptype)
  - [`coo_matrix.astype`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.astype.html#scipy.sparse.coo_matrix.astype)
  - [`coo_matrix.ceil`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.ceil.html#scipy.sparse.coo_matrix.ceil)
  - [`coo_matrix.conj`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.conj.html#scipy.sparse.coo_matrix.conj)
  - [`coo_matrix.conjugate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.conjugate.html#scipy.sparse.coo_matrix.conjugate)
  - [`coo_matrix.copy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.copy.html#scipy.sparse.coo_matrix.copy)
  - [`coo_matrix.count_nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.count_nonzero.html#scipy.sparse.coo_matrix.count_nonzero)
  - [`coo_matrix.deg2rad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.deg2rad.html#scipy.sparse.coo_matrix.deg2rad)
  - [`coo_matrix.diagonal`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.diagonal.html#scipy.sparse.coo_matrix.diagonal)
  - [`coo_matrix.dot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.dot.html#scipy.sparse.coo_matrix.dot)
  - [`coo_matrix.eliminate_zeros`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.eliminate_zeros.html#scipy.sparse.coo_matrix.eliminate_zeros)
  - [`coo_matrix.expm1`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.expm1.html#scipy.sparse.coo_matrix.expm1)
  - [`coo_matrix.floor`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.floor.html#scipy.sparse.coo_matrix.floor)
  - [`coo_matrix.getH`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.getH.html#scipy.sparse.coo_matrix.geth)
  - [`coo_matrix.get_shape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.get_shape.html#scipy.sparse.coo_matrix.get_shape)
  - [`coo_matrix.getcol`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.getcol.html#scipy.sparse.coo_matrix.getcol)
  - [`coo_matrix.getformat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.getformat.html#scipy.sparse.coo_matrix.getformat)
  - [`coo_matrix.getmaxprint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.getmaxprint.html#scipy.sparse.coo_matrix.getmaxprint)
  - [`coo_matrix.getnnz`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.getnnz.html#scipy.sparse.coo_matrix.getnnz)
  - [`coo_matrix.getrow`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.getrow.html#scipy.sparse.coo_matrix.getrow)
  - [`coo_matrix.log1p`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.log1p.html#scipy.sparse.coo_matrix.log1p)
  - [`coo_matrix.max`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.max.html#scipy.sparse.coo_matrix.max)
  - [`coo_matrix.maximum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.maximum.html#scipy.sparse.coo_matrix.maximum)
  - [`coo_matrix.mean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.mean.html#scipy.sparse.coo_matrix.mean)
  - [`coo_matrix.min`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.min.html#scipy.sparse.coo_matrix.min)
  - [`coo_matrix.minimum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.minimum.html#scipy.sparse.coo_matrix.minimum)
  - [`coo_matrix.multiply`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.multiply.html#scipy.sparse.coo_matrix.multiply)
  - [`coo_matrix.nanmax`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.nanmax.html#scipy.sparse.coo_matrix.nanmax)
  - [`coo_matrix.nanmin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.nanmin.html#scipy.sparse.coo_matrix.nanmin)
  - [`coo_matrix.nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.nonzero.html#scipy.sparse.coo_matrix.nonzero)
  - [`coo_matrix.power`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.power.html#scipy.sparse.coo_matrix.power)
  - [`coo_matrix.rad2deg`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.rad2deg.html#scipy.sparse.coo_matrix.rad2deg)
  - [`coo_matrix.reshape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.reshape.html#scipy.sparse.coo_matrix.reshape)
  - [`coo_matrix.resize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.resize.html#scipy.sparse.coo_matrix.resize)
  - [`coo_matrix.rint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.rint.html#scipy.sparse.coo_matrix.rint)
  - [`coo_matrix.set_shape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.set_shape.html#scipy.sparse.coo_matrix.set_shape)
  - [`coo_matrix.setdiag`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.setdiag.html#scipy.sparse.coo_matrix.setdiag)
  - [`coo_matrix.sign`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.sign.html#scipy.sparse.coo_matrix.sign)
  - [`coo_matrix.sin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.sin.html#scipy.sparse.coo_matrix.sin)
  - [`coo_matrix.sinh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.sinh.html#scipy.sparse.coo_matrix.sinh)
  - [`coo_matrix.sqrt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.sqrt.html#scipy.sparse.coo_matrix.sqrt)
  - [`coo_matrix.sum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.sum.html#scipy.sparse.coo_matrix.sum)
  - [`coo_matrix.sum_duplicates`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.sum_duplicates.html#scipy.sparse.coo_matrix.sum_duplicates)
  - [`coo_matrix.tan`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.tan.html#scipy.sparse.coo_matrix.tan)
  - [`coo_matrix.tanh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.tanh.html#scipy.sparse.coo_matrix.tanh)
  - [`coo_matrix.tensordot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.tensordot.html#scipy.sparse.coo_matrix.tensordot)
  - [`coo_matrix.toarray`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.toarray.html#scipy.sparse.coo_matrix.toarray)
  - [`coo_matrix.tobsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.tobsr.html#scipy.sparse.coo_matrix.tobsr)
  - [`coo_matrix.tocoo`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.tocoo.html#scipy.sparse.coo_matrix.tocoo)
  - [`coo_matrix.tocsc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.tocsc.html#scipy.sparse.coo_matrix.tocsc)
  - [`coo_matrix.tocsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.tocsr.html#scipy.sparse.coo_matrix.tocsr)
  - [`coo_matrix.todense`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.todense.html#scipy.sparse.coo_matrix.todense)
  - [`coo_matrix.todia`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.todia.html#scipy.sparse.coo_matrix.todia)
  - [`coo_matrix.todok`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.todok.html#scipy.sparse.coo_matrix.todok)
  - [`coo_matrix.tolil`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.tolil.html#scipy.sparse.coo_matrix.tolil)
  - [`coo_matrix.trace`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.trace.html#scipy.sparse.coo_matrix.trace)
  - [`coo_matrix.transpose`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.transpose.html#scipy.sparse.coo_matrix.transpose)
  - [`coo_matrix.trunc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.trunc.html#scipy.sparse.coo_matrix.trunc)
- [`csc_array`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.html#scipy.sparse.csc_array)
  - [`csc_array.arcsin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.arcsin.html#scipy.sparse.csc_array.arcsin)
  - [`csc_array.arcsinh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.arcsinh.html#scipy.sparse.csc_array.arcsinh)
  - [`csc_array.arctan`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.arctan.html#scipy.sparse.csc_array.arctan)
  - [`csc_array.arctanh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.arctanh.html#scipy.sparse.csc_array.arctanh)
  - [`csc_array.argmax`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.argmax.html#scipy.sparse.csc_array.argmax)
  - [`csc_array.argmin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.argmin.html#scipy.sparse.csc_array.argmin)
  - [`csc_array.asformat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.asformat.html#scipy.sparse.csc_array.asformat)
  - [`csc_array.astype`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.astype.html#scipy.sparse.csc_array.astype)
  - [`csc_array.ceil`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.ceil.html#scipy.sparse.csc_array.ceil)
  - [`csc_array.check_format`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.check_format.html#scipy.sparse.csc_array.check_format)
  - [`csc_array.conj`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.conj.html#scipy.sparse.csc_array.conj)
  - [`csc_array.conjugate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.conjugate.html#scipy.sparse.csc_array.conjugate)
  - [`csc_array.copy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.copy.html#scipy.sparse.csc_array.copy)
  - [`csc_array.count_nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.count_nonzero.html#scipy.sparse.csc_array.count_nonzero)
  - [`csc_array.deg2rad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.deg2rad.html#scipy.sparse.csc_array.deg2rad)
  - [`csc_array.diagonal`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.diagonal.html#scipy.sparse.csc_array.diagonal)
  - [`csc_array.dot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.dot.html#scipy.sparse.csc_array.dot)
  - [`csc_array.eliminate_zeros`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.eliminate_zeros.html#scipy.sparse.csc_array.eliminate_zeros)
  - [`csc_array.expm1`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.expm1.html#scipy.sparse.csc_array.expm1)
  - [`csc_array.floor`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.floor.html#scipy.sparse.csc_array.floor)
  - [`csc_array.log1p`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.log1p.html#scipy.sparse.csc_array.log1p)
  - [`csc_array.max`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.max.html#scipy.sparse.csc_array.max)
  - [`csc_array.maximum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.maximum.html#scipy.sparse.csc_array.maximum)
  - [`csc_array.mean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.mean.html#scipy.sparse.csc_array.mean)
  - [`csc_array.min`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.min.html#scipy.sparse.csc_array.min)
  - [`csc_array.minimum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.minimum.html#scipy.sparse.csc_array.minimum)
  - [`csc_array.multiply`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.multiply.html#scipy.sparse.csc_array.multiply)
  - [`csc_array.nanmax`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.nanmax.html#scipy.sparse.csc_array.nanmax)
  - [`csc_array.nanmin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.nanmin.html#scipy.sparse.csc_array.nanmin)
  - [`csc_array.nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.nonzero.html#scipy.sparse.csc_array.nonzero)
  - [`csc_array.power`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.power.html#scipy.sparse.csc_array.power)
  - [`csc_array.prune`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.prune.html#scipy.sparse.csc_array.prune)
  - [`csc_array.rad2deg`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.rad2deg.html#scipy.sparse.csc_array.rad2deg)
  - [`csc_array.reshape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.reshape.html#scipy.sparse.csc_array.reshape)
  - [`csc_array.resize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.resize.html#scipy.sparse.csc_array.resize)
  - [`csc_array.rint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.rint.html#scipy.sparse.csc_array.rint)
  - [`csc_array.setdiag`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.setdiag.html#scipy.sparse.csc_array.setdiag)
  - [`csc_array.sign`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.sign.html#scipy.sparse.csc_array.sign)
  - [`csc_array.sin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.sin.html#scipy.sparse.csc_array.sin)
  - [`csc_array.sinh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.sinh.html#scipy.sparse.csc_array.sinh)
  - [`csc_array.sort_indices`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.sort_indices.html#scipy.sparse.csc_array.sort_indices)
  - [`csc_array.sorted_indices`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.sorted_indices.html#scipy.sparse.csc_array.sorted_indices)
  - [`csc_array.sqrt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.sqrt.html#scipy.sparse.csc_array.sqrt)
  - [`csc_array.sum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.sum.html#scipy.sparse.csc_array.sum)
  - [`csc_array.sum_duplicates`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.sum_duplicates.html#scipy.sparse.csc_array.sum_duplicates)
  - [`csc_array.tan`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.tan.html#scipy.sparse.csc_array.tan)
  - [`csc_array.tanh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.tanh.html#scipy.sparse.csc_array.tanh)
  - [`csc_array.toarray`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.toarray.html#scipy.sparse.csc_array.toarray)
  - [`csc_array.tobsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.tobsr.html#scipy.sparse.csc_array.tobsr)
  - [`csc_array.tocoo`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.tocoo.html#scipy.sparse.csc_array.tocoo)
  - [`csc_array.tocsc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.tocsc.html#scipy.sparse.csc_array.tocsc)
  - [`csc_array.tocsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.tocsr.html#scipy.sparse.csc_array.tocsr)
  - [`csc_array.todense`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.todense.html#scipy.sparse.csc_array.todense)
  - [`csc_array.todia`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.todia.html#scipy.sparse.csc_array.todia)
  - [`csc_array.todok`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.todok.html#scipy.sparse.csc_array.todok)
  - [`csc_array.tolil`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.tolil.html#scipy.sparse.csc_array.tolil)
  - [`csc_array.trace`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.trace.html#scipy.sparse.csc_array.trace)
  - [`csc_array.transpose`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.transpose.html#scipy.sparse.csc_array.transpose)
  - [`csc_array.trunc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.trunc.html#scipy.sparse.csc_array.trunc)
- [`csc_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html#scipy.sparse.csc_matrix)
  - [`csc_matrix.arcsin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.arcsin.html#scipy.sparse.csc_matrix.arcsin)
  - [`csc_matrix.arcsinh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.arcsinh.html#scipy.sparse.csc_matrix.arcsinh)
  - [`csc_matrix.arctan`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.arctan.html#scipy.sparse.csc_matrix.arctan)
  - [`csc_matrix.arctanh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.arctanh.html#scipy.sparse.csc_matrix.arctanh)
  - [`csc_matrix.argmax`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.argmax.html#scipy.sparse.csc_matrix.argmax)
  - [`csc_matrix.argmin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.argmin.html#scipy.sparse.csc_matrix.argmin)
  - [`csc_matrix.asformat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.asformat.html#scipy.sparse.csc_matrix.asformat)
  - [`csc_matrix.asfptype`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.asfptype.html#scipy.sparse.csc_matrix.asfptype)
  - [`csc_matrix.astype`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.astype.html#scipy.sparse.csc_matrix.astype)
  - [`csc_matrix.ceil`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.ceil.html#scipy.sparse.csc_matrix.ceil)
  - [`csc_matrix.check_format`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.check_format.html#scipy.sparse.csc_matrix.check_format)
  - [`csc_matrix.conj`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.conj.html#scipy.sparse.csc_matrix.conj)
  - [`csc_matrix.conjugate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.conjugate.html#scipy.sparse.csc_matrix.conjugate)
  - [`csc_matrix.copy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.copy.html#scipy.sparse.csc_matrix.copy)
  - [`csc_matrix.count_nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.count_nonzero.html#scipy.sparse.csc_matrix.count_nonzero)
  - [`csc_matrix.deg2rad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.deg2rad.html#scipy.sparse.csc_matrix.deg2rad)
  - [`csc_matrix.diagonal`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.diagonal.html#scipy.sparse.csc_matrix.diagonal)
  - [`csc_matrix.dot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.dot.html#scipy.sparse.csc_matrix.dot)
  - [`csc_matrix.eliminate_zeros`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.eliminate_zeros.html#scipy.sparse.csc_matrix.eliminate_zeros)
  - [`csc_matrix.expm1`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.expm1.html#scipy.sparse.csc_matrix.expm1)
  - [`csc_matrix.floor`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.floor.html#scipy.sparse.csc_matrix.floor)
  - [`csc_matrix.getH`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.getH.html#scipy.sparse.csc_matrix.geth)
  - [`csc_matrix.get_shape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.get_shape.html#scipy.sparse.csc_matrix.get_shape)
  - [`csc_matrix.getcol`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.getcol.html#scipy.sparse.csc_matrix.getcol)
  - [`csc_matrix.getformat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.getformat.html#scipy.sparse.csc_matrix.getformat)
  - [`csc_matrix.getmaxprint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.getmaxprint.html#scipy.sparse.csc_matrix.getmaxprint)
  - [`csc_matrix.getnnz`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.getnnz.html#scipy.sparse.csc_matrix.getnnz)
  - [`csc_matrix.getrow`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.getrow.html#scipy.sparse.csc_matrix.getrow)
  - [`csc_matrix.log1p`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.log1p.html#scipy.sparse.csc_matrix.log1p)
  - [`csc_matrix.max`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.max.html#scipy.sparse.csc_matrix.max)
  - [`csc_matrix.maximum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.maximum.html#scipy.sparse.csc_matrix.maximum)
  - [`csc_matrix.mean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.mean.html#scipy.sparse.csc_matrix.mean)
  - [`csc_matrix.min`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.min.html#scipy.sparse.csc_matrix.min)
  - [`csc_matrix.minimum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.minimum.html#scipy.sparse.csc_matrix.minimum)
  - [`csc_matrix.multiply`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.multiply.html#scipy.sparse.csc_matrix.multiply)
  - [`csc_matrix.nanmax`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.nanmax.html#scipy.sparse.csc_matrix.nanmax)
  - [`csc_matrix.nanmin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.nanmin.html#scipy.sparse.csc_matrix.nanmin)
  - [`csc_matrix.nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.nonzero.html#scipy.sparse.csc_matrix.nonzero)
  - [`csc_matrix.power`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.power.html#scipy.sparse.csc_matrix.power)
  - [`csc_matrix.prune`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.prune.html#scipy.sparse.csc_matrix.prune)
  - [`csc_matrix.rad2deg`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.rad2deg.html#scipy.sparse.csc_matrix.rad2deg)
  - [`csc_matrix.reshape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.reshape.html#scipy.sparse.csc_matrix.reshape)
  - [`csc_matrix.resize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.resize.html#scipy.sparse.csc_matrix.resize)
  - [`csc_matrix.rint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.rint.html#scipy.sparse.csc_matrix.rint)
  - [`csc_matrix.set_shape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.set_shape.html#scipy.sparse.csc_matrix.set_shape)
  - [`csc_matrix.setdiag`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.setdiag.html#scipy.sparse.csc_matrix.setdiag)
  - [`csc_matrix.sign`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.sign.html#scipy.sparse.csc_matrix.sign)
  - [`csc_matrix.sin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.sin.html#scipy.sparse.csc_matrix.sin)
  - [`csc_matrix.sinh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.sinh.html#scipy.sparse.csc_matrix.sinh)
  - [`csc_matrix.sort_indices`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.sort_indices.html#scipy.sparse.csc_matrix.sort_indices)
  - [`csc_matrix.sorted_indices`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.sorted_indices.html#scipy.sparse.csc_matrix.sorted_indices)
  - [`csc_matrix.sqrt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.sqrt.html#scipy.sparse.csc_matrix.sqrt)
  - [`csc_matrix.sum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.sum.html#scipy.sparse.csc_matrix.sum)
  - [`csc_matrix.sum_duplicates`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.sum_duplicates.html#scipy.sparse.csc_matrix.sum_duplicates)
  - [`csc_matrix.tan`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.tan.html#scipy.sparse.csc_matrix.tan)
  - [`csc_matrix.tanh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.tanh.html#scipy.sparse.csc_matrix.tanh)
  - [`csc_matrix.toarray`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.toarray.html#scipy.sparse.csc_matrix.toarray)
  - [`csc_matrix.tobsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.tobsr.html#scipy.sparse.csc_matrix.tobsr)
  - [`csc_matrix.tocoo`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.tocoo.html#scipy.sparse.csc_matrix.tocoo)
  - [`csc_matrix.tocsc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.tocsc.html#scipy.sparse.csc_matrix.tocsc)
  - [`csc_matrix.tocsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.tocsr.html#scipy.sparse.csc_matrix.tocsr)
  - [`csc_matrix.todense`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.todense.html#scipy.sparse.csc_matrix.todense)
  - [`csc_matrix.todia`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.todia.html#scipy.sparse.csc_matrix.todia)
  - [`csc_matrix.todok`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.todok.html#scipy.sparse.csc_matrix.todok)
  - [`csc_matrix.tolil`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.tolil.html#scipy.sparse.csc_matrix.tolil)
  - [`csc_matrix.trace`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.trace.html#scipy.sparse.csc_matrix.trace)
  - [`csc_matrix.transpose`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.transpose.html#scipy.sparse.csc_matrix.transpose)
  - [`csc_matrix.trunc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.trunc.html#scipy.sparse.csc_matrix.trunc)
- [`csr_array`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html#scipy.sparse.csr_array)
  - [`csr_array.arcsin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.arcsin.html#scipy.sparse.csr_array.arcsin)
  - [`csr_array.arcsinh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.arcsinh.html#scipy.sparse.csr_array.arcsinh)
  - [`csr_array.arctan`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.arctan.html#scipy.sparse.csr_array.arctan)
  - [`csr_array.arctanh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.arctanh.html#scipy.sparse.csr_array.arctanh)
  - [`csr_array.argmax`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.argmax.html#scipy.sparse.csr_array.argmax)
  - [`csr_array.argmin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.argmin.html#scipy.sparse.csr_array.argmin)
  - [`csr_array.asformat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.asformat.html#scipy.sparse.csr_array.asformat)
  - [`csr_array.astype`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.astype.html#scipy.sparse.csr_array.astype)
  - [`csr_array.ceil`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.ceil.html#scipy.sparse.csr_array.ceil)
  - [`csr_array.check_format`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.check_format.html#scipy.sparse.csr_array.check_format)
  - [`csr_array.conj`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.conj.html#scipy.sparse.csr_array.conj)
  - [`csr_array.conjugate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.conjugate.html#scipy.sparse.csr_array.conjugate)
  - [`csr_array.copy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.copy.html#scipy.sparse.csr_array.copy)
  - [`csr_array.count_nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.count_nonzero.html#scipy.sparse.csr_array.count_nonzero)
  - [`csr_array.deg2rad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.deg2rad.html#scipy.sparse.csr_array.deg2rad)
  - [`csr_array.diagonal`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.diagonal.html#scipy.sparse.csr_array.diagonal)
  - [`csr_array.dot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.dot.html#scipy.sparse.csr_array.dot)
  - [`csr_array.eliminate_zeros`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.eliminate_zeros.html#scipy.sparse.csr_array.eliminate_zeros)
  - [`csr_array.expm1`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.expm1.html#scipy.sparse.csr_array.expm1)
  - [`csr_array.floor`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.floor.html#scipy.sparse.csr_array.floor)
  - [`csr_array.log1p`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.log1p.html#scipy.sparse.csr_array.log1p)
  - [`csr_array.max`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.max.html#scipy.sparse.csr_array.max)
  - [`csr_array.maximum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.maximum.html#scipy.sparse.csr_array.maximum)
  - [`csr_array.mean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.mean.html#scipy.sparse.csr_array.mean)
  - [`csr_array.min`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.min.html#scipy.sparse.csr_array.min)
  - [`csr_array.minimum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.minimum.html#scipy.sparse.csr_array.minimum)
  - [`csr_array.multiply`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.multiply.html#scipy.sparse.csr_array.multiply)
  - [`csr_array.nanmax`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.nanmax.html#scipy.sparse.csr_array.nanmax)
  - [`csr_array.nanmin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.nanmin.html#scipy.sparse.csr_array.nanmin)
  - [`csr_array.nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.nonzero.html#scipy.sparse.csr_array.nonzero)
  - [`csr_array.power`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.power.html#scipy.sparse.csr_array.power)
  - [`csr_array.prune`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.prune.html#scipy.sparse.csr_array.prune)
  - [`csr_array.rad2deg`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.rad2deg.html#scipy.sparse.csr_array.rad2deg)
  - [`csr_array.reshape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.reshape.html#scipy.sparse.csr_array.reshape)
  - [`csr_array.resize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.resize.html#scipy.sparse.csr_array.resize)
  - [`csr_array.rint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.rint.html#scipy.sparse.csr_array.rint)
  - [`csr_array.setdiag`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.setdiag.html#scipy.sparse.csr_array.setdiag)
  - [`csr_array.sign`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.sign.html#scipy.sparse.csr_array.sign)
  - [`csr_array.sin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.sin.html#scipy.sparse.csr_array.sin)
  - [`csr_array.sinh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.sinh.html#scipy.sparse.csr_array.sinh)
  - [`csr_array.sort_indices`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.sort_indices.html#scipy.sparse.csr_array.sort_indices)
  - [`csr_array.sorted_indices`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.sorted_indices.html#scipy.sparse.csr_array.sorted_indices)
  - [`csr_array.sqrt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.sqrt.html#scipy.sparse.csr_array.sqrt)
  - [`csr_array.sum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.sum.html#scipy.sparse.csr_array.sum)
  - [`csr_array.sum_duplicates`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.sum_duplicates.html#scipy.sparse.csr_array.sum_duplicates)
  - [`csr_array.tan`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.tan.html#scipy.sparse.csr_array.tan)
  - [`csr_array.tanh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.tanh.html#scipy.sparse.csr_array.tanh)
  - [`csr_array.toarray`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.toarray.html#scipy.sparse.csr_array.toarray)
  - [`csr_array.tobsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.tobsr.html#scipy.sparse.csr_array.tobsr)
  - [`csr_array.tocoo`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.tocoo.html#scipy.sparse.csr_array.tocoo)
  - [`csr_array.tocsc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.tocsc.html#scipy.sparse.csr_array.tocsc)
  - [`csr_array.tocsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.tocsr.html#scipy.sparse.csr_array.tocsr)
  - [`csr_array.todense`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.todense.html#scipy.sparse.csr_array.todense)
  - [`csr_array.todia`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.todia.html#scipy.sparse.csr_array.todia)
  - [`csr_array.todok`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.todok.html#scipy.sparse.csr_array.todok)
  - [`csr_array.tolil`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.tolil.html#scipy.sparse.csr_array.tolil)
  - [`csr_array.trace`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.trace.html#scipy.sparse.csr_array.trace)
  - [`csr_array.transpose`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.transpose.html#scipy.sparse.csr_array.transpose)
  - [`csr_array.trunc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.trunc.html#scipy.sparse.csr_array.trunc)
- [`csr_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix)
  - [`csr_matrix.arcsin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.arcsin.html#scipy.sparse.csr_matrix.arcsin)
  - [`csr_matrix.arcsinh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.arcsinh.html#scipy.sparse.csr_matrix.arcsinh)
  - [`csr_matrix.arctan`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.arctan.html#scipy.sparse.csr_matrix.arctan)
  - [`csr_matrix.arctanh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.arctanh.html#scipy.sparse.csr_matrix.arctanh)
  - [`csr_matrix.argmax`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.argmax.html#scipy.sparse.csr_matrix.argmax)
  - [`csr_matrix.argmin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.argmin.html#scipy.sparse.csr_matrix.argmin)
  - [`csr_matrix.asformat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.asformat.html#scipy.sparse.csr_matrix.asformat)
  - [`csr_matrix.asfptype`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.asfptype.html#scipy.sparse.csr_matrix.asfptype)
  - [`csr_matrix.astype`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.astype.html#scipy.sparse.csr_matrix.astype)
  - [`csr_matrix.ceil`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.ceil.html#scipy.sparse.csr_matrix.ceil)
  - [`csr_matrix.check_format`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.check_format.html#scipy.sparse.csr_matrix.check_format)
  - [`csr_matrix.conj`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.conj.html#scipy.sparse.csr_matrix.conj)
  - [`csr_matrix.conjugate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.conjugate.html#scipy.sparse.csr_matrix.conjugate)
  - [`csr_matrix.copy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.copy.html#scipy.sparse.csr_matrix.copy)
  - [`csr_matrix.count_nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.count_nonzero.html#scipy.sparse.csr_matrix.count_nonzero)
  - [`csr_matrix.deg2rad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.deg2rad.html#scipy.sparse.csr_matrix.deg2rad)
  - [`csr_matrix.diagonal`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.diagonal.html#scipy.sparse.csr_matrix.diagonal)
  - [`csr_matrix.dot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.dot.html#scipy.sparse.csr_matrix.dot)
  - [`csr_matrix.eliminate_zeros`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.eliminate_zeros.html#scipy.sparse.csr_matrix.eliminate_zeros)
  - [`csr_matrix.expm1`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.expm1.html#scipy.sparse.csr_matrix.expm1)
  - [`csr_matrix.floor`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.floor.html#scipy.sparse.csr_matrix.floor)
  - [`csr_matrix.getH`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.getH.html#scipy.sparse.csr_matrix.geth)
  - [`csr_matrix.get_shape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.get_shape.html#scipy.sparse.csr_matrix.get_shape)
  - [`csr_matrix.getcol`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.getcol.html#scipy.sparse.csr_matrix.getcol)
  - [`csr_matrix.getformat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.getformat.html#scipy.sparse.csr_matrix.getformat)
  - [`csr_matrix.getmaxprint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.getmaxprint.html#scipy.sparse.csr_matrix.getmaxprint)
  - [`csr_matrix.getnnz`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.getnnz.html#scipy.sparse.csr_matrix.getnnz)
  - [`csr_matrix.getrow`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.getrow.html#scipy.sparse.csr_matrix.getrow)
  - [`csr_matrix.log1p`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.log1p.html#scipy.sparse.csr_matrix.log1p)
  - [`csr_matrix.max`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.max.html#scipy.sparse.csr_matrix.max)
  - [`csr_matrix.maximum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.maximum.html#scipy.sparse.csr_matrix.maximum)
  - [`csr_matrix.mean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.mean.html#scipy.sparse.csr_matrix.mean)
  - [`csr_matrix.min`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.min.html#scipy.sparse.csr_matrix.min)
  - [`csr_matrix.minimum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.minimum.html#scipy.sparse.csr_matrix.minimum)
  - [`csr_matrix.multiply`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.multiply.html#scipy.sparse.csr_matrix.multiply)
  - [`csr_matrix.nanmax`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.nanmax.html#scipy.sparse.csr_matrix.nanmax)
  - [`csr_matrix.nanmin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.nanmin.html#scipy.sparse.csr_matrix.nanmin)
  - [`csr_matrix.nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.nonzero.html#scipy.sparse.csr_matrix.nonzero)
  - [`csr_matrix.power`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.power.html#scipy.sparse.csr_matrix.power)
  - [`csr_matrix.prune`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.prune.html#scipy.sparse.csr_matrix.prune)
  - [`csr_matrix.rad2deg`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.rad2deg.html#scipy.sparse.csr_matrix.rad2deg)
  - [`csr_matrix.reshape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.reshape.html#scipy.sparse.csr_matrix.reshape)
  - [`csr_matrix.resize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.resize.html#scipy.sparse.csr_matrix.resize)
  - [`csr_matrix.rint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.rint.html#scipy.sparse.csr_matrix.rint)
  - [`csr_matrix.set_shape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.set_shape.html#scipy.sparse.csr_matrix.set_shape)
  - [`csr_matrix.setdiag`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.setdiag.html#scipy.sparse.csr_matrix.setdiag)
  - [`csr_matrix.sign`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.sign.html#scipy.sparse.csr_matrix.sign)
  - [`csr_matrix.sin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.sin.html#scipy.sparse.csr_matrix.sin)
  - [`csr_matrix.sinh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.sinh.html#scipy.sparse.csr_matrix.sinh)
  - [`csr_matrix.sort_indices`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.sort_indices.html#scipy.sparse.csr_matrix.sort_indices)
  - [`csr_matrix.sorted_indices`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.sorted_indices.html#scipy.sparse.csr_matrix.sorted_indices)
  - [`csr_matrix.sqrt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.sqrt.html#scipy.sparse.csr_matrix.sqrt)
  - [`csr_matrix.sum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.sum.html#scipy.sparse.csr_matrix.sum)
  - [`csr_matrix.sum_duplicates`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.sum_duplicates.html#scipy.sparse.csr_matrix.sum_duplicates)
  - [`csr_matrix.tan`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.tan.html#scipy.sparse.csr_matrix.tan)
  - [`csr_matrix.tanh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.tanh.html#scipy.sparse.csr_matrix.tanh)
  - [`csr_matrix.toarray`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.toarray.html#scipy.sparse.csr_matrix.toarray)
  - [`csr_matrix.tobsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.tobsr.html#scipy.sparse.csr_matrix.tobsr)
  - [`csr_matrix.tocoo`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.tocoo.html#scipy.sparse.csr_matrix.tocoo)
  - [`csr_matrix.tocsc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.tocsc.html#scipy.sparse.csr_matrix.tocsc)
  - [`csr_matrix.tocsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.tocsr.html#scipy.sparse.csr_matrix.tocsr)
  - [`csr_matrix.todense`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.todense.html#scipy.sparse.csr_matrix.todense)
  - [`csr_matrix.todia`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.todia.html#scipy.sparse.csr_matrix.todia)
  - [`csr_matrix.todok`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.todok.html#scipy.sparse.csr_matrix.todok)
  - [`csr_matrix.tolil`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.tolil.html#scipy.sparse.csr_matrix.tolil)
  - [`csr_matrix.trace`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.trace.html#scipy.sparse.csr_matrix.trace)
  - [`csr_matrix.transpose`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.transpose.html#scipy.sparse.csr_matrix.transpose)
  - [`csr_matrix.trunc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.trunc.html#scipy.sparse.csr_matrix.trunc)
- [`dia_array`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.html#scipy.sparse.dia_array)
  - [`dia_array.arcsin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.arcsin.html#scipy.sparse.dia_array.arcsin)
  - [`dia_array.arcsinh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.arcsinh.html#scipy.sparse.dia_array.arcsinh)
  - [`dia_array.arctan`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.arctan.html#scipy.sparse.dia_array.arctan)
  - [`dia_array.arctanh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.arctanh.html#scipy.sparse.dia_array.arctanh)
  - [`dia_array.asformat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.asformat.html#scipy.sparse.dia_array.asformat)
  - [`dia_array.astype`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.astype.html#scipy.sparse.dia_array.astype)
  - [`dia_array.ceil`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.ceil.html#scipy.sparse.dia_array.ceil)
  - [`dia_array.conj`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.conj.html#scipy.sparse.dia_array.conj)
  - [`dia_array.conjugate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.conjugate.html#scipy.sparse.dia_array.conjugate)
  - [`dia_array.copy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.copy.html#scipy.sparse.dia_array.copy)
  - [`dia_array.count_nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.count_nonzero.html#scipy.sparse.dia_array.count_nonzero)
  - [`dia_array.deg2rad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.deg2rad.html#scipy.sparse.dia_array.deg2rad)
  - [`dia_array.diagonal`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.diagonal.html#scipy.sparse.dia_array.diagonal)
  - [`dia_array.dot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.dot.html#scipy.sparse.dia_array.dot)
  - [`dia_array.expm1`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.expm1.html#scipy.sparse.dia_array.expm1)
  - [`dia_array.floor`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.floor.html#scipy.sparse.dia_array.floor)
  - [`dia_array.log1p`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.log1p.html#scipy.sparse.dia_array.log1p)
  - [`dia_array.maximum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.maximum.html#scipy.sparse.dia_array.maximum)
  - [`dia_array.mean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.mean.html#scipy.sparse.dia_array.mean)
  - [`dia_array.minimum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.minimum.html#scipy.sparse.dia_array.minimum)
  - [`dia_array.multiply`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.multiply.html#scipy.sparse.dia_array.multiply)
  - [`dia_array.nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.nonzero.html#scipy.sparse.dia_array.nonzero)
  - [`dia_array.power`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.power.html#scipy.sparse.dia_array.power)
  - [`dia_array.rad2deg`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.rad2deg.html#scipy.sparse.dia_array.rad2deg)
  - [`dia_array.reshape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.reshape.html#scipy.sparse.dia_array.reshape)
  - [`dia_array.resize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.resize.html#scipy.sparse.dia_array.resize)
  - [`dia_array.rint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.rint.html#scipy.sparse.dia_array.rint)
  - [`dia_array.setdiag`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.setdiag.html#scipy.sparse.dia_array.setdiag)
  - [`dia_array.sign`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.sign.html#scipy.sparse.dia_array.sign)
  - [`dia_array.sin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.sin.html#scipy.sparse.dia_array.sin)
  - [`dia_array.sinh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.sinh.html#scipy.sparse.dia_array.sinh)
  - [`dia_array.sqrt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.sqrt.html#scipy.sparse.dia_array.sqrt)
  - [`dia_array.sum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.sum.html#scipy.sparse.dia_array.sum)
  - [`dia_array.tan`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.tan.html#scipy.sparse.dia_array.tan)
  - [`dia_array.tanh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.tanh.html#scipy.sparse.dia_array.tanh)
  - [`dia_array.toarray`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.toarray.html#scipy.sparse.dia_array.toarray)
  - [`dia_array.tobsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.tobsr.html#scipy.sparse.dia_array.tobsr)
  - [`dia_array.tocoo`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.tocoo.html#scipy.sparse.dia_array.tocoo)
  - [`dia_array.tocsc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.tocsc.html#scipy.sparse.dia_array.tocsc)
  - [`dia_array.tocsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.tocsr.html#scipy.sparse.dia_array.tocsr)
  - [`dia_array.todense`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.todense.html#scipy.sparse.dia_array.todense)
  - [`dia_array.todia`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.todia.html#scipy.sparse.dia_array.todia)
  - [`dia_array.todok`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.todok.html#scipy.sparse.dia_array.todok)
  - [`dia_array.tolil`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.tolil.html#scipy.sparse.dia_array.tolil)
  - [`dia_array.trace`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.trace.html#scipy.sparse.dia_array.trace)
  - [`dia_array.transpose`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.transpose.html#scipy.sparse.dia_array.transpose)
  - [`dia_array.trunc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_array.trunc.html#scipy.sparse.dia_array.trunc)
- [`dia_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.html#scipy.sparse.dia_matrix)
  - [`dia_matrix.arcsin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.arcsin.html#scipy.sparse.dia_matrix.arcsin)
  - [`dia_matrix.arcsinh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.arcsinh.html#scipy.sparse.dia_matrix.arcsinh)
  - [`dia_matrix.arctan`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.arctan.html#scipy.sparse.dia_matrix.arctan)
  - [`dia_matrix.arctanh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.arctanh.html#scipy.sparse.dia_matrix.arctanh)
  - [`dia_matrix.asformat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.asformat.html#scipy.sparse.dia_matrix.asformat)
  - [`dia_matrix.asfptype`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.asfptype.html#scipy.sparse.dia_matrix.asfptype)
  - [`dia_matrix.astype`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.astype.html#scipy.sparse.dia_matrix.astype)
  - [`dia_matrix.ceil`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.ceil.html#scipy.sparse.dia_matrix.ceil)
  - [`dia_matrix.conj`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.conj.html#scipy.sparse.dia_matrix.conj)
  - [`dia_matrix.conjugate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.conjugate.html#scipy.sparse.dia_matrix.conjugate)
  - [`dia_matrix.copy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.copy.html#scipy.sparse.dia_matrix.copy)
  - [`dia_matrix.count_nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.count_nonzero.html#scipy.sparse.dia_matrix.count_nonzero)
  - [`dia_matrix.deg2rad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.deg2rad.html#scipy.sparse.dia_matrix.deg2rad)
  - [`dia_matrix.diagonal`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.diagonal.html#scipy.sparse.dia_matrix.diagonal)
  - [`dia_matrix.dot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.dot.html#scipy.sparse.dia_matrix.dot)
  - [`dia_matrix.expm1`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.expm1.html#scipy.sparse.dia_matrix.expm1)
  - [`dia_matrix.floor`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.floor.html#scipy.sparse.dia_matrix.floor)
  - [`dia_matrix.getH`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.getH.html#scipy.sparse.dia_matrix.geth)
  - [`dia_matrix.get_shape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.get_shape.html#scipy.sparse.dia_matrix.get_shape)
  - [`dia_matrix.getcol`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.getcol.html#scipy.sparse.dia_matrix.getcol)
  - [`dia_matrix.getformat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.getformat.html#scipy.sparse.dia_matrix.getformat)
  - [`dia_matrix.getmaxprint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.getmaxprint.html#scipy.sparse.dia_matrix.getmaxprint)
  - [`dia_matrix.getnnz`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.getnnz.html#scipy.sparse.dia_matrix.getnnz)
  - [`dia_matrix.getrow`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.getrow.html#scipy.sparse.dia_matrix.getrow)
  - [`dia_matrix.log1p`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.log1p.html#scipy.sparse.dia_matrix.log1p)
  - [`dia_matrix.maximum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.maximum.html#scipy.sparse.dia_matrix.maximum)
  - [`dia_matrix.mean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.mean.html#scipy.sparse.dia_matrix.mean)
  - [`dia_matrix.minimum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.minimum.html#scipy.sparse.dia_matrix.minimum)
  - [`dia_matrix.multiply`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.multiply.html#scipy.sparse.dia_matrix.multiply)
  - [`dia_matrix.nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.nonzero.html#scipy.sparse.dia_matrix.nonzero)
  - [`dia_matrix.power`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.power.html#scipy.sparse.dia_matrix.power)
  - [`dia_matrix.rad2deg`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.rad2deg.html#scipy.sparse.dia_matrix.rad2deg)
  - [`dia_matrix.reshape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.reshape.html#scipy.sparse.dia_matrix.reshape)
  - [`dia_matrix.resize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.resize.html#scipy.sparse.dia_matrix.resize)
  - [`dia_matrix.rint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.rint.html#scipy.sparse.dia_matrix.rint)
  - [`dia_matrix.set_shape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.set_shape.html#scipy.sparse.dia_matrix.set_shape)
  - [`dia_matrix.setdiag`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.setdiag.html#scipy.sparse.dia_matrix.setdiag)
  - [`dia_matrix.sign`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.sign.html#scipy.sparse.dia_matrix.sign)
  - [`dia_matrix.sin`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.sin.html#scipy.sparse.dia_matrix.sin)
  - [`dia_matrix.sinh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.sinh.html#scipy.sparse.dia_matrix.sinh)
  - [`dia_matrix.sqrt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.sqrt.html#scipy.sparse.dia_matrix.sqrt)
  - [`dia_matrix.sum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.sum.html#scipy.sparse.dia_matrix.sum)
  - [`dia_matrix.tan`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.tan.html#scipy.sparse.dia_matrix.tan)
  - [`dia_matrix.tanh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.tanh.html#scipy.sparse.dia_matrix.tanh)
  - [`dia_matrix.toarray`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.toarray.html#scipy.sparse.dia_matrix.toarray)
  - [`dia_matrix.tobsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.tobsr.html#scipy.sparse.dia_matrix.tobsr)
  - [`dia_matrix.tocoo`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.tocoo.html#scipy.sparse.dia_matrix.tocoo)
  - [`dia_matrix.tocsc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.tocsc.html#scipy.sparse.dia_matrix.tocsc)
  - [`dia_matrix.tocsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.tocsr.html#scipy.sparse.dia_matrix.tocsr)
  - [`dia_matrix.todense`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.todense.html#scipy.sparse.dia_matrix.todense)
  - [`dia_matrix.todia`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.todia.html#scipy.sparse.dia_matrix.todia)
  - [`dia_matrix.todok`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.todok.html#scipy.sparse.dia_matrix.todok)
  - [`dia_matrix.tolil`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.tolil.html#scipy.sparse.dia_matrix.tolil)
  - [`dia_matrix.trace`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.trace.html#scipy.sparse.dia_matrix.trace)
  - [`dia_matrix.transpose`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.transpose.html#scipy.sparse.dia_matrix.transpose)
  - [`dia_matrix.trunc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.trunc.html#scipy.sparse.dia_matrix.trunc)
- [`dok_array`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.html#scipy.sparse.dok_array)
  - [`dok_array.asformat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.asformat.html#scipy.sparse.dok_array.asformat)
  - [`dok_array.astype`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.astype.html#scipy.sparse.dok_array.astype)
  - [`dok_array.clear`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.clear.html#scipy.sparse.dok_array.clear)
  - [`dok_array.conj`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.conj.html#scipy.sparse.dok_array.conj)
  - [`dok_array.conjugate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.conjugate.html#scipy.sparse.dok_array.conjugate)
  - [`dok_array.copy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.copy.html#scipy.sparse.dok_array.copy)
  - [`dok_array.count_nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.count_nonzero.html#scipy.sparse.dok_array.count_nonzero)
  - [`dok_array.diagonal`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.diagonal.html#scipy.sparse.dok_array.diagonal)
  - [`dok_array.dot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.dot.html#scipy.sparse.dok_array.dot)
  - [`dok_array.fromkeys`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.fromkeys.html#scipy.sparse.dok_array.fromkeys)
  - [`dok_array.get`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.get.html#scipy.sparse.dok_array.get)
  - [`dok_array.items`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.items.html#scipy.sparse.dok_array.items)
  - [`dok_array.keys`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.keys.html#scipy.sparse.dok_array.keys)
  - [`dok_array.maximum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.maximum.html#scipy.sparse.dok_array.maximum)
  - [`dok_array.mean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.mean.html#scipy.sparse.dok_array.mean)
  - [`dok_array.minimum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.minimum.html#scipy.sparse.dok_array.minimum)
  - [`dok_array.multiply`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.multiply.html#scipy.sparse.dok_array.multiply)
  - [`dok_array.nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.nonzero.html#scipy.sparse.dok_array.nonzero)
  - [`dok_array.pop`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.pop.html#scipy.sparse.dok_array.pop)
  - [`dok_array.popitem`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.popitem.html#scipy.sparse.dok_array.popitem)
  - [`dok_array.power`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.power.html#scipy.sparse.dok_array.power)
  - [`dok_array.reshape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.reshape.html#scipy.sparse.dok_array.reshape)
  - [`dok_array.resize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.resize.html#scipy.sparse.dok_array.resize)
  - [`dok_array.setdefault`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.setdefault.html#scipy.sparse.dok_array.setdefault)
  - [`dok_array.setdiag`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.setdiag.html#scipy.sparse.dok_array.setdiag)
  - [`dok_array.sum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.sum.html#scipy.sparse.dok_array.sum)
  - [`dok_array.toarray`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.toarray.html#scipy.sparse.dok_array.toarray)
  - [`dok_array.tobsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.tobsr.html#scipy.sparse.dok_array.tobsr)
  - [`dok_array.tocoo`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.tocoo.html#scipy.sparse.dok_array.tocoo)
  - [`dok_array.tocsc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.tocsc.html#scipy.sparse.dok_array.tocsc)
  - [`dok_array.tocsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.tocsr.html#scipy.sparse.dok_array.tocsr)
  - [`dok_array.todense`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.todense.html#scipy.sparse.dok_array.todense)
  - [`dok_array.todia`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.todia.html#scipy.sparse.dok_array.todia)
  - [`dok_array.todok`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.todok.html#scipy.sparse.dok_array.todok)
  - [`dok_array.tolil`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.tolil.html#scipy.sparse.dok_array.tolil)
  - [`dok_array.trace`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.trace.html#scipy.sparse.dok_array.trace)
  - [`dok_array.transpose`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.transpose.html#scipy.sparse.dok_array.transpose)
  - [`dok_array.update`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.update.html#scipy.sparse.dok_array.update)
  - [`dok_array.values`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_array.values.html#scipy.sparse.dok_array.values)
- [`dok_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.html#scipy.sparse.dok_matrix)
  - [`dok_matrix.asformat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.asformat.html#scipy.sparse.dok_matrix.asformat)
  - [`dok_matrix.asfptype`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.asfptype.html#scipy.sparse.dok_matrix.asfptype)
  - [`dok_matrix.astype`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.astype.html#scipy.sparse.dok_matrix.astype)
  - [`dok_matrix.clear`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.clear.html#scipy.sparse.dok_matrix.clear)
  - [`dok_matrix.conj`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.conj.html#scipy.sparse.dok_matrix.conj)
  - [`dok_matrix.conjugate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.conjugate.html#scipy.sparse.dok_matrix.conjugate)
  - [`dok_matrix.copy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.copy.html#scipy.sparse.dok_matrix.copy)
  - [`dok_matrix.count_nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.count_nonzero.html#scipy.sparse.dok_matrix.count_nonzero)
  - [`dok_matrix.diagonal`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.diagonal.html#scipy.sparse.dok_matrix.diagonal)
  - [`dok_matrix.dot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.dot.html#scipy.sparse.dok_matrix.dot)
  - [`dok_matrix.fromkeys`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.fromkeys.html#scipy.sparse.dok_matrix.fromkeys)
  - [`dok_matrix.get`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.get.html#scipy.sparse.dok_matrix.get)
  - [`dok_matrix.getH`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.getH.html#scipy.sparse.dok_matrix.geth)
  - [`dok_matrix.get_shape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.get_shape.html#scipy.sparse.dok_matrix.get_shape)
  - [`dok_matrix.getcol`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.getcol.html#scipy.sparse.dok_matrix.getcol)
  - [`dok_matrix.getformat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.getformat.html#scipy.sparse.dok_matrix.getformat)
  - [`dok_matrix.getmaxprint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.getmaxprint.html#scipy.sparse.dok_matrix.getmaxprint)
  - [`dok_matrix.getnnz`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.getnnz.html#scipy.sparse.dok_matrix.getnnz)
  - [`dok_matrix.getrow`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.getrow.html#scipy.sparse.dok_matrix.getrow)
  - [`dok_matrix.items`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.items.html#scipy.sparse.dok_matrix.items)
  - [`dok_matrix.keys`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.keys.html#scipy.sparse.dok_matrix.keys)
  - [`dok_matrix.maximum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.maximum.html#scipy.sparse.dok_matrix.maximum)
  - [`dok_matrix.mean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.mean.html#scipy.sparse.dok_matrix.mean)
  - [`dok_matrix.minimum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.minimum.html#scipy.sparse.dok_matrix.minimum)
  - [`dok_matrix.multiply`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.multiply.html#scipy.sparse.dok_matrix.multiply)
  - [`dok_matrix.nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.nonzero.html#scipy.sparse.dok_matrix.nonzero)
  - [`dok_matrix.pop`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.pop.html#scipy.sparse.dok_matrix.pop)
  - [`dok_matrix.popitem`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.popitem.html#scipy.sparse.dok_matrix.popitem)
  - [`dok_matrix.power`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.power.html#scipy.sparse.dok_matrix.power)
  - [`dok_matrix.reshape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.reshape.html#scipy.sparse.dok_matrix.reshape)
  - [`dok_matrix.resize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.resize.html#scipy.sparse.dok_matrix.resize)
  - [`dok_matrix.set_shape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.set_shape.html#scipy.sparse.dok_matrix.set_shape)
  - [`dok_matrix.setdefault`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.setdefault.html#scipy.sparse.dok_matrix.setdefault)
  - [`dok_matrix.setdiag`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.setdiag.html#scipy.sparse.dok_matrix.setdiag)
  - [`dok_matrix.sum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.sum.html#scipy.sparse.dok_matrix.sum)
  - [`dok_matrix.toarray`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.toarray.html#scipy.sparse.dok_matrix.toarray)
  - [`dok_matrix.tobsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.tobsr.html#scipy.sparse.dok_matrix.tobsr)
  - [`dok_matrix.tocoo`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.tocoo.html#scipy.sparse.dok_matrix.tocoo)
  - [`dok_matrix.tocsc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.tocsc.html#scipy.sparse.dok_matrix.tocsc)
  - [`dok_matrix.tocsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.tocsr.html#scipy.sparse.dok_matrix.tocsr)
  - [`dok_matrix.todense`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.todense.html#scipy.sparse.dok_matrix.todense)
  - [`dok_matrix.todia`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.todia.html#scipy.sparse.dok_matrix.todia)
  - [`dok_matrix.todok`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.todok.html#scipy.sparse.dok_matrix.todok)
  - [`dok_matrix.tolil`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.tolil.html#scipy.sparse.dok_matrix.tolil)
  - [`dok_matrix.trace`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.trace.html#scipy.sparse.dok_matrix.trace)
  - [`dok_matrix.transpose`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.transpose.html#scipy.sparse.dok_matrix.transpose)
  - [`dok_matrix.update`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.update.html#scipy.sparse.dok_matrix.update)
  - [`dok_matrix.values`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.values.html#scipy.sparse.dok_matrix.values)
- [`lil_array`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.html#scipy.sparse.lil_array)
  - [`lil_array.asformat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.asformat.html#scipy.sparse.lil_array.asformat)
  - [`lil_array.astype`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.astype.html#scipy.sparse.lil_array.astype)
  - [`lil_array.conj`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.conj.html#scipy.sparse.lil_array.conj)
  - [`lil_array.conjugate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.conjugate.html#scipy.sparse.lil_array.conjugate)
  - [`lil_array.copy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.copy.html#scipy.sparse.lil_array.copy)
  - [`lil_array.count_nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.count_nonzero.html#scipy.sparse.lil_array.count_nonzero)
  - [`lil_array.diagonal`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.diagonal.html#scipy.sparse.lil_array.diagonal)
  - [`lil_array.dot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.dot.html#scipy.sparse.lil_array.dot)
  - [`lil_array.getrow`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.getrow.html#scipy.sparse.lil_array.getrow)
  - [`lil_array.getrowview`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.getrowview.html#scipy.sparse.lil_array.getrowview)
  - [`lil_array.maximum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.maximum.html#scipy.sparse.lil_array.maximum)
  - [`lil_array.mean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.mean.html#scipy.sparse.lil_array.mean)
  - [`lil_array.minimum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.minimum.html#scipy.sparse.lil_array.minimum)
  - [`lil_array.multiply`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.multiply.html#scipy.sparse.lil_array.multiply)
  - [`lil_array.nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.nonzero.html#scipy.sparse.lil_array.nonzero)
  - [`lil_array.power`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.power.html#scipy.sparse.lil_array.power)
  - [`lil_array.reshape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.reshape.html#scipy.sparse.lil_array.reshape)
  - [`lil_array.resize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.resize.html#scipy.sparse.lil_array.resize)
  - [`lil_array.setdiag`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.setdiag.html#scipy.sparse.lil_array.setdiag)
  - [`lil_array.sum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.sum.html#scipy.sparse.lil_array.sum)
  - [`lil_array.toarray`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.toarray.html#scipy.sparse.lil_array.toarray)
  - [`lil_array.tobsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.tobsr.html#scipy.sparse.lil_array.tobsr)
  - [`lil_array.tocoo`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.tocoo.html#scipy.sparse.lil_array.tocoo)
  - [`lil_array.tocsc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.tocsc.html#scipy.sparse.lil_array.tocsc)
  - [`lil_array.tocsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.tocsr.html#scipy.sparse.lil_array.tocsr)
  - [`lil_array.todense`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.todense.html#scipy.sparse.lil_array.todense)
  - [`lil_array.todia`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.todia.html#scipy.sparse.lil_array.todia)
  - [`lil_array.todok`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.todok.html#scipy.sparse.lil_array.todok)
  - [`lil_array.tolil`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.tolil.html#scipy.sparse.lil_array.tolil)
  - [`lil_array.trace`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.trace.html#scipy.sparse.lil_array.trace)
  - [`lil_array.transpose`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.transpose.html#scipy.sparse.lil_array.transpose)
- [`lil_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.html#scipy.sparse.lil_matrix)
  - [`lil_matrix.asformat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.asformat.html#scipy.sparse.lil_matrix.asformat)
  - [`lil_matrix.asfptype`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.asfptype.html#scipy.sparse.lil_matrix.asfptype)
  - [`lil_matrix.astype`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.astype.html#scipy.sparse.lil_matrix.astype)
  - [`lil_matrix.conj`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.conj.html#scipy.sparse.lil_matrix.conj)
  - [`lil_matrix.conjugate`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.conjugate.html#scipy.sparse.lil_matrix.conjugate)
  - [`lil_matrix.copy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.copy.html#scipy.sparse.lil_matrix.copy)
  - [`lil_matrix.count_nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.count_nonzero.html#scipy.sparse.lil_matrix.count_nonzero)
  - [`lil_matrix.diagonal`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.diagonal.html#scipy.sparse.lil_matrix.diagonal)
  - [`lil_matrix.dot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.dot.html#scipy.sparse.lil_matrix.dot)
  - [`lil_matrix.getH`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.getH.html#scipy.sparse.lil_matrix.geth)
  - [`lil_matrix.get_shape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.get_shape.html#scipy.sparse.lil_matrix.get_shape)
  - [`lil_matrix.getcol`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.getcol.html#scipy.sparse.lil_matrix.getcol)
  - [`lil_matrix.getformat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.getformat.html#scipy.sparse.lil_matrix.getformat)
  - [`lil_matrix.getmaxprint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.getmaxprint.html#scipy.sparse.lil_matrix.getmaxprint)
  - [`lil_matrix.getnnz`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.getnnz.html#scipy.sparse.lil_matrix.getnnz)
  - [`lil_matrix.getrow`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.getrow.html#scipy.sparse.lil_matrix.getrow)
  - [`lil_matrix.getrowview`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.getrowview.html#scipy.sparse.lil_matrix.getrowview)
  - [`lil_matrix.maximum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.maximum.html#scipy.sparse.lil_matrix.maximum)
  - [`lil_matrix.mean`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.mean.html#scipy.sparse.lil_matrix.mean)
  - [`lil_matrix.minimum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.minimum.html#scipy.sparse.lil_matrix.minimum)
  - [`lil_matrix.multiply`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.multiply.html#scipy.sparse.lil_matrix.multiply)
  - [`lil_matrix.nonzero`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.nonzero.html#scipy.sparse.lil_matrix.nonzero)
  - [`lil_matrix.power`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.power.html#scipy.sparse.lil_matrix.power)
  - [`lil_matrix.reshape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.reshape.html#scipy.sparse.lil_matrix.reshape)
  - [`lil_matrix.resize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.resize.html#scipy.sparse.lil_matrix.resize)
  - [`lil_matrix.set_shape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.set_shape.html#scipy.sparse.lil_matrix.set_shape)
  - [`lil_matrix.setdiag`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.setdiag.html#scipy.sparse.lil_matrix.setdiag)
  - [`lil_matrix.sum`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.sum.html#scipy.sparse.lil_matrix.sum)
  - [`lil_matrix.toarray`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.toarray.html#scipy.sparse.lil_matrix.toarray)
  - [`lil_matrix.tobsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.tobsr.html#scipy.sparse.lil_matrix.tobsr)
  - [`lil_matrix.tocoo`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.tocoo.html#scipy.sparse.lil_matrix.tocoo)
  - [`lil_matrix.tocsc`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.tocsc.html#scipy.sparse.lil_matrix.tocsc)
  - [`lil_matrix.tocsr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.tocsr.html#scipy.sparse.lil_matrix.tocsr)
  - [`lil_matrix.todense`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.todense.html#scipy.sparse.lil_matrix.todense)
  - [`lil_matrix.todia`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.todia.html#scipy.sparse.lil_matrix.todia)
  - [`lil_matrix.todok`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.todok.html#scipy.sparse.lil_matrix.todok)
  - [`lil_matrix.tolil`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.tolil.html#scipy.sparse.lil_matrix.tolil)
  - [`lil_matrix.trace`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.trace.html#scipy.sparse.lil_matrix.trace)
  - [`lil_matrix.transpose`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.transpose.html#scipy.sparse.lil_matrix.transpose)
- [`sparray`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.sparray.html#scipy.sparse.sparray)
- [`spmatrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.spmatrix.html#scipy.sparse.spmatrix)
  - [`spmatrix.asfptype`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.spmatrix.asfptype.html#scipy.sparse.spmatrix.asfptype)
  - [`spmatrix.getH`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.spmatrix.getH.html#scipy.sparse.spmatrix.geth)
  - [`spmatrix.get_shape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.spmatrix.get_shape.html#scipy.sparse.spmatrix.get_shape)
  - [`spmatrix.getcol`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.spmatrix.getcol.html#scipy.sparse.spmatrix.getcol)
  - [`spmatrix.getformat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.spmatrix.getformat.html#scipy.sparse.spmatrix.getformat)
  - [`spmatrix.getmaxprint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.spmatrix.getmaxprint.html#scipy.sparse.spmatrix.getmaxprint)
  - [`spmatrix.getnnz`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.spmatrix.getnnz.html#scipy.sparse.spmatrix.getnnz)
  - [`spmatrix.getrow`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.spmatrix.getrow.html#scipy.sparse.spmatrix.getrow)
  - [`spmatrix.set_shape`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.spmatrix.set_shape.html#scipy.sparse.spmatrix.set_shape)
  - [`spmatrix.todense`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.spmatrix.todense.html#scipy.sparse.spmatrix.todense)

### scipy.sparse.csgraph

[Module reference](https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html#module-scipy.sparse.csgraph)

Function entries: **25**. Callable objects: **0**. Classes: **1**.

#### Functions

- [`bellman_ford`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.bellman_ford.html#scipy.sparse.csgraph.bellman_ford)
- [`breadth_first_order`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.breadth_first_order.html#scipy.sparse.csgraph.breadth_first_order)
- [`breadth_first_tree`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.breadth_first_tree.html#scipy.sparse.csgraph.breadth_first_tree)
- [`connected_components`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.connected_components.html#scipy.sparse.csgraph.connected_components)
- [`construct_dist_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.construct_dist_matrix.html#scipy.sparse.csgraph.construct_dist_matrix)
- [`csgraph_from_dense`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.csgraph_from_dense.html#scipy.sparse.csgraph.csgraph_from_dense)
- [`csgraph_from_masked`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.csgraph_from_masked.html#scipy.sparse.csgraph.csgraph_from_masked)
- [`csgraph_masked_from_dense`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.csgraph_masked_from_dense.html#scipy.sparse.csgraph.csgraph_masked_from_dense)
- [`csgraph_to_dense`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.csgraph_to_dense.html#scipy.sparse.csgraph.csgraph_to_dense)
- [`csgraph_to_masked`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.csgraph_to_masked.html#scipy.sparse.csgraph.csgraph_to_masked)
- [`depth_first_order`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.depth_first_order.html#scipy.sparse.csgraph.depth_first_order)
- [`depth_first_tree`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.depth_first_tree.html#scipy.sparse.csgraph.depth_first_tree)
- [`dijkstra`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.dijkstra.html#scipy.sparse.csgraph.dijkstra)
- [`floyd_warshall`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.floyd_warshall.html#scipy.sparse.csgraph.floyd_warshall)
- [`johnson`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.johnson.html#scipy.sparse.csgraph.johnson)
- [`laplacian`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.laplacian.html#scipy.sparse.csgraph.laplacian)
- [`maximum_bipartite_matching`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.maximum_bipartite_matching.html#scipy.sparse.csgraph.maximum_bipartite_matching)
- [`maximum_flow`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.maximum_flow.html#scipy.sparse.csgraph.maximum_flow)
- [`min_weight_full_bipartite_matching`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.min_weight_full_bipartite_matching.html#scipy.sparse.csgraph.min_weight_full_bipartite_matching)
- [`minimum_spanning_tree`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.minimum_spanning_tree.html#scipy.sparse.csgraph.minimum_spanning_tree)
- [`reconstruct_path`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.reconstruct_path.html#scipy.sparse.csgraph.reconstruct_path)
- [`reverse_cuthill_mckee`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.reverse_cuthill_mckee.html#scipy.sparse.csgraph.reverse_cuthill_mckee)
- [`shortest_path`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.shortest_path.html#scipy.sparse.csgraph.shortest_path)
- [`structural_rank`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.structural_rank.html#scipy.sparse.csgraph.structural_rank)
- [`yen`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.yen.html#scipy.sparse.csgraph.yen)

#### Classes and their public methods

- [`NegativeCycleError`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.NegativeCycleError.html#scipy.sparse.csgraph.negativecycleerror)
  - [`NegativeCycleError.add_note`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.NegativeCycleError.html#scipy.sparse.csgraph.negativecycleerror.add_note)
  - [`NegativeCycleError.with_traceback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.NegativeCycleError.html#scipy.sparse.csgraph.negativecycleerror.with_traceback)

### scipy.sparse.linalg

[Module reference](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html#module-scipy.sparse.linalg)

Function entries: **32**. Callable objects: **0**. Classes: **6**.

#### Functions

- [`aslinearoperator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.aslinearoperator.html#scipy.sparse.linalg.aslinearoperator)
- [`bicg`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.bicg.html#scipy.sparse.linalg.bicg)
- [`bicgstab`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.bicgstab.html#scipy.sparse.linalg.bicgstab)
- [`cg`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html#scipy.sparse.linalg.cg)
- [`cgs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cgs.html#scipy.sparse.linalg.cgs)
- [`eigs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html#scipy.sparse.linalg.eigs)
- [`eigsh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html#scipy.sparse.linalg.eigsh)
- [`expm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm.html#scipy.sparse.linalg.expm)
- [`expm_multiply`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm_multiply.html#scipy.sparse.linalg.expm_multiply)
- [`factorized`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.factorized.html#scipy.sparse.linalg.factorized)
- [`funm_multiply_krylov`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.funm_multiply_krylov.html#scipy.sparse.linalg.funm_multiply_krylov)
- [`gcrotmk`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gcrotmk.html#scipy.sparse.linalg.gcrotmk)
- [`gmres`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html#scipy.sparse.linalg.gmres)
- [`inv`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.inv.html#scipy.sparse.linalg.inv)
- [`is_sptriangular`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.is_sptriangular.html#scipy.sparse.linalg.is_sptriangular)
- [`lgmres`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lgmres.html#scipy.sparse.linalg.lgmres)
- [`lobpcg`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lobpcg.html#scipy.sparse.linalg.lobpcg)
- [`lsmr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html#scipy.sparse.linalg.lsmr)
- [`lsqr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html#scipy.sparse.linalg.lsqr)
- [`matrix_power`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.matrix_power.html#scipy.sparse.linalg.matrix_power)
- [`minres`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.minres.html#scipy.sparse.linalg.minres)
- [`norm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.norm.html#scipy.sparse.linalg.norm)
- [`onenormest`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.onenormest.html#scipy.sparse.linalg.onenormest)
- [`qmr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.qmr.html#scipy.sparse.linalg.qmr)
- [`spbandwidth`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spbandwidth.html#scipy.sparse.linalg.spbandwidth)
- [`spilu`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spilu.html#scipy.sparse.linalg.spilu)
- [`splu`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html#scipy.sparse.linalg.splu)
- [`spsolve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html#scipy.sparse.linalg.spsolve)
- [`spsolve_triangular`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve_triangular.html#scipy.sparse.linalg.spsolve_triangular)
- [`svds`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html#scipy.sparse.linalg.svds)
- [`tfqmr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.tfqmr.html#scipy.sparse.linalg.tfqmr)
- [`use_solver`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.use_solver.html#scipy.sparse.linalg.use_solver)

#### Classes and their public methods

- [`ArpackError`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.ArpackError.html#scipy.sparse.linalg.arpackerror)
  - [`ArpackError.add_note`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.ArpackError.html#scipy.sparse.linalg.arpackerror.add_note)
  - [`ArpackError.with_traceback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.ArpackError.html#scipy.sparse.linalg.arpackerror.with_traceback)
- [`ArpackNoConvergence`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.ArpackNoConvergence.html#scipy.sparse.linalg.arpacknoconvergence)
  - [`ArpackNoConvergence.add_note`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.ArpackNoConvergence.html#scipy.sparse.linalg.arpacknoconvergence.add_note)
  - [`ArpackNoConvergence.with_traceback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.ArpackNoConvergence.html#scipy.sparse.linalg.arpacknoconvergence.with_traceback)
- [`LaplacianNd`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LaplacianNd.html#scipy.sparse.linalg.laplaciannd)
  - [`LaplacianNd.adjoint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LaplacianNd.adjoint.html#scipy.sparse.linalg.laplaciannd.adjoint)
  - [`LaplacianNd.dot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LaplacianNd.dot.html#scipy.sparse.linalg.laplaciannd.dot)
  - [`LaplacianNd.eigenvalues`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LaplacianNd.eigenvalues.html#scipy.sparse.linalg.laplaciannd.eigenvalues)
  - [`LaplacianNd.eigenvectors`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LaplacianNd.eigenvectors.html#scipy.sparse.linalg.laplaciannd.eigenvectors)
  - [`LaplacianNd.matmat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LaplacianNd.matmat.html#scipy.sparse.linalg.laplaciannd.matmat)
  - [`LaplacianNd.matvec`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LaplacianNd.matvec.html#scipy.sparse.linalg.laplaciannd.matvec)
  - [`LaplacianNd.rdot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LaplacianNd.rdot.html#scipy.sparse.linalg.laplaciannd.rdot)
  - [`LaplacianNd.rmatmat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LaplacianNd.rmatmat.html#scipy.sparse.linalg.laplaciannd.rmatmat)
  - [`LaplacianNd.rmatvec`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LaplacianNd.rmatvec.html#scipy.sparse.linalg.laplaciannd.rmatvec)
  - [`LaplacianNd.toarray`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LaplacianNd.toarray.html#scipy.sparse.linalg.laplaciannd.toarray)
  - [`LaplacianNd.tosparse`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LaplacianNd.tosparse.html#scipy.sparse.linalg.laplaciannd.tosparse)
  - [`LaplacianNd.transpose`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LaplacianNd.transpose.html#scipy.sparse.linalg.laplaciannd.transpose)
- [`LinearOperator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html#scipy.sparse.linalg.linearoperator)
  - [`LinearOperator.adjoint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.adjoint.html#scipy.sparse.linalg.linearoperator.adjoint)
  - [`LinearOperator.dot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.dot.html#scipy.sparse.linalg.linearoperator.dot)
  - [`LinearOperator.matmat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.matmat.html#scipy.sparse.linalg.linearoperator.matmat)
  - [`LinearOperator.matvec`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.matvec.html#scipy.sparse.linalg.linearoperator.matvec)
  - [`LinearOperator.rdot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.rdot.html#scipy.sparse.linalg.linearoperator.rdot)
  - [`LinearOperator.rmatmat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.rmatmat.html#scipy.sparse.linalg.linearoperator.rmatmat)
  - [`LinearOperator.rmatvec`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.rmatvec.html#scipy.sparse.linalg.linearoperator.rmatvec)
  - [`LinearOperator.transpose`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.transpose.html#scipy.sparse.linalg.linearoperator.transpose)
- [`MatrixRankWarning`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.MatrixRankWarning.html#scipy.sparse.linalg.matrixrankwarning)
  - [`MatrixRankWarning.add_note`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.MatrixRankWarning.html#scipy.sparse.linalg.matrixrankwarning.add_note)
  - [`MatrixRankWarning.with_traceback`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.MatrixRankWarning.html#scipy.sparse.linalg.matrixrankwarning.with_traceback)
- [`SuperLU`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.SuperLU.html#scipy.sparse.linalg.superlu)
  - [`SuperLU.solve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.SuperLU.solve.html#scipy.sparse.linalg.superlu.solve)

## Additional and compatibility namespaces

Many of these paths are deprecated compatibility aliases. Class and object method lists are omitted in these sections to avoid repeating the public API inventory.

### scipy.sparse.base

Function entries: **2**. Callable objects: **0**. Classes: **3**.

#### Functions

- `issparse`
- `isspmatrix`

#### Classes

- `SparseEfficiencyWarning`
- `SparseWarning`
- `spmatrix`

### scipy.sparse.bsr

Function entries: **1**. Callable objects: **0**. Classes: **2**.

#### Functions

- `isspmatrix_bsr`

#### Classes

- `bsr_matrix`
- `spmatrix`

### scipy.sparse.compressed

Function entries: **0**. Callable objects: **0**. Classes: **1**.

#### Classes

- `SparseEfficiencyWarning`

### scipy.sparse.construct

Function entries: **14**. Callable objects: **0**. Classes: **5**.

#### Functions

- `block_diag`
- `bmat`
- `diags`
- `eye`
- `get_index_dtype`
- `hstack`
- `identity`
- `issparse`
- `kron`
- `kronsum`
- `rand`
- `random`
- `spdiags`
- `vstack`

#### Classes

- `bsr_matrix`
- `coo_matrix`
- `csc_matrix`
- `csr_matrix`
- `dia_matrix`

### scipy.sparse.coo

Function entries: **1**. Callable objects: **0**. Classes: **3**.

#### Functions

- `isspmatrix_coo`

#### Classes

- `SparseEfficiencyWarning`
- `coo_matrix`
- `spmatrix`

### scipy.sparse.csc

Function entries: **1**. Callable objects: **0**. Classes: **2**.

#### Functions

- `isspmatrix_csc`

#### Classes

- `csc_matrix`
- `spmatrix`

### scipy.sparse.csr

Function entries: **1**. Callable objects: **0**. Classes: **2**.

#### Functions

- `isspmatrix_csr`

#### Classes

- `csr_matrix`
- `spmatrix`

### scipy.sparse.data

Function entries: **0**. Callable objects: **0**. Classes: **0**.

This namespace contains submodules or has no exported callable entries in this runtime.

### scipy.sparse.dia

Function entries: **1**. Callable objects: **0**. Classes: **2**.

#### Functions

- `isspmatrix_dia`

#### Classes

- `dia_matrix`
- `spmatrix`

### scipy.sparse.dok

Function entries: **1**. Callable objects: **0**. Classes: **2**.

#### Functions

- `isspmatrix_dok`

#### Classes

- `dok_matrix`
- `spmatrix`

### scipy.sparse.extract

Function entries: **3**. Callable objects: **0**. Classes: **1**.

#### Functions

- `find`
- `tril`
- `triu`

#### Classes

- `coo_matrix`

### scipy.sparse.lil

Function entries: **1**. Callable objects: **0**. Classes: **2**.

#### Functions

- `isspmatrix_lil`

#### Classes

- `lil_array`
- `lil_matrix`

### scipy.sparse.linalg.dsolve

Function entries: **7**. Callable objects: **0**. Classes: **2**.

#### Functions

- `factorized`
- `spilu`
- `splu`
- `spsolve`
- `spsolve_triangular`
- `test`
- `use_solver`

#### Classes

- `MatrixRankWarning`
- `SuperLU`

### scipy.sparse.linalg.eigen

Function entries: **5**. Callable objects: **0**. Classes: **2**.

#### Functions

- `eigs`
- `eigsh`
- `lobpcg`
- `svds`
- `test`

#### Classes

- `ArpackError`
- `ArpackNoConvergence`

### scipy.sparse.linalg.interface

Function entries: **1**. Callable objects: **0**. Classes: **1**.

#### Functions

- `aslinearoperator`

#### Classes

- `LinearOperator`

### scipy.sparse.linalg.isolve

Function entries: **13**. Callable objects: **0**. Classes: **0**.

#### Functions

- `bicg`
- `bicgstab`
- `cg`
- `cgs`
- `gcrotmk`
- `gmres`
- `lgmres`
- `lsmr`
- `lsqr`
- `minres`
- `qmr`
- `test`
- `tfqmr`

### scipy.sparse.linalg.matfuncs

Function entries: **3**. Callable objects: **0**. Classes: **1**.

#### Functions

- `expm`
- `inv`
- `spsolve`

#### Classes

- `LinearOperator`

### scipy.sparse.sparsetools

Function entries: **0**. Callable objects: **0**. Classes: **0**.

This namespace contains submodules or has no exported callable entries in this runtime.

### scipy.sparse.spfuncs

Function entries: **0**. Callable objects: **0**. Classes: **0**.

This namespace contains submodules or has no exported callable entries in this runtime.

### scipy.sparse.sputils

Function entries: **0**. Callable objects: **0**. Classes: **0**.

This namespace contains submodules or has no exported callable entries in this runtime.
