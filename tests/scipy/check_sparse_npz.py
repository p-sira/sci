# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Verify the Rust-written NPZ with SciPy after cargo test --test sparse_parity."""
from pathlib import Path
import numpy as np
import scipy
from scipy import sparse
assert (np.__version__,scipy.__version__)==('2.5.4','1.18.1')
root=Path(__file__).resolve().parents[2]
a=sparse.load_npz(root/'target/sparse-rust.npz')
b=sparse.load_npz(root/'tests/fixtures/save_npz_0.npz')
np.testing.assert_array_equal(a.toarray(),b.toarray())
assert a.format=='csr' and isinstance(a,sparse.csr_array)
print('SciPy successfully loaded and verified the Rust NPZ archive.')
