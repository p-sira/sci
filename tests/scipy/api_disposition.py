"""Intentional SciPy inventory dispositions for the Rust public API.

Fixture labels keep SciPy spellings when the canonical Rust API reuses their
independent numerical references. Exclusions are never counted as exports.
"""
ALIASES = {
    'linalg': {'solve_lyapunov': 'solve_continuous_lyapunov'},
    'cluster': {'py_vq': 'vq', 'maxRstat': 'max_rstat'},
    'interpolate': {'splrep': 'make_splrep', 'splprep': 'make_splprep'},
    'signal': {'freqresp': 'freqs', 'dfreqresp': 'freqz', 'sosfreqz': 'freqz_sos',
               'check_COLA': 'check_cola', 'check_NOLA': 'check_nola'},
    'sparse': {'bmat': 'block_array', 'diags_array': 'diags', 'eye_array': 'eye',
               'matrix_transpose': 'transpose', 'factorized': 'splu'},
    'special': {
        'c_roots': 'roots_chebyc', 'cg_roots': 'roots_gegenbauer',
        'h_roots': 'roots_hermite', 'he_roots': 'roots_hermitenorm',
        'j_roots': 'roots_jacobi', 'js_roots': 'roots_sh_jacobi',
        'l_roots': 'roots_laguerre', 'la_roots': 'roots_genlaguerre',
        'p_roots': 'roots_legendre', 'ps_roots': 'roots_sh_legendre',
        's_roots': 'roots_chebys', 't_roots': 'roots_chebyt',
        'ts_roots': 'roots_sh_chebyt', 'u_roots': 'roots_chebyu',
        'us_roots': 'roots_sh_chebyu',
    },
}
EXCLUDED = {
    'linalg': {'find_best_blas_type', 'get_blas_funcs', 'get_lapack_funcs'},
    'special': {'geterr', 'seterr', 'cbrt', 'exp2', 'exp10', 'expm1', 'log1p', 'round'},
    'stats': {'abs', 'exp', 'log'},
    'sparse': {'issparse', 'get_index_dtype', 'safely_cast_index_arrays', 'aslinearoperator'},
}
