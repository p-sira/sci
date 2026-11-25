# Scientific I/O coverage

With facade defaults disabled, formats are selected independently by `io-matrix-market`, `io-wav`, and `io-matlab`; `io` alone exposes shared I/O types. Domain features remain enabled by default. Matrix Market does not enable sparse NPZ; MATLAB and NPZ share flate2’s Rust/miniz backend. Supported functions, numerical conventions, and reference fixtures are unchanged.

Scope: Matrix Market, WAV, and a dense numeric MATLAB subset. The source
[inventory](scipy_function_list/scipy.io.md) lists 28 qualified function entries
across 18 namespaces, including deprecated aliases. The public namespaces contain
17 qualified entries and 14 distinct function names. There are no Cython-only
I/O functions in this inventory.

## Inventory disposition

| SciPy function | Sci API | Status and adaptation |
| --- | --- | --- |
| `scipy.io.mminfo` | `io::mminfo` | Supported header subset; named fields replace the six-element tuple |
| `scipy.io.mmread` | `io::mmread` | Dense real/integer and real/integer/pattern CSR; real symmetry variants; complex excluded |
| `scipy.io.mmwrite` | `io::mmwrite` | Explicit dense/CSR input; writes real/general, without symmetry inference or Python options |
| `scipy.io.loadmat` | `io::loadmat` / `io::matlab::loadmat` | Dense numeric Level-5 subset; typed arrays and map, no synthetic metadata variables |
| `scipy.io.savemat` | `io::savemat` / `io::matlab::savemat` | Same subset, deterministic little-endian output; explicit shape and compression |
| `scipy.io.whosmat` | `io::whosmat` / `io::matlab::whosmat` | Same subset; metadata in file order, with an extra complexity flag; validates full file |
| `scipy.io.matlab.matfile_version` | `io::matlab::matfile_version` | Header detection for supported Level-5 version `(1,0)`; other versions return errors |
| `scipy.io.wavfile.read` | `io::wavfile::read` | Typed audio with rate, channels and bit depth; PCM8/16/24/32 and float32/64 |
| `scipy.io.wavfile.write` | `io::wavfile::write` | Same subset, explicit bit depth; seekable destination at position zero |
| `scipy.io.hb_read`, `hb_write` | — | Deferred Harwell–Boeing format |
| `scipy.io.readsav` | — | Deferred IDL format |
| `scipy.io.arff.loadarff` | — | Deferred ARFF format |
| `scipy.io.matlab.varmats_from_mat` | — | Deferred variable-level MAT container splitting |

Nine distinct inventory function names have tested implementations within the
stated subsets; five are deferred. The three public MATLAB aliases reuse the
same functions, giving mappings for 12 of 17 qualified public entries. This is
not full SciPy behavioral parity or full `scipy.io` coverage.

Deprecated `arffread`, `harwell_boeing`, `idl`, `matlab.mio*`, `mmio`, and `netcdf`
paths are omitted. Empty compatibility namespaces are not counted as work.
SciPy classes and Python/NumPy inherited methods are not reproduced. NetCDF
file/variable objects and Fortran records are deferred; errors use Rust enums
and retained stream sources. `NumericData`, `Audio`, `MatArray`, `MatVariableInfo`,
`MarketMatrix`, and conversion helpers are Rust adaptations, not extra inventory
coverage. NPZ remains in `sci::sparse`, outside the SciPy I/O inventory.

## Validation

Input-only cases are in `tests/scipy/io_cases.json`. The pinned generator
`tests/scipy/generate_io.py` uses SciPy 1.18.1 and NumPy 2.5.4 to generate
`io_inputs.csv` byte streams and separate reference CSVs. Fixtures include 11
Matrix Market, eight WAV and 68 MATLAB cases. MATLAB tests cover both endian
orders with and without compression, all ten numeric storage types, complex
components, signed zeros, floating subnormals, empty shapes and multidimensional
ordering. Additional properties check mixed-type variables and metadata file order. Big-endian fixtures re-encode
SciPy-generated words using the documented Level-5 layout, and SciPy reads them
back before they are committed.

`tests/scipy/audit_io.py` checks the inventory disposition, exported definitions,
and public test calls. `tests/io.rs` exercises every exported inventory function through `sci::io`,
checks dtype/shape/metadata and exact integer values, tests special floating
values and signed zero, validates invalid domains and truncation, checks stream
error sources, and round-trips all supported types. It also checks WAV signal
processing and shared matrix conversions. Negative tests include malformed
storage, unsupported features, declared lengths, coordinate bounds, duplicate
names, input/work limits and invalid public records.

`tests/scipy/check_io_writes.py` independently reads all Rust-written files using
SciPy and compares against the committed reference CSVs. Rust tests remain
independent of Python; this additional check runs after `cargo test --test io`.
Exact numerical comparisons apply: typed I/O has no approximation tolerance.
NaN payload bits are not part of the contract. Conversion to f64 may round
large integers as documented; PCM power-of-two normalization is exact.

SciPy 1.18.1's default C++ Matrix Market reader raises SIGFPE on `(0,3)` dense
array storage, and its retained Python reader rejects that boundary. The fixture
uses the unique empty `(0,3)` reference produced by SciPy sparse storage; the
writer check validates its header and empty payload without invoking the faulty
reader. SciPy's text reader canonicalizes `-0.0` to `+0.0`; Rust explicitly tests
preservation against the source token, while all other values use SciPy results.

## Implementation boundaries

Read the [crate README](../crates/sci-io/README.md) and API documentation for full
subsets, normalization, endian/order conventions, limits and errors. WAV PCM/float32 decoding
uses private Hound; float64 uses native byte encoding. MATLAB metadata and
numeric decoding share one bounded native parser, with zlib decompression
through flate2. It decodes both endian orders and small tags directly, retaining
all scalar bits without an intermediate normalized file. Matrix Market and
deterministic MAT numeric encoding use straightforward format serialization.
