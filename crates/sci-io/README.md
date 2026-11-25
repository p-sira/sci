# sci-io

The Sci facade exposes each format through `io-matrix-market`, `io-wav`, and `io-matlab`. With facade defaults disabled, `io` alone provides shared I/O types without codecs. Matrix Market avoids sparse NPZ dependencies. MATLAB and NPZ share the Rust/miniz compression backend.

Scientific file interchange with typed values, explicit shapes, bounded parsing,
and conversions into Sci's shared matrix and signal types.

```rust
use sci_io::matrix_market::{mmread, MarketMatrix};

let bytes = b"%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 2\n2 2 3\n";
let MarketMatrix::Sparse(a) = mmread(&bytes[..])? else { unreachable!() };
assert_eq!(a.data, vec![2.0, 3.0]);
# Ok::<(), sci_io::Error>(())
```

## Formats and features

All three format features are enabled by default. Select individual formats with
`default-features = false` and `features = ["matrix-market"]`, `["wav"]`, or
`["matlab"]`. The Sci facade exposes the crate through `sci::io` with feature `io`.
Format dependencies and backend types remain private. Matrix Market uses
the shared `sci-core` CSR support without depending on `sci-sparse`.

| Feature | APIs | Supported behavior |
| --- | --- | --- |
| `matrix-market` | `matrix_market::{mminfo, mmread, mmwrite}` | Real/integer/pattern fields, array and coordinate storage, general/symmetric/skew/real-Hermitian input; real/general output |
| `wav` | `wavfile::{read, write, signal}` | Little-endian RIFF, packed PCM8/16/24/32, IEEE float32/64, channels and sample rate |
| `matlab` | `matlab::{loadmat, savemat, whosmat, matfile_version}` | Dense numeric Level-5 containers, both endian orders, f32/f64 and all 8/16/32/64-bit integer types, floating complex data, optional zlib compression |

Readers consume Rust `Read` streams. WAV writers also require `Seek` and an empty
destination positioned at byte zero. Callers open files and flush buffers; no
extensions are inferred. Files can contain floating NaN/infinities/signed zero,
although numerical consumers can reject these. Errors preserve underlying stream
errors and distinguish malformed data, unsupported features and codec failures.

## Types and conversions

`NumericData` retains scalar types. Calling `to_f64` explicitly converts scalar
values; 64-bit integers beyond ±2^53 can round.

`matrix_market::MarketMatrix` carries row-major dense values with explicit dimensions or shared
`SparseMatrix` CSR. Coordinate duplicates are summed by shared CSR helpers. Dense input
preserves signed zero; SciPy's text reader can canonicalize it to positive zero.
The real/integer CSR interface rejects nonfinite entries; integer Matrix Market
fields outside ±2^53 are rejected. `to_matrix` returns shared matrix rows.

`wavfile::Audio` preserves the rate in Hz, channels, bit depth and interleaved
samples. PCM24 is left-aligned in i32, matching SciPy. `wavfile::signal` extracts
one channel into normalized f64 samples for signal processing, with explicit
power-of-two PCM scaling. Floating samples retain their original amplitudes.

`matlab::MatArray` retains explicit rank 2..32 shape, column-major data, and
optional imaginary data. `to_matrix` converts real rank-two arrays into shared
f64 rows. Both matrix conversions reject shape `(0,n)` when `n>0`, since row
storage cannot represent its column count. File records themselves retain it.

## Limits and unsupported behavior

Inputs and expanded MATLAB data are bounded to 64 MiB. Arrays, dense matrices,
and audio are bounded to 1,000,000 scalar values; sparse coordinate expansion is
also bounded to 1,000,000 entries. MATLAB bounds apply across all real and
imaginary components, with at most 1,024 variables. Axes are at most 1,000,000;
every dimension prefix product must also remain within the scalar limit.

Matrix Market complex fields, automatic symmetry inference, integer-preserving
Market storage, and built-in gzip handling are unsupported. Callers may supply
an independently decompressed reader, subject to the decoded input limit.
WAV RF64, big-endian RIFX, compressed codecs, memory mapping, packed
nonstandard bit depths, and padded sample containers are unsupported. WAV
metadata chunks are validated structurally but not retained. Float64 uses the
standard IEEE format chunk (16 or 18 bytes); extensible float64 is unsupported.
MATLAB v4, v7.3/HDF5, sparse, logical, string/character, cell, struct, object,
global/subsystem data, compact integer encodings of floating arrays, and complex
integer arrays are rejected. Names must be ASCII identifiers of 1..63 bytes;
duplicate names are rejected. Loading never silently drops variables. Variable
inspection validates the entire supported payload and preserves original file
order; writing sorts names and uses a deterministic header. Python dictionaries,
metadata keys, squeezing, dtype conversion options, and path conveniences are
replaced by explicit Rust records and byte streams.

Fortran records, NetCDF, IDL, Harwell–Boeing, ARFF and MATLAB variable splitting
are outside this release's scope. See the repository's I/O coverage report.
