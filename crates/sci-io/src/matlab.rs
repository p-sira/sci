//! Dense numeric MATLAB Level-5 interchange (MATLAB v6/v7–7.2 containers).
//!
//! Preserves f32/f64 and signed/unsigned 8/16/32/64-bit integer storage, shape,
//! column-major order, and split floating complex components. Compressed input
//! is decoded under a cumulative 64 MiB bound before the private parser runs.
//! Writers produce deterministic little-endian Level-5 data, optionally zlib
//! compressed. Character/logical/cell/struct/object/sparse arrays, v4, v7.3/HDF5,
//! global variables, and subsystem data are rejected. No data is silently skipped.
//! Numeric payload types must match their MATLAB class (no compact integer
//! encoding of floating arrays). Names are ASCII MATLAB identifiers of ≤63 bytes.
use crate::{BYTES, Error, Matrix, NumericData, Result, VALUES, bounded_read, volume};
use flate2::{Compression, bufread::ZlibDecoder, write::ZlibEncoder};
use std::{
    collections::BTreeMap,
    io::{Read, Write},
};

/// A named collection of MATLAB arrays, ordered lexicographically when written.
pub type MatFile = BTreeMap<String, MatArray>;

/// Typed MATLAB numerical data in column-major order (first axis varies fastest).
#[derive(Clone, Debug, PartialEq)]
pub struct MatArray {
    /// Explicit rank 2..=32 shape, including unit and empty axes; never squeezed.
    pub shape: Vec<usize>,
    /// Exactly the product of `shape` scalar values, at most 1,000,000.
    pub real: NumericData,
    /// Optional imaginary component, matching the real length and f32/f64 type.
    pub imag: Option<NumericData>,
}

/// MATLAB variable metadata, in original file order when read by [`whosmat`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatVariableInfo {
    /// ASCII MATLAB variable name.
    pub name: String,
    /// Original shape; no squeezing or axis reordering.
    pub shape: Vec<usize>,
    /// MATLAB class: `double`, `single`, `int8`…`uint64`.
    pub class: &'static str,
    /// Whether an imaginary component is stored.
    pub complex: bool,
}
fn invalid() -> Error {
    Error::InvalidData("malformed MATLAB Level-5 data")
}
fn name_valid(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 63
        && s.as_bytes()[0].is_ascii_alphabetic()
        && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}
// (MATLAB class, mi storage type, scalar width, class name)
fn repr(d: &NumericData) -> (u32, u32, usize, &'static str) {
    match d {
        NumericData::F64(_) => (6, 9, 8, "double"),
        NumericData::F32(_) => (7, 7, 4, "single"),
        NumericData::I8(_) => (8, 1, 1, "int8"),
        NumericData::U8(_) => (9, 2, 1, "uint8"),
        NumericData::I16(_) => (10, 3, 2, "int16"),
        NumericData::U16(_) => (11, 4, 2, "uint16"),
        NumericData::I32(_) => (12, 5, 4, "int32"),
        NumericData::U32(_) => (13, 6, 4, "uint32"),
        NumericData::I64(_) => (14, 12, 8, "int64"),
        NumericData::U64(_) => (15, 13, 8, "uint64"),
    }
}
fn class_repr(c: u32) -> Result<(u32, usize, &'static str)> {
    Ok(match c {
        6 => (9, 8, "double"),
        7 => (7, 4, "single"),
        8 => (1, 1, "int8"),
        9 => (2, 1, "uint8"),
        10 => (3, 2, "int16"),
        11 => (4, 2, "uint16"),
        12 => (5, 4, "int32"),
        13 => (6, 4, "uint32"),
        14 => (12, 8, "int64"),
        15 => (13, 8, "uint64"),
        _ => {
            return Err(Error::Unsupported(
                "only dense numeric MATLAB arrays are supported",
            ));
        }
    })
}
fn validate(a: &MatArray) -> Result<usize> {
    if a.shape.len() < 2 {
        return Err(Error::InvalidData("MATLAB arrays require rank 2..32"));
    }
    let n = volume(&a.shape)?;
    if n != a.real.len() {
        return Err(Error::InvalidData(
            "MATLAB shape does not match data length",
        ));
    }
    if let Some(im) = &a.imag
        && (!matches!(&a.real, NumericData::F32(_) | NumericData::F64(_))
            || repr(im).0 != repr(&a.real).0
            || im.len() != n)
    {
        return Err(Error::InvalidData(
            "imaginary data must match the floating real component",
        ));
    }
    Ok(n * (1 + usize::from(a.imag.is_some())))
}
#[derive(Clone, Copy)]
struct Endian(bool);
impl Endian {
    fn u32(self, b: &[u8]) -> Result<u32> {
        let a = b
            .get(..4)
            .ok_or_else(invalid)?
            .try_into()
            .map_err(|_| invalid())?;
        Ok(if self.0 {
            u32::from_le_bytes(a)
        } else {
            u32::from_be_bytes(a)
        })
    }
    fn u16(self, b: &[u8]) -> u16 {
        let a = [b[0], b[1]];
        if self.0 {
            u16::from_le_bytes(a)
        } else {
            u16::from_be_bytes(a)
        }
    }
}
fn file_header(b: &[u8]) -> Result<Endian> {
    if b.len() < 128 {
        return Err(invalid());
    }
    let endian = match &b[126..128] {
        b"IM" => Endian(true),
        b"MI" => Endian(false),
        _ => return Err(invalid()),
    };
    let version = endian.u16(&b[124..126]);
    if version == 0x200 {
        return Err(Error::Unsupported("MATLAB v7.3/HDF5 is unsupported"));
    }
    if version != 0x100 {
        return Err(Error::Unsupported(
            "only MATLAB Level-5 version 1.0 is supported",
        ));
    }
    if b[116..124].iter().any(|&v| v != 0) {
        return Err(Error::Unsupported("MATLAB subsystem data is unsupported"));
    }
    Ok(endian)
}
// Split one normal or small data element. Compressed elements have no padding.
fn element<'a>(b: &mut &'a [u8], e: Endian) -> Result<(u32, &'a [u8])> {
    if b.len() < 8 {
        return Err(invalid());
    }
    let tag = e.u32(b)?;
    if tag > u32::from(u16::MAX) {
        let t = tag & 0xffff;
        let n = (tag >> 16) as usize;
        if n == 0 || n > 4 {
            return Err(invalid());
        }
        let data = &b[4..4 + n];
        *b = &b[8..];
        return Ok((t, data));
    }
    let n = e.u32(&b[4..])? as usize;
    let padded = if tag == 15 {
        n
    } else {
        n.checked_add(7).ok_or_else(invalid)? & !7
    };
    let end = 8usize
        .checked_add(padded)
        .filter(|&n| n <= b.len())
        .ok_or_else(invalid)?;
    let data = &b[8..8 + n];
    *b = &b[end..];
    Ok((tag, data))
}
fn metadata(mut b: &[u8], e: Endian, total: &mut usize) -> Result<MatVariableInfo> {
    let (t, flags) = element(&mut b, e)?;
    if t != 6 || flags.len() != 8 {
        return Err(invalid());
    }
    let flags = e.u32(flags)?;
    if flags & !0x8ff != 0 {
        return Err(Error::Unsupported(
            "logical, global, or unknown MATLAB flags are unsupported",
        ));
    }
    let class = flags & 255;
    let (storage, width, class_name) = class_repr(class)?;
    let complex = flags & 0x800 != 0;
    if complex && !matches!(class, 6 | 7) {
        return Err(Error::Unsupported(
            "complex integer MATLAB arrays are unsupported",
        ));
    }
    let (t, dims) = element(&mut b, e)?;
    if t != 5 || dims.len() % 4 != 0 || !(8..=128).contains(&dims.len()) {
        return Err(invalid());
    }
    let shape: Vec<usize> = dims
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| e.u32(v).map(|v| v as usize))
        .collect::<Result<_>>()?;
    let n = volume(&shape)?;
    *total = total
        .checked_add(n * (1 + usize::from(complex)))
        .filter(|&n| n <= VALUES)
        .ok_or(Error::InvalidData(
            "MATLAB file exceeds 1,000,000 scalar values",
        ))?;
    let (t, name) = element(&mut b, e)?;
    if t != 1 {
        return Err(invalid());
    }
    let name = std::str::from_utf8(name).map_err(|_| invalid())?;
    if !name_valid(name) {
        return Err(Error::Unsupported(
            "MATLAB names must be ASCII identifiers of 1..63 bytes",
        ));
    }
    for _ in 0..1 + usize::from(complex) {
        let (t, data) = element(&mut b, e)?;
        if t != storage {
            return Err(Error::Unsupported(
                "MATLAB numeric payload type must match its class",
            ));
        }
        if data.len() != n * width {
            return Err(invalid());
        }
    }
    if !b.is_empty() {
        return Err(invalid());
    }
    Ok(MatVariableInfo {
        name: name.to_string(),
        shape,
        class: class_name,
        complex,
    })
}
struct Prepared {
    bytes: Vec<u8>,
    info: Vec<MatVariableInfo>,
}
fn prepare(reader: impl Read) -> Result<Prepared> {
    let bytes = bounded_read(reader)?;
    let e = file_header(&bytes)?;
    let mut rest = &bytes[128..];
    let mut out = bytes[..128].to_vec();
    let signature = b"MATLAB 5.0 MAT-file, ";
    out[..signature.len()].copy_from_slice(signature);
    out[124..128].copy_from_slice(&[0, 1, b'I', b'M']);
    let mut info = Vec::new();
    let mut names = std::collections::BTreeSet::new();
    let mut total = 0;
    while !rest.is_empty() {
        if info.len() >= 1024 {
            return Err(Error::InvalidData("MATLAB file exceeds 1,024 variables"));
        }
        let (t, data) = element(&mut rest, e)?;
        let matrix;
        let mut decoded;
        let payload = match t {
            14 => data,
            15 => {
                let mut decoder = ZlibDecoder::new(data);
                decoded = Vec::new();
                (&mut decoder)
                    .take((BYTES - out.len() + 1) as u64)
                    .read_to_end(&mut decoded)?;
                if decoded.len() > BYTES - out.len() {
                    return Err(Error::InvalidData("MATLAB decompression exceeds 64 MiB"));
                }
                if decoder.total_in() != data.len() as u64 {
                    return Err(invalid());
                }
                let mut d = decoded.as_slice();
                let (inner, p) = element(&mut d, e)?;
                if inner != 14 || !d.is_empty() {
                    return Err(invalid());
                }
                matrix = p;
                matrix
            }
            _ => return Err(Error::Unsupported("unsupported top-level MATLAB element")),
        };
        let m = metadata(payload, e, &mut total)?;
        if !names.insert(m.name.clone()) {
            return Err(Error::InvalidData("duplicate MATLAB variable name"));
        }
        if out.len() + 8 + payload.len() + 7 > BYTES {
            return Err(Error::InvalidData("MATLAB expanded file exceeds 64 MiB"));
        }
        // Give the private backend normal tags and little-endian scalar words.
        // This also avoids backend-specific big-endian small-tag assumptions.
        let mut rest = payload;
        let mut normalized = Vec::new();
        while !rest.is_empty() {
            let (kind, data) = element(&mut rest, e)?;
            let width = match kind {
                3 | 4 => 2,
                5..=7 => 4,
                9 | 12 | 13 => 8,
                _ => 1,
            };
            let data = if e.0 {
                data.to_vec()
            } else {
                data.chunks_exact(width)
                    .flat_map(|v| v.iter().rev().copied())
                    .collect()
            };
            put_element(&mut normalized, kind, &data);
        }
        put_element(&mut out, 14, &normalized);
        info.push(m);
    }
    Ok(Prepared { bytes: out, info })
}

/// Identify a MATLAB Level-5 container's `(major, minor)` version.
///
/// # Parameters and conventions
/// `reader` is consumed for exactly the 128-byte header; numerical data is not read.
/// Both byte orders are recognized. Versions describe the container, not MATLAB releases.
/// # Returns
/// `(1,0)` for the supported Level-5 version.
/// # Errors
/// Truncated/invalid header, stream errors, v4/v7.3, or subsystem data.
/// # Examples
/// ```
/// let mut b=Vec::new();
/// sci_io::savemat(&mut b,&Default::default(),false)?;
/// assert_eq!(sci_io::matlab::matfile_version(b.as_slice())?,(1,0));
/// # Ok::<(), sci_io::Error>(())
/// ```
/// # References
/// [SciPy matfile_version](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.matlab.matfile_version.html).
pub fn matfile_version(mut reader: impl Read) -> Result<(u8, u8)> {
    let mut header = [0; 128];
    reader.read_exact(&mut header)?;
    file_header(&header)?;
    Ok((1, 0))
}

/// Inspect the names, shapes, classes, and complexity of MATLAB numeric arrays.
///
/// # Parameters and conventions
/// `reader` supplies a supported Level-5 file; see module-level restrictions.
/// Shapes retain all axes; results follow file order. Compressed data is inflated
/// and structurally checked, without allocating typed numerical arrays.
/// # Returns
/// Variable metadata, extending SciPy's `(name,shape,class)` with `complex`.
/// # Errors
/// All malformed or unsupported data is rejected, including payloads, duplicate
/// names, >64 MiB input/expanded bytes, >1,024 variables, >1,000,000 aggregate
/// real plus imaginary scalars, or rank/axis limits documented by [`MatArray`].
/// # Examples
/// ```
/// let mut b=Vec::new();sci_io::savemat(&mut b,&Default::default(),true)?;
/// assert!(sci_io::whosmat(b.as_slice())?.is_empty());
/// # Ok::<(), sci_io::Error>(())
/// ```
/// # References
/// [SciPy whosmat](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.whosmat.html).
pub fn whosmat(reader: impl Read) -> Result<Vec<MatVariableInfo>> {
    Ok(prepare(reader)?.info)
}

/// Load MATLAB numeric values `A[i₀,…,iₖ]` without dtype conversion or squeezing.
///
/// # Parameters and conventions
/// `reader` supplies Level-5 bytes; supports both endian orders and compressed
/// variables. Values use column-major index `i₀ + d₀*(i₁ + d₁*(…))`, with separate
/// real/imaginary components. No synthetic `__header__` metadata variables are added.
/// # Domain and special values
/// Dense integer/floating arrays only, as described by this module. Floating
/// NaN/infinities/signed zeros, exact 64-bit integers, unit axes and empty axes persist.
/// # Returns
/// A [`MatFile`] keyed by original names. Backend APIs remain private.
/// # Errors
/// Unsupported features and work limits in [`whosmat`], I/O errors, or backend
/// decoding failures. All variables are validated before backend decoding.
/// # Numerical behavior
/// Bounded zlib decompression and structural validation precede private numeric
/// parsing. No precision conversion, transposition, arithmetic, or silent skipping.
/// # Examples
/// ```
/// use sci_io::{NumericData, matlab::{MatArray, MatFile}};
/// let a=MatArray {shape:vec![1,2],real:NumericData::F64(vec![1.0,2.0]),imag:None};
/// let f=MatFile::from([("x".into(),a)]);
/// let mut b=Vec::new();sci_io::savemat(&mut b,&f,true)?;
/// assert_eq!(sci_io::loadmat(b.as_slice())?,f);
/// # Ok::<(), sci_io::Error>(())
/// ```
/// # References
/// [SciPy loadmat](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html).
pub fn loadmat(reader: impl Read) -> Result<MatFile> {
    let p = prepare(reader)?;
    let parsed = matrw::load_matfile_from_u8(&p.bytes).map_err(|e| Error::Codec(e.to_string()))?;
    let mut out = MatFile::new();
    fn convert(value: matrw::MatlabType) -> Result<NumericData> {
        macro_rules! numeric {
            ($($v:ident),+) => { match value {
                $(matrw::MatlabType::$v(v)=>Ok(NumericData::$v(v))),+,
                _=>Err(invalid()),
            } };
        }
        numeric!(I8, U8, I16, U16, I32, U32, I64, U64, F32, F64)
    }
    for (name, a) in parsed {
        let matrw::MatVariable::NumericArray(a) = a else {
            return Err(invalid());
        };
        let real = convert(a.value)?;
        let imag = a.value_cmp.map(convert).transpose()?;
        out.insert(
            name,
            MatArray {
                shape: a.dim,
                real,
                imag,
            },
        );
    }
    if out.len() != p.info.len() {
        return Err(invalid());
    }
    for info in p.info {
        let a = out.get(&info.name).ok_or_else(invalid)?;
        validate(a)?;
        if a.shape != info.shape
            || repr(&a.real).3 != info.class
            || a.imag.is_some() != info.complex
        {
            return Err(invalid());
        }
    }
    Ok(out)
}

fn put_element(out: &mut Vec<u8>, kind: u32, bytes: &[u8]) {
    out.extend(kind.to_le_bytes());
    out.extend((bytes.len() as u32).to_le_bytes());
    out.extend(bytes);
    if kind != 15 {
        out.resize((out.len() + 7) & !7, 0);
    }
}
fn numeric_bytes(d: &NumericData) -> Vec<u8> {
    macro_rules! bytes { ($($v:ident),+) => { match d { $(NumericData::$v(v)=>v.iter().flat_map(|x|x.to_le_bytes()).collect()),+ } }; }
    bytes!(I8, U8, I16, U16, I32, U32, I64, U64, F32, F64)
}

/// Save typed numerical arrays as a deterministic little-endian Level-5 file.
///
/// # Parameters and conventions
/// `writer` receives bytes at its current position; no seeking is needed.
/// `variables` maps valid names to column-major [`MatArray`] records. `compressed`
/// selects zlib per variable. Explicit shapes replace SciPy's `oned_as` option;
/// singleton axes are retained. No path extensions or Python options are inferred.
/// # Domain and special values
/// Preserves integer types, floating special values, complex components, and empty
/// axes. Only floating arrays can be complex. See module-level feature restrictions.
/// # Returns
/// `()` after writing the complete file. Callers control flushing and closing.
/// # Errors
/// Invalid names, types, lengths, shapes or work limits from [`whosmat`]; I/O or
/// compression failure. Validation and encoding finish before output starts;
/// a destination failure may still leave a partial file.
/// # Numerical behavior
/// Exact little-endian typed encoding and lossless zlib. Names are written in
/// lexicographic order; output has no timestamps or platform-dependent header text.
/// # Examples
/// ```
/// let mut b=Vec::new();sci_io::savemat(&mut b,&Default::default(),false)?;
/// assert!(sci_io::loadmat(b.as_slice())?.is_empty());
/// # Ok::<(), sci_io::Error>(())
/// ```
/// # References
/// [MAT-file specification](https://www.mathworks.com/help/pdf_doc/matlab/matfile_format.pdf),
/// [SciPy savemat](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.savemat.html).
pub fn savemat(mut writer: impl Write, variables: &MatFile, compressed: bool) -> Result<()> {
    if variables.len() > 1024 {
        return Err(Error::InvalidData("MATLAB file exceeds 1,024 variables"));
    }
    let mut total = 0usize;
    for (name, a) in variables {
        if !name_valid(name) {
            return Err(Error::InvalidData("invalid MATLAB variable name"));
        }
        total = total
            .checked_add(validate(a)?)
            .filter(|&n| n <= VALUES)
            .ok_or(Error::InvalidData(
                "MATLAB file exceeds 1,000,000 scalar values",
            ))?;
    }
    let mut out = vec![b' '; 128];
    let text = b"MATLAB 5.0 MAT-file, written by Sci";
    out[..text.len()].copy_from_slice(text);
    out[116..124].fill(0);
    out[124..128].copy_from_slice(&[0, 1, b'I', b'M']);
    for (name, a) in variables {
        let mut payload = Vec::new();
        let (class, storage, _, _) = repr(&a.real);
        let flags = class | if a.imag.is_some() { 0x800 } else { 0 };
        let mut flag_bytes = flags.to_le_bytes().to_vec();
        flag_bytes.extend([0; 4]);
        put_element(&mut payload, 6, &flag_bytes);
        let dims: Vec<u8> = a
            .shape
            .iter()
            .flat_map(|&n| (n as i32).to_le_bytes())
            .collect();
        put_element(&mut payload, 5, &dims);
        put_element(&mut payload, 1, name.as_bytes());
        put_element(&mut payload, storage, &numeric_bytes(&a.real));
        if let Some(im) = &a.imag {
            put_element(&mut payload, storage, &numeric_bytes(im));
        }
        if compressed {
            let mut matrix = Vec::new();
            put_element(&mut matrix, 14, &payload);
            let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(&matrix)?;
            put_element(&mut out, 15, &encoder.finish()?);
        } else {
            put_element(&mut out, 14, &payload);
        }
        if out.len() > BYTES {
            return Err(Error::InvalidData("MATLAB output exceeds 64 MiB"));
        }
    }
    writer.write_all(&out)?;
    Ok(())
}

impl MatArray {
    /// Convert a rank-two real array into shared `f64` matrix rows.
    ///
    /// # Parameters and conventions
    /// `self` is column-major; returned `rows[i][j] = A[i,j]`. Original data is
    /// retained. This is an explicit dtype conversion; complex arrays are rejected.
    /// # Domain and special values
    /// Floating special values persist; 64-bit integers outside ±2^53 may round.
    /// # Returns
    /// Shared [`Matrix`] with row-major logical indexing.
    /// # Errors
    /// Invalid shape/storage, work limits of [`MatArray`], complex data, rank other
    /// than two, or `(0,n)` with `n>0` (unrepresentable in shared row storage).
    /// # Numerical behavior
    /// Reordering is exact; scalar conversion follows [`NumericData::to_f64`].
    /// # Examples
    /// ```
    /// let a=sci_io::matlab::MatArray {shape:vec![2,2],imag:None,
    ///     real:sci_io::NumericData::I16(vec![1,3,2,4])};
    /// assert_eq!(a.to_matrix()?,vec![vec![1.0,2.0],vec![3.0,4.0]]);
    /// # Ok::<(), sci_io::Error>(())
    /// ```
    pub fn to_matrix(&self) -> Result<Matrix> {
        validate(self)?;
        if self.shape.len() != 2 || self.imag.is_some() || (self.shape[0] == 0 && self.shape[1] > 0)
        {
            return Err(Error::Unsupported(
                "expected real rank-two array representable as shared rows",
            ));
        }
        let v = self.real.to_f64();
        let (rows, cols) = (self.shape[0], self.shape[1]);
        Ok((0..rows)
            .map(|r| (0..cols).map(|c| v[r + c * rows]).collect())
            .collect())
    }
}
