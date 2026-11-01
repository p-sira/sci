//! Bounded, pickle-free SciPy NPZ interchange.
use super::*;
use crate::construct::from_csc;
use std::io::{Read, Seek, SeekFrom, Write};
use zip::{ZipArchive, ZipWriter, write::SimpleFileOptions};

const ARCHIVE_LIMIT: u64 = 64 * 1024 * 1024;
fn io_error(_: impl std::fmt::Display) -> Error {
    Error::InvalidInput("invalid, unsupported, or unreadable sparse NPZ archive")
}
struct Npy {
    dtype: String,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}
fn read_npy<R: Read + Seek>(z: &mut ZipArchive<R>, name: &str) -> Result<Npy> {
    let mut f = z.by_name(name).map_err(io_error)?;
    if f.size() > ARCHIVE_LIMIT {
        return Err(Error::InvalidInput("NPY member exceeds 64 MiB"));
    }
    let mut b = Vec::new();
    f.by_ref()
        .take(ARCHIVE_LIMIT + 1)
        .read_to_end(&mut b)
        .map_err(io_error)?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(io_error("magic"));
    }
    let (start, len): (usize, usize) = match b[6] {
        1 => (10, u16::from_le_bytes([b[8], b[9]]) as usize),
        2 | 3 if b.len() >= 12 => (
            12,
            u32::from_le_bytes(b[8..12].try_into().map_err(io_error)?) as usize,
        ),
        _ => return Err(io_error("NPY version")),
    };
    let end = start
        .checked_add(len)
        .filter(|&e| e <= b.len())
        .ok_or(io_error("header"))?;
    let header = std::str::from_utf8(&b[start..end]).map_err(io_error)?;
    fn field<'a>(h: &'a str, key: &str) -> Result<&'a str> {
        h.split_once(key)
            .and_then(|(_, s)| s.split_once(':'))
            .map(|(_, s)| s.trim_start())
            .ok_or(io_error("header field"))
    }
    let d = field(header, "descr")?;
    let quote = d.chars().next().ok_or(io_error("dtype"))?;
    if quote != '\'' && quote != '"' {
        return Err(io_error("structured dtype"));
    }
    let dtype = d[1..]
        .split(quote)
        .next()
        .ok_or(io_error("dtype"))?
        .to_string();
    if !field(header, "fortran_order")?.starts_with("False") {
        return Err(io_error("Fortran-order NPY unsupported"));
    }
    let sh = field(header, "shape")?;
    let sh = sh
        .strip_prefix('(')
        .and_then(|s| s.split_once(')'))
        .ok_or(io_error("shape"))?
        .0;
    let shape: Vec<usize> = sh
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().map_err(io_error))
        .collect::<Result<_>>()?;
    if shape.len() > 32 || shape.iter().any(|&d| d > LIMIT + 1) {
        return Err(io_error("shape limit"));
    }
    let count = shape
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))
        .filter(|&v| v <= WORK)
        .ok_or(io_error("volume limit"))?;
    let width: usize = dtype
        .get(2..)
        .ok_or(io_error("dtype"))?
        .parse()
        .map_err(io_error)?;
    if count.checked_mul(width) != Some(b.len() - end) {
        return Err(io_error("payload size"));
    }
    Ok(Npy {
        dtype,
        shape,
        bytes: b[end..].to_vec(),
    })
}
impl Npy {
    fn ints(&self) -> Result<Vec<usize>> {
        let width: usize = self
            .dtype
            .get(2..)
            .ok_or(io_error("dtype"))?
            .parse()
            .map_err(io_error)?;
        if !matches!(width, 1 | 2 | 4 | 8)
            || !matches!(self.dtype.as_bytes().get(1), Some(b'i' | b'u'))
        {
            return Err(io_error("integer dtype required"));
        }
        self.bytes
            .chunks_exact(width)
            .map(|b| {
                let mut word = [0u8; 8];
                let big = self.dtype.starts_with('>');
                if big {
                    for (i, &v) in b.iter().rev().enumerate() {
                        word[i] = v;
                    }
                } else {
                    word[..width].copy_from_slice(b);
                }
                if self.dtype.as_bytes()[1] == b'i' && word[width - 1] & 0x80 != 0 {
                    return Err(io_error("negative index"));
                }
                usize::try_from(u64::from_le_bytes(word)).map_err(io_error)
            })
            .collect()
    }
    fn values(&self) -> Result<Vec<f64>> {
        let width: usize = self
            .dtype
            .get(2..)
            .ok_or(io_error("dtype"))?
            .parse()
            .map_err(io_error)?;
        if self.dtype.as_bytes().get(1) == Some(&b'f') && matches!(width, 4 | 8) {
            self.bytes
                .chunks_exact(width)
                .map(|b| {
                    let v = if width == 8 {
                        let b: [u8; 8] = b.try_into().map_err(io_error)?;
                        if self.dtype.starts_with('>') {
                            f64::from_be_bytes(b)
                        } else {
                            f64::from_le_bytes(b)
                        }
                    } else {
                        let b: [u8; 4] = b.try_into().map_err(io_error)?;
                        if self.dtype.starts_with('>') {
                            f32::from_be_bytes(b) as f64
                        } else {
                            f32::from_le_bytes(b) as f64
                        }
                    };
                    if v.is_finite() {
                        Ok(v)
                    } else {
                        Err(Error::NonFiniteEvaluation)
                    }
                })
                .collect()
        } else if matches!(self.dtype.as_bytes().get(1), Some(b'i' | b'u'))
            && matches!(width, 1 | 2 | 4 | 8)
        {
            self.bytes
                .chunks_exact(width)
                .map(|b| {
                    let mut word = [0u8; 8];
                    if self.dtype.starts_with('>') {
                        for (i, &v) in b.iter().rev().enumerate() {
                            word[i] = v;
                        }
                    } else {
                        word[..width].copy_from_slice(b);
                    }
                    let signed = self.dtype.as_bytes()[1] == b'i';
                    if signed && word[width - 1] & 0x80 != 0 {
                        word[width..].fill(0xff);
                    }
                    Ok(if signed {
                        i64::from_le_bytes(word) as f64
                    } else {
                        u64::from_le_bytes(word) as f64
                    })
                })
                .collect()
        } else if self.dtype == "|b1" {
            Ok(self.bytes.iter().map(|&b| f64::from(b != 0)).collect())
        } else {
            Err(io_error(
                "only real floating, integer, and boolean data supported",
            ))
        }
    }
}
fn write_npy<W: Write + Seek>(
    z: &mut ZipWriter<W>,
    name: &str,
    dtype: &str,
    shape: &[usize],
    data: &[u8],
    compressed: bool,
) -> Result<()> {
    let dims = if shape.len() == 1 {
        format!("{},", shape[0])
    } else {
        shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut h = format!("{{'descr': '{dtype}', 'fortran_order': False, 'shape': ({dims}), }}");
    let padding = (64 - (10 + h.len() + 1) % 64) % 64;
    h.extend(std::iter::repeat_n(' ', padding));
    h.push('\n');
    z.start_file(
        name,
        SimpleFileOptions::default().compression_method(if compressed {
            zip::CompressionMethod::Deflated
        } else {
            zip::CompressionMethod::Stored
        }),
    )
    .map_err(io_error)?;
    z.write_all(b"\x93NUMPY\x01\x00").map_err(io_error)?;
    z.write_all(&(h.len() as u16).to_le_bytes())
        .map_err(io_error)?;
    z.write_all(h.as_bytes()).map_err(io_error)?;
    z.write_all(data).map_err(io_error)
}

/// Write a SciPy-compatible NPZ archive containing `A` in canonical CSR format.
///
/// # Parameters and conventions
/// `writer` is a seekable byte destination; `compressed` selects DEFLATE or ZIP
/// storage. Callers open paths explicitly; extensions are not appended.
/// # Domain and special values
/// Valid finite CSR under crate limits; explicit zeros and empty axes persist.
/// # Returns
/// `()` after the ZIP directory is finished; `_is_array` is true for SciPy.
/// # Errors
/// Invalid CSR or any write/seek/compression error, represented by shared errors.
/// # Numerical behavior
/// Lossless f64 and signed-64 index encoding. No Python or pickle execution.
/// # Examples
/// ```
/// let mut bytes=std::io::Cursor::new(Vec::new());
/// sci_sparse::io::save_npz(&mut bytes, &sci_sparse::construct::identity(2)?, true)?;
/// bytes.set_position(0);
/// assert_eq!(sci_sparse::io::load_npz(bytes)?, sci_sparse::construct::identity(2)?);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy save_npz](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.save_npz.html),
/// [NumPy NPY specification](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html).
pub fn save_npz<W: Write + Seek>(writer: W, a: &SparseMatrix, compressed: bool) -> Result<()> {
    validate(a)?;
    let mut z = ZipWriter::new(writer);
    let idx = |v: &[usize]| {
        v.iter()
            .flat_map(|&i| (i as i64).to_le_bytes())
            .collect::<Vec<_>>()
    };
    write_npy(&mut z, "format.npy", "|S3", &[], b"csr", compressed)?;
    write_npy(
        &mut z,
        "shape.npy",
        "<i8",
        &[2],
        &idx(&[a.rows, a.cols]),
        compressed,
    )?;
    write_npy(
        &mut z,
        "indptr.npy",
        "<i8",
        &[a.indptr.len()],
        &idx(&a.indptr),
        compressed,
    )?;
    write_npy(
        &mut z,
        "indices.npy",
        "<i8",
        &[a.indices.len()],
        &idx(&a.indices),
        compressed,
    )?;
    write_npy(
        &mut z,
        "data.npy",
        "<f8",
        &[a.data.len()],
        &a.data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
        compressed,
    )?;
    write_npy(&mut z, "_is_array.npy", "|b1", &[], &[1], compressed)?;
    z.finish().map_err(io_error)?;
    Ok(())
}

/// Read a real two-dimensional SciPy NPZ archive as canonical CSR.
///
/// # Parameters and conventions
/// `reader` is seekable. CSR, CSC, COO, DIA and BSR archives are accepted,
/// including legacy matrix archives, and converted to the shared CSR record.
/// Real f32/f64, integer and bool data are converted to f64; no object dtype.
/// # Domain and special values
/// Finite data only; zero dimensions and explicit zeros are supported. Integer
/// data above 2^53 may round. Duplicate entries are summed by [`from_coo`].
/// # Returns
/// A validated [`SparseMatrix`], independent of the archive's format tag.
/// # Errors
/// Malformed/unsupported NPZ/NPY, invalid indices, nonfinite data, I/O failure,
/// rank other than two, duplicate ZIP member names, more than 16 ZIP members,
/// more than 64 MiB total uncompressed payload,
/// or crate limits. Encrypted archives and pickle data are never evaluated.
/// # Numerical behavior
/// Bounded decompression and allocation precede format conversion; duplicate
/// summation follows canonical constructor conventions.
/// # Examples
/// ```
/// let mut buf=std::io::Cursor::new(Vec::new());
/// sci_sparse::io::save_npz(&mut buf, &sci_sparse::construct::identity(1)?, false)?;
/// buf.set_position(0);
/// assert_eq!(sci_sparse::io::load_npz(buf)?.data, [1.0]);
/// # Ok::<(), sci_core::Error>(())
/// ```
/// # References
/// [SciPy load_npz](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.load_npz.html).
pub fn load_npz<R: Read + Seek>(reader: R) -> Result<SparseMatrix> {
    let mut z = ZipArchive::new(reader).map_err(io_error)?;
    if z.len() > 16 {
        return Err(io_error("member count"));
    }
    let mut total = 0u64;
    let mut headers = Vec::with_capacity(z.len());
    for i in 0..z.len() {
        let f = z.by_index(i).map_err(io_error)?;
        total = total
            .checked_add(f.size())
            .filter(|&v| v <= ARCHIVE_LIMIT)
            .ok_or(io_error("archive size"))?;
        headers.push(f.central_header_start());
    }
    // zip indexes names with last-member-wins semantics. A replaced member
    // leaves a gap among the retained central headers; inspecting those gaps
    // detects duplicates even though by_index() can no longer enumerate them.
    // Only fixed-size headers are read, with at most 16 iterations.
    headers.sort_unstable();
    let mut next = z.central_directory_start();
    let mut reader = z.into_inner();
    for start in headers {
        if start != next {
            return Err(io_error("duplicate member"));
        }
        reader.seek(SeekFrom::Start(start)).map_err(io_error)?;
        let mut header = [0; 46];
        reader.read_exact(&mut header).map_err(io_error)?;
        // ZIP central-header name, extra-field and comment lengths.
        let variable: u64 = [28, 30, 32]
            .into_iter()
            .map(|i| u64::from(u16::from_le_bytes([header[i], header[i + 1]])))
            .sum();
        next = start
            .checked_add(46 + variable)
            .ok_or(io_error("header offset"))?;
    }
    let mut z = ZipArchive::new(reader).map_err(io_error)?;
    let format = read_npy(&mut z, "format.npy")?;
    if format.dtype != "|S3" || !format.shape.is_empty() {
        return Err(io_error("format tag"));
    }
    let dims = read_npy(&mut z, "shape.npy")?.ints()?;
    if dims.len() != 2 {
        return Err(io_error("only rank-two archives supported"));
    }
    let (m, n) = (dims[0], dims[1]);
    shape(m, n)?;
    let data = read_npy(&mut z, "data.npy")?;
    if data.bytes.len() > LIMIT * 8 || data.shape.iter().product::<usize>() > LIMIT {
        return Err(io_error("storage limit"));
    }
    let values = data.values()?;
    match format.bytes.as_slice() {
        b"csr" | b"csc" => {
            let p = read_npy(&mut z, "indptr.npy")?.ints()?;
            let i = read_npy(&mut z, "indices.npy")?.ints()?;
            if format.bytes == b"csr" {
                from_csr(m, n, &p, &i, &values)
            } else {
                from_csc(m, n, &p, &i, &values)
            }
        }
        b"coo" => {
            let r = read_npy(&mut z, "row.npy")?.ints()?;
            let c = read_npy(&mut z, "col.npy")?.ints()?;
            if r.len() != values.len() || c.len() != values.len() {
                return Err(Error::DimensionMismatch);
            }
            from_coo(
                m,
                n,
                &r.into_iter()
                    .zip(c)
                    .zip(values)
                    .map(|((r, c), v)| (r, c, v))
                    .collect::<Vec<_>>(),
            )
        }
        b"dia" => {
            let off = read_npy(&mut z, "offsets.npy")?;
            let offs = off.values()?;
            if data.shape.len() != 2 || data.shape[0] != offs.len() {
                return Err(Error::DimensionMismatch);
            }
            let width = data.shape[1];
            let mut t = vec![];
            for (d, &k) in offs.iter().enumerate() {
                if k.fract() != 0. || k.abs() > LIMIT as f64 {
                    return Err(io_error("offset"));
                }
                for c in 0..width {
                    let r = c as i64 - k as i64;
                    if r >= 0 && (r as usize) < m && c < n {
                        t.push((r as usize, c, values[d * width + c]));
                    }
                }
            }
            from_coo(m, n, &t)
        }
        b"bsr" => {
            if data.shape.len() != 3 || data.shape[1] == 0 || data.shape[2] == 0 {
                return Err(io_error("block shape"));
            }
            let (br, bc) = (data.shape[1], data.shape[2]);
            if m % br != 0 || n % bc != 0 {
                return Err(io_error("block dimensions"));
            }
            let p = read_npy(&mut z, "indptr.npy")?.ints()?;
            let cols = read_npy(&mut z, "indices.npy")?.ints()?;
            let _validated = from_csr(m / br, n / bc, &p, &cols, &vec![1.; data.shape[0]])?;
            let mut t = vec![];
            for r in 0..m / br {
                for b in p[r]..p[r + 1] {
                    for i in 0..br {
                        for j in 0..bc {
                            t.push((r * br + i, cols[b] * bc + j, values[(b * br + i) * bc + j]));
                        }
                    }
                }
            }
            from_coo(m, n, &t)
        }
        _ => Err(io_error("format")),
    }
}
