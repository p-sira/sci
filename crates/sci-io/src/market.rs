use crate::{Error, Matrix, Result, SparseMatrix, VALUES, bounded_read};
use std::io::{Read, Write};

/// A real Matrix Market matrix, with explicit dense shape or shared CSR storage.
#[derive(Clone, Debug, PartialEq)]
pub enum MarketMatrix {
    /// Dense values in row-major order, including explicitly shaped empty axes.
    Dense {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        cols: usize,
        /// Exactly `rows * cols` values; `data[i * cols + j] = A[i,j]`.
        data: Vec<f64>,
    },
    /// Canonical finite real CSR; duplicate input coordinates are summed.
    Sparse(SparseMatrix),
}

/// Matrix Market header fields, following SciPy's `mminfo` ordering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatrixMarketInfo {
    /// Row count, at most 1,000,000.
    pub rows: usize,
    /// Column count, at most 1,000,000.
    pub cols: usize,
    /// Stored coordinate records, or `rows * cols` for array storage.
    pub entries: usize,
    /// `coordinate` or `array`.
    pub format: &'static str,
    /// `real`, `integer`, or `pattern`; complex fields are unsupported.
    pub field: &'static str,
    /// `general`, `symmetric`, `skew-symmetric`, or real `hermitian`.
    pub symmetry: &'static str,
}

fn invalid() -> Error {
    Error::InvalidData("malformed Matrix Market data")
}
fn number(s: &str) -> Result<usize> {
    s.parse().map_err(|_| invalid())
}
fn lines(s: &str) -> impl Iterator<Item = &str> {
    s.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('%'))
}
fn header(s: &str) -> Result<(MatrixMarketInfo, &str)> {
    let (first, rest) = s.split_once('\n').ok_or_else(invalid)?;
    let h: Vec<_> = first.split_whitespace().take(6).collect();
    if h.len() != 5
        || !h[0].eq_ignore_ascii_case("%%MatrixMarket")
        || !h[1].eq_ignore_ascii_case("matrix")
    {
        return Err(invalid());
    }
    let format = match h[2].to_ascii_lowercase().as_str() {
        "array" => "array",
        "coordinate" => "coordinate",
        _ => return Err(invalid()),
    };
    let field = match h[3].to_ascii_lowercase().as_str() {
        "real" => "real",
        "integer" => "integer",
        "pattern" if format == "coordinate" => "pattern",
        "complex" => {
            return Err(Error::Unsupported(
                "complex Matrix Market matrices are unsupported",
            ));
        }
        _ => return Err(invalid()),
    };
    let symmetry = match h[4].to_ascii_lowercase().as_str() {
        "general" => "general",
        "symmetric" => "symmetric",
        "skew-symmetric" => "skew-symmetric",
        "hermitian" => "hermitian",
        _ => return Err(invalid()),
    };
    let mut offset = 0;
    for line in rest.split_inclusive('\n') {
        offset += line.len();
        let t = line.trim();
        if t.is_empty() || t.starts_with('%') {
            continue;
        }
        let dims: Vec<_> = t
            .split_whitespace()
            .take(4)
            .map(number)
            .collect::<Result<_>>()?;
        if dims.len() != if format == "array" { 2 } else { 3 } {
            return Err(invalid());
        }
        let (rows, cols) = (dims[0], dims[1]);
        if rows > VALUES || cols > VALUES || (symmetry != "general" && rows != cols) {
            return Err(Error::InvalidData("invalid Matrix Market shape"));
        }
        let entries = if format == "array" {
            rows.checked_mul(cols).ok_or_else(invalid)?
        } else {
            dims[2]
        };
        if entries > VALUES {
            return Err(Error::InvalidData(
                "Matrix Market storage exceeds 1,000,000",
            ));
        }
        return Ok((
            MatrixMarketInfo {
                rows,
                cols,
                entries,
                format,
                field,
                symmetry,
            },
            &rest[offset..],
        ));
    }
    Err(invalid())
}
fn value(s: &str, field: &str) -> Result<f64> {
    if field == "integer" {
        let v: i64 = s.parse().map_err(|_| invalid())?;
        if v.unsigned_abs() > 1u64 << 53 {
            return Err(Error::Unsupported(
                "Matrix Market integers outside ±2^53 require typed storage",
            ));
        }
        Ok(v as f64)
    } else {
        s.parse().map_err(|_| invalid())
    }
}

/// Inspect a Matrix Market header without decoding its numerical payload.
///
/// # Parameters and conventions
/// `reader` supplies UTF-8/ASCII Matrix Market bytes; the entire stream is consumed.
/// Coordinate indices on disk are one-based. Symmetric arrays store a lower triangle.
/// # Returns
/// Shape, stored entry count, format, field, and symmetry in named fields.
/// # Errors
/// I/O failure, invalid header/shape, unsupported complex field, >64 MiB input,
/// axes or stored entries above 1,000,000. Payload contents are not validated.
/// # Examples
/// ```
/// let h = sci_io::mminfo(&b"%%MatrixMarket matrix array real general\n1 1\n2\n"[..])?;
/// assert_eq!((h.rows, h.cols, h.entries), (1, 1, 1));
/// # Ok::<(), sci_io::Error>(())
/// ```
/// # References
/// [NIST Matrix Market](https://math.nist.gov/MatrixMarket/formats.html).
pub fn mminfo(reader: impl Read) -> Result<MatrixMarketInfo> {
    let b = bounded_read(reader)?;
    Ok(header(std::str::from_utf8(&b).map_err(|_| invalid())?)?.0)
}

/// Read `A[i,j]` from Matrix Market array values or coordinate triples.
///
/// # Parameters and conventions
/// `reader` supplies uncompressed bytes. Array files are column-major on disk;
/// returned dense data is row-major. Coordinates become shared canonical CSR.
/// Real, integer (within ±2^53), and coordinate pattern fields are supported.
/// General, symmetric, skew-symmetric, and real Hermitian storage are accepted.
/// # Domain and special values
/// Empty shapes are supported. Dense real data preserves NaN, infinity, and
/// signed zero; CSR requires finite values. Pattern entries equal one.
/// Symmetric coordinates must lie in the lower triangle; skew diagonals are zero.
/// # Returns
/// [`MarketMatrix`]; off-diagonal triangular entries are reflected, negated for
/// skew symmetry. Duplicate coordinates are summed in input order by Sci sparse.
/// # Errors
/// Invalid/truncated/excess data, unsupported complex or large integer fields,
/// invalid indices/symmetry, I/O failure, >64 MiB bytes, or more than 1,000,000
/// dense cells, coordinate records, or expanded sparse entries.
/// # Numerical behavior
/// Real text is rounded to nearest f64; dense reordering is exact. Sparse duplicate
/// addition follows [`sci_sparse::from_coo`]. Comments and blank lines are skipped.
/// # Examples
/// ```
/// let a = sci_io::mmread(&b"%%MatrixMarket matrix coordinate real general\n1 1 1\n1 1 3\n"[..])?;
/// assert!(matches!(a, sci_io::MarketMatrix::Sparse(_)));
/// # Ok::<(), sci_io::Error>(())
/// ```
/// # References
/// [NIST Matrix Market](https://math.nist.gov/MatrixMarket/formats.html).
pub fn mmread(reader: impl Read) -> Result<MarketMatrix> {
    let b = bounded_read(reader)?;
    let (h, payload) = header(std::str::from_utf8(&b).map_err(|_| invalid())?)?;
    if h.format == "coordinate" {
        let mut coordinates = Vec::new();
        let mut records = lines(payload);
        for _ in 0..h.entries {
            let t: Vec<_> = records
                .next()
                .ok_or_else(invalid)?
                .split_whitespace()
                .take(4)
                .collect();
            if t.len() != if h.field == "pattern" { 2 } else { 3 } {
                return Err(invalid());
            }
            let r = number(t[0])?.checked_sub(1).ok_or_else(invalid)?;
            let c = number(t[1])?.checked_sub(1).ok_or_else(invalid)?;
            let v = if h.field == "pattern" {
                1.0
            } else {
                value(t[2], h.field)?
            };
            if r >= h.rows
                || c >= h.cols
                || (h.symmetry != "general" && r < c)
                || (h.symmetry == "skew-symmetric" && r == c && v != 0.0)
            {
                return Err(invalid());
            }
            coordinates.push((r, c, v));
            if h.symmetry != "general" && r != c {
                coordinates.push((
                    c,
                    r,
                    if h.symmetry == "skew-symmetric" {
                        -v
                    } else {
                        v
                    },
                ));
            }
            if coordinates.len() > VALUES {
                return Err(Error::InvalidData(
                    "expanded sparse storage exceeds 1,000,000",
                ));
            }
        }
        if records.next().is_some() {
            return Err(invalid());
        }
        return Ok(MarketMatrix::Sparse(sci_sparse::from_coo(
            h.rows,
            h.cols,
            &coordinates,
        )?));
    }
    let mut tokens = lines(payload).flat_map(str::split_whitespace);
    let mut data = vec![0.0; h.entries];
    for c in 0..h.cols {
        let start = match h.symmetry {
            "general" => 0,
            "skew-symmetric" => c + 1,
            _ => c,
        };
        for r in start..h.rows {
            let v = value(tokens.next().ok_or_else(invalid)?, h.field)?;
            data[r * h.cols + c] = v;
            if h.symmetry != "general" && r != c {
                data[c * h.cols + r] = if h.symmetry == "skew-symmetric" {
                    -v
                } else {
                    v
                };
            }
        }
    }
    if tokens.next().is_some() {
        return Err(invalid());
    }
    Ok(MarketMatrix::Dense {
        rows: h.rows,
        cols: h.cols,
        data,
    })
}

/// Write `A[i,j]` as real general Matrix Market data with round-trip f64 text.
///
/// # Parameters and conventions
/// `writer` receives uncompressed ASCII. `a` selects array (dense row-major input,
/// column-major output) or coordinate (shared CSR, one-based output) storage.
/// Symmetry is always `general`; no automatic symmetry inference is performed.
/// # Domain and special values
/// Dense data preserves floating special values and signed zero. Sparse data
/// must satisfy shared finite canonical CSR invariants. Empty axes are supported.
/// # Returns
/// `()` after writing all bytes. The caller owns flushing/closing its stream.
/// # Errors
/// Invalid shapes/storage, limits in [`mmread`], or stream errors. Input is
/// validated before writing; stream failures may leave partial output.
/// # Numerical behavior
/// Shortest round-trip exponential f64 formatting; no arithmetic or densification.
/// # Examples
/// ```
/// let a = sci_io::MarketMatrix::Dense { rows:1, cols:1, data:vec![2.0] };
/// let mut bytes = Vec::new();
/// sci_io::mmwrite(&mut bytes, &a)?;
/// assert_eq!(sci_io::mmread(bytes.as_slice())?, a);
/// # Ok::<(), sci_io::Error>(())
/// ```
/// # References
/// [NIST Matrix Market](https://math.nist.gov/MatrixMarket/formats.html).
pub fn mmwrite(mut writer: impl Write, a: &MarketMatrix) -> Result<()> {
    match a {
        MarketMatrix::Dense { rows, cols, data } => {
            if *rows > VALUES
                || *cols > VALUES
                || rows.checked_mul(*cols) != Some(data.len())
                || data.len() > VALUES
            {
                return Err(Error::InvalidData("invalid dense Matrix Market shape"));
            }
            writeln!(
                writer,
                "%%MatrixMarket matrix array real general\n{rows} {cols}"
            )?;
            for c in 0..*cols {
                for r in 0..*rows {
                    writeln!(writer, "{:e}", data[r * cols + c])?;
                }
            }
        }
        MarketMatrix::Sparse(a) => {
            sci_sparse::validate(a)?;
            writeln!(
                writer,
                "%%MatrixMarket matrix coordinate real general\n{} {} {}",
                a.rows,
                a.cols,
                a.data.len()
            )?;
            for r in 0..a.rows {
                for p in a.indptr[r]..a.indptr[r + 1] {
                    writeln!(writer, "{} {} {:e}", r + 1, a.indices[p] + 1, a.data[p])?;
                }
            }
        }
    }
    Ok(())
}

impl MarketMatrix {
    /// Convert to shared real rows, preserving matrix indices and values.
    ///
    /// # Errors
    /// Invalid public storage, >1,000,000 dense cells, or a `(0,n)` shape with
    /// `n > 0`, which shared row storage cannot represent. Sparse input must be finite.
    /// # Examples
    /// ```
    /// let a = sci_io::MarketMatrix::Dense {rows:1, cols:2, data:vec![1.0,2.0]};
    /// assert_eq!(a.to_matrix()?, vec![vec![1.0,2.0]]);
    /// # Ok::<(), sci_io::Error>(())
    /// ```
    pub fn to_matrix(&self) -> Result<Matrix> {
        match self {
            Self::Sparse(a) => {
                if a.rows == 0 && a.cols > 0 {
                    return Err(Error::Unsupported(
                        "shared rows cannot represent shape (0,n)",
                    ));
                }
                Ok(sci_sparse::to_dense(a)?)
            }
            Self::Dense { rows, cols, data } => {
                if *rows > VALUES
                    || *cols > VALUES
                    || data.len() > VALUES
                    || rows.checked_mul(*cols) != Some(data.len())
                {
                    return Err(Error::InvalidData("invalid dense matrix shape"));
                }
                if *rows == 0 && *cols > 0 {
                    return Err(Error::Unsupported(
                        "shared rows cannot represent shape (0,n)",
                    ));
                }
                Ok((0..*rows)
                    .map(|r| data[r * cols..(r + 1) * cols].to_vec())
                    .collect())
            }
        }
    }
}
