use sci::io::{self as io, NumericData as D, matlab as mat, wavfile as wav};
use std::{collections::BTreeMap, io::Cursor, path::Path};

fn records(csv: &str) -> BTreeMap<&str, Vec<&str>> {
    csv.lines()
        .skip(1)
        .map(|line| {
            let v: Vec<_> = line.split(',').collect();
            (v[0], v)
        })
        .collect()
}
fn hex(s: &str) -> Vec<u8> {
    assert_eq!(s.len() % 2, 0);
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}
fn floats(s: &str) -> Vec<f64> {
    if s.is_empty() {
        vec![]
    } else {
        s.split(';').map(|x| x.parse().unwrap()).collect()
    }
}
fn sizes(s: &str) -> Vec<usize> {
    s.split(';').map(|x| x.parse().unwrap()).collect()
}
fn equal(a: &[f64], b: &[f64], id: &str) {
    assert_eq!(a.len(), b.len(), "{id}");
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        if b.is_nan() {
            assert!(a.is_nan(), "{id}[{i}]");
        } else {
            assert_eq!(a.to_bits(), b.to_bits(), "{id}[{i}]: {a} != {b}");
        }
    }
}
fn data(a: &D, dtype: &str, reference: &str, id: &str) {
    macro_rules! ints { ($($v:ident($t:ty,$s:literal)),+) => { match a {
        $(D::$v(v) => { assert_eq!(dtype,$s,"{id}"); let expected:Vec<$t>=if reference.is_empty(){vec![]}else{reference.split(';').map(|x|x.parse().unwrap()).collect()};assert_eq!(v,&expected,"{id}"); }),+,
        D::F32(v) => {assert_eq!(dtype,"float32","{id}");equal(&v.iter().map(|&v|f64::from(v)).collect::<Vec<_>>(),&floats(reference).into_iter().map(|v|f64::from(v as f32)).collect::<Vec<_>>(),id);},
        D::F64(v) => {assert_eq!(dtype,"float64","{id}");equal(v,&floats(reference),id);}
    } }; }
    ints!(
        I8(i8, "int8"),
        U8(u8, "uint8"),
        I16(i16, "int16"),
        U16(u16, "uint16"),
        I32(i32, "int32"),
        U32(u32, "uint32"),
        I64(i64, "int64"),
        U64(u64, "uint64")
    );
}
fn inputs() -> BTreeMap<&'static str, Vec<&'static str>> {
    records(include_str!("fixtures/io_inputs.csv"))
}
fn source(id: &str) -> Vec<u8> {
    hex(inputs()[id][2])
}
fn output(id: &str, extension: &str, b: &[u8]) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/io-rust");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(format!("{id}.{extension}")), b).unwrap();
}

#[test]
fn scipy_matrix_market_references_and_writes() {
    let expected = records(include_str!("fixtures/io_market_expected.csv"));
    for (id, e) in &expected {
        let b = source(id);
        let h = io::mminfo(b.as_slice()).unwrap();
        assert_eq!(
            (h.rows, h.cols, h.entries),
            (
                e[1].parse().unwrap(),
                e[2].parse().unwrap(),
                e[3].parse().unwrap()
            ),
            "{id}"
        );
        assert_eq!((h.format, h.field, h.symmetry), (e[4], e[5], e[6]), "{id}");
        let a = io::mmread(b.as_slice()).unwrap();
        let check = |a: &io::MarketMatrix| {
            match a {
                io::MarketMatrix::Dense { rows, cols, data } => {
                    assert_eq!((*rows, *cols), (h.rows, h.cols));
                    let mut reference = floats(e[7]);
                    // SciPy's text parser discards the sign of textual zero; Sci preserves it.
                    if *id == "specials" {
                        reference[3] = -0.0;
                    }
                    equal(data, &reference, id);
                }
                io::MarketMatrix::Sparse(s) => {
                    sci::sparse::validate(s).unwrap();
                    assert_eq!((s.rows, s.cols), (h.rows, h.cols));
                    equal(
                        &sci::sparse::to_dense(s)
                            .unwrap()
                            .into_iter()
                            .flatten()
                            .collect::<Vec<_>>(),
                        &floats(e[7]),
                        id,
                    );
                }
            }
            if h.rows == 0 && h.cols > 0 {
                assert!(a.to_matrix().is_err());
            } else {
                let m = a.to_matrix().unwrap();
                assert_eq!(m.len(), h.rows);
                assert!(m.iter().all(|r| r.len() == h.cols));
                let mut reference = floats(e[7]);
                if *id == "specials" {
                    reference[3] = -0.0;
                }
                equal(&m.into_iter().flatten().collect::<Vec<_>>(), &reference, id);
            }
        };
        check(&a);
        let mut b = Vec::new();
        io::mmwrite(&mut b, &a).unwrap();
        check(&io::mmread(b.as_slice()).unwrap());
        output(id, "mtx", &b);
    }
    assert_eq!(expected.len(), 11);
}

#[test]
fn scipy_wav_references_signal_interop_and_writes() {
    let expected = records(include_str!("fixtures/io_wav_expected.csv"));
    for (id, e) in &expected {
        let a = wav::read(source(id).as_slice()).unwrap();
        assert_eq!(
            (a.sample_rate, a.channels, a.bits_per_sample),
            (
                e[1].parse().unwrap(),
                e[2].parse().unwrap(),
                e[3].parse().unwrap()
            ),
            "{id}"
        );
        data(&a.samples, e[4], e[5], id);
        let normalized = floats(e[6]);
        for c in 0..usize::from(a.channels) {
            let x = wav::signal(&a, c).unwrap();
            equal(
                &x,
                &normalized
                    .iter()
                    .skip(c)
                    .step_by(usize::from(a.channels))
                    .copied()
                    .collect::<Vec<_>>(),
                id,
            );
            if !x.is_empty() && x.iter().all(|v| v.is_finite()) {
                let y = sci::signal::convolve(&x, &[1.0], sci::Mode::Full).unwrap();
                // Arithmetic may turn negative zero into positive zero.
                assert_eq!(y, x);
            }
        }
        let mut b = Cursor::new(Vec::new());
        wav::write(&mut b, &a).unwrap();
        let round = wav::read(b.get_ref().as_slice()).unwrap();
        assert_eq!(
            (round.sample_rate, round.channels, round.bits_per_sample),
            (a.sample_rate, a.channels, a.bits_per_sample)
        );
        data(&round.samples, e[4], e[5], id);
        output(id, "wav", b.get_ref());
    }
    assert_eq!(expected.len(), 8);
}

#[test]
fn scipy_matlab_references_metadata_endianness_and_writes() {
    let expected = records(include_str!("fixtures/io_matlab_expected.csv"));
    for (id, e) in &expected {
        let b = source(id);
        assert_eq!(mat::matfile_version(b.as_slice()).unwrap(), (1, 0));
        let info = io::whosmat(b.as_slice()).unwrap();
        assert_eq!(info.len(), 1);
        assert_eq!(
            info[0],
            mat::MatVariableInfo {
                name: e[1].into(),
                shape: sizes(e[2]),
                class: e[6],
                complex: e[7] == "1"
            }
        );
        let f = io::loadmat(b.as_slice()).unwrap_or_else(|err| panic!("{id}: {err}"));
        assert_eq!(f.len(), 1);
        let a = &f[e[1]];
        let check = |a: &mat::MatArray| {
            assert_eq!(a.shape, sizes(e[2]), "{id}");
            data(&a.real, e[3], e[4], id);
            if e[7] == "1" {
                data(a.imag.as_ref().unwrap(), e[3], e[5], id);
            } else {
                assert!(a.imag.is_none());
            }
            if a.shape.len() == 2 && a.imag.is_none() && !(a.shape[0] == 0 && a.shape[1] > 0) {
                let m = a.to_matrix().unwrap();
                let v = a.real.to_f64();
                assert_eq!(m.len(), a.shape[0]);
                for (r, row) in m.iter().enumerate() {
                    equal(
                        row,
                        &(0..a.shape[1])
                            .map(|c| v[r + c * a.shape[0]])
                            .collect::<Vec<_>>(),
                        id,
                    );
                }
            } else {
                assert!(a.to_matrix().is_err());
            }
        };
        check(a);
        for compressed in [false, true] {
            let mut b = Vec::new();
            io::savemat(&mut b, &f, compressed).unwrap();
            check(&io::loadmat(b.as_slice()).unwrap()[e[1]]);
            assert_eq!(io::whosmat(b.as_slice()).unwrap(), info);
            output(&format!("{id}_{}", usize::from(compressed)), "mat", &b);
        }
    }
    assert_eq!(expected.len(), 45);
}

#[test]
fn matrix_market_invalid_domains_and_limits() {
    let cases = [
        "",
        "not a matrix\n",
        "%%MatrixMarket matrix coordinate complex general\n1 1 1\n1 1 2 3\n",
        "%%MatrixMarket matrix array pattern general\n1 1\n1\n",
        "%%MatrixMarket matrix array real general\n1000000 1000000\n",
        "%%MatrixMarket matrix coordinate real general\n1000001 0 0\n",
        "%%MatrixMarket matrix coordinate real general\n1 1 1000001\n",
        "%%MatrixMarket matrix coordinate real general\n1 1 1\n0 1 2\n",
        "%%MatrixMarket matrix coordinate real general\n1 1 1\n2 1 2\n",
        "%%MatrixMarket matrix coordinate real general\n1 1 1\n1 1 nan\n",
        "%%MatrixMarket matrix coordinate real general\n1 1 1\n1 1 inf\n",
        "%%MatrixMarket matrix coordinate real general\n1 1 0\n1 1 3\n",
        "%%MatrixMarket matrix coordinate real general\n1 1 1\n",
        "%%MatrixMarket matrix coordinate real symmetric\n2 3 0\n",
        "%%MatrixMarket matrix coordinate real symmetric\n2 2 1\n1 2 3\n",
        "%%MatrixMarket matrix coordinate real skew-symmetric\n1 1 1\n1 1 2\n",
        "%%MatrixMarket matrix array integer general\n1 1\n9007199254740993\n",
        "%%MatrixMarket matrix array real general\n1 1\n1 2\n",
        "%%MatrixMarket matrix array real general\n1 1\n",
    ];
    for c in cases {
        assert!(io::mmread(c.as_bytes()).is_err(), "{c}");
    }
    let mut b = Vec::new();
    assert!(
        io::mmwrite(
            &mut b,
            &io::MarketMatrix::Dense {
                rows: 1,
                cols: 2,
                data: vec![1.0]
            }
        )
        .is_err()
    );
    assert!(b.is_empty());
    let mut s = sci::sparse::identity(2).unwrap();
    s.indptr.clear();
    assert!(io::mmwrite(&mut b, &io::MarketMatrix::Sparse(s)).is_err());
    assert!(b.is_empty());
    assert!(
        io::MarketMatrix::Dense {
            rows: 2,
            cols: 2,
            data: vec![1.0]
        }
        .to_matrix()
        .is_err()
    );
}

#[test]
fn wav_invalid_domains_and_truncation() {
    let base = wav::Audio {
        sample_rate: 8000,
        channels: 1,
        bits_per_sample: 16,
        samples: D::I16(vec![1, 2]),
    };
    for a in [
        wav::Audio {
            sample_rate: 0,
            ..base.clone()
        },
        wav::Audio {
            channels: 0,
            ..base.clone()
        },
        wav::Audio {
            channels: 3,
            ..base.clone()
        },
        wav::Audio {
            bits_per_sample: 24,
            ..base.clone()
        },
        wav::Audio {
            bits_per_sample: 24,
            samples: D::I32(vec![1]),
            ..base.clone()
        },
        wav::Audio {
            sample_rate: u32::MAX,
            ..base.clone()
        },
        wav::Audio {
            samples: D::F64(vec![1.0]),
            ..base.clone()
        },
        wav::Audio {
            samples: D::I16(vec![0; 1_000_001]),
            ..base.clone()
        },
    ] {
        let mut b = Cursor::new(Vec::new());
        assert!(wav::write(&mut b, &a).is_err());
        assert!(b.get_ref().is_empty());
        assert!(wav::signal(&a, 0).is_err());
    }
    assert!(wav::signal(&base, 1).is_err());
    let mut at_end = Cursor::new(vec![0]);
    at_end.set_position(1);
    assert!(wav::write(&mut at_end, &base).is_err());
    for id in ["u8", "i16", "f32", "f64", "i24"] {
        let b = source(id);
        for n in [0, 1, 12, b.len() - 1] {
            assert!(wav::read(&b[..n]).is_err());
        }
    }
    let mut b = source("i16");
    b[22..24].fill(0);
    assert!(wav::read(b.as_slice()).is_err());
    let mut b = source("i16");
    b[34..36].fill(0);
    assert!(wav::read(b.as_slice()).is_err());
    let mut b = source("i16");
    b[20..22].copy_from_slice(&6u16.to_le_bytes());
    assert!(wav::read(b.as_slice()).is_err());
}

fn saved(a: mat::MatArray) -> io::Result<Vec<u8>> {
    let mut b = Vec::new();
    io::savemat(&mut b, &mat::MatFile::from([("x".into(), a)]), false)?;
    Ok(b)
}
fn array() -> mat::MatArray {
    mat::MatArray {
        shape: vec![1, 2],
        real: D::F64(vec![1.0, 2.0]),
        imag: None,
    }
}

#[test]
fn matlab_invalid_domains_truncation_and_limits() {
    for a in [
        mat::MatArray {
            shape: vec![],
            ..array()
        },
        mat::MatArray {
            shape: vec![2],
            ..array()
        },
        mat::MatArray {
            shape: vec![1; 33],
            ..array()
        },
        mat::MatArray {
            shape: vec![1_000_001, 0],
            real: D::F64(vec![]),
            imag: None,
        },
        mat::MatArray {
            shape: vec![1000, 1001],
            ..array()
        },
        mat::MatArray {
            shape: vec![2, 2],
            ..array()
        },
        mat::MatArray {
            imag: Some(D::F32(vec![1.0, 2.0])),
            ..array()
        },
        mat::MatArray {
            imag: Some(D::F64(vec![1.0])),
            ..array()
        },
        mat::MatArray {
            real: D::I16(vec![1, 2]),
            imag: Some(D::I16(vec![1, 2])),
            ..array()
        },
        mat::MatArray {
            shape: vec![1_000_000, 1_000_000, 0],
            real: D::F64(vec![]),
            imag: None,
        },
    ] {
        assert!(saved(a.clone()).is_err());
        assert!(a.to_matrix().is_err());
    }
    for name in ["", "_x", "1x", "é", "a-b", &"x".repeat(64)] {
        let mut b = Vec::new();
        assert!(io::savemat(&mut b, &mat::MatFile::from([(name.into(), array())]), false).is_err());
        assert!(b.is_empty());
    }
    let mut b = Vec::new();
    let f: mat::MatFile = (0..1025)
        .map(|i| {
            (
                format!("x{i}"),
                mat::MatArray {
                    shape: vec![0, 0],
                    real: D::F64(vec![]),
                    imag: None,
                },
            )
        })
        .collect();
    assert!(io::savemat(&mut b, &f, false).is_err());
    assert!(b.is_empty());
    for id in ["double_plain", "double_big", "double_compressed"] {
        let b = source(id);
        for n in [0, 1, 127, 129, b.len() - 1] {
            assert!(io::loadmat(&b[..n]).is_err(), "{id}: {n}");
            assert!(io::whosmat(&b[..n]).is_err());
        }
    }
    let b = source("double_plain");
    // Unsupported classes, logical/global flags, bad rank, byte counts and version.
    for (index, value) in [
        (144, 1),
        (144, 2),
        (144, 3),
        (144, 4),
        (144, 5),
        (145, 2),
        (145, 4),
        (160, 255),
        (132, 255),
        (124, 1),
    ] {
        let mut c = b.clone();
        c[index] = value;
        assert!(io::loadmat(c.as_slice()).is_err(), "offset {index}");
        assert!(io::whosmat(c.as_slice()).is_err());
    }
    let mut c = b.clone();
    c[124..126].copy_from_slice(&0x200u16.to_le_bytes());
    assert!(mat::matfile_version(c.as_slice()).is_err());
    let mut c = b.clone();
    c[116] = 1;
    assert!(io::loadmat(c.as_slice()).is_err());
    // Duplicate names are rejected before decoding.
    let mut c = b.clone();
    c.extend_from_slice(&b[128..]);
    assert!(io::loadmat(c.as_slice()).is_err());
    // Valid header-only empty files and multi-variable ordering.
    for compressed in [false, true] {
        let mut b = Vec::new();
        io::savemat(&mut b, &Default::default(), compressed).unwrap();
        assert!(io::loadmat(b.as_slice()).unwrap().is_empty());
        let f = mat::MatFile::from([("z".into(), array()), ("a".into(), array())]);
        b.clear();
        io::savemat(&mut b, &f, compressed).unwrap();
        assert_eq!(io::loadmat(b.as_slice()).unwrap(), f);
        assert_eq!(
            io::whosmat(b.as_slice())
                .unwrap()
                .iter()
                .map(|v| v.name.as_str())
                .collect::<Vec<_>>(),
            ["a", "z"]
        );
    }
}

struct Broken;
impl std::io::Read for Broken {
    fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
        Err(std::io::Error::other("test failure"))
    }
}
impl std::io::Write for Broken {
    fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
        Err(std::io::Error::other("test failure"))
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
impl std::io::Seek for Broken {
    fn seek(&mut self, _: std::io::SeekFrom) -> std::io::Result<u64> {
        Ok(0)
    }
}
#[test]
fn stream_errors_keep_their_source() {
    use std::error::Error as _;
    for err in [
        io::mmread(Broken).unwrap_err(),
        io::mminfo(Broken).unwrap_err(),
        io::loadmat(Broken).unwrap_err(),
        io::whosmat(Broken).unwrap_err(),
        mat::matfile_version(Broken).unwrap_err(),
        wav::read(Broken).unwrap_err(),
        io::savemat(Broken, &Default::default(), false).unwrap_err(),
        io::mmwrite(
            Broken,
            &io::MarketMatrix::Dense {
                rows: 0,
                cols: 0,
                data: vec![],
            },
        )
        .unwrap_err(),
        wav::write(
            Broken,
            &wav::Audio {
                sample_rate: 8000,
                channels: 1,
                bits_per_sample: 16,
                samples: D::I16(vec![]),
            },
        )
        .unwrap_err(),
    ] {
        assert!(matches!(err, io::Error::Io(_)), "{err:?}");
        assert!(err.source().is_some());
        assert_eq!(err.to_string(), "test failure");
    }
}

#[test]
fn malformed_compression_and_unsupported_file_features() {
    let mut checked = 0;
    for (id, r) in inputs() {
        if r[1] == "invalid_matlab" {
            let b = hex(r[2]);
            assert!(io::loadmat(b.as_slice()).is_err(), "{id}");
            assert!(io::whosmat(b.as_slice()).is_err(), "{id}");
            checked += 1;
        } else if r[1] == "invalid_wav" {
            assert!(wav::read(hex(r[2]).as_slice()).is_err(), "{id}");
            checked += 1;
        }
    }
    assert_eq!(checked, 11);
}

#[test]
fn byte_limits_and_public_numeric_conversions() {
    use std::io::Read as _;
    for format in ["market", "wav", "matlab"] {
        let bytes = std::io::repeat(0).take(64 * 1024 * 1024 + 1);
        let error = match format {
            "market" => io::mmread(bytes).unwrap_err(),
            "wav" => wav::read(bytes).unwrap_err(),
            _ => io::loadmat(bytes).unwrap_err(),
        };
        assert_eq!(error.to_string(), "input exceeds 64 MiB");
    }
    assert!(D::F64(vec![]).is_empty());
    assert_eq!(D::U64(vec![u64::MAX]).len(), 1);
    assert_eq!(
        D::I64(vec![-9_007_199_254_740_993, 9_007_199_254_740_993]).to_f64(),
        [-9_007_199_254_740_992.0, 9_007_199_254_740_992.0]
    );
    let a = wav::Audio {
        sample_rate: 8000,
        channels: 1,
        bits_per_sample: 16,
        samples: D::I16(vec![1]),
    };
    let mut nonempty = Cursor::new(vec![1, 2, 3]);
    assert!(wav::write(&mut nonempty, &a).is_err());
    assert_eq!(nonempty.into_inner(), [1, 2, 3]);
}

#[test]
fn shared_sparse_interoperability_and_matrix_properties() {
    for rows in 0..5 {
        for cols in 0..5 {
            let entries: Vec<_> = (0..rows)
                .flat_map(|r| {
                    (0..cols)
                        .filter(move |&c| (r + c) % 2 == 0)
                        .map(move |c| (r, c, (r * cols + c) as f64 - 3.0))
                })
                .collect();
            let a = sci::sparse::from_coo(rows, cols, &entries).unwrap();
            let mut b = Vec::new();
            io::mmwrite(&mut b, &io::MarketMatrix::Sparse(a.clone())).unwrap();
            let io::MarketMatrix::Sparse(loaded) = io::mmread(b.as_slice()).unwrap() else {
                panic!("expected CSR");
            };
            assert_eq!(loaded, a);
            let x = vec![2.0; cols];
            let expected: Vec<_> = (0..rows)
                .map(|r| entries.iter().filter(|e| e.0 == r).map(|e| e.2 * 2.0).sum())
                .collect();
            assert_eq!(sci::sparse::matvec(&loaded, &x).unwrap(), expected);
        }
    }
    let a = io::loadmat(source("double_plain").as_slice()).unwrap()["x"]
        .to_matrix()
        .unwrap();
    let a = sci::sparse::from_dense(&a).unwrap();
    assert_eq!(
        sci::sparse::matvec(&a, &[1.0, 2.0, 3.0]).unwrap(),
        [22.0, 28.0]
    );
}
