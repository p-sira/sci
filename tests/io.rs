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
    assert_eq!(expected.len(), 68);
}

#[test]
fn matlab_multiple_variables_preserve_types_and_metadata_order() {
    let variables = mat::MatFile::from([
        (
            "a".into(),
            mat::MatArray {
                shape: vec![1, 2],
                real: D::U64(vec![9007199254740993, u64::MAX]),
                imag: None,
            },
        ),
        (
            "b".into(),
            mat::MatArray {
                shape: vec![2, 1],
                real: D::F32(vec![-0., 1.25]),
                imag: Some(D::F32(vec![-2.5, 0.])),
            },
        ),
        (
            "c".into(),
            mat::MatArray {
                shape: vec![2, 0, 3],
                real: D::I16(vec![]),
                imag: None,
            },
        ),
    ]);
    for compressed in [false, true] {
        let mut bytes = Vec::new();
        io::savemat(&mut bytes, &variables, compressed).unwrap();
        let loaded = io::loadmat(bytes.as_slice()).unwrap();
        assert_eq!(loaded, variables);
        let D::F32(values) = &loaded["b"].real else {
            panic!("expected f32");
        };
        assert_eq!(values[0].to_bits(), (-0.0f32).to_bits());
        let info = io::whosmat(bytes.as_slice()).unwrap();
        assert_eq!(
            info.iter().map(|v| v.name.as_str()).collect::<Vec<_>>(),
            ["a", "b", "c"]
        );
        assert_eq!(
            info.iter()
                .map(|v| (v.class, v.complex))
                .collect::<Vec<_>>(),
            [("uint64", false), ("single", true), ("int16", false)]
        );
        // Metadata follows file order, independently of the returned map's sorting.
        if !compressed {
            let element_len = |at: usize| {
                let n = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap()) as usize;
                8 + ((n + 7) & !7)
            };
            let first_end = 128 + element_len(128);
            let second_end = first_end + element_len(first_end);
            let reordered = [
                &bytes[..128],
                &bytes[first_end..second_end],
                &bytes[128..first_end],
                &bytes[second_end..],
            ]
            .concat();
            assert_eq!(io::loadmat(reordered.as_slice()).unwrap(), variables);
            assert_eq!(
                io::whosmat(reordered.as_slice())
                    .unwrap()
                    .iter()
                    .map(|v| v.name.as_str())
                    .collect::<Vec<_>>(),
                ["b", "a", "c"]
            );
        }
    }
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

#[test]
fn matrix_market_rejects_malformed_headers_and_expanded_storage() {
    for text in [
        "%%MatrixMarket vector array real general\n1 1\n0\n",
        "%%MatrixMarket matrix unknown real general\n1 1\n0\n",
        "%%MatrixMarket matrix array real unknown\n1 1\n0\n",
        "%%MatrixMarket matrix array real general\n% only a comment\n",
        "%%MatrixMarket matrix array real general\n1\n",
        "%%MatrixMarket matrix array real general\nx 1\n",
        "%%MatrixMarket matrix array integer general\n1 1\nx\n",
        "%%MatrixMarket matrix array real general\n1 1\nx\n",
        "%%MatrixMarket matrix coordinate real general\n1 1 1\n1 1\n",
    ] {
        assert!(io::mmread(text.as_bytes()).is_err(), "{text}");
    }
    for read in [
        io::mminfo(&b"\xff"[..]).map(|_| ()),
        io::mmread(&b"\xff"[..]).map(|_| ()),
    ] {
        assert!(matches!(read, Err(io::Error::InvalidData(_))));
    }
    let mut text = String::from("%%MatrixMarket matrix coordinate real symmetric\n2 2 500001\n");
    for _ in 0..500001 {
        text.push_str("2 1 1\n");
    }
    assert_eq!(
        io::mmread(text.as_bytes()).unwrap_err().to_string(),
        "expanded sparse storage exceeds 1,000,000"
    );
}

#[test]
fn matlab_rejects_malformed_metadata_and_aggregate_limits() {
    let original = saved(array()).unwrap();
    // Normal tags emitted by Sci: flags at 136, dimensions at 152, name at
    // 168, and real payload at 184. Each mutation violates one invariant.
    for (offset, value) in [
        (126, 0),               // Invalid endian marker.
        (128, 1),               // Unsupported top-level tag.
        (136, 5),               // Wrong flags storage type.
        (140, 4),               // Flags have the wrong byte count.
        (144, 0x808),           // Complex integer class.
        (152, 6),               // Wrong dimensions storage type.
        (156, 4),               // Rank one.
        (168, 2),               // Wrong name storage type.
        (176, u32::from(b'1')), // Invalid identifier.
        (184, 7),               // Payload does not match the double class.
        (188, 8),               // Payload is shorter than the declared shape.
    ] {
        let mut bytes = original.clone();
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        assert!(io::loadmat(bytes.as_slice()).is_err(), "offset {offset}");
        assert!(io::whosmat(bytes.as_slice()).is_err(), "offset {offset}");
    }
    // Small data tags cannot contain more than four bytes.
    let mut bytes = original.clone();
    bytes[168..172].copy_from_slice(&0x50001u32.to_le_bytes());
    assert!(io::whosmat(bytes.as_slice()).is_err());
    // The outer element includes an unexpected trailing inner element.
    let mut bytes = original.clone();
    bytes.extend_from_slice(&[0; 8]);
    let size = (bytes.len() - 136) as u32;
    bytes[132..136].copy_from_slice(&size.to_le_bytes());
    assert!(io::whosmat(bytes.as_slice()).is_err());

    let a = mat::MatArray {
        shape: vec![1, 500001],
        real: D::U8(vec![0; 500001]),
        imag: None,
    };
    let file = mat::MatFile::from([("a".into(), a.clone()), ("b".into(), a.clone())]);
    let mut bytes = Vec::new();
    assert_eq!(
        io::savemat(&mut bytes, &file, false)
            .unwrap_err()
            .to_string(),
        "MATLAB file exceeds 1,000,000 scalar values"
    );
    assert!(bytes.is_empty());
    // Independently valid variables can exceed the aggregate reader limit.
    let first = saved(a.clone()).unwrap();
    let mut second = saved(a).unwrap();
    second[176] = b'y';
    let mut joined = first;
    joined.extend_from_slice(&second[128..]);
    assert_eq!(
        io::whosmat(joined.as_slice()).unwrap_err().to_string(),
        "MATLAB file exceeds 1,000,000 scalar values"
    );

    let mut bytes = Vec::new();
    for i in 0..1025 {
        let mut variable = Vec::new();
        let file = mat::MatFile::from([(format!("x{i}"), array())]);
        io::savemat(&mut variable, &file, false).unwrap();
        if i == 0 {
            bytes.extend_from_slice(&variable[..128]);
        }
        bytes.extend_from_slice(&variable[128..]);
    }
    assert_eq!(
        io::whosmat(bytes.as_slice()).unwrap_err().to_string(),
        "MATLAB file exceeds 1,024 variables"
    );
}

#[test]
fn public_error_display_and_sources() {
    use std::error::Error as _;
    let numerical = io::Error::from(sci::Error::DimensionMismatch);
    assert_eq!(
        numerical.to_string(),
        sci::Error::DimensionMismatch.to_string()
    );
    assert!(numerical.source().is_some());
    for error in [
        io::Error::InvalidData("invalid"),
        io::Error::Unsupported("unsupported"),
        io::Error::Codec("codec".into()),
    ] {
        assert!(!error.to_string().is_empty());
        assert!(error.source().is_none());
    }
}

#[test]
fn wav_rejects_invalid_chunk_structure() {
    let original = source("i16");
    let check = |mut bytes: Vec<u8>| {
        let size = (bytes.len() - 8) as u32;
        bytes[4..8].copy_from_slice(&size.to_le_bytes());
        assert!(wav::read(bytes.as_slice()).is_err());
    };
    let mut bytes = original.clone();
    bytes.extend_from_slice(b"x");
    check(bytes); // Partial chunk header.
    let mut bytes = original.clone();
    bytes[40..44].copy_from_slice(&u32::MAX.to_le_bytes());
    check(bytes); // Data chunk extends beyond the container.
    let mut bytes = original.clone();
    bytes.extend_from_slice(&original[12..36]);
    check(bytes); // Duplicate format chunk.
    let mut bytes = original.clone();
    bytes.extend_from_slice(&original[36..]);
    check(bytes); // Duplicate data chunk.
    let mut bytes = original.clone();
    bytes.drain(12..36);
    check(bytes); // Data precedes the format.
    let mut bytes = original.clone();
    bytes.truncate(36);
    check(bytes); // Missing data.
    let mut bytes = original.clone();
    bytes[16..20].copy_from_slice(&14u32.to_le_bytes());
    check(bytes); // Short format chunk.
    let mut bytes = source("f64");
    bytes[16..20].copy_from_slice(&20u32.to_le_bytes());
    bytes.splice(36..36, [0; 4]);
    check(bytes); // Unsupported float64 format extension.
}

#[test]
fn writers_propagate_failures_during_payload_and_finalization() {
    use std::io::{self as stdio, Seek, SeekFrom, Write};
    struct Destination {
        inner: Cursor<Vec<u8>>,
        remaining: usize,
        failed: bool,
    }
    impl Destination {
        fn step(&mut self) -> stdio::Result<()> {
            if self.remaining == 0 {
                self.failed = true;
                Err(stdio::Error::other("injected destination failure"))
            } else {
                self.remaining -= 1;
                Ok(())
            }
        }
    }
    impl Write for Destination {
        fn write(&mut self, bytes: &[u8]) -> stdio::Result<usize> {
            self.step()?;
            self.inner.write(bytes)
        }
        fn flush(&mut self) -> stdio::Result<()> {
            self.step()?;
            self.inner.flush()
        }
    }
    impl Seek for Destination {
        fn seek(&mut self, pos: SeekFrom) -> stdio::Result<u64> {
            self.step()?;
            self.inner.seek(pos)
        }
    }
    // Fail each successive stream operation, including sample writes and
    // header patching, until a complete file can be read back successfully.
    for id in ["u8", "i16", "i24", "i32", "f32", "f64", "dense", "sparse"] {
        let mut completed = false;
        for budget in 0..256 {
            let mut destination = Destination {
                inner: Cursor::new(Vec::new()),
                remaining: budget,
                failed: false,
            };
            let result = match id {
                "dense" => io::mmwrite(
                    &mut destination,
                    &io::MarketMatrix::Dense {
                        rows: 1,
                        cols: 2,
                        data: vec![1., 2.],
                    },
                ),
                "sparse" => io::mmwrite(
                    &mut destination,
                    &io::MarketMatrix::Sparse(sci::sparse::identity(2).unwrap()),
                ),
                _ => wav::write(&mut destination, &wav::read(source(id).as_slice()).unwrap()),
            };
            if destination.failed {
                assert!(
                    matches!(result, Err(io::Error::Io(_))),
                    "{id}, operation {budget}: {result:?}"
                );
            } else {
                result.unwrap();
                if matches!(id, "dense" | "sparse") {
                    io::mmread(destination.inner.get_ref().as_slice()).unwrap();
                } else {
                    wav::read(destination.inner.get_ref().as_slice()).unwrap();
                }
                completed = true;
                break;
            }
        }
        assert!(completed, "{id} exceeded the operation bound");
    }
}
