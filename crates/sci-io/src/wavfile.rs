//! Little-endian RIFF/WAVE PCM8/16/24/32 and IEEE float32/64 audio.
//!
//! Integer values follow SciPy conventions: unsigned PCM8, signed PCM16/32,
//! and PCM24 left-aligned in i32. Samples are interleaved by frame then channel.
//! Compressed codecs, RF64, RIFX, and memory mapping are unsupported.
use crate::{Error, NumericData, Result, VALUES, bounded_read};
use std::io::{Read, Seek, Write};

/// Typed interleaved audio samples and their sampling metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct Audio {
    /// Samples per second per channel, in hertz; must be positive.
    pub sample_rate: u32,
    /// Channels per frame; must be positive.
    pub channels: u16,
    /// PCM bit depth (8, 16, 24, 32), or 32/64 for floating samples.
    pub bits_per_sample: u16,
    /// `U8`, `I16`, `I32`, `F32`, or `F64`, matching the declared bit depth.
    /// Length is a multiple of `channels`; at most 1,000,000 values.
    /// PCM24 uses i32 with its low eight bits zero, as in SciPy.
    pub samples: NumericData,
}
fn codec(e: hound::Error) -> Error {
    match e {
        hound::Error::IoError(e) => Error::Io(e),
        e => Error::Codec(e.to_string()),
    }
}
struct Wave<'a> {
    format: u16,
    channels: u16,
    sample_rate: u32,
    bits: u16,
    data: &'a [u8],
}
fn container(bytes: &[u8]) -> Result<Wave<'_>> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(Error::Unsupported(
            "only little-endian RIFF/WAVE is supported",
        ));
    }
    let u32_at = |p: usize| {
        u32::from_le_bytes(bytes[p..p + 4].try_into().expect("checked chunk bounds")) as usize
    };
    let u16_at = |p: usize| u16::from_le_bytes([bytes[p], bytes[p + 1]]);
    if u32_at(4).checked_add(8) != Some(bytes.len()) {
        return Err(Error::InvalidData("RIFF size does not match input length"));
    }
    let (mut fmt, mut data) = (None, None);
    let mut pos = 12;
    while pos < bytes.len() {
        if bytes.len() - pos < 8 {
            return Err(Error::InvalidData("truncated WAV chunk header"));
        }
        let len = u32_at(pos + 4);
        let start = pos + 8;
        let end = start
            .checked_add(len)
            .filter(|&n| n <= bytes.len())
            .ok_or(Error::InvalidData("truncated WAV chunk"))?;
        match &bytes[pos..pos + 4] {
            b"fmt " => {
                if fmt.is_some() || data.is_some() || len < 16 {
                    return Err(Error::InvalidData("invalid WAV format chunk"));
                }
                let channels = u16_at(start + 2);
                let bits = u16_at(start + 14);
                let align = u16_at(start + 12);
                let format = u16_at(start);
                let rate = u32_at(start + 4) as u32;
                if channels == 0
                    || rate == 0
                    || (!matches!(bits, 8 | 16 | 24 | 32) && !(bits == 64 && format == 3))
                    || u32::from(align) != u32::from(channels) * u32::from(bits / 8)
                    || rate.checked_mul(u32::from(align)) != Some(u32_at(start + 8) as u32)
                {
                    return Err(Error::Unsupported(
                        "WAV requires packed PCM8/16/24/32 or float32/64 samples",
                    ));
                }
                if bits == 64 && !(len == 16 || (len == 18 && u16_at(start + 16) == 0)) {
                    return Err(Error::Unsupported(
                        "float64 WAV requires a standard IEEE format chunk",
                    ));
                }
                fmt = Some((format, channels, rate, bits, align));
            }
            b"data" => {
                if fmt.is_none() || data.is_some() {
                    return Err(Error::InvalidData(
                        "missing WAV format or duplicate data chunk",
                    ));
                }
                data = Some(&bytes[start..end]);
            }
            _ => {}
        }
        // Accept final unpadded odd data, which SciPy and common WAV writers emit.
        pos = if end == bytes.len() {
            end
        } else {
            end + len % 2
        };
    }
    let (Some((format, channels, sample_rate, bits, align)), Some(data)) = (fmt, data) else {
        return Err(Error::InvalidData("WAV requires format and data chunks"));
    };
    if data.len() % usize::from(align) != 0 || data.len() / usize::from(bits / 8) > VALUES {
        return Err(Error::InvalidData(
            "incomplete WAV frames or more than 1,000,000 samples",
        ));
    }
    Ok(Wave {
        format,
        channels,
        sample_rate,
        bits,
        data,
    })
}
fn validate(a: &Audio) -> Result<hound::WavSpec> {
    let format = match (&a.samples, a.bits_per_sample) {
        (NumericData::U8(_), 8) | (NumericData::I16(_), 16) | (NumericData::I32(_), 32) => {
            hound::SampleFormat::Int
        }
        (NumericData::I32(v), 24) if v.iter().all(|x| x & 255 == 0) => hound::SampleFormat::Int,
        (NumericData::F32(_), 32) | (NumericData::F64(_), 64) => hound::SampleFormat::Float,
        _ => {
            return Err(Error::InvalidData(
                "WAV sample type does not match bit depth, or PCM24 low bits are nonzero",
            ));
        }
    };
    if a.channels == 0
        || a.sample_rate == 0
        || a.samples.len() > VALUES
        || !a.samples.len().is_multiple_of(usize::from(a.channels))
    {
        return Err(Error::InvalidData(
            "invalid WAV rate, channels, or sample count",
        ));
    }
    let align = u32::from(a.channels) * u32::from(a.bits_per_sample / 8);
    if align > u32::from(u16::MAX) || a.sample_rate.checked_mul(align).is_none() {
        return Err(Error::InvalidData(
            "WAV block alignment or byte rate overflows",
        ));
    }
    Ok(hound::WavSpec {
        channels: a.channels,
        sample_rate: a.sample_rate,
        bits_per_sample: a.bits_per_sample,
        sample_format: format,
    })
}

/// Read WAV samples `x[frame,channel]` with their sample rate and original type.
///
/// # Parameters and conventions
/// `reader` supplies little-endian RIFF/WAVE data. Samples remain interleaved;
/// PCM24 is left-aligned i32, PCM8 is unsigned with silence at 128. No scaling
/// is applied. The returned [`Audio`] contains the rate in Hz and channel count.
/// # Domain and special values
/// PCM8/16/24/32 and IEEE float32/64 are supported, including empty audio. Floating
/// NaN, infinities, out-of-unit-range values, and signed zero are preserved.
/// # Returns
/// Typed samples and metadata, replacing SciPy's `(rate, ndarray)` tuple.
/// # Errors
/// Malformed/truncated data, unsupported codecs/depths, zero rate/channels,
/// incomplete frames, I/O failure, >64 MiB input, or >1,000,000 scalar samples.
/// # Numerical behavior
/// Lossless PCM/float32 decoding through a private codec and native float64
/// decoding; PCM24 alignment is an exact eight-bit shift. Extra RIFF metadata
/// is not retained.
/// # Examples
/// ```
/// use sci_io::{NumericData, wavfile::{Audio, read, write}};
/// let a = Audio {sample_rate:8000, channels:1, bits_per_sample:16,
///                samples:NumericData::I16(vec![0,123,-456])};
/// let mut b = std::io::Cursor::new(Vec::new());
/// write(&mut b,&a)?;
/// assert_eq!(read(b.get_ref().as_slice())?,a);
/// # Ok::<(), sci_io::Error>(())
/// ```
/// # References
/// [SciPy WAV conventions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html).
pub fn read(reader: impl Read) -> Result<Audio> {
    let bytes = bounded_read(reader)?;
    let raw = container(&bytes)?;
    if raw.format == 3 && raw.bits == 64 {
        let samples = raw
            .data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| f64::from_le_bytes(*b))
            .collect();
        let audio = Audio {
            sample_rate: raw.sample_rate,
            channels: raw.channels,
            bits_per_sample: 64,
            samples: NumericData::F64(samples),
        };
        validate(&audio)?;
        return Ok(audio);
    }
    let mut r = hound::WavReader::new(bytes.as_slice()).map_err(codec)?;
    let spec = r.spec();
    if r.len() as usize > VALUES {
        return Err(Error::InvalidData("WAV exceeds 1,000,000 samples"));
    }
    let samples = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 8) => NumericData::U8(
            r.samples::<i8>()
                .map(|v| v.map(|x| (i16::from(x) + 128) as u8))
                .collect::<std::result::Result<_, _>>()
                .map_err(codec)?,
        ),
        (hound::SampleFormat::Int, 16) => NumericData::I16(
            r.samples::<i16>()
                .collect::<std::result::Result<_, _>>()
                .map_err(codec)?,
        ),
        (hound::SampleFormat::Int, 24) => NumericData::I32(
            r.samples::<i32>()
                .map(|v| v.map(|x| x << 8))
                .collect::<std::result::Result<_, _>>()
                .map_err(codec)?,
        ),
        (hound::SampleFormat::Int, 32) => NumericData::I32(
            r.samples::<i32>()
                .collect::<std::result::Result<_, _>>()
                .map_err(codec)?,
        ),
        (hound::SampleFormat::Float, 32) => NumericData::F32(
            r.samples::<f32>()
                .collect::<std::result::Result<_, _>>()
                .map_err(codec)?,
        ),
        _ => return Err(Error::Unsupported("unsupported WAV sample encoding")),
    };
    let a = Audio {
        sample_rate: spec.sample_rate,
        channels: spec.channels,
        bits_per_sample: spec.bits_per_sample,
        samples,
    };
    validate(&a)?;
    Ok(a)
}

/// Write interleaved `x[frame,channel]` as typed little-endian WAV audio.
///
/// # Parameters and conventions
/// `writer` is seekable and positioned at byte zero; RIFF sizes are patched at
/// finalization. `audio` supplies sample rate in Hz, channels and exact bit depth.
/// PCM24 is stored in three bytes from left-aligned i32; its low eight bits must
/// be zero. Other types and normalization follow [`read`]. Use an empty destination.
/// # Domain and special values
/// Supports empty audio and all floating special values, without clipping.
/// # Returns
/// `()` after finalizing RIFF headers. The caller controls flushing/closing.
/// # Errors
/// Invalid sample type, rate, channels, depth or incomplete frames; >1,000,000
/// scalar samples; byte-rate overflow; nonempty destination or nonzero initial
/// position; stream failures.
/// Input is validated before output; stream errors can leave partial files.
/// # Numerical behavior
/// Exact typed encoding; no quantization, scaling, or conversion to f64.
/// # Examples
/// ```
/// let a = sci_io::wavfile::Audio {sample_rate:8000, channels:1,
///     bits_per_sample:8, samples:sci_io::NumericData::U8(vec![0,128,255])};
/// sci_io::wavfile::write(std::io::Cursor::new(Vec::new()),&a)?;
/// # Ok::<(), sci_io::Error>(())
/// ```
/// # References
/// [SciPy WAV writing](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html).
pub fn write<W: Write + Seek>(mut writer: W, audio: &Audio) -> Result<()> {
    let spec = validate(audio)?;
    if writer.stream_position()? != 0 {
        return Err(Error::InvalidData(
            "WAV destination must start at byte zero",
        ));
    }
    let size = writer.seek(std::io::SeekFrom::End(0))?;
    writer.seek(std::io::SeekFrom::Start(0))?;
    if size != 0 {
        return Err(Error::InvalidData("WAV destination must be empty"));
    }
    if let NumericData::F64(samples) = &audio.samples {
        // Standard IEEE float64 needs only native byte encoding; Hound handles
        // PCM and float32. Include the non-PCM fact chunk with the frame count.
        let length = (samples.len() * 8) as u32;
        let align = audio.channels * 8;
        let mut b = Vec::with_capacity(56 + length as usize);
        b.extend(b"RIFF");
        b.extend((48 + length).to_le_bytes());
        b.extend(b"WAVEfmt ");
        b.extend(16u32.to_le_bytes());
        b.extend(3u16.to_le_bytes());
        b.extend(audio.channels.to_le_bytes());
        b.extend(audio.sample_rate.to_le_bytes());
        b.extend((audio.sample_rate * u32::from(align)).to_le_bytes());
        b.extend(align.to_le_bytes());
        b.extend(64u16.to_le_bytes());
        b.extend(b"fact");
        b.extend(4u32.to_le_bytes());
        b.extend((samples.len() as u32 / u32::from(audio.channels)).to_le_bytes());
        b.extend(b"data");
        b.extend(length.to_le_bytes());
        b.extend(samples.iter().flat_map(|v| v.to_le_bytes()));
        writer.write_all(&b)?;
        return Ok(());
    }
    let mut w = hound::WavWriter::new(writer, spec).map_err(codec)?;
    match &audio.samples {
        NumericData::U8(v) => {
            for &x in v {
                w.write_sample((i16::from(x) - 128) as i8).map_err(codec)?;
            }
        }
        NumericData::I16(v) => {
            for &x in v {
                w.write_sample(x).map_err(codec)?;
            }
        }
        NumericData::I32(v) => {
            for &x in v {
                w.write_sample(if audio.bits_per_sample == 24 {
                    x >> 8
                } else {
                    x
                })
                .map_err(codec)?;
            }
        }
        NumericData::F32(v) => {
            for &x in v {
                w.write_sample(x).map_err(codec)?;
            }
        }
        _ => unreachable!("validated sample type"),
    }
    w.finalize().map_err(codec)
}

/// Extract channel `c` as an `f64` signal with normalized PCM amplitude.
///
/// # Parameters and conventions
/// `audio` uses [`Audio`] conventions; `channel` is zero-based. Frame ordering
/// is preserved. Sample time is `t[n] = n / audio.sample_rate` seconds.
/// # Domain and special values
/// Floating samples pass through unchanged, including NaN/infinity/signed zero.
/// # Returns
/// PCM8: `(x-128)/128`; PCM16: `x/2^15`; PCM24/32 (left-aligned): `x/2^31`.
/// The PCM amplitude range is [-1,1); float samples are not clipped.
/// # Errors
/// Invalid audio fields, sample limits in [`read`], or out-of-range channel.
/// # Numerical behavior
/// Exact integer-to-f64 conversion and power-of-two scaling for supported PCM;
/// floating conversion is exact. No filtering or resampling is performed.
/// # Examples
/// ```
/// let a = sci_io::wavfile::Audio {sample_rate:8000, channels:1,
///     bits_per_sample:8, samples:sci_io::NumericData::U8(vec![0,128])};
/// assert_eq!(sci_io::wavfile::signal(&a,0)?, [-1.0,0.0]);
/// # Ok::<(), sci_io::Error>(())
/// ```
/// # References
/// [SciPy PCM data ranges](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html).
pub fn signal(audio: &Audio, channel: usize) -> Result<Vec<f64>> {
    validate(audio)?;
    if channel >= usize::from(audio.channels) {
        return Err(Error::InvalidData("WAV channel out of range"));
    }
    let (offset, scale) = match audio.samples {
        NumericData::U8(_) => (128.0, 128.0),
        NumericData::I16(_) => (0.0, 32768.0),
        NumericData::I32(_) => (0.0, 2147483648.0),
        _ => (0.0, 1.0),
    };
    Ok(audio
        .samples
        .to_f64()
        .into_iter()
        .skip(channel)
        .step_by(usize::from(audio.channels))
        .map(|v| (v - offset) / scale)
        .collect())
}
