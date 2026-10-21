# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Generate byte inputs and independent SciPy reference CSVs from io_cases.json.

Run with uv run tests/scipy/generate_io.py. Rust reads committed CSVs without
Python. Numerical I/O is exact (including integer dtypes, signed zero and
infinities); NaN payloads are not compared. WAV normalized signal references
use the documented PCM power-of-two scaling, without any Sci backend.
"""
import csv
import io
import json
import struct
import wave
import zlib
from pathlib import Path

import numpy as np
import scipy
from scipy import io as sio, sparse

assert (np.__version__, scipy.__version__) == ("2.5.4", "1.18.1")
ROOT = Path(__file__).resolve().parents[1] / 'fixtures'
CASES = json.loads(Path(__file__).with_name('io_cases.json').read_text())


def encoded(v):
    return ';'.join(str(x) for x in np.asarray(v).ravel())


def writer(stack, name, fields):
    f = stack.enter_context((ROOT / name).open('w', newline=''))
    w = csv.writer(f, lineterminator='\n')
    w.writerow(fields)
    return w


def big_endian(raw):
    """Re-encode an uncompressed SciPy numeric MAT fixture in big-endian order."""
    def elements(data):
        out = bytearray()
        while data:
            tag, = struct.unpack_from('<I', data)
            if tag >> 16:
                kind, count = struct.unpack_from('<HH', data)
                payload, data = data[4:4+count], data[8:]
                small = True
            else:
                kind, count = struct.unpack_from('<II', data)
                end = 8 + ((count + 7) & ~7)
                payload, data = data[8:8+count], data[end:]
                small = False
            if kind == 14:
                payload = elements(payload)
            elif kind in (3, 4, 5, 6, 7, 9, 12, 13):
                width = {3:2, 4:2, 5:4, 6:4, 7:4, 9:8, 12:8, 13:8}[kind]
                payload = b''.join(payload[i:i+width][::-1] for i in range(0,len(payload),width))
            if small:
                out += struct.pack('>I',(count<<16)|kind) + payload + bytes(4-count)
            else:
                out += struct.pack('>II',kind,count) + payload + bytes((-count)%8)
        return out
    return raw[:124] + b'\x01\x00MI' + elements(raw[128:])


if __name__ == '__main__':
    from contextlib import ExitStack
    with ExitStack() as stack:
        inputs = writer(stack,'io_inputs.csv',['id','format','hex'])
        mm = writer(stack,'io_market_expected.csv',['id','rows','cols','entries','storage','field','symmetry','values'])
        wav = writer(stack,'io_wav_expected.csv',['id','rate','channels','bits','dtype','values','signal'])
        mat = writer(stack,'io_matlab_expected.csv',['id','name','shape','dtype','real','imag','class','complex'])
        for c in CASES['market']:
            raw = c['text'].encode()
            # The C++ reader in SciPy 1.18.1 raises SIGFPE on zero-row array input.
            # The retained Python reader also rejects it. Generate the uniquely
            # defined empty reference through SciPy sparse storage instead.
            if c['id']=='empty_dense':
                a = sparse.coo_array((0,3)).toarray()
            else:
                a = sio.mmread(io.BytesIO(raw), spmatrix=False)
            shape = a.shape
            if sparse.issparse(a):
                a = a.toarray()
            inputs.writerow([c['id'],'market',raw.hex()])
            h = sio.mminfo(io.BytesIO(raw))
            mm.writerow([c['id'],*h,encoded(a)])
        for c in CASES['wav']:
            a = np.array(c['samples'],dtype=c['dtype'])
            b = io.BytesIO()
            if c['bits']==24:
                with wave.open(b,'wb') as w:
                    w.setnchannels(c['channels']);w.setframerate(c['rate']);w.setsampwidth(3)
                    w.writeframes(b''.join(int(x>>8).to_bytes(3,'little',signed=True) for x in a))
            else:
                sio.wavfile.write(b,c['rate'],a.reshape(-1,c['channels']) if c['channels']>1 else a)
            raw=b.getvalue()
            rate, a=sio.wavfile.read(io.BytesIO(raw))
            channels=1 if a.ndim==1 else a.shape[1]
            norm=a.astype(float)
            if c['dtype']=='uint8':norm=(norm-128)/128
            elif c['dtype']=='int16':norm/=32768
            elif c['dtype']=='int32':norm/=2147483648
            inputs.writerow([c['id'],'wav',raw.hex()])
            wav.writerow([c['id'],rate,channels,c['bits'],a.dtype.name,encoded(a),encoded(norm)])
        for c in CASES['matlab']:
            a=np.array(c['real'],dtype=c['dtype'])
            if 'imag' in c:
                im=np.array(c['imag'],dtype=c['dtype'])
                a=a.astype('complex64' if c['dtype']=='float32' else 'complex128')
                a.imag=im
            a=a.reshape(c['shape'],order='F')
            for compress in (False,True):
                b=io.BytesIO();sio.savemat(b,{'x':a},do_compression=compress)
                raw=b.getvalue()
                # Replace SciPy's timestamp-bearing descriptive header for reproducibility.
                raw=b'MATLAB 5.0 MAT-file, SciPy fixture'.ljust(116,b' ')+raw[116:]
                variants=[('compressed' if compress else 'plain',raw)]
                if not compress:variants.append(('big',big_endian(raw)))
                for suffix, raw in variants:
                    case_id=f"{c['id']}_{suffix}"
                    loaded=sio.loadmat(io.BytesIO(raw))['x']
                    info=sio.whosmat(io.BytesIO(raw))[0]
                    assert np.array_equal(loaded,a,equal_nan=True)
                    inputs.writerow([case_id,'matlab',raw.hex()])
                    mat.writerow([case_id,'x',encoded(info[1]),c['dtype'],encoded(loaded.real.ravel(order='F')),
                                  encoded(loaded.imag.ravel(order='F')) if 'imag' in c else '',info[2],int('imag' in c)])
        # Specification-invalid inputs test rejection before backend parsing.
        b=io.BytesIO();sio.savemat(b,{'x':np.array([[1.0,2.0]])})
        header=b.getvalue()[:128]
        header=b'MATLAB 5.0 MAT-file, SciPy fixture'.ljust(116,b' ')+header[116:]
        def compressed(payload):
            return header+struct.pack('<II',15,len(payload))+payload
        bomb=zlib.compress(bytes(64*1024*1024+1))
        inputs.writerow(['zlib_bomb','invalid_matlab',compressed(bomb).hex()])
        normal=zlib.compress(b.getvalue()[128:])
        inputs.writerow(['zlib_trailing','invalid_matlab',compressed(normal+b'excess').hex()])
        damaged=normal[:-1]+bytes([normal[-1]^1])
        inputs.writerow(['zlib_corrupt','invalid_matlab',compressed(damaged).hex()])
        inputs.writerow(['zlib_nested','invalid_matlab',compressed(zlib.compress(struct.pack('<II',15,0))).hex()])
        for name,array in [('logical',np.array([[True]])),('char',np.array(['abc'])),
                           ('cell',np.array([[np.array([1,2])]],dtype=object)),
                           ('struct',{'field':np.array([1.0])}),
                           ('sparse',sparse.eye_array(2))]:
            b=io.BytesIO();sio.savemat(b,{'x':array})
            raw=header+b.getvalue()[128:]
            inputs.writerow([name+'_unsupported','invalid_matlab',raw.hex()])
        b=io.BytesIO();sio.savemat(b,{'x':np.array([[1.0]])},format='4')
        inputs.writerow(['v4_unsupported','invalid_matlab',b.getvalue().hex()])
        b=io.BytesIO();sio.wavfile.write(b,8000,np.array([1,2],dtype='int16'))
        raw=bytearray(b.getvalue());raw[20:22]=struct.pack('<H',6)
        inputs.writerow(['mulaw_unsupported','invalid_wav',raw.hex()])
    print('Regenerated scientific I/O fixtures with SciPy',scipy.__version__)
