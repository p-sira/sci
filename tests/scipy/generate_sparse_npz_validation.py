# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Generate deterministic NPZ validation fixtures and independent SciPy references.

Run: uv run tests/scipy/generate_sparse_npz_validation.py
Input cases live in sparse_npz_validation_cases.json. Valid archives use NumPy
NPY versions 2/3, nonnative byte order and signed/unsigned indices; NumPy loads
each finished archive and SciPy computes its dense reference after byte-order
normalization (SciPy constructors require native-endian data). Rejection cases
exercise Sci's documented bounded, real, rank-two archive subset, including
malformed headers/storage. Some (e.g. Fortran order) are valid NumPy encodings
outside Sci's supported subset; no SciPy rejection parity is claimed.
"""
import csv
import io
import json
from pathlib import Path
import warnings
import zipfile

import numpy as np
import scipy
from scipy import sparse

assert (np.__version__, scipy.__version__) == ("2.5.4", "1.18.1")
ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"
CASES = json.loads(Path(__file__).with_name("sparse_npz_validation_cases.json").read_text())


def npy(array, version=(1, 0)):
    out = io.BytesIO()
    np.lib.format.write_array(out, np.asarray(array), version=version, allow_pickle=False)
    return out.getvalue()


def archive(name, members, metadata=False):
    path = FIXTURES / f"sparse_npz_{name}.npz"
    with zipfile.ZipFile(path, "w") as out:
        for key, payload in members:
            info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            if metadata:
                info.extra = b"\xfe\xca\x02\x00ok"
                info.comment = "Member comment: \u6e2c\u8a66".encode()
            # Duplicate members are deliberately malformed test inputs.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                out.writestr(info, payload)
    if metadata:
        # Exercise the ZIP backend's prepended-data offset convention too.
        path.write_bytes(b"NPZ prefix\n" + path.read_bytes())
    return path


with (FIXTURES / "sparse_npz_validation.csv").open("w", newline="") as out:
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["name", "supported", "rows", "cols", "values"])
    for case in CASES["valid"]:
        a = sparse.csr_array(np.array(case["values"], dtype=np.dtype(case["dtype"]).newbyteorder("=")))
        arrays = dict(format=np.array(b"csr"), shape=np.array(a.shape, dtype=case["index_dtype"]),
                      data=a.data.astype(case["dtype"]), indices=a.indices.astype(case["index_dtype"]),
                      indptr=a.indptr.astype(case["index_dtype"]))
        path = archive(case["id"], [(key, npy(value, tuple(case["version"]))) for key, value in arrays.items()],
                       metadata=case.get("zip_metadata", False))
        with zipfile.ZipFile(path) as archive_file:
            loaded = {key: np.load(io.BytesIO(archive_file.read(key + ".npy")), allow_pickle=False)
                      for key in ("data", "indices", "indptr", "shape")}
            data = loaded["data"].astype(np.dtype(case["dtype"]).newbyteorder("="))
            ref = sparse.csr_array((data, loaded["indices"], loaded["indptr"]),
                                   shape=tuple(loaded["shape"])).toarray()
        writer.writerow([path.stem, 1, *ref.shape, ";".join(str(float(v)) for v in ref.ravel())])

    for case in CASES["invalid"]:
        arrays = dict(format=np.array(b"csr"), shape=np.array([1, 1]), data=np.array([1.]),
                      indices=np.array([0]), indptr=np.array([0, 1]))
        raw = {}
        if case == "magic":
            raw["format"] = b"not an NPY array"
        elif case == "version":
            payload = bytearray(npy(arrays["format"]))
            payload[6] = 9
            raw["format"] = bytes(payload)
        elif case == "structured_dtype":
            arrays["data"] = np.array([(1.,)], dtype=[("value", "f8")])
        elif case == "fortran_order":
            # Both spellings have equal length, preserving the header size.
            raw["format"] = npy(arrays["format"]).replace(b"False", b"True ")
        elif case == "shape_limit":
            # Header-only mutation avoids allocating an oversized array.
            raw["data"] = npy(arrays["data"]).replace(b"(1,)", b"(1000002,)")
        elif case == "payload_size":
            raw["data"] = npy(arrays["data"])[:-1]
        elif case == "float_indices":
            arrays["indices"] = np.array([0.])
        elif case == "format_tag":
            arrays["format"] = np.array([b"csr"])
        elif case == "rank":
            arrays["shape"] = np.array([1, 1, 1])
        elif case == "storage_limit":
            arrays["data"] = np.zeros(1_000_001, dtype="u1")
        elif case == "coo_lengths":
            arrays.update(format=np.array(b"coo"), row=np.array([], dtype="i8"), col=np.array([0]))
        elif case == "dia_shape":
            arrays.update(format=np.array(b"dia"), offsets=np.array([0]))
        elif case == "dia_offset":
            arrays.update(format=np.array(b"dia"), data=np.array([[1.]]), offsets=np.array([0.5]))
        elif case == "bsr_shape":
            arrays["format"] = np.array(b"bsr")
        elif case == "bsr_dimensions":
            arrays.update(format=np.array(b"bsr"), data=np.ones((1, 2, 2)))
        elif case == "unknown_format":
            arrays["format"] = np.array(b"lil")
        elif case not in ("member_count", "duplicate_member", "duplicate_middle_member", "duplicate_last_member"):
            raise ValueError(case)
        members = [(key, raw.get(key, npy(value))) for key, value in arrays.items()]
        if case == "member_count":
            members.extend((f"extra{i}", npy(np.array(0))) for i in range(12))
        elif case.startswith("duplicate_"):
            index = {"duplicate_member": 0, "duplicate_middle_member": 2, "duplicate_last_member": -1}[case]
            members.append(members[index])
        path = archive(case, members)
        writer.writerow([path.stem, 0, "", "", ""])

print(f"Generated {len(CASES['valid'])} SciPy references and {len(CASES['invalid'])} rejection cases.")
