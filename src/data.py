"""MOT17 training sequences: public detections, ground truth, and (for the video) frames read out of the big zip."""
import configparser
import json
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
LABELS_URL = "https://motchallenge.net/data/MOT17Labels.zip"
FRAMES_URL = "https://motchallenge.net/data/MOT17.zip"  # 5.5 GB: only the needed frames are fetched, by HTTP range
SEQUENCES = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]
DETECTORS = ["DPM", "FRCNN", "SDP"]


def labels_dir() -> Path:
    root = DATA / "labels"
    if not root.exists():
        DATA.mkdir(exist_ok=True)
        z = DATA / "MOT17Labels.zip"
        urllib.request.urlretrieve(LABELS_URL, z)
        with zipfile.ZipFile(z) as f:
            f.extractall(root)
    return root / "train"


def seq_dir(seq: str, det: str) -> Path:
    return labels_dir() / f"{seq}-{det}"


def info(seq: str) -> dict:
    c = configparser.ConfigParser()
    c.read(seq_dir(seq, "FRCNN") / "seqinfo.ini")
    s = c["Sequence"]
    return {"frames": int(s["seqLength"]), "fps": int(s["frameRate"]), "w": int(s["imWidth"]), "h": int(s["imHeight"])}


def detections(seq: str, det: str) -> dict[int, np.ndarray]:
    """frame -> (k, 5) array of x1, y1, x2, y2, score."""
    a = np.loadtxt(seq_dir(seq, det) / "det" / "det.txt", delimiter=",", ndmin=2)
    boxes = np.c_[a[:, 2], a[:, 3], a[:, 2] + a[:, 4], a[:, 3] + a[:, 5], a[:, 6]]
    return {int(f): boxes[a[:, 0] == f] for f in np.unique(a[:, 0])}


def gt_file(seq: str) -> Path:
    return seq_dir(seq, "FRCNN") / "gt" / "gt.txt"  # identical for the three detectors


def _get(url, start, end) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _index(seq: str) -> dict:
    """Offsets of one sequence's frames inside the remote zip, read from its central directory (cached)."""
    cache = DATA / f"index_{seq}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    with urllib.request.urlopen(urllib.request.Request(FRAMES_URL, method="HEAD")) as r:
        size = int(r.headers["Content-Length"])
    tail = _get(FRAMES_URL, size - (1 << 16), size - 1)
    i = tail.rfind(b"PK")  # ZIP64 end of central directory (the archive is over 4 GB)
    cd_size, cd_off = struct.unpack("<QQ", tail[i + 40:i + 56])
    cd = _get(FRAMES_URL, cd_off, cd_off + cd_size - 1)
    prefix, out, pos = f"MOT17/train/{seq}-FRCNN/img1/".encode(), {}, 0
    while pos < len(cd) and cd[pos:pos + 4] == b"PK":
        method, = struct.unpack("<H", cd[pos + 10:pos + 12])
        csize, usize, nlen, xlen, clen = struct.unpack("<IIHHH", cd[pos + 20:pos + 34])
        off, = struct.unpack("<I", cd[pos + 42:pos + 46])
        name, extra = cd[pos + 46:pos + 46 + nlen], cd[pos + 46 + nlen:pos + 46 + nlen + xlen]
        if name.startswith(prefix):
            vals, j = [], 0  # ZIP64 extra field holds the 64-bit values set to 0xFFFFFFFF above, in order
            while j < len(extra):
                tag, ln = struct.unpack("<HH", extra[j:j + 4])
                if tag == 1:
                    vals = list(struct.unpack(f"<{ln // 8}Q", extra[j + 4:j + 4 + ln]))
                j += 4 + ln
            if usize == 0xFFFFFFFF: usize = vals.pop(0)
            if csize == 0xFFFFFFFF: csize = vals.pop(0)
            if off == 0xFFFFFFFF: off = vals.pop(0)
            out[name[len(prefix):].decode()] = [off, csize, method]
        pos += 46 + nlen + xlen + clen
    cache.write_text(json.dumps(out))
    return out


def _member(off, csize, method) -> bytes:
    head = _get(FRAMES_URL, off, off + 29)
    nlen, xlen = struct.unpack("<HH", head[26:30])
    data = _get(FRAMES_URL, off + 30 + nlen + xlen, off + 30 + nlen + xlen + csize - 1)
    return data if method == 0 else zlib.decompress(data, -15)


def frames(seq: str, first: int, last: int, step: int = 1) -> Path:
    """Fetch frames first..last (every step-th) of a training sequence into data/frames/<seq>/ (once), in parallel."""
    out = DATA / "frames" / seq
    want = [f"{i:06d}.jpg" for i in range(first, last + 1, step) if not (out / f"{i:06d}.jpg").exists()]
    if want:
        out.mkdir(parents=True, exist_ok=True)
        idx = _index(seq)
        with ThreadPoolExecutor(16) as ex:
            for n, data in zip(want, ex.map(lambda n: _member(*idx[n]), want)):
                (out / n).write_bytes(data)
    return out
