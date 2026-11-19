import numpy as np

from src.run import pooled
from src.tracker import Params, iou, to_box, to_z, track


def walk(n, x0, y0, dx, score=1.0, skip=()):
    """One object moving right; frames listed in skip have no detection (occlusion)."""
    return {f: np.array([[x0 + dx * f, y0, x0 + dx * f + 40, y0 + 100, score]]) for f in range(1, n + 1) if f not in skip}


def merge(*ds):
    out = {}
    for d in ds:
        for f, b in d.items():
            out[f] = np.vstack([out[f], b]) if f in out else b
    return out


def ids(rows):
    return {r[1] for r in rows}


def test_iou_and_box_round_trip():
    a = np.array([[0, 0, 10, 10.0]])
    assert np.isclose(iou(a, a)[0, 0], 1) and np.isclose(iou(a, a + [5, 0, 5, 0])[0, 0], 50 / 150)
    b = np.array([12, 30, 52, 130.0])
    assert np.allclose(to_box(to_z(b)), b)


def test_two_walkers_keep_their_identity():
    d = merge(walk(30, 0, 0, 3), walk(30, 0, 400, 3))
    rows = track(d, 30, Params())
    assert len(ids(rows)) == 2


def test_occlusion_needs_max_age():
    d = walk(30, 0, 0, 3, skip=range(12, 16))
    assert len(ids(track(d, 30, Params(max_age=1)))) == 2   # identity lost after the gap
    assert len(ids(track(d, 30, Params(max_age=5)))) == 1   # bridged by the Kalman prediction


def test_second_pass_uses_weak_detections():
    weak = {f: b.copy() for f, b in walk(30, 0, 0, 3).items()}
    for f in range(12, 16):
        weak[f][:, 4] = 0.2
    no = track(weak, 30, Params(min_score=0.5, max_age=1))
    yes = track(weak, 30, Params(min_score=0.5, max_age=1, low_score=0.1))
    assert len(ids(no)) == 2 and len(ids(yes)) == 1


def test_pooled_metrics():
    c = {"idtp": 80, "idfp": 20, "idfn": 20, "num_switches": 5, "num_false_positives": 10, "num_misses": 15,
         "num_objects": 100, "mostly_tracked": 1, "mostly_lost": 0, "num_fragmentations": 2, "num_unique_objects": 2}
    m = pooled([c, c])
    assert np.isclose(m["mota"], 1 - 30 / 100) and np.isclose(m["idf1"], 160 / 200)
