"""Tracking metrics with py-motmetrics, following the MOT17 protocol (IoU >= 0.5, distractors removed)."""
import warnings

import numpy as np

# shortcut: py-motmetrics 1.4.0 still calls np.asfarray, removed in NumPy 2; drop this once a fixed release is out
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

import motmetrics as mm
import pandas as pd
from motmetrics.preprocess import preprocessResult

from .data import gt_file, seq_dir

METRICS = ["mota", "idf1", "num_switches", "num_false_positives", "num_misses", "num_fragmentations",
           "mostly_tracked", "mostly_lost", "num_unique_objects"]
_GT = {}


def gt(seq):
    if seq not in _GT:
        _GT[seq] = mm.io.loadtxt(gt_file(seq), fmt="mot15-2D", min_confidence=1)
    return _GT[seq]


def to_frame(rows) -> pd.DataFrame:
    a = np.asarray(rows, float).reshape(-1, 6)
    df = pd.DataFrame({"FrameId": a[:, 0].astype(int), "Id": a[:, 1].astype(int), "X": a[:, 2], "Y": a[:, 3],
                       "Width": a[:, 4] - a[:, 2], "Height": a[:, 5] - a[:, 3], "Confidence": 1.0})
    return df.set_index(["FrameId", "Id"])


def accumulate(seq, rows):
    """Event accumulator for one sequence (counts can be summed over sequences)."""
    g = gt(seq)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = preprocessResult(to_frame(rows), g, str(seq_dir(seq, "FRCNN") / "seqinfo.ini"))
    return mm.utils.compare_to_groundtruth(g, res, "iou", distth=0.5)


def score(accs: dict) -> dict:
    """Overall metrics over several sequences (counts pooled, as the benchmark does)."""
    mh = mm.metrics.create()
    s = mh.compute_many(list(accs.values()), names=list(accs), metrics=METRICS, generate_overall=True)
    o = s.loc["OVERALL"]
    return {k: (float(o[k]) if k in ("mota", "idf1") else int(o[k])) for k in METRICS}
