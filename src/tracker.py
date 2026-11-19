"""Tracking by detection: a constant-velocity Kalman filter per object and IoU matching with the Hungarian algorithm.

Same structure as SORT (Bewley et al., 2016), written from scratch in NumPy, plus an optional second matching
pass for low-score detections (the idea behind ByteTrack, Zhang et al., 2022).
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Params:
    min_score: float = -np.inf   # detections below this are dropped
    iou_match: float = 0.3       # minimum IoU to accept a match
    max_age: int = 1             # frames a track survives without a match
    min_hits: int = 3            # matches before a track is reported
    low_score: float | None = None  # if set: detections in [low_score, min_score) get a second matching pass


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (n, 4) and b (m, 4), boxes as x1, y1, x2, y2."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], axis=1)
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def to_z(box):  # x1, y1, x2, y2 -> centre x, centre y, area, aspect ratio
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + w / 2, box[1] + h / 2, w * h, w / h])


def to_box(x):
    w = np.sqrt(max(x[2] * x[3], 1e-6))
    h = x[2] / w
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


# Kalman model of SORT: state (cx, cy, area, ratio, vx, vy, v_area), constant velocity, ratio constant.
F = np.eye(7)
F[0, 4] = F[1, 5] = F[2, 6] = 1
H = np.eye(4, 7)
R = np.diag([1, 1, 10, 10.0])
Q = np.diag([1, 1, 1, 1, 0.01, 0.01, 1e-4])
P0 = np.diag([10, 10, 10, 10, 1e4, 1e4, 1e4])


class Track:
    def __init__(self, box, tid):
        self.x = np.r_[to_z(box), 0, 0, 0]
        self.P = P0.copy()
        self.id, self.hits, self.age_unmatched, self.streak = tid, 1, 0, 1

    def predict(self):
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.age_unmatched += 1
        if self.age_unmatched > 1:
            self.streak = 0
        return to_box(self.x)

    def update(self, box):
        y = to_z(box) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ H) @ self.P
        self.hits += 1
        self.streak += 1
        self.age_unmatched = 0


def match(tracks_boxes, dets, thr):
    """Hungarian assignment on IoU; returns matched (track, det) pairs, unmatched tracks, unmatched dets."""
    if len(tracks_boxes) == 0 or len(dets) == 0:
        return [], list(range(len(tracks_boxes))), list(range(len(dets)))
    m = iou(tracks_boxes, dets)
    r, c = linear_sum_assignment(-m)
    pairs = [(i, j) for i, j in zip(r, c) if m[i, j] >= thr]
    mt, md = {i for i, _ in pairs}, {j for _, j in pairs}
    return pairs, [i for i in range(len(tracks_boxes)) if i not in mt], [j for j in range(len(dets)) if j not in md]


def track(frames: dict[int, np.ndarray], n_frames: int, p: Params) -> list[tuple]:
    """frames: frame -> array (k, 5) of x1, y1, x2, y2, score. Returns rows (frame, id, x1, y1, x2, y2)."""
    tracks, out, next_id = [], [], 1
    for f in range(1, n_frames + 1):
        d = frames.get(f, np.zeros((0, 5)))
        high = d[d[:, 4] >= p.min_score, :4]
        low = d[(d[:, 4] < p.min_score) & (d[:, 4] >= p.low_score), :4] if p.low_score is not None else np.zeros((0, 4))
        pred = np.array([t.predict() for t in tracks]).reshape(-1, 4)

        pairs, un_t, un_d = match(pred, high, p.iou_match)
        for i, j in pairs:
            tracks[i].update(high[j])
        if len(low):  # second pass: tracks left over may still be explained by a weak detection
            pairs2, rest, _ = match(pred[un_t], low, p.iou_match)
            for i, j in pairs2:
                tracks[un_t[i]].update(low[j])
            un_t = [un_t[i] for i in rest]
        for j in un_d:  # only confident detections start tracks
            tracks.append(Track(high[j], next_id))
            next_id += 1

        tracks = [t for t in tracks if t.age_unmatched <= p.max_age]
        for t in tracks:
            if t.age_unmatched == 0 and (t.streak >= p.min_hits or f <= p.min_hits):
                out.append((f, t.id, *to_box(t.x)))
    return out
