"""Every figure of the README and the demo page comes from here: python -m src.run"""
import itertools
import json
import platform
import subprocess
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .data import DETECTORS, SEQUENCES, detections, info
from .tracker import Params, track

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = Params()  # SORT's published settings: IoU 0.3, max_age 1, min_hits 3, every detection used
IOUS, AGES, HITS = [0.2, 0.3, 0.4], [1, 5, 15, 30], [1, 3]
SCORE_QUANTILES = [0.0, 0.25, 0.5]
COUNTS = ["idtp", "idfp", "idfn", "num_switches", "num_false_positives", "num_misses", "num_objects",
          "mostly_tracked", "mostly_lost", "num_fragmentations", "num_unique_objects"]
VIDEO = {"seq": "MOT17-09", "detector": "SDP", "first": 1, "last": 179, "step": 2}  # fixed before looking at results


def grid(det):
    """Score thresholds are quantiles of the detector's own scores (DPM scores are not probabilities)."""
    s = np.concatenate([d[:, 4] for q in SEQUENCES for d in detections(q, det).values()])
    th = [float(np.quantile(s, q)) for q in SCORE_QUANTILES]
    out = []
    for i, m in enumerate(th):
        for low in ([None] if i == 0 else [None, th[0]]):  # second pass only when some detections were dropped
            for iou_m, age, hits in itertools.product(IOUS, AGES, HITS):
                out.append(Params(min_score=-np.inf if i == 0 else m, iou_match=iou_m, max_age=age, min_hits=hits, low_score=low))
    return out


def counts(job):
    """Run one configuration on every sequence; per-sequence event counts, so any subset can be pooled later."""
    det, p = job
    warnings.simplefilter("ignore")
    from .evaluate import accumulate
    import motmetrics as mm
    mh = mm.metrics.create()
    out = {}
    for s in SEQUENCES:
        acc = accumulate(s, track(detections(s, det), info(s)["frames"], p))
        r = mh.compute(acc, metrics=COUNTS, name=s)
        out[s] = {k: int(r[k].iloc[0]) for k in COUNTS}
    return det, p, out


def pooled(c: list[dict]) -> dict:
    t = {k: sum(x[k] for x in c) for k in COUNTS}
    return {"mota": 1 - (t["num_misses"] + t["num_false_positives"] + t["num_switches"]) / t["num_objects"],
            "idf1": 2 * t["idtp"] / (2 * t["idtp"] + t["idfp"] + t["idfn"]),
            "id_switches": t["num_switches"], "false_positives": t["num_false_positives"], "misses": t["num_misses"],
            "fragmentations": t["num_fragmentations"], "mostly_tracked": t["mostly_tracked"], "mostly_lost": t["mostly_lost"],
            "gt_objects": t["num_unique_objects"]}


def describe(p: Params, det_th: list[float]) -> dict:
    return {"min_score": None if p.min_score == -np.inf else round(p.min_score, 3), "iou_match": p.iou_match,
            "max_age": p.max_age, "min_hits": p.min_hits, "second_pass": p.low_score is not None}


def cpu_name() -> str:
    if platform.system() == "Windows":
        r = subprocess.run(["powershell", "-NoProfile", "-c", "(Get-CimInstance Win32_Processor).Name"],
                           capture_output=True, text=True)
        if r.stdout.strip():
            return r.stdout.strip()
    return platform.processor() or platform.machine()


def video(params: dict[str, Params], path: Path):
    """Side by side, default vs tuned, on the first seconds of one sequence. Same colour = same identity."""
    import cv2
    from PIL import Image
    from .data import frames
    v = VIDEO
    folder = frames(v["seq"], v["first"], v["last"], v["step"])
    d, n = detections(v["seq"], v["detector"]), info(v["seq"])["frames"]
    runs = {k: track(d, n, p) for k, p in params.items()}
    rng = np.random.default_rng(0)
    palette = rng.integers(60, 255, (5000, 3))
    out = []
    for f in range(v["first"], v["last"] + 1, v["step"]):
        img = cv2.imread(str(folder / f"{f:06d}.jpg"))
        panels = []
        for name, rows in runs.items():
            im = img.copy()
            for r in rows:
                if r[0] == f:
                    c = tuple(int(x) for x in palette[r[1] % 5000])
                    cv2.rectangle(im, (int(r[2]), int(r[3])), (int(r[4]), int(r[5])), c, 4)
                    cv2.putText(im, str(r[1]), (int(r[2]), int(r[3]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.4, c, 4)
            ids = len({r[1] for r in rows if r[0] <= f})
            cv2.rectangle(im, (0, 0), (im.shape[1], 70), (24, 24, 24), -1)
            cv2.putText(im, f"{name}: {ids} identities so far", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
            panels.append(cv2.resize(im, (640, 360), interpolation=cv2.INTER_AREA))
        out.append(Image.fromarray(cv2.cvtColor(np.vstack(panels), cv2.COLOR_BGR2RGB)))
    out[0].save(path, save_all=True, append_images=out[1:], duration=1000 * v["step"] // info(v["seq"])["fps"], loop=0, quality=60, method=4)


def readme_block(R) -> str:
    out = ["Leave-one-sequence-out over the 7 MOT17 training sequences: for each sequence, the settings are chosen on the "
           "other 6 (highest IDF1), then scored on it. Counts pooled over the 7 held-out sequences.", "",
           "| detector | tracker | MOTA | IDF1 | identity switches | false positives | misses | mostly tracked | mostly lost |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for det in DETECTORS:
        for k, name in (("default", "SORT defaults"), ("cross_validated", "tuned (held out)")):
            m = R["detectors"][det][k]
            out.append(f"| {det} | {name} | {100 * m['mota']:.1f} % | {100 * m['idf1']:.1f} % | {m['id_switches']:,} | "
                       f"{m['false_positives']:,} | {m['misses']:,} | {m['mostly_tracked']} / {m['gt_objects']} | {m['mostly_lost']} / {m['gt_objects']} |")
    out += ["", "Settings chosen when tuning on all 7 sequences (what would ship):", ""]
    for det in DETECTORS:
        s = R["detectors"][det]["tuned_on_all"]["settings"]
        out.append(f"- **{det}**: IoU {s['iou_match']}, max age {s['max_age']} frames, min hits {s['min_hits']}, "
                   f"score threshold {s['min_score']}, second pass for weak detections: {'yes' if s['second_pass'] else 'no'}.")
    t = R["timing"]
    out.append(f"\n{len(R['grid_size'])} detectors x {max(R['grid_size'].values())} settings searched. Tracker speed: "
               f"{t['frames_per_s']:,.0f} frames/s (median over sequences, detections given), pure NumPy, one thread, {t['cpu']}.")
    return "\n".join(out)


def main():
    jobs = [(det, p) for det in DETECTORS for p in grid(det)] + [(det, DEFAULT) for det in DETECTORS]
    t0 = time.time()
    with Pool() as pool:
        results = pool.map(counts, jobs, chunksize=1)
    print(f"{len(jobs)} runs in {time.time() - t0:.0f} s", flush=True)

    table = {}
    for det, p, c in results:
        table.setdefault(det, {})[p] = c
    R = {"sequences": SEQUENCES, "grid_size": {d: len(grid(d)) for d in DETECTORS}, "detectors": {}}
    for det in DETECTORS:
        T, cands = table[det], grid(det)
        default = pooled([table[det][DEFAULT][s] for s in SEQUENCES])
        held, chosen = [], {}
        for s in SEQUENCES:
            others = [q for q in SEQUENCES if q != s]
            best = max(cands, key=lambda p: pooled([T[p][q] for q in others])["idf1"])
            held.append(T[best][s])
            chosen[s] = describe(best, [])
        best_all = max(cands, key=lambda p: pooled([T[p][q] for q in SEQUENCES])["idf1"])
        R["detectors"][det] = {
            "default": default, "cross_validated": pooled(held), "chosen_per_fold": chosen,
            "tuned_on_all": {"settings": describe(best_all, []), **pooled([T[best_all][q] for q in SEQUENCES])},
            "per_sequence": {s: {"default": pooled([table[det][DEFAULT][s]]), "held_out": pooled([held[i]])}
                             for i, s in enumerate(SEQUENCES)},
        }
        print(det, "default IDF1 %.3f -> held-out %.3f" % (default["idf1"], R["detectors"][det]["cross_validated"]["idf1"]), flush=True)

    fps = []
    for s in SEQUENCES:
        d, n = detections(s, "SDP"), info(s)["frames"]
        t = time.perf_counter()
        track(d, n, DEFAULT)
        fps.append(n / (time.perf_counter() - t))
    R["timing"] = {"frames_per_s": float(np.median(fps)), "cpu": cpu_name()}

    best_v = max([p for p in table[VIDEO["detector"]] if p in set(grid(VIDEO["detector"]))],
                 key=lambda p: pooled([table[VIDEO["detector"]][p][q] for q in SEQUENCES if q != VIDEO["seq"]])["idf1"])
    R["video"] = {**VIDEO, "tuned_settings": describe(best_v, []), "note": "tuned settings chosen without this sequence"}
    (ROOT / "docs").mkdir(exist_ok=True)
    video({"SORT defaults": DEFAULT, "tuned": best_v}, ROOT / "docs" / "tracks.webp")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "results.json").write_text(json.dumps(R, indent=2))
    (ROOT / "docs" / "results.js").write_text("window.RESULTS = " + json.dumps(R) + ";\n")
    readme = ROOT / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        a, b = text.index("<!-- results -->"), text.index("<!-- /results -->")
        readme.write_text(text[:a] + "<!-- results -->\n" + readme_block(R) + "\n" + text[b:], encoding="utf-8")
    print(readme_block(R))


if __name__ == "__main__":
    main()
