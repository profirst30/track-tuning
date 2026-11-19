# track-tuning: tuning a multi-object tracker with metrics, without fooling yourself

## The problem

A detector draws boxes in each frame; a robot needs more than that. It needs to know that the person in frame 120 is the same as in frame 119, where each object is going, and when to act. That is the tracker's job, and a tracker has settings: how much two boxes must overlap to be the same object, how long a lost object is kept alive, how many sightings before an object is reported, which detections to trust.

Those settings are often left at the values of the paper the code came from. Two questions:

1. **How much do identity consistency and identity switches improve when the settings are tuned with proper tracking metrics?**
2. **Does the gain hold on sequences the tuning never saw, with three different detectors?**

## Method

- **Data**: the 7 training sequences of MOT17 (pedestrians; static and moving cameras; 5,316 frames), with the three sets of public detections that come with the benchmark (DPM, Faster R-CNN, SDP) and the ground truth. No detector is trained here: the tracker is evaluated on fixed detections, so any change comes from the tracker.
- **Tracker** (`src/tracker.py`), written from scratch in NumPy, same structure as SORT:
  - a constant-velocity Kalman filter per object (centre, area, aspect ratio and their speeds);
  - matching between predicted tracks and detections with the Hungarian algorithm on IoU;
  - a track is reported after `min_hits` matches and deleted after `max_age` frames without one;
  - optional second matching pass: detections below the score threshold may still extend an existing track, but never start one (the idea behind ByteTrack).
- **Settings searched** (`src/run.py`): IoU threshold, max age, min hits, score threshold (quantiles of each detector's scores, since DPM scores are not probabilities) and the second pass. Starting point: SORT's published settings.
- **Metrics** (`src/evaluate.py`): py-motmetrics with the MOT17 protocol (match at IoU 0.5, distractor classes removed). **IDF1** measures how consistently each person keeps one identity; **MOTA** combines misses, false positives and identity switches; plus the raw count of **identity switches**.
- **No leakage between tuning and testing**: leave-one-sequence-out. For each sequence, the settings with the best IDF1 on the other six are chosen, then scored on the held-out one. The figures below pool the seven held-out results.

## Results

All figures below are written by `python -m src.run`.

<!-- results -->
Leave-one-sequence-out over the 7 MOT17 training sequences: for each sequence, the settings are chosen on the other 6 (highest IDF1), then scored on it. Counts pooled over the 7 held-out sequences.

| detector | tracker | MOTA | IDF1 | identity switches | false positives | misses | mostly tracked | mostly lost |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DPM | SORT defaults | 22.3 % | 29.0 % | 1,253 | 16,424 | 69,621 | 44 / 546 | 287 / 546 |
| DPM | tuned (held out) | 24.0 % | 37.8 % | 772 | 16,925 | 67,664 | 59 / 546 | 274 / 546 |
| FRCNN | SORT defaults | 45.7 % | 49.7 % | 797 | 5,548 | 54,610 | 96 / 546 | 179 / 546 |
| FRCNN | tuned (held out) | 44.2 % | 52.6 % | 946 | 6,838 | 54,879 | 109 / 546 | 176 / 546 |
| SDP | SORT defaults | 59.3 % | 55.5 % | 1,028 | 4,098 | 40,579 | 157 / 546 | 141 / 546 |
| SDP | tuned (held out) | 59.5 % | 63.3 % | 656 | 3,642 | 41,146 | 157 / 546 | 155 / 546 |

Settings chosen when tuning on all 7 sequences (what would ship):

- **DPM**: IoU 0.3, max age 30 frames, min hits 1, score threshold 0.195, second pass for weak detections: yes.
- **FRCNN**: IoU 0.3, max age 30 frames, min hits 1, score threshold 0.999, second pass for weak detections: yes.
- **SDP**: IoU 0.3, max age 30 frames, min hits 1, score threshold 0.995, second pass for weak detections: yes.

3 detectors x 120 settings searched. Tracker speed: 1,881 frames/s (median over sequences, detections given), pure NumPy, one thread, AMD Ryzen 7 5825U with Radeon Graphics.
<!-- /results -->

What the figures say:

- **Identities hold much better with all three detectors** on sequences the tuning never saw: IDF1 rises, mostly because lost tracks are kept alive longer (every fold chose the longest track life searched) and weak detections may extend a track without starting one.
- **MOTA barely moves, and with Faster R-CNN detections identity switches go up.** IDF1 and the switch count do not always agree; which one to optimise is a product decision, not a detail.
- **The failures are on the fast moving-camera sequences** (MOT17-10 and MOT17-13, per-sequence figures in `results/results.json`): a constant-velocity Kalman filter in image coordinates cannot follow when the camera itself turns, and a long track life then links the wrong people. The fix is camera-motion compensation (or tracking in world coordinates), not more tuning.
- **Tuning is cheap; the protocol is what matters.** Tuned and scored on the same sequences, the gains look slightly larger than the held-out ones above (`tuned_on_all` in `results/results.json`); with more settings to tune, that gap would grow.

## Limits

- **Fixed public detections**: the gains are the tracker's alone; a better detector would move every number.
- **Pedestrians, 2D, one camera**: no multi-camera fusion, no 3D positions or speeds, no robot-specific classes.
- **Grid search over a handful of settings**, not a full optimiser; the Kalman noise values stay those of SORT. Every fold chose the largest max age searched (30 frames), so the grid should be widened.
- **Training sequences only**, since the MOT17 test ground truth is not public; leave-one-sequence-out is the honest substitute, with seven sequences that differ a lot (crowds, moving cameras, night street).
- **CPU timing of the tracker alone**, detections given; no embedded target was measured.

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/activate   # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m src.run        # downloads the labels (10 MB) and 90 frames for the video; about 20 minutes on 8 cores
```

## Data

MOT17, [MOTChallenge](https://motchallenge.net/data/MOT17/): Milan, A., Leal-Taixé, L., Reid, I., Roth, S., Schindler, K. (2016). *MOT16: A Benchmark for Multi-Object Tracking*. arXiv:1603.00831. License: [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). This repository is non-commercial and shares its code and results under the same terms for the data-derived parts.
