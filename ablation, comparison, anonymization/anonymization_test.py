#!/usr/bin/env python3
"""
Anonymization-before-pose test (COMP825 Assignment 2).

Question: if the image is anonymized BEFORE MediaPipe Pose sees it, does the knee angle change?
In the main pipeline, anonymization happens AFTER pose estimation, so it cannot change the angle.
This test puts anonymization first, as it must be when the pose model runs on privacy-safe input
(for example a low-resolution privacy camera or a remote pose service).

Conditions (each one is the ONLY input MediaPipe Pose receives):
  raw                 original frame (reference, same as the main pipeline)
  face_blur           face box blurred (MediaPipe Face Detection; last box held for 15 frames)
  background_removed  background replaced with grey (MediaPipe Selfie Segmentation)
  silhouette          person shown as a flat grey shape on black (no face, clothing or room)
  pixelate_192        whole frame reduced to 192 px on the long side, then enlarged
  pixelate_96         whole frame reduced to 96 px on the long side, then enlarged

Accuracy metrics (condition minus raw, frames where both detect a pose):
  detection rate, bias, 95% limits of agreement (Bland-Altman), MAE, max error,
  % frames with error > 5 and > 10 deg, Pearson r, repetition-peak MAE.
Privacy proxy: % of sampled frames where a face detector still finds a face
(lower = more private; a proxy, not a proof of anonymity).

Usage:
  python anonymization_test.py --video Sagital_left.mov:left --video Sagital_right.mov:right --out_dir ablation
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from knee_pipeline import LANDMARK_NAMES, calculate_angle, detect_reps, mp, mp_pose

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402

CONDITIONS = ["raw", "face_blur", "background_removed", "silhouette", "pixelate_192", "pixelate_96"]
LABELS = {"raw": "Raw (reference)", "face_blur": "Face blurred", "background_removed": "Background removed",
          "silhouette": "Silhouette only", "pixelate_192": "Pixelated 192 px", "pixelate_96": "Pixelated 96 px"}
FACE_HOLD_FRAMES = 15
PRIVACY_SAMPLE_EVERY = 3
MAX_CONSECUTIVE_READ_FAILS = 30


def _face_box(detections, w, h, scale=1.6):
    """Largest detected face as an expanded pixel box (x0, y0, x1, y1)."""
    best = None
    for d in detections or []:
        b = d.location_data.relative_bounding_box
        area = b.width * b.height
        if best is None or area > best[0]:
            best = (area, b)
    if best is None:
        return None
    b = best[1]
    cx, cy = (b.xmin + b.width / 2) * w, (b.ymin + b.height / 2) * h
    half = max(b.width * w, b.height * h) * scale / 2
    return (max(0, int(cx - half)), max(0, int(cy - half)), min(w, int(cx + half)), min(h, int(cy + half)))


def _pixelate(img, long_side):
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def make_variants(rgb, face_box, mask_u8):
    """Build every anonymized version of one RGB frame."""
    out = {"raw": rgb}
    fb = rgb.copy()
    if face_box is not None:
        x0, y0, x1, y1 = face_box
        if x1 > x0 and y1 > y0:
            roi = fb[y0:y1, x0:x1]
            fb[y0:y1, x0:x1] = _pixelate(cv2.GaussianBlur(roi, (0, 0), max(3, (x1 - x0) / 10)), 8)
    out["face_blur"] = fb
    bg = np.full_like(rgb, 128)
    cv2.copyTo(rgb, mask_u8, bg)
    out["background_removed"] = bg
    out["silhouette"] = cv2.merge([mask_u8 * 170] * 3)
    out["pixelate_192"] = _pixelate(rgb, 192)
    out["pixelate_96"] = _pixelate(rgb, 96)
    return out


def run_video(path, side, out_dir, max_frames=None, sample_frame=150):
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video '{path}'.")
    h_i, k_i, a_i = LANDMARK_NAMES[side]
    poses = {c: mp_pose.Pose(static_image_mode=False, model_complexity=1,
                             min_detection_confidence=0.5, min_tracking_confidence=0.5) for c in CONDITIONS}
    face_det = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    face_probe = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)

    angles = {c: [] for c in CONDITIONS}
    vis = {c: [] for c in CONDITIONS}
    times, face_found, face_boxed = [], [], []
    privacy_hits = {c: 0 for c in CONDITIONS}
    privacy_n = 0
    last_box, box_age = None, 10 ** 9
    fails, prev_t, n = 0, None, 0
    t0 = time.perf_counter()
    try:
        while max_frames is None or n < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                fails += 1
                if fails >= MAX_CONSECUTIVE_READ_FAILS:
                    break
                continue
            fails = 0
            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if not math.isfinite(t_ms) or (prev_t is not None and t_ms <= prev_t):
                t_ms = (prev_t + 1000 / 30) if prev_t is not None else 0.0
            prev_t = t_ms
            times.append(t_ms)
            H, W = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            del frame

            box = _face_box(face_det.process(rgb).detections, W, H)
            face_found.append(box is not None)
            if box is not None:
                last_box, box_age = box, 0
            else:
                box_age += 1
            use_box = last_box if box_age <= FACE_HOLD_FRAMES else None
            face_boxed.append(use_box is not None)

            mask = segmenter.process(rgb).segmentation_mask
            mask_u8 = (mask > 0.5).astype(np.uint8)
            variants = make_variants(rgb, use_box, mask_u8)

            if n == sample_frame:
                tiles = []
                for c in CONDITIONS:
                    tile = cv2.resize(cv2.cvtColor(variants[c], cv2.COLOR_RGB2BGR), (270, 480))
                    cv2.putText(tile, LABELS[c], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                    tiles.append(tile)
                cv2.imwrite(str(Path(out_dir) / f"{path.stem}_anonymization_examples.jpg"), np.hstack(tiles))

            probe = n % PRIVACY_SAMPLE_EVERY == 0
            privacy_n += probe
            for c in CONDITIONS:
                try:
                    lm = poses[c].process(variants[c]).pose_landmarks
                except Exception:
                    lm = None
                if lm is None:
                    angles[c].append(np.nan)
                    vis[c].append(np.nan)
                else:
                    p = lm.landmark
                    angles[c].append(calculate_angle((p[h_i].x * W, p[h_i].y * H), (p[k_i].x * W, p[k_i].y * H),
                                                     (p[a_i].x * W, p[a_i].y * H)))
                    vis[c].append(min(p[h_i].visibility, p[k_i].visibility, p[a_i].visibility))
                if probe:
                    privacy_hits[c] += bool(face_probe.process(variants[c]).detections)
            n += 1
    finally:
        cap.release()
        for p in poses.values():
            p.close()
        face_det.close()
        face_probe.close()
        segmenter.close()

    if n == 0:
        raise ValueError(f"No frames decoded from '{path}'.")
    t = (np.array(times) - times[0]) / 1000.0
    A = {c: np.array(angles[c]) for c in CONDITIONS}
    V = {c: np.array(vis[c]) for c in CONDITIONS}

    # Per-frame CSV
    with open(Path(out_dir) / f"{path.stem}_ablation_frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s"] + [f"angle_{c}" for c in CONDITIONS] + [f"vis_{c}" for c in CONDITIONS]
                   + ["face_detected_raw", "face_box_used"])
        for i in range(n):
            w.writerow([i, round(float(t[i]), 3)]
                       + ["" if np.isnan(A[c][i]) else round(float(A[c][i]), 2) for c in CONDITIONS]
                       + ["" if np.isnan(V[c][i]) else round(float(V[c][i]), 3) for c in CONDITIONS]
                       + [int(face_found[i]), int(face_boxed[i])])

    # Metrics vs raw
    ref = A["raw"]
    raw_reps, _ = detect_reps(t, ref)
    rows = []
    for c in CONDITIONS:
        a = A[c]
        both = np.isfinite(ref) & np.isfinite(a)
        d = a[both] - ref[both]
        row = {"video": path.name, "side": side, "condition": c, "frames": n,
               "detection_rate_pct": round(float(np.isfinite(a).mean() * 100), 1),
               "paired_frames": int(both.sum()),
               "face_detectable_pct": round(privacy_hits[c] / max(privacy_n, 1) * 100, 1)}
        if both.sum() >= 3:
            sd = float(d.std(ddof=1))
            row.update({
                "bias_deg": round(float(d.mean()), 2),
                "loa_low_deg": round(float(d.mean() - 1.96 * sd), 2),
                "loa_high_deg": round(float(d.mean() + 1.96 * sd), 2),
                "mae_deg": round(float(np.abs(d).mean()), 2),
                "max_abs_err_deg": round(float(np.abs(d).max()), 1),
                "pct_err_gt5": round(float((np.abs(d) > 5).mean() * 100), 1),
                "pct_err_gt10": round(float((np.abs(d) > 10).mean() * 100), 1),
                "pearson_r": round(float(np.corrcoef(a[both], ref[both])[0, 1]), 3) if c != "raw" else 1.0,
            })
            if raw_reps and np.isfinite(a).sum() >= 5:
                idx = np.arange(n)
                ok_ = np.isfinite(a)
                sm = median_filter(np.interp(idx, idx[ok_], a[ok_]), size=5, mode="nearest")
                peak_err = [abs(float(sm[r["start_frame"]:r["end_frame"] + 1].max()) - r["peak_deg"]) for r in raw_reps]
                row["rep_peak_mae_deg"] = round(float(np.mean(peak_err)), 2)
        rows.append(row)
    meta = {"video": path.name, "side": side, "frames": n,
            "raw_face_detected_pct": round(float(np.mean(face_found) * 100), 1),
            "face_blur_box_available_pct": round(float(np.mean(face_boxed) * 100), 1),
            "raw_reps": len(raw_reps), "wall_time_s": round(time.perf_counter() - t0, 1)}
    return {"rows": rows, "meta": meta, "t": t, "A": A}


def make_outputs(results, out_dir):
    out_dir = Path(out_dir)
    fields = ["video", "side", "condition", "frames", "detection_rate_pct", "paired_frames", "bias_deg",
              "loa_low_deg", "loa_high_deg", "mae_deg", "max_abs_err_deg", "pct_err_gt5", "pct_err_gt10",
              "pearson_r", "rep_peak_mae_deg", "face_detectable_pct"]
    all_rows = [r for res in results for r in res["rows"]]
    with open(out_dir / "ablation_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    with open(out_dir / "ablation_meta.json", "w") as f:
        json.dump([res["meta"] for res in results], f, indent=2)

    def g(r, k, fmt="{:.2f}"):
        return fmt.format(r[k]) if k in r else "n/a"
    md = ["| Video | Condition | Detection | MAE (deg) | Bias [95% LoA] (deg) | Max err | >5° | r | Rep-peak MAE | Face detectable |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in all_rows:
        loa = f"{g(r, 'bias_deg')} [{g(r, 'loa_low_deg')}, {g(r, 'loa_high_deg')}]" if "bias_deg" in r else "n/a"
        md.append(f"| {r['video']} | {LABELS[r['condition']]} | {r['detection_rate_pct']:.1f}% | {g(r, 'mae_deg')} | "
                  f"{loa} | {g(r, 'max_abs_err_deg', '{:.1f}')} | {g(r, 'pct_err_gt5', '{:.1f}%')} | "
                  f"{g(r, 'pearson_r', '{:.3f}')} | {g(r, 'rep_peak_mae_deg')} | {r['face_detectable_pct']:.1f}% |")
    (out_dir / "ablation_summary.md").write_text("\n".join(md) + "\n")

    # Figure 1: privacy vs accuracy trade-off
    colors = {"raw": "#555555", "face_blur": "#1f77b4", "background_removed": "#2ca02c",
              "silhouette": "#9467bd", "pixelate_192": "#ff7f0e", "pixelate_96": "#d62728"}
    markers = ["o", "s", "^"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    ax = axes[0]
    for k, res in enumerate(results):
        for r in res["rows"]:
            if "mae_deg" not in r:
                continue
            ax.scatter(r["face_detectable_pct"], r["mae_deg"], s=70, color=colors[r["condition"]],
                       marker=markers[k % 3], edgecolor="black", linewidth=0.5,
                       label=LABELS[r["condition"]] if k == 0 else None)
    ax.set_xlabel("Face still detectable (% of sampled frames)  →  less private")
    ax.set_ylabel("Mean absolute angle error vs raw (deg)")
    ax.set_title("(a) Privacy vs accuracy (circle = left video, square = right video)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax = axes[1]
    x = np.arange(len(CONDITIONS))
    width = 0.8 / max(len(results), 1)
    for k, res in enumerate(results):
        det = [next(r for r in res["rows"] if r["condition"] == c)["detection_rate_pct"] for c in CONDITIONS]
        ax.bar(x + (k - (len(results) - 1) / 2) * width, det, width, label=res["meta"]["video"])
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONDITIONS], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Pose detection rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(b) Pose detection rate per condition")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_tradeoff.png")
    plt.close(fig)

    # Figure 2: angle traces, raw vs each condition
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 3.6 * len(results)), dpi=150, squeeze=False)
    for k, res in enumerate(results):
        ax = axes[k, 0]
        for c in CONDITIONS:
            ax.plot(res["t"], res["A"][c], lw=2.2 if c == "raw" else 1, color=colors[c],
                    alpha=1 if c == "raw" else 0.8, label=LABELS[c])
        ax.set_title(f"{res['meta']['video']} ({res['meta']['side']} knee): angle when MediaPipe sees each input")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Knee angle (deg)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_traces.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Anonymization-before-pose test")
    ap.add_argument("--video", action="append", required=True, help="PATH:SIDE, SIDE = left or right")
    ap.add_argument("--out_dir", default="ablation")
    ap.add_argument("--max_frames", type=int, default=None, help="For quick tests only")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    results = []
    for spec in args.video:
        path, _, side = spec.rpartition(":")
        if side not in ("left", "right"):
            print(f"ERROR: give PATH:left or PATH:right, got '{spec}'", file=sys.stderr)
            continue
        print(f"Processing {path} ({side}) ...", flush=True)
        try:
            res = run_video(path, side, args.out_dir, args.max_frames)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            continue
        results.append(res)
        np.savez(Path(args.out_dir) / f"{Path(path).stem}_ablation.npz", t=res["t"],
                 **{c: res["A"][c] for c in CONDITIONS})
        with open(Path(args.out_dir) / f"{Path(path).stem}_ablation_rows.json", "w") as f:
            json.dump({"rows": res["rows"], "meta": res["meta"]}, f, indent=2)
        print("  ", res["meta"], flush=True)
    if results:
        make_outputs(results, args.out_dir)
        print((Path(args.out_dir) / "ablation_summary.md").read_text())
    return 0 if results else 1


def rebuild(out_dir, stems):
    """Combine per-video saved results (lets each video run in a separate process)."""
    results = []
    for s in stems:
        d = np.load(Path(out_dir) / f"{s}_ablation.npz")
        j = json.load(open(Path(out_dir) / f"{s}_ablation_rows.json"))
        results.append({"rows": j["rows"], "meta": j["meta"], "t": d["t"], "A": {c: d[c] for c in CONDITIONS}})
    make_outputs(results, out_dir)
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--rebuild":
        rebuild(sys.argv[2], sys.argv[3:])
        print(open(Path(sys.argv[2]) / "ablation_summary.md").read())
    else:
        sys.exit(main())
