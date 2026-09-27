#!/usr/bin/env python3
"""
Privacy-preserving markerless knee-angle pipeline (COMP825 Assignment 2, Volunteer 2).

Pipeline: Camera -> MediaPipe Pose -> Anonymization (skeleton on black canvas) -> Feedback

CORE METHOD = TEAMMATE'S METHOD (unchanged):
  * MediaPipe Pose legacy Solutions API, static_image_mode=False, model_complexity=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5.
  * Angle at the knee from hip/knee/ankle in PIXEL coordinates (dot product / arccos).
  * Per-frame feedback rule feedback_for_angle() (target 90 deg +/- 10 deg).
  * Summary stats computed from angles rounded to 2 dp (same as teammate).
  Verified: angles match the teammate script output to 0.005 deg (rounding) on both videos.

CHANGES (all flagged in the report, none silent):
  1. time_s uses each frame's decoder timestamp (CAP_PROP_POS_MSEC), not frame_idx / CAP_PROP_FPS.
     On iPhone .mov files CAP_PROP_FPS returned 29.51 / 29.69 fps, but the decoded stream is 30 fps.
  2. Output video is H.264 (browser-playable) when ffmpeg is available. Teammate's mp4v is the fallback.
  3. Canvas size is taken from the first decoded frame, not from container metadata.
  4. Read errors and inference errors are counted and skipped instead of stopping or crashing.
  5. Added outputs: landmark visibility, clinical flexion (180 - angle), repetition detection,
     a per-repetition extension check (PROPOSED FIX for the inverted feedback rule), PNG graph,
     Table-1-format summary, and an optional left-vs-right comparison.
  6. The anonymized video is rendered at max 1280 px on the long side (angle maths still uses
     full-resolution coordinates). The measured leg is highlighted in magenta.

Privacy: the raw frame exists in memory only for one MediaPipe call. It is never written or shown.

Requirements: Python 3.10-3.12, mediapipe==0.10.21 (newer releases do not include mp.solutions).

Usage:
  python knee_pipeline.py --video Sagital_left.mov:left --video Sagital_right.mov:right \
      --out_dir outputs --compare
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except AttributeError as exc:  # mediapipe >= 0.10.30 removed the legacy Solutions API
    raise ImportError(
        "This pipeline needs the MediaPipe legacy Solutions API (mp.solutions.pose). "
        "Install it with: pip install mediapipe==0.10.21"
    ) from exc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

VIDEO_EXTS = (".mov", ".mp4", ".m4v", ".avi", ".mkv")
LOW_VISIBILITY = 0.5          # MediaPipe also hides landmarks below 0.5 when drawing
MAX_CONSECUTIVE_READ_FAILS = 30

LANDMARK_NAMES = {            # unchanged from teammate
    "left": (mp_pose.PoseLandmark.LEFT_HIP,
             mp_pose.PoseLandmark.LEFT_KNEE,
             mp_pose.PoseLandmark.LEFT_ANKLE),
    "right": (mp_pose.PoseLandmark.RIGHT_HIP,
              mp_pose.PoseLandmark.RIGHT_KNEE,
              mp_pose.PoseLandmark.RIGHT_ANKLE),
}


# ----------------------------------------------------------------------------------------
# Teammate functions (UNCHANGED)
# ----------------------------------------------------------------------------------------
def calculate_angle(a, b, c):
    """Angle at point b (in degrees), given three (x, y) points. Unchanged from teammate."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def feedback_for_angle(angle, target_min_angle, tolerance=10):
    """Per-frame rule from the report. Unchanged from teammate.
    FLAGGED BUG: this rule rewards a bent knee. For a seated knee extension it tells the
    user to 'bend further' at full extension and 'good' while resting."""
    if angle > target_min_angle + tolerance:
        return "Bend your knee further to reach target depth"
    elif angle < target_min_angle - tolerance:
        return "Over-flexing - ease back slightly"
    else:
        return "Good range of motion"


# ----------------------------------------------------------------------------------------
# Proposed fix (additional output, the teammate rule stays in the 'feedback' column)
# ----------------------------------------------------------------------------------------
def extension_rep_feedback(peak_angle, ext_target, tolerance):
    """Judge one repetition by its most extended angle. 180 deg = straight leg.
    ext_target is a PLACEHOLDER. The team MUST choose and justify a clinical target."""
    if peak_angle >= ext_target - tolerance:
        return "Good range of motion"
    return "Straighten your knee further"


def detect_reps(t, angle, min_prominence=30.0, min_separation_s=1.0, kernel=5):
    """Find extension repetitions. Peaks = most extended points of a 5-frame median-filtered
    signal. A repetition runs from the lowest point before its peak to the lowest point after it."""
    angle = np.asarray(angle, dtype=float)
    valid = np.isfinite(angle)
    if valid.sum() < max(kernel, 3):
        return [], np.full(len(angle), np.nan)
    idx = np.arange(len(angle))
    filled = np.interp(idx, idx[valid], angle[valid])
    smooth = median_filter(filled, size=kernel, mode="nearest")
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    distance = max(1, int(round(min_separation_s / max(dt, 1e-6))))
    peaks, _ = find_peaks(smooth, prominence=min_prominence, distance=distance)
    if len(peaks) == 0:
        return [], smooth
    bounds = [int(np.argmin(smooth[: peaks[0] + 1]))]
    for a, b in zip(peaks[:-1], peaks[1:]):
        bounds.append(int(a + np.argmin(smooth[a: b + 1])))
    bounds.append(int(peaks[-1] + np.argmin(smooth[peaks[-1]:])))
    reps = []
    for i, p in enumerate(peaks):
        s, e = bounds[i], bounds[i + 1]
        reps.append({
            "rep": i + 1, "start_frame": s, "peak_frame": int(p), "end_frame": e,
            "start_s": round(float(t[s]), 3), "peak_s": round(float(t[p]), 3),
            "end_s": round(float(t[e]), 3),
            "peak_deg": round(float(smooth[p]), 1),
            "start_deg": round(float(smooth[s]), 1),
            "rom_deg": round(float(smooth[p] - smooth[s]), 1),
            "duration_s": round(float(t[e] - t[s]), 2),
        })
    return reps, smooth


# ----------------------------------------------------------------------------------------
# Pass 1: decode + pose estimation (raw pixels are discarded immediately)
# ----------------------------------------------------------------------------------------
def _extract_landmarks(input_path, progress_cb=None):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video '{input_path}'. The file is missing, damaged or not a video.")
    c_fps = cap.get(cv2.CAP_PROP_FPS)
    c_fps = c_fps if (c_fps and math.isfinite(c_fps) and c_fps > 0) else None
    c_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    nominal_dt_ms = 1000.0 / (c_fps or 30.0)

    width = height = None
    landmarks, times_ms, infer_ms = [], [], []
    read_failures = inference_errors = consecutive_fail = 0
    prev_t = None
    wall_start = time.perf_counter()

    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                consecutive_fail += 1
                read_failures += 1
                if consecutive_fail >= MAX_CONSECUTIVE_READ_FAILS:
                    read_failures -= consecutive_fail   # trailing failures = end of file
                    break
                continue
            consecutive_fail = 0

            if frame.ndim == 2 or frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            if width is None:
                height, width = frame.shape[:2]
            elif frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))

            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if not math.isfinite(t_ms) or (prev_t is not None and t_ms <= prev_t):
                t_ms = prev_t + nominal_dt_ms if prev_t is not None else 0.0
            prev_t = t_ms

            # --- Markerless pose estimation: the only use of the raw frame ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            del frame
            t0 = time.perf_counter()
            try:
                lm = pose.process(rgb).pose_landmarks
            except Exception:  # damaged frame content
                lm = None
                inference_errors += 1
            infer_ms.append((time.perf_counter() - t0) * 1000)
            del rgb
            # --- Anonymization boundary: only landmark coordinates are kept ---

            landmarks.append(lm)
            times_ms.append(t_ms)
            if progress_cb is not None:
                progress_cb(min(len(landmarks) / c_count, 0.99) if c_count else 0.5,
                            f"Pose estimation: frame {len(landmarks)}")
    cap.release()
    t = np.asarray(times_ms, dtype=float)
    if len(t):
        t = (t - t[0]) / 1000.0
    return {
        "landmarks": landmarks, "t": t, "width": width, "height": height,
        "container_fps": c_fps, "container_frame_count": c_count,
        "read_failures": read_failures, "inference_errors": inference_errors,
        "infer_ms": infer_ms, "wall_s": time.perf_counter() - wall_start,
    }


def _leg_series(landmarks, side, width, height):
    h_i, k_i, a_i = LANDMARK_NAMES[side]
    n = len(landmarks)
    ang, vis = np.full(n, np.nan), np.full(n, np.nan)
    pts = [None] * n
    for i, lm in enumerate(landmarks):
        if lm is None:
            continue
        p = lm.landmark
        hip = (p[h_i].x * width, p[h_i].y * height)
        knee = (p[k_i].x * width, p[k_i].y * height)
        ankle = (p[a_i].x * width, p[a_i].y * height)
        ang[i] = calculate_angle(hip, knee, ankle)
        vis[i] = min(p[h_i].visibility, p[k_i].visibility, p[a_i].visibility)
        pts[i] = ((p[h_i].x, p[h_i].y), (p[k_i].x, p[k_i].y), (p[a_i].x, p[a_i].y))
    return ang, vis, pts


# ----------------------------------------------------------------------------------------
# Pass 2: render the anonymized video from stored landmarks only
# ----------------------------------------------------------------------------------------
def _ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def _render_video(out_path, landmarks, pts, angles, feedback, rep_msgs, width, height,
                  fps, render_long_side=1280):
    scale = 1.0
    if render_long_side and max(width, height) > render_long_side:
        scale = render_long_side / max(width, height)
    w = max(2, int(round(width * scale / 2)) * 2)
    h = max(2, int(round(height * scale / 2)) * 2)
    tmp_path = str(out_path) + ".mp4v.mp4"
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV cannot create the output video '{tmp_path}'.")
    for i, lm in enumerate(landmarks):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        if lm is not None:
            mp_drawing.draw_landmarks(   # same drawing specs as teammate
                canvas, lm, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=2),
            )
            hk = [(int(x * w), int(y * h)) for x, y in pts[i]]
            cv2.line(canvas, hk[0], hk[1], (255, 0, 255), 4)
            cv2.line(canvas, hk[1], hk[2], (255, 0, 255), 4)
            cv2.putText(canvas, f"Knee angle: {angles[i]:.1f} deg", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(canvas, feedback[i], (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(canvas, "No pose detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if rep_msgs[i]:
            cv2.putText(canvas, rep_msgs[i], (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        writer.write(canvas)
    writer.release()

    exe = _ffmpeg_exe()
    if exe:
        cmd = [exe, "-y", "-loglevel", "error", "-i", tmp_path, "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
               "-movflags", "+faststart", str(out_path)]
        try:
            subprocess.run(cmd, check=True, timeout=900, capture_output=True)
            os.remove(tmp_path)
            return "h264"
        except Exception:
            pass
    os.replace(tmp_path, out_path)
    return "mp4v (may not play in web browsers)"


# ----------------------------------------------------------------------------------------
# Graph and tables
# ----------------------------------------------------------------------------------------
def _pct(x):
    return f"{x * 100:.1f}".rstrip("0").rstrip(".") + "%"


def table1_row(summary):
    """Same columns and rounding as Table 1 of the draft report."""
    def f(v):
        return "n/a" if v is None else f"{v:.1f}"
    return {"Detection rate": _pct(summary["detection_rate"]),
            "Min angle (deg)": f(summary["min_angle_deg"]),
            "Max angle (deg)": f(summary["max_angle_deg"]),
            "Mean angle (deg)": f(summary["mean_angle_deg"])}


def plot_angle(t, angle, vis, reps, summary, png_path):
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
    valid = np.isfinite(angle)
    ax.axhspan(summary["target_min_angle"] - summary["tolerance"],
               summary["target_min_angle"] + summary["tolerance"], color="#999999", alpha=0.18,
               label=f"Report rule 'good' band ({summary['target_min_angle']:.0f} ± {summary['tolerance']:.0f}°)")
    ax.axhline(summary["ext_target"] - summary["tolerance"], color="#b8860b", ls="--", lw=1,
               label=f"Extension check threshold ({summary['ext_target'] - summary['tolerance']:.0f}°, placeholder)")
    if valid.any():
        ax.plot(t, angle, color="#1f4e79", lw=1.2, label="Knee angle per frame")
        low = valid & (vis < LOW_VISIBILITY)
        if low.any():
            ax.scatter(t[low], angle[low], s=10, color="#c0392b", zorder=3,
                       label=f"Landmark visibility < {LOW_VISIBILITY}")
    else:
        ax.text(0.5, 0.5, "No pose detected in any frame", transform=ax.transAxes, ha="center")
    if reps:
        ax.plot([r["peak_s"] for r in reps], [r["peak_deg"] for r in reps], "v", color="#d35400",
                ms=7, label="Repetition peak (5-frame median)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hip-knee-ankle angle (deg)\n180 = straight leg")
    ax.set_title(f"{summary['video']}: {summary['side']} knee, {summary['total_frames']} frames, "
                 f"{summary['duration_s']:.2f} s, detection {_pct(summary['detection_rate'])}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def _stats(values):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return None
    return {"mean": round(float(v.mean()), 1), "sd": round(float(v.std(ddof=1)), 1) if len(v) > 1 else 0.0,
            "min": round(float(v.min()), 1), "max": round(float(v.max()), 1)}


# ----------------------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------------------
def process_video(input_path, out_dir, side="auto", target_min_angle=90.0, tolerance=10.0,
                  ext_target=170.0, render_long_side=1280, progress_cb=None, label=None):
    """Run the full pipeline on one video. side: 'left', 'right' or 'auto'.
    Returns a dict with summary, per-frame table and output file paths."""
    input_path = Path(input_path)
    stem = label or input_path.stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ex = _extract_landmarks(input_path, progress_cb)
    lms, t = ex["landmarks"], ex["t"]
    n = len(lms)
    if n == 0:
        raise ValueError(f"No frames could be decoded from '{input_path}'.")
    width, height = ex["width"], ex["height"]

    both = {s: _leg_series(lms, s, width, height) for s in ("left", "right")}
    side_mode = "manual"
    if side == "auto":
        side_mode = "auto"
        mv = {s: np.nanmean(both[s][1]) if np.isfinite(both[s][1]).any() else -1 for s in both}
        side = "left" if mv["left"] > mv["right"] else "right"
    other = "right" if side == "left" else "left"
    angle_raw, vis, pts = both[side]
    other_angle, other_vis, _ = both[other]

    # Teammate rounding: 2 dp, stats from rounded values
    angle = np.round(angle_raw, 2)
    detected = int(np.isfinite(angle).sum())
    feedback = [feedback_for_angle(a, target_min_angle, tolerance) if np.isfinite(a) else ""
                for a in angle_raw]

    reps, _ = detect_reps(t, angle)
    rep_id = [""] * n
    rep_fb = [""] * n
    overlay = [""] * n
    for r in reps:
        r["feedback"] = extension_rep_feedback(r["peak_deg"], ext_target, tolerance)
        last = r["end_frame"] if r["rep"] == len(reps) else r["end_frame"] - 1
        for i in range(r["start_frame"], last + 1):
            rep_id[i] = r["rep"]
            rep_fb[i] = r["feedback"]
        for i in range(r["peak_frame"], last + 1):  # message is known once the peak has passed
            overlay[i] = f"Rep {r['rep']}: {r['feedback']} (peak {r['peak_deg']:.0f} deg)"

    dts = np.diff(t)
    measured_fps = float(1.0 / np.median(dts)) if len(dts) else (ex["container_fps"] or 30.0)
    duration = float(t[-1] + (np.median(dts) if len(dts) else 0.0))
    render_fps = round(measured_fps, 3)

    prefix = out_dir / stem
    video_path = Path(f"{prefix}_anonymized.mp4")
    csv_path = Path(f"{prefix}_knee_angle_data.csv")
    png_path = Path(f"{prefix}_knee_angle.png")
    table_path = Path(f"{prefix}_summary_table.csv")
    json_path = Path(f"{prefix}_summary.json")

    if progress_cb is not None:
        progress_cb(0.99, "Rendering anonymized video")
    codec = _render_video(video_path, lms, pts, angle_raw, feedback, overlay, width, height,
                          render_fps, render_long_side)

    fieldnames = ["frame", "time_s", "knee_angle_deg", "feedback", "clinical_flexion_deg",
                  "min_landmark_visibility", "rep", "rep_feedback",
                  "other_leg_angle_deg", "other_leg_min_visibility"]
    rows = []
    for i in range(n):
        ok = np.isfinite(angle[i])
        rows.append({
            "frame": i,
            "time_s": round(float(t[i]), 3),
            "knee_angle_deg": float(angle[i]) if ok else "",
            "feedback": feedback[i],
            "clinical_flexion_deg": round(180.0 - float(angle[i]), 2) if ok else "",
            "min_landmark_visibility": round(float(vis[i]), 3) if ok else "",
            "rep": rep_id[i],
            "rep_feedback": rep_fb[i],
            "other_leg_angle_deg": round(float(other_angle[i]), 2) if np.isfinite(other_angle[i]) else "",
            "other_leg_min_visibility": round(float(other_vis[i]), 3) if np.isfinite(other_vis[i]) else "",
        })
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    valid = angle[np.isfinite(angle)]
    fb_counts = {}
    for m in feedback:
        if m:
            fb_counts[m] = fb_counts.get(m, 0) + 1
    ov = other_angle[np.isfinite(other_angle)]
    summary = {
        "video": input_path.name, "side": side, "side_selection": side_mode,
        "mediapipe_version": mp.__version__, "opencv_version": cv2.__version__,
        "resolution": f"{width}x{height}",
        "total_frames": n, "frames_with_pose_detected": detected,
        "detection_rate": round(detected / n, 3),
        "min_angle_deg": round(float(valid.min()), 2) if len(valid) else None,
        "max_angle_deg": round(float(valid.max()), 2) if len(valid) else None,
        "mean_angle_deg": round(float(valid.mean()), 2) if len(valid) else None,
        "duration_s": round(duration, 3),
        "measured_fps": round(measured_fps, 3),
        "container_fps_reported": round(ex["container_fps"], 3) if ex["container_fps"] else None,
        "container_frame_count_reported": ex["container_frame_count"],
        "teammate_time_error_at_last_frame_s": (
            round((n - 1) / ex["container_fps"] - float(t[-1]), 3) if ex["container_fps"] else None),
        "read_failures_skipped": ex["read_failures"], "inference_errors": ex["inference_errors"],
        "mean_min_visibility": round(float(np.nanmean(vis)), 3) if detected else None,
        "frames_low_visibility_pct": round(float(np.mean(vis[np.isfinite(vis)] < LOW_VISIBILITY)) * 100, 1)
        if detected else None,
        "other_leg": {
            "side": other,
            "mean_min_visibility": round(float(np.nanmean(other_vis)), 3) if len(ov) else None,
            "frames_low_visibility_pct": round(float(np.mean(other_vis[np.isfinite(other_vis)] < LOW_VISIBILITY)) * 100, 1)
            if len(ov) else None,
            "min_angle_deg": round(float(ov.min()), 1) if len(ov) else None,
            "max_angle_deg": round(float(ov.max()), 1) if len(ov) else None,
        },
        "target_min_angle": target_min_angle, "tolerance": tolerance, "ext_target": ext_target,
        "teammate_feedback_pct_of_detected_frames": {
            k: round(v / detected * 100, 1) for k, v in sorted(fb_counts.items())} if detected else {},
        "reps": reps,
        "rep_count": len(reps),
        "rep_peak_deg": _stats([r["peak_deg"] for r in reps]),
        "rep_start_deg": _stats([r["start_deg"] for r in reps]),
        "rep_rom_deg": _stats([r["rom_deg"] for r in reps]),
        "rep_duration_s": _stats([r["duration_s"] for r in reps]),
        "reps_good": sum(r["feedback"] == "Good range of motion" for r in reps),
        "median_inference_ms_per_frame": round(float(np.median(ex["infer_ms"])), 1) if ex["infer_ms"] else None,
        "pipeline_wall_time_s": round(ex["wall_s"], 1),
        "video_codec": codec,
    }
    summary["table1"] = table1_row(summary)

    with open(table_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary["table1"].keys()))
        w.writeheader()
        w.writerow(summary["table1"])
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    plot_angle(t, angle, vis, reps, summary, png_path)
    if progress_cb is not None:
        progress_cb(1.0, "Done")

    return {"summary": summary, "rows": rows, "t": t, "angle": angle, "vis": vis,
            "other_angle": other_angle, "other_vis": other_vis,
            "paths": {"video": str(video_path), "csv": str(csv_path), "png": str(png_path),
                      "table": str(table_path), "json": str(json_path)}}


# ----------------------------------------------------------------------------------------
# Left vs right comparison
# ----------------------------------------------------------------------------------------
def compare_runs(results, out_dir):
    out_dir = Path(out_dir)

    def ms(s):
        return "n/a" if s is None else f"{s['mean']:.1f} ± {s['sd']:.1f}"

    def num(v, spec=".1f", suffix=""):
        return "n/a" if v is None else f"{v:{spec}}{suffix}"

    metrics = [
        ("Measured leg (camera side)", lambda s: f"{s['side']} ({s['video']})"),
        ("Frames / duration", lambda s: f"{s['total_frames']} / {s['duration_s']:.2f} s"),
        ("Detection rate", lambda s: s["table1"]["Detection rate"]),
        ("Min / max / mean angle (deg)", lambda s: f"{s['table1']['Min angle (deg)']} / "
         f"{s['table1']['Max angle (deg)']} / {s['table1']['Mean angle (deg)']}"),
        ("Mean landmark visibility, measured leg", lambda s: num(s["mean_min_visibility"], ".2f")),
        ("Frames with visibility < 0.5, measured leg", lambda s: num(s["frames_low_visibility_pct"], ".1f", "%")),
        ("Repetitions detected", lambda s: str(s["rep_count"])),
        ("Peak extension per rep (deg), mean ± SD", lambda s: ms(s["rep_peak_deg"])),
        ("Start (flexed) angle per rep (deg), mean ± SD", lambda s: ms(s["rep_start_deg"])),
        ("Range of motion per rep (deg), mean ± SD", lambda s: ms(s["rep_rom_deg"])),
        ("Rep duration (s), mean ± SD", lambda s: ms(s["rep_duration_s"])),
        ("Report rule: frames 'Good range of motion'",
         lambda s: f"{s['teammate_feedback_pct_of_detected_frames'].get('Good range of motion', 0):.1f}%"),
        ("Report rule: frames 'Bend your knee further'",
         lambda s: f"{s['teammate_feedback_pct_of_detected_frames'].get('Bend your knee further to reach target depth', 0):.1f}%"),
        ("Extension check: reps 'Good range of motion'", lambda s: f"{s['reps_good']} of {s['rep_count']}"),
        ("Far (occluded) leg: frames with visibility < 0.5",
         lambda s: num(s["other_leg"]["frames_low_visibility_pct"], ".1f", "%")),
        ("Far (occluded) leg: angle range (deg)",
         lambda s: f"{num(s['other_leg']['min_angle_deg'])} to {num(s['other_leg']['max_angle_deg'])}"),
    ]
    names = [r["summary"]["video"] for r in results]
    table = [{"Metric": m, **{nm: fn(r["summary"]) for nm, r in zip(names, results)}} for m, fn in metrics]
    with open(out_dir / "comparison_left_vs_right.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Metric"] + names)
        w.writeheader()
        w.writerows(table)
    md = ["| Metric | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    md += ["| " + row["Metric"] + " | " + " | ".join(row[nm] for nm in names) + " |" for row in table]
    t1 = ["| Video | Detection rate | Min angle (deg) | Max angle (deg) | Mean angle (deg) |",
          "|---|---|---|---|---|"]
    t1 += [f"| {r['summary']['video']} ({r['summary']['side']} knee) | " +
           " | ".join(r["summary"]["table1"].values()) + " |" for r in results]
    (out_dir / "comparison_left_vs_right.md").write_text(
        "## Table 1 format (Volunteer 2)\n\n" + "\n".join(t1) +
        "\n\n## Left vs right sagittal comparison\n\n" + "\n".join(md) + "\n")

    colors = ["#1f4e79", "#c0392b", "#27ae60", "#8e44ad"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    ax = axes[0, 0]
    for k, r in enumerate(results):
        s = r["summary"]
        for j, rep in enumerate(s["reps"]):
            seg = r["angle"][rep["start_frame"]: rep["end_frame"] + 1]
            if len(seg) < 3 or not np.isfinite(seg).any():
                continue
            x = np.linspace(0, 100, len(seg))
            ax.plot(x, seg, color=colors[k], alpha=0.6, lw=1,
                    label=f"{s['video']} ({s['side']} knee)" if j == 0 else None)
    ax.set_title("(a) Each repetition, time-normalised")
    ax.set_xlabel("Repetition progress (%)")
    ax.set_ylabel("Knee angle (deg)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for k, r in enumerate(results):
        s = r["summary"]
        reps = s["reps"]
        xs = np.arange(1, len(reps) + 1) + (k - (len(results) - 1) / 2) * 0.15
        ax.plot(xs, [x["peak_deg"] for x in reps], "o", color=colors[k], label=f"{s['side']}: peak extension")
        ax.plot(xs, [x["rom_deg"] for x in reps], "s", mfc="none", color=colors[k], label=f"{s['side']}: range of motion")
    ax.set_title("(b) Per-repetition peak extension and range of motion")
    ax.set_xlabel("Repetition")
    ax.set_ylabel("Degrees")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    for k, r in enumerate(results[:2]):
        ax = axes[1, k]
        s = r["summary"]
        ax.plot(r["t"], r["angle"], color=colors[k], lw=1.2,
                label=f"Near leg ({s['side']}), vis {num(s['mean_min_visibility'], '.2f')}")
        ax.plot(r["t"], r["other_angle"], color="#7f8c8d", lw=1,
                label=f"Far leg ({s['other_leg']['side']}), vis {num(s['other_leg']['mean_min_visibility'], '.2f')}")
        low = np.isfinite(r["other_angle"]) & (r["other_vis"] < LOW_VISIBILITY)
        ax.scatter(r["t"][low], r["other_angle"][low], s=6, color="#e67e22", zorder=3,
                   label="Far leg, visibility < 0.5")
        ax.set_title(f"({'cd'[k]}) {s['video']}: both legs 'detected' in every frame")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Knee angle (deg)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_left_vs_right.png")
    plt.close(fig)
    return table


def load_result(json_path):
    """Rebuild a result from saved *_summary.json and *_knee_angle_data.csv files."""
    json_path = Path(json_path)
    with open(json_path) as f:
        summary = json.load(f)
    csv_path = Path(str(json_path).replace("_summary.json", "_knee_angle_data.csv"))
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    def col(name):
        return np.array([float(r[name]) if r[name] != "" else np.nan for r in rows])
    return {"summary": summary, "rows": rows, "t": col("time_s"), "angle": col("knee_angle_deg"),
            "vis": col("min_landmark_visibility"), "other_angle": col("other_leg_angle_deg"),
            "other_vis": col("other_leg_min_visibility"), "paths": {"json": str(json_path)}}


def main():
    ap = argparse.ArgumentParser(description="Anonymized knee-angle pipeline (teammate method).")
    ap.add_argument("--video", action="append", default=[],
                    help="PATH or PATH:SIDE (SIDE = left, right or auto). Repeat for several videos.")
    ap.add_argument("--compare_from", nargs="+", default=[],
                    help="Saved *_summary.json files to compare without reprocessing.")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--target_min_angle", type=float, default=90.0, help="Teammate rule target (deg)")
    ap.add_argument("--tolerance", type=float, default=10.0)
    ap.add_argument("--ext_target", type=float, default=170.0,
                    help="Per-rep extension target (deg). PLACEHOLDER: justify before reporting.")
    ap.add_argument("--render_long_side", type=int, default=1280, help="0 = native resolution")
    ap.add_argument("--compare", action="store_true", help="Build the left-vs-right comparison")
    args = ap.parse_args()

    if not args.video and not args.compare_from:
        ap.error("give at least one --video or --compare_from")
    results = [load_result(p) for p in args.compare_from]
    if args.compare_from:
        args.compare = True
    for spec in args.video:
        path, side = spec, "auto"
        head, sep, tail = spec.rpartition(":")
        if sep and tail.lower() in ("left", "right", "auto"):
            path, side = head, tail.lower()
        print(f"Processing {path} (side={side}) ...", flush=True)
        try:
            res = process_video(path, Path(args.out_dir) / Path(path).stem, side=side,
                                target_min_angle=args.target_min_angle, tolerance=args.tolerance,
                                ext_target=args.ext_target, render_long_side=args.render_long_side)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            continue
        s = res["summary"]
        print("  Table 1:", s["table1"])
        print(f"  frames={s['total_frames']} fps={s['measured_fps']} reps={s['rep_count']} "
              f"codec={s['video_codec']}")
        for k, v in res["paths"].items():
            print(f"  {k}: {v}")
        results.append(res)
    if args.compare and len(results) >= 2:
        compare_runs(results, args.out_dir)
        print(f"Comparison written to {args.out_dir}/comparison_left_vs_right.(csv|md|png)")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
