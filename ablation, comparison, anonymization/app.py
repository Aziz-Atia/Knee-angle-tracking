"""
Streamlit UI for the anonymized knee-angle pipeline.

Run:  streamlit run app.py
Optional: set KNEE_VIDEO_DIR to the folder that holds your videos (default: ./videos).

Privacy: the app never displays the raw video. An uploaded file is saved to a temporary
folder only for processing and is deleted when processing ends.
"""
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from knee_pipeline import VIDEO_EXTS, process_video

st.set_page_config(page_title="Anonymised knee-angle tracker", layout="wide")
st.markdown(
    """<style>
    .block-container {max-width: 1200px; padding-top: 2rem;}
    h1 {font-weight: 650; letter-spacing: -0.01em;}
    .fb-good {border-left: 4px solid #2e7d32; padding: 0.4rem 0.8rem; margin: 0.25rem 0;}
    .fb-fix  {border-left: 4px solid #b8860b; padding: 0.4rem 0.8rem; margin: 0.25rem 0;}
    </style>""",
    unsafe_allow_html=True,
)

st.title("Anonymised knee-angle tracker")
st.caption("MediaPipe Pose, skeleton-only output. The raw video is never shown or kept.")

VIDEO_DIR = Path(os.environ.get("KNEE_VIDEO_DIR", "videos"))

# ------------------------------------------------------------------ sidebar: inputs
with st.sidebar:
    st.header("1. Choose a video")
    source = st.radio("Source", ["Upload a video", "Pick from folder"], label_visibility="collapsed")
    uploaded, picked = None, None
    if source == "Upload a video":
        uploaded = st.file_uploader("Video file", type=[e.lstrip(".") for e in VIDEO_EXTS])
    else:
        folder = Path(st.text_input("Folder", value=str(VIDEO_DIR)))
        files = sorted(p for p in folder.glob("*") if p.suffix.lower() in VIDEO_EXTS) if folder.is_dir() else []
        if files:
            picked = st.selectbox("Video", files, format_func=lambda p: p.name)
        else:
            st.info(f"No videos found in '{folder}'. Add .mov or .mp4 files, or upload one.")

    st.header("2. Settings")
    side_label = st.selectbox("Measured leg", ["Auto (most visible leg)", "Left", "Right"],
                              help="Use the leg nearest the camera. Auto picks the leg with the "
                                   "higher mean landmark visibility.")
    with st.expander("Feedback thresholds"):
        target = st.number_input("Report rule target (deg)", value=90.0, step=5.0,
                                 help="Teammate rule: 'good' when angle is target ± tolerance.")
        tol = st.number_input("Tolerance (deg)", value=10.0, min_value=0.0, step=1.0)
        ext_target = st.number_input("Extension target (deg, placeholder)", value=170.0,
                                     min_value=90.0, max_value=180.0, step=1.0,
                                     help="Per-repetition check: 'good' when the peak angle "
                                          "reaches target - tolerance. Justify this value in the report.")
    ready = uploaded is not None or picked is not None
    run = st.button("Run analysis", type="primary", disabled=not ready, width="stretch")

side = {"Auto (most visible leg)": "auto", "Left": "left", "Right": "right"}[side_label]

# ------------------------------------------------------------------ run pipeline
if run:
    tmp_in = None
    out_dir = tempfile.mkdtemp(prefix="knee_out_")
    bar = st.progress(0.0, text="Starting")

    def progress(frac, msg):
        bar.progress(float(min(max(frac, 0.0), 1.0)), text=msg)

    try:
        if uploaded is not None:
            tmp_in = tempfile.mkdtemp(prefix="knee_in_")
            in_path = Path(tmp_in) / Path(uploaded.name).name
            with open(in_path, "wb") as f:
                shutil.copyfileobj(uploaded, f)
            name = uploaded.name
        else:
            in_path, name = picked, picked.name
        res = process_video(in_path, out_dir, side=side, target_min_angle=target,
                            tolerance=tol, ext_target=ext_target, progress_cb=progress,
                            label=Path(name).stem)
        st.session_state["result"] = res
        bar.empty()
    except Exception as exc:  # show a clear message instead of a stack trace
        bar.empty()
        st.session_state.pop("result", None)
        st.error(f"Processing stopped: {exc}")
    finally:
        if tmp_in:
            shutil.rmtree(tmp_in, ignore_errors=True)   # delete the raw upload

# ------------------------------------------------------------------ results
res = st.session_state.get("result")
if res is None:
    st.info("Choose a video in the sidebar, then select **Run analysis**.")
    st.stop()

s, paths = res["summary"], res["paths"]
st.subheader(f"{s['video']}: {s['side']} knee ({s['side_selection']} selection)")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Detection rate", s["table1"]["Detection rate"])
m2.metric("Frames", f"{s['total_frames']}", f"{s['duration_s']:.2f} s at {s['measured_fps']:.2f} fps",
          delta_color="off")
m3.metric("Repetitions", s["rep_count"])
m4.metric("Mean landmark visibility", "n/a" if s["mean_min_visibility"] is None
          else f"{s['mean_min_visibility']:.2f}")

left, right = st.columns([1, 1.6])
with left:
    st.markdown("**Anonymised video**")
    st.video(paths["video"])
    if not s["video_codec"].startswith("h264"):
        st.warning("ffmpeg was not found, so the video uses mp4v. Your browser may not play it. "
                   "Download it or install imageio-ffmpeg.")
with right:
    st.markdown("**Knee angle over time**")
    st.image(paths["png"], width="stretch")
    st.markdown("**Summary (Table 1 format)**")
    st.dataframe(pd.DataFrame([s["table1"]]), hide_index=True, width="stretch")

st.subheader("Feedback")
fb_col1, fb_col2 = st.columns(2)
with fb_col1:
    st.markdown(f"**Report rule, per frame** (target {s['target_min_angle']:.0f} ± {s['tolerance']:.0f}°)")
    pct = s["teammate_feedback_pct_of_detected_frames"]
    if pct:
        st.dataframe(pd.DataFrame({"Message": list(pct.keys()), "% of frames": list(pct.values())}),
                     hide_index=True, width="stretch")
    else:
        st.write("No frames with a detected pose.")
    st.caption("This rule rewards a bent knee. For a seated knee extension it says "
               "'bend further' at full extension. Use the repetition check for this exercise.")
with fb_col2:
    thr = s["ext_target"] - s["tolerance"]
    st.markdown(f"**Repetition check, extension** (peak ≥ {thr:.0f}°)")
    if s["reps"]:
        st.markdown(f"{s['reps_good']} of {s['rep_count']} repetitions reached the target.")
        for r in s["reps"]:
            css = "fb-good" if r["feedback"] == "Good range of motion" else "fb-fix"
            st.markdown(f"<div class='{css}'>Rep {r['rep']}: <b>{r['feedback']}</b> "
                        f"(peak {r['peak_deg']:.0f}°, range {r['rom_deg']:.0f}°, "
                        f"{r['duration_s']:.1f} s)</div>", unsafe_allow_html=True)
    else:
        st.write("No repetitions found. The angle must change by at least 30° to count.")

with st.expander("Per-frame data"):
    frame_df = pd.DataFrame(res["rows"])
    for c in frame_df.columns:
        if c not in ("feedback", "rep_feedback"):
            frame_df[c] = pd.to_numeric(frame_df[c], errors="coerce")
    st.dataframe(frame_df, hide_index=True, width="stretch", height=300)

d1, d2, d3, d4 = st.columns(4)
for col, key, label, mime in [(d1, "video", "Anonymised video", "video/mp4"),
                              (d2, "png", "Angle graph", "image/png"),
                              (d3, "csv", "Per-frame CSV", "text/csv"),
                              (d4, "json", "Summary JSON", "application/json")]:
    with open(paths[key], "rb") as f:
        col.download_button(f"Download {label.lower()}", f.read(), file_name=Path(paths[key]).name,
                            mime=mime, width="stretch")
