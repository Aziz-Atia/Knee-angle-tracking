# Anonymized knee-angle pipeline (COMP825 Assignment 2, Volunteer 2)

## Install
    python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

## Process both videos and build the comparison
    python knee_pipeline.py --video Sagital_left.mov:left --video Sagital_right.mov:right --out_dir outputs --compare

## Rebuild the comparison from saved results (no reprocessing)
    python knee_pipeline.py --compare_from outputs/Sagital_left/Sagital_left_summary.json outputs/Sagital_right/Sagital_right_summary.json --out_dir outputs

## Run the UI
    streamlit run app.py
Put videos in ./videos (or set KNEE_VIDEO_DIR) to use "Pick from folder", or upload a file.

## Outputs per video
- `<name>_anonymized.mp4`: skeleton-only video (measured leg in magenta)
- `<name>_knee_angle.png`: angle-vs-time graph
- `<name>_knee_angle_data.csv`: per-frame data (teammate columns first)
- `<name>_summary_table.csv`: Table 1 format
- `<name>_summary.json`: all metrics, repetitions, timing, flagged checks

## Method notes
The core method is the teammate's (same MediaPipe settings, angle maths, feedback rule).
Verified: 0.0 deg angle difference and identical feedback text on both videos.
Changes and flagged bugs are listed at the top of knee_pipeline.py.
