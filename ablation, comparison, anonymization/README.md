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

## Anonymization-before-pose test
Tests whether anonymizing the image BEFORE MediaPipe changes the knee angle
(face blur, background removal, silhouette, pixelation at 192 px and 96 px), compared with the raw frame.
    python anonymization_test.py --video Sagital_left.mov:left --video Sagital_right.mov:right --out_dir ablation
About 0.5 s per frame (about 4 to 5 min per video on one CPU core).
Outputs: ablation_summary.md/.csv (MAE, bias, 95% limits of agreement, r, detection, face-detectable %),
ablation_tradeoff.png, ablation_traces.png, <video>_anonymization_examples.jpg, <video>_ablation_frames.csv.
Note: "face detectable %" is a proxy. A detector can still find a blurred face shape that is not recognisable.
