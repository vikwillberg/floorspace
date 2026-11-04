"""Shared paths, court geometry and visual constants."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "clips" / "raw"
MP4_DIR = ROOT / "clips" / "mp4"
CLIPS_CSV = ROOT / "clips" / "clips.csv"
WORK_DIR = ROOT / "work"          # big intermediate files (gitignored)
DATA_DIR = ROOT / "data"          # tidy tables for SQL / Power BI
OUT_DIR = ROOT / "outputs"        # rendered videos
DOCS_DIR = ROOT / "docs"          # GIF + charts for the README
WEIGHTS_DIR = ROOT / "weights"
FONTS_DIR = ROOT / "assets" / "fonts"

FPS = 30

# --- Court geometry (feet). x runs along the court (0 = left baseline as seen
# on the broadcast, 94 = right baseline); y runs across (0 = far sideline at the
# top of the picture, 50 = near sideline by the camera).
COURT_L = 94.0
COURT_W = 50.0
HALF_L = COURT_L / 2
HOOP_FROM_BASELINE = 5.25          # rim centre is 63 in from the baseline
THREE_R = 23.75                    # arc radius from rim centre
THREE_CORNER = 22.0                # corner 3 distance (lines 3 ft inside the sidelines)
LANE_W = 16.0
LANE_L = 19.0                      # baseline to free-throw line
FT_CIRCLE_R = 6.0
CENTER_CIRCLE_R = 6.0
RESTRICTED_R = 4.0
HALF_COURT_AREA = HALF_L * COURT_W  # 2,350 sq ft

TEAM_COLORS = {                    # BGR for OpenCV
    "MIA": (30, 38, 179),          # Heat red  #b3261e-ish
    "LAL": (40, 185, 253),         # Lakers gold #fdb927
}
SPACING_RED = (30, 38, 179)        # #b3261e
SPACING_GREEN = (80, 161, 18)      # #12a150
