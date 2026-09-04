from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================
# CONFIGURATION
# ==================================================

INPUT_PATH = Path(
    "data/test_landmarks_filled.csv"
)

OUTPUT_PATH = Path(
    "data/test_landmarks_sampled.csv"
)

NUM_FRAMES = 20


# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(INPUT_PATH)


# ==================================================
# LANDMARK COLUMNS
# ==================================================

landmark_columns = [
    column
    for column in df.columns
    if (
        column.startswith("left_")
        or column.startswith("right_")
    )
    and column not in [
        "left_detected",
        "right_detected",
    ]
]


# ==================================================
# CHECK
# ==================================================

if len(landmark_columns) != 126:
    raise ValueError(
        f"Expected 126 landmark features, "
        f"found {len(landmark_columns)}"
    )


# ==================================================
# SAMPLE FRAME INDICES
# ==================================================

total_frames = len(df)

if total_frames < NUM_FRAMES:
    raise ValueError(
        f"Video has only {total_frames} frames. "
        f"Cannot sample {NUM_FRAMES} frames."
    )


sample_indices = np.linspace(
    0,
    total_frames - 1,
    NUM_FRAMES,
    dtype=int
)


sampled_df = df.iloc[
    sample_indices
].copy()


# ==================================================
# ADD SAMPLE INDEX
# ==================================================

sampled_df.insert(
    1,
    "sample_index",
    range(NUM_FRAMES)
)


# ==================================================
# SAVE
# ==================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

sampled_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ==================================================
# REPORT
# ==================================================

print("\n")
print("=" * 60)
print("             TEMPORAL SAMPLING")
print("=" * 60)

print(
    f"\nOriginal frames: "
    f"{total_frames}"
)

print(
    f"Sampled frames: "
    f"{NUM_FRAMES}"
)

print(
    f"Landmark features/frame: "
    f"{len(landmark_columns)}"
)

print(
    f"Total landmark values/video: "
    f"{NUM_FRAMES * len(landmark_columns)}"
)

print(
    "\nSelected original frame numbers:"
)

print(
    sampled_df["frame"].tolist()
)

print(
    f"\nSaved to: "
    f"{OUTPUT_PATH}"
)

print("\n" + "=" * 60)