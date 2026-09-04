from pathlib import Path

import pandas as pd


# ==================================================
# CONFIGURATION
# ==================================================

INPUT_PATH = Path(
    "data/test_landmarks_normalized.csv"
)

OUTPUT_PATH = Path(
    "data/test_landmarks_filled.csv"
)


# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(INPUT_PATH)


# ==================================================
# IDENTIFY LANDMARK COLUMNS
# ==================================================

feature_columns = [
    column
    for column in df.columns
    if (
        column.startswith("left_")
        or column.startswith("right_")
    )
    and column not in [
        "left_detected",
        "right_detected"
    ]
]


# ==================================================
# INTERPOLATE MISSING VALUES
# ==================================================

df[feature_columns] = (
    df[feature_columns]
    .interpolate(
        method="linear",
        limit_direction="both"
    )
)


# ==================================================
# SAVE RESULT
# ==================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ==================================================
# REPORT
# ==================================================

remaining_missing = (
    df[feature_columns]
    .isna()
    .sum()
    .sum()
)


print("\n")
print("=" * 60)
print("          MISSING VALUE HANDLING")
print("=" * 60)

print(
    f"\nTotal frames: "
    f"{len(df)}"
)

print(
    f"Landmark features: "
    f"{len(feature_columns)}"
)

print(
    f"Remaining missing values: "
    f"{remaining_missing}"
)

print(
    f"\nSaved to: "
    f"{OUTPUT_PATH}"
)

print("\n" + "=" * 60)