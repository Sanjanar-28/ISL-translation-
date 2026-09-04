import pandas as pd
import numpy as np


PATH = "data/processed/full_landmark_dataset_clean.csv"


print("\n" + "=" * 70)
print("             FINAL DATASET CHECK")
print("=" * 70)

df = pd.read_csv(PATH)

feature_columns = [
    c for c in df.columns
    if c.startswith("feature_")
]

print("\nDataset shape:")
print(f"  Videos:           {len(df)}")
print(f"  Feature columns:  {len(feature_columns)}")

# ------------------------------------------------------------
# Basic checks
# ------------------------------------------------------------

print("\nBasic checks:")

print(
    f"  Expected videos: 862"
)

print(
    f"  Expected features/video: 2520"
)

# ------------------------------------------------------------
# Missing values
# ------------------------------------------------------------

missing = (
    df[feature_columns]
    .isna()
    .sum()
    .sum()
)

infinite = np.isinf(
    df[feature_columns].to_numpy()
).sum()

print("\nFeature checks:")

print(
    f"  Missing values:   {missing}"
)

print(
    f"  Infinite values:  {infinite}"
)

# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

print("\nClasses:")

print(
    f"  Number of classes: {df['class_id'].nunique()}"
)

print(
    f"  Labels: {df['label'].nunique()}"
)

# ------------------------------------------------------------
# Split
# ------------------------------------------------------------

print("\nSplit distribution:")

print(
    df["split"].value_counts()
)

# ------------------------------------------------------------
# Class distribution
# ------------------------------------------------------------

print("\nClass distribution:")

class_counts = (
    df.groupby(
        ["class_id", "label", "split"]
    )
    .size()
    .unstack(fill_value=0)
)

print(class_counts.to_string())

# ------------------------------------------------------------
# Duplicate videos
# ------------------------------------------------------------

duplicates = df.duplicated(
    subset=["label", "video_name"]
).sum()

print("\nDuplicate video entries:")
print(f"  {duplicates}")

# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

print("\n" + "=" * 70)

success = (
    len(df) == 862
    and len(feature_columns) == 2520
    and missing == 0
    and infinite == 0
    and df["class_id"].nunique() == 49
    and df["label"].nunique() == 49
    and duplicates == 0
)

if success:
    print("FINAL DATASET CHECK PASSED ✅")
    print("Dataset is ready for ML preparation.")
else:
    print("FINAL DATASET CHECK FAILED ❌")

print("=" * 70)