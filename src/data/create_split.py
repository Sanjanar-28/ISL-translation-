from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

METADATA_FILE = Path("data/video_metadata.csv")
OUTPUT_FILE = Path("data/video_metadata_split.csv")

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# --------------------------------------------------
# LOAD METADATA
# --------------------------------------------------

df = pd.read_csv(METADATA_FILE)

print(f"Total videos: {len(df)}")
print(f"Total classes: {df['class_id'].nunique()}")


# --------------------------------------------------
# FIRST SPLIT
# Train = 70%
# Temporary = 30%
# --------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=(1 - TRAIN_SIZE),
    stratify=df["class_id"],
    random_state=RANDOM_STATE
)


# --------------------------------------------------
# SECOND SPLIT
# Validation = 15%
# Test = 15%
#
# temp = 30%
# So validation = 50% of temp
# and test = 50% of temp
# --------------------------------------------------

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["class_id"],
    random_state=RANDOM_STATE
)


# --------------------------------------------------
# ADD SPLIT LABELS
# --------------------------------------------------

train_df = train_df.copy()
validation_df = validation_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
validation_df["split"] = "validation"
test_df["split"] = "test"


# --------------------------------------------------
# COMBINE
# --------------------------------------------------

final_df = pd.concat(
    [
        train_df,
        validation_df,
        test_df
    ],
    ignore_index=True
)


# Sort for easier inspection

final_df = final_df.sort_values(
    ["class_id", "split", "video_name"]
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n========== SPLIT SUMMARY ==========")

print(
    final_df["split"].value_counts()
)


print("\n========== CLASS / SPLIT COUNTS ==========")

print(
    pd.crosstab(
        final_df["class_id"],
        final_df["split"]
    )
)


print(
    f"\nSaved to: {OUTPUT_FILE}"
)