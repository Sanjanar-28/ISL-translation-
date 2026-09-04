import pandas as pd


METADATA = "data/video_metadata_split.csv"
PROCESSED = "data/processed/full_landmark_dataset.csv"


original = pd.read_csv(METADATA)
processed = pd.read_csv(PROCESSED)


# ------------------------------------------------------------
# Create unique video identifiers
# ------------------------------------------------------------

original_ids = set(
    zip(
        original["label"],
        original["video_name"]
    )
)

processed_ids = set(
    zip(
        processed["label"],
        processed["video_name"]
    )
)


# ------------------------------------------------------------
# Missing videos
# ------------------------------------------------------------

missing = original_ids - processed_ids

extra = processed_ids - original_ids


print("\n" + "=" * 60)
print("             DATASET VERIFICATION")
print("=" * 60)

print(
    f"\nOriginal videos:  {len(original_ids)}"
)

print(
    f"Processed videos: {len(processed_ids)}"
)

print(
    f"Missing videos:   {len(missing)}"
)

print(
    f"Extra videos:     {len(extra)}"
)


# ------------------------------------------------------------
# Show missing videos
# ------------------------------------------------------------

if missing:

    print("\nMISSING VIDEOS:")

    for label, video in sorted(missing):

        print(
            f"  {label}/{video}"
        )

else:

    print(
        "\nAll original videos were processed! ✅"
    )


# ------------------------------------------------------------
# Feature verification
# ------------------------------------------------------------

feature_columns = [
    column
    for column in processed.columns
    if column.startswith("feature_")
]


print(
    f"\nFeature columns: "
    f"{len(feature_columns)}"
)


if len(feature_columns) == 2520:

    print(
        "Feature count: 2520 ✅"
    )

else:

    print(
        "Feature count is WRONG ❌"
    )


# ------------------------------------------------------------
# Missing feature values
# ------------------------------------------------------------

missing_values = (
    processed[feature_columns]
    .isna()
    .sum()
    .sum()
)


print(
    f"Missing feature values: "
    f"{missing_values}"
)


# ------------------------------------------------------------
# Split verification
# ------------------------------------------------------------

print("\nSplit counts:")

print(
    processed["split"]
    .value_counts()
)


print("\n" + "=" * 60)