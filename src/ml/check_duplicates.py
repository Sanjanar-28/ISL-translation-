import pandas as pd


DATASET = "data/processed/full_landmark_dataset_clean.csv"


def main():

    df = pd.read_csv(DATASET)

    feature_cols = [
        c for c in df.columns
        if c.startswith("feature_")
    ]

    print(f"Feature columns: {len(feature_cols)}")

    # Create a deterministic feature signature
    df["feature_hash"] = (
        df[feature_cols]
        .astype(str)
        .agg("|".join, axis=1)
        .map(hash)
    )

    # Find duplicated feature vectors
    counts = df["feature_hash"].value_counts()

    duplicate_hashes = counts[counts > 1].index

    duplicates = df[
        df["feature_hash"].isin(duplicate_hashes)
    ].copy()

    print("\nDuplicate feature groups:")
    print("=" * 70)

    for feature_hash, group in duplicates.groupby(
        "feature_hash"
    ):

        splits = set(group["split"])

        # Only show cross-split duplicates
        if len(splits) <= 1:
            continue

        print("\n" + "-" * 70)

        print(
            group[
                [
                    "video_name",
                    "label",
                    "split",
                    "original_frames",
                    "left_detections",
                    "right_detections",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()