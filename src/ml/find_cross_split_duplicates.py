import pandas as pd
import hashlib


INPUT_PATH = "data/processed/full_landmark_dataset_clean.csv"


def feature_hash(row, feature_cols):
    values = row[feature_cols].to_numpy(dtype="float32")
    return hashlib.sha256(values.tobytes()).hexdigest()


def main():

    df = pd.read_csv(INPUT_PATH)

    feature_cols = [
        c for c in df.columns
        if c.startswith("feature_")
    ]

    print("=" * 70)
    print("       CROSS-SPLIT DUPLICATE VIDEO ANALYSIS")
    print("=" * 70)

    print(f"Rows: {len(df)}")
    print(f"Features: {len(feature_cols)}")

    # Create hash without repeatedly inserting columns
    hashes = df[feature_cols].astype("float32").apply(
        lambda row: hashlib.sha256(
            row.to_numpy().tobytes()
        ).hexdigest(),
        axis=1
    )

    work = df[
        ["class_id", "label", "video_name", "split"]
    ].copy()

    work["feature_hash"] = hashes.values

    # Only hashes appearing in more than one split
    split_counts = (
        work.groupby("feature_hash")["split"]
        .nunique()
    )

    duplicate_hashes = split_counts[
        split_counts > 1
    ].index

    duplicates = work[
        work.feature_hash.isin(duplicate_hashes)
    ].sort_values(
        ["feature_hash", "split", "label", "video_name"]
    )

    print()
    print(
        f"Cross-split duplicate feature vectors: "
        f"{len(duplicate_hashes)}"
    )

    print()
    print("=" * 70)
    print("DETAILS")
    print("=" * 70)

    for h, group in duplicates.groupby("feature_hash"):

        print()
        print("-" * 70)

        for _, row in group.iterrows():

            print(
                f"{row['split']:12} | "
                f"{row['label']:15} | "
                f"{row['video_name']}"
            )

        labels = group["label"].unique()

        if len(labels) == 1:
            print("  --> SAME LABEL")
        else:
            print(
                "  --> DIFFERENT LABELS: "
                + ", ".join(labels)
            )


if __name__ == "__main__":
    main()