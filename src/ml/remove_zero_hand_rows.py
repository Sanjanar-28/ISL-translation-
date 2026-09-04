import pandas as pd


INPUT_PATH = "data/processed/full_landmark_dataset_clean.csv"
OUTPUT_PATH = "data/processed/full_landmark_dataset_ml.csv"


def main():

    df = pd.read_csv(INPUT_PATH)

    print("=" * 70)
    print("        REMOVING ZERO-HAND VIDEOS")
    print("=" * 70)

    zero_hand = (
        (df["left_detections"] == 0)
        & (df["right_detections"] == 0)
    )

    bad = df[zero_hand].copy()

    print(f"\nOriginal rows: {len(df)}")
    print(f"Zero-hand rows: {len(bad)}")

    if not bad.empty:
        print("\nVideos being removed:")
        print(
            bad[
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

    clean = df.loc[~zero_hand].copy()

    clean.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    print(f"Rows remaining: {len(clean)}")
    print(f"Output: {OUTPUT_PATH}")

    print("\nSplit counts:")
    print(
        clean["split"]
        .value_counts()
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    main()