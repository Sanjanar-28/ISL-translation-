from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/processed/full_landmark_dataset.csv"
)

OUTPUT_PATH = Path(
    "data/processed/full_landmark_dataset_clean.csv"
)


def main():

    print("\n" + "=" * 70)
    print("              FAST LANDMARK CLEANING")
    print("=" * 70)

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_PATH)

    feature_columns = [
        c for c in df.columns
        if c.startswith("feature_")
    ]

    print(f"Videos: {len(df)}")
    print(f"Feature columns: {len(feature_columns)}")

    if len(df) != 862:
        raise RuntimeError(
            f"Expected 862 videos, found {len(df)}"
        )

    if len(feature_columns) != 2520:
        raise RuntimeError(
            f"Expected 2520 features, found {len(feature_columns)}"
        )

    # --------------------------------------------------------
    # Count missing values
    # --------------------------------------------------------

    missing_before = (
        df[feature_columns]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Missing values before: {missing_before}"
    )

    # --------------------------------------------------------
    # Replace infinite values with NaN
    # --------------------------------------------------------

    df[feature_columns] = (
        df[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    # --------------------------------------------------------
    # Fill all remaining missing landmark values with 0
    # --------------------------------------------------------

    df[feature_columns] = (
        df[feature_columns]
        .fillna(0.0)
    )

    # --------------------------------------------------------
    # Final checks
    # --------------------------------------------------------

    missing_after = (
        df[feature_columns]
        .isna()
        .sum()
        .sum()
    )

    infinite_after = np.isinf(
        df[feature_columns].to_numpy()
    ).sum()

    print(
        f"Missing values after: {missing_after}"
    )

    print(
        f"Infinite values after: {infinite_after}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("\nSaving clean dataset...")

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    if (
        len(df) == 862
        and len(feature_columns) == 2520
        and missing_after == 0
        and infinite_after == 0
    ):

        print("DATASET CLEANING SUCCESSFUL! ✅")

    else:

        print("DATASET CHECK FAILED ❌")

    print("=" * 70)


if __name__ == "__main__":
    main()