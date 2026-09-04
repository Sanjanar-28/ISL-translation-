from pathlib import Path
import pandas as pd
import numpy as np


RESULT_FILE = Path(
    "data/ml_normalized/all_test_feature_comparison.csv"
)


def main():

    print("\n" + "=" * 70)
    print("              FEATURE MISMATCH ANALYSIS")
    print("=" * 70)

    if not RESULT_FILE.exists():
        raise FileNotFoundError(
            f"Comparison results not found:\n{RESULT_FILE}"
        )

    df = pd.read_csv(RESULT_FILE)

    print("\nColumns:")
    print(df.columns.tolist())

    print(f"\nTotal rows: {len(df)}")

    # ------------------------------------------------------------
    # Find MAE column
    # ------------------------------------------------------------

    mae_candidates = [
        "mae",
        "MAE",
        "mean_absolute_difference",
        "mean_absolute_error"
    ]

    mae_col = None

    for col in mae_candidates:
        if col in df.columns:
            mae_col = col
            break

    if mae_col is None:
        raise ValueError(
            "Could not find MAE column.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # ------------------------------------------------------------
    # Convert to numeric
    # ------------------------------------------------------------

    df[mae_col] = pd.to_numeric(
        df[mae_col],
        errors="coerce"
    )

    df = df.dropna(
        subset=[mae_col]
    ).copy()

    # ------------------------------------------------------------
    # Print high precision statistics
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("                  MAE STATISTICS")
    print("=" * 70)

    print(
        f"\nMinimum MAE : {df[mae_col].min():.15f}"
    )

    print(
        f"Maximum MAE : {df[mae_col].max():.15f}"
    )

    print(
        f"Mean MAE    : {df[mae_col].mean():.15f}"
    )

    print(
        f"Median MAE  : {df[mae_col].median():.15f}"
    )

    # ------------------------------------------------------------
    # Exact / near-exact / significant
    # ------------------------------------------------------------

    exact = df[mae_col] == 0

    near_exact = df[mae_col] <= 1e-7

    small_difference = (
        (df[mae_col] > 1e-7) &
        (df[mae_col] <= 1e-4)
    )

    significant = df[mae_col] > 1e-4

    print("\n" + "=" * 70)
    print("                  DIFFERENCE GROUPS")
    print("=" * 70)

    print(
        f"\nExactly zero      : "
        f"{exact.sum()}/{len(df)}"
    )

    print(
        f"MAE <= 1e-7      : "
        f"{near_exact.sum()}/{len(df)}"
    )

    print(
        f"1e-7 < MAE <=1e-4: "
        f"{small_difference.sum()}/{len(df)}"
    )

    print(
        f"MAE > 1e-4       : "
        f"{significant.sum()}/{len(df)}"
    )

    # ------------------------------------------------------------
    # Sort by MAE
    # ------------------------------------------------------------

    worst = df.sort_values(
        mae_col,
        ascending=False
    )

    print("\n" + "=" * 70)
    print("                 ALL NON-ZERO DIFFERENCES")
    print("=" * 70)

    nonzero = worst[
        worst[mae_col] > 0
    ]

    if len(nonzero) == 0:

        print("\nAll features match exactly.")

    else:

        for _, row in nonzero.iterrows():

            label = row.get(
                "label",
                "unknown"
            )

            video = row.get(
                "video_name",
                "unknown"
            )

            match = row.get(
                "matching_percentage",
                row.get(
                    "match_percentage",
                    ""
                )
            )

            print(
                f"{str(label):18s} "
                f"{str(video):35s} "
                f"MAE={row[mae_col]:.15f} "
                f"Match={match}"
            )

    # ------------------------------------------------------------
    # Save detailed report
    # ------------------------------------------------------------

    output = Path(
        "data/ml_normalized/"
        "feature_mismatch_analysis.csv"
    )

    worst.to_csv(
        output,
        index=False
    )

    print("\n" + "=" * 70)
    print("Analysis saved to:")
    print(output)
    print("=" * 70)


if __name__ == "__main__":
    main()